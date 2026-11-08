"""토이 환경으로 인코더/평가기/GRPO 의 핵심 성질을 확인한다."""

import numpy as np
import torch

from sts2ai import schema as S
from sts2ai.agents.evaluator import IGNORE, survival_dp
from sts2ai.batching import collate_combat, collate_run
from sts2ai.envs.toy import ToyRun
from sts2ai.train.evaluator_train import RunRecorder, make_labels
from sts2ai.train.grpo import advantages, play
from sts2ai.train.loop import Agent

torch.manual_seed(0)


def _run_at_combat(seed=0):
    run = ToyRun()
    run.reset(seed)
    run.choose_node(run.map_choices()[0])  # 0층은 항상 일반 전투
    assert run.phase == "combat"
    return run


def test_encoders_shapes():
    run = _run_at_combat()
    agent = Agent(run.vocab, 32)
    batch = collate_run([run.run_state(), run.run_state()], run.vocab)
    ev = agent.evaluator(batch)
    V = batch["node_type"].shape[1]
    assert ev.hazard.shape == (2, V) and ev.survive.shape == (2, V)
    assert ev.p_act.shape == (2, run.vocab.n_acts)
    out = agent.policy(collate_combat([run.combat_obs()]))
    assert out.action_logits.shape == (1, S.MAX_ACTIONS)


def test_deck_encoder_is_order_invariant():
    run = _run_at_combat()
    agent = Agent(run.vocab, 32).evaluator.eval()
    s1 = run.run_state()
    s2 = run.run_state()
    s2.deck = list(reversed(s2.deck))
    b1, b2 = collate_run([s1], run.vocab), collate_run([s2], run.vocab)
    d1, _ = agent.encoder.deck(b1)
    d2, _ = agent.encoder.deck(b2)
    assert torch.allclose(d1, d2, atol=1e-5)


def test_survival_dp_takes_safest_path():
    # 0 -> 1 -> 3, 0 -> 2 -> 3. 노드 1 은 위험, 2 는 안전.
    adj = torch.zeros(1, 4, 4)
    adj[0, 0, 1] = adj[0, 0, 2] = adj[0, 1, 3] = adj[0, 2, 3] = 1
    h = torch.tensor([[0.1, 0.5, 0.0, 0.2]])
    s = survival_dp(h, adj, torch.ones(1, 4))
    assert torch.allclose(s[0, 3], torch.tensor(0.8))
    assert torch.allclose(s[0, 1], torch.tensor(0.4))
    assert torch.allclose(s[0, 2], torch.tensor(0.8))
    assert torch.allclose(s[0, 0], torch.tensor(0.9 * 0.8))


def test_future_head_is_empty_in_last_act():
    run = _run_at_combat()
    agent = Agent(run.vocab, 32)
    st = run.run_state()
    st.act = run.vocab.n_acts - 1
    ev = agent.evaluator(collate_run([st], run.vocab))
    assert not ev.future_mask.any()
    assert float(ev.future_death.detach()) == 0.0
    st.act = 0
    ev = agent.evaluator(collate_run([st], run.vocab))
    assert ev.future_mask[0].tolist() == [False, True, True]


def test_fork_does_not_touch_real_run():
    run = _run_at_combat()
    agent = Agent(run.vocab, 32)
    before = run.combat_obs()
    tr = play(agent.policy, run.fork_combat(123))
    assert tr.actions
    after = run.combat_obs()
    assert np.array_equal(before.types, after.types) and np.allclose(before.feats, after.feats)


def test_advantages():
    a = advantages([0.0, 0.5, 0.5, 1.0], "zscore")
    assert abs(sum(a)) < 1e-6 and a[0] < a[1] < a[3]
    b = advantages([0.0, 0.2, 0.9, 0.3], "best_vs_rest")
    assert b[2] == 1.0 and abs(sum(b)) < 1e-6
    assert advantages([0.3, 0.3, 0.3]) == [0.0, 0.0, 0.0]


def test_labels_are_censored_after_death():
    rec = RunRecorder(3)
    st = _run_at_combat().run_state()
    rec.snapshot(st)
    rec.visit(0, 1)
    rec.visit(0, 4)  # 여기서 사망
    snaps = rec.finish(died=True, death_act=0, death_node=4)
    node, act = make_labels(snaps, 20, 3)
    assert node[0, 1] == 0 and node[0, 4] == 1
    assert (node[0, [0, 2, 3]] == IGNORE).all()
    assert (act[0] == IGNORE).all()  # 2, 3막에 도착하지 못했으니 라벨 없음


def test_log_value_separates_near_zero_states():
    """P(클리어) 가 둘 다 0 에 가까워도 log V 로는 비교가 된다."""
    from sts2ai.agents.evaluator import log_survival_dp
    adj = torch.zeros(1, 2, 2)
    adj[0, 0, 1] = 1
    a = log_survival_dp(torch.tensor([[-20.0, -20.0]]), adj, torch.ones(1, 2))
    b = log_survival_dp(torch.tensor([[-21.0, -20.0]]), adj, torch.ones(1, 2))
    assert a[0, 0].exp() < 1e-16 and b[0, 0].exp() < 1e-16  # 확률로는 둘 다 사실상 0
    assert float(a[0, 0] - b[0, 0]) == 1.0  # log 로는 정확히 1 nat 차이


def test_policy_sees_next_node_through_run_token():
    """같은 전투 상태라도 다음 노드(휴식 vs 엘리트)가 다르면 정책 입력이 달라져야 한다."""
    from sts2ai import schema as S

    run = _run_at_combat()
    agent = Agent(run.vocab, 32)
    agent.policy.eval()
    state = run.run_state()
    succ = state.map.successors()[state.map.current]
    rest, elite = S.NODE_REST, S.NODE_ELITE
    a, b = run.run_state(), run.run_state()
    a.map.node_types = list(a.map.node_types)
    b.map.node_types = list(b.map.node_types)
    for v in succ:
        a.map.node_types[v] = rest
        b.map.node_types[v] = elite
    batch = collate_combat([run.combat_obs()])
    la = agent.policy(batch, agent.run_vec(a).unsqueeze(0)).action_logits
    lb = agent.policy(batch, agent.run_vec(b).unsqueeze(0)).action_logits
    legal = batch["legal"][0]
    assert not torch.allclose(la[0, legal], lb[0, legal])
    # RUN 토큰을 붙여도 행동 번호 규약(토큰 1+i = 행동 i)과 출력 모양은 그대로
    assert la.shape == (1, S.MAX_ACTIONS)
    assert torch.equal(la[0] > -1e8, legal)


class _FakeEvaluation:
    def __init__(self, logv, hazard):
        self.log_run_value = logv
        self.hazard = hazard


class _StrikeHater:
    """가짜 평가기: 덱의 타격(id 1) 한 장마다 log V −0.5, Heavy Blade(id 6) 가 있으면 +2."""

    def __call__(self, batch):
        ids = batch["deck_ids"] * batch["deck_mask"]
        logv = -0.5 * (ids == 1).sum(-1).double() + 2.0 * (ids == 6).any(-1).double()
        return _FakeEvaluation(logv, torch.zeros(batch["node_type"].shape))


def _run_at_screen(make):
    run = ToyRun()
    run.reset(0)
    run._open(make(run))
    return run


def test_chooser_removes_the_card_the_evaluator_dislikes():
    from sts2ai.agents.chooser import ExpectedValueChooser

    run = _run_at_screen(lambda r: {"kind": "deck", "op": "remove", "eligible": list(range(len(r.deck)))})
    ch = ExpectedValueChooser(_StrikeHater(), run.vocab, samples=4)
    a = ch.choose(run, np.random.default_rng(0))
    assert run.deck[a].id == 1  # 타격을 뺀다


def test_chooser_follows_shop_removal_into_the_next_screen():
    """'카드 제거' 구입의 가치는 다음 화면에서 가장 좋은 카드를 뺐을 때의 가치여야 한다."""
    from sts2ai.agents.chooser import ExpectedValueChooser

    run = _run_at_screen(lambda r: {"kind": "shop", "cards": [2, 7], "bought": [False, False], "removed": False})
    ch = ExpectedValueChooser(_StrikeHater(), run.vocab, samples=2)
    vals = ch.evaluate(run, np.random.default_rng(0))
    remove, leave = 2, 3
    assert abs((vals[remove] - vals[leave]) - 0.5) < 1e-6  # 타격 한 장 제거 = +0.5
    assert ch.choose(run, np.random.default_rng(0)) == remove


def test_chooser_values_random_transform_by_expectation():
    """무작위 변환: 결과를 몰라도 '가끔 Heavy Blade 가 나온다' 를 기대값으로 반영한다."""
    from sts2ai.agents.chooser import ExpectedValueChooser

    run = _run_at_screen(lambda r: {"kind": "event_transform"})
    ch = ExpectedValueChooser(_StrikeHater(), run.vocab, samples=32, follow_samples=32)
    vals = ch.evaluate(run, np.random.default_rng(0))
    transform, leave = 0, 1
    # 타격 하나를 변환하면 최소 +0.5, 8장 중 1장 꼴로 Heavy Blade 가 나와 +2 가 더해진다
    assert vals[transform] > vals[leave] + 0.5


def _tiny_cfg(out_dir, iterations):
    from sts2ai.train.loop import Config

    return Config(iterations=iterations, runs_per_iter=1, group_size=2, grpo_every_n_combats=3, d_model=16,
                  eval_epochs=1, grpo_epochs=1, warmup_snapshots=3, val_every=2, out_dir=str(out_dir),
                  freeze_policy=False)


def _train_tiny(out_dir, iterations, resume=False):
    from sts2ai.train.loop import train

    return train(ToyRun(), _tiny_cfg(out_dir, iterations), log=lambda *_: None, resume=resume)


def test_resume_matches_uninterrupted_training(tmp_path):
    """2 반복 + 이어서 2 반복 == 끊기지 않은 4 반복 (가중치·기록 모두)."""
    import json
    import random

    def seed_globals():  # 학습 코드가 쓰는 전역 난수(셔플)도 같은 출발점에서
        random.seed(1)
        np.random.seed(1)

    seed_globals()
    full = _train_tiny(tmp_path / "full", 4)
    seed_globals()
    _train_tiny(tmp_path / "split", 2)
    random.seed(999)  # 끊긴 사이에 다른 코드가 난수를 써도 체크포인트에서 되살린다
    np.random.seed(999)
    torch.manual_seed(999)
    resumed = _train_tiny(tmp_path / "split", 4, resume=True)

    a, b = full.state_dict(), resumed.state_dict()
    for part in ("evaluator", "policy"):
        for k in a[part]:
            assert torch.equal(a[part][k], b[part][k]), (part, k)
    hist = lambda d: [{k: v for k, v in r.items() if k != "sec"} for r in json.loads((d / "history.json").read_text())]
    assert hist(tmp_path / "full") == hist(tmp_path / "split")
    assert len(hist(tmp_path / "split")) == 4


def test_resume_rejects_changed_config(tmp_path):
    import pytest

    from sts2ai.train.loop import train

    _train_tiny(tmp_path, 1)
    cfg = _tiny_cfg(tmp_path, 2)
    cfg.lr_policy = 1e-2
    with pytest.raises(ValueError, match="lr_policy"):
        train(ToyRun(), cfg, log=lambda *_: None, resume=True)


def test_batched_rollout_matches_one_by_one():
    """여러 전투를 한 배치로 같이 굴려도(greedy) 하나씩 굴린 것과 행동·log 확률이 같다. 패딩 영향 없음."""
    from sts2ai.train.grpo import drive, play_steps

    agent = Agent(ToyRun.vocab, 32)
    runs = [_run_at_combat(seed) for seed in range(3)]
    rvs = [agent.run_vec(r.run_state()) for r in runs]
    alone = [play(agent.policy, r.fork_combat(k), greedy=True, run_vec=rv)
             for r, rv in zip(runs, rvs) for k in range(3)]
    together = [t for ts in drive(agent.policy, [
        play_steps([r.fork_combat(k) for k in range(3)], greedy=True, run_vec=rv) for r, rv in zip(runs, rvs)
    ]) for t in ts]
    for a, b in zip(alone, together):
        assert a.actions == b.actions and a.targets == b.targets and a.won == b.won and a.hp == b.hp
        assert np.allclose(a.logp, b.logp, atol=1e-4)


def test_parallel_runs_match_sequential_runs():
    """런 여러 판을 나란히 진행해도 판마다 따로 돌린 결과와 같다 (greedy, 런마다 난수 생성기)."""
    from sts2ai.train.grpo import drive
    from sts2ai.train.loop import Config, play_run, play_run_steps

    agent = Agent(ToyRun.vocab, 32)
    cfg = Config()
    seeds = [3, 4, 5]
    alone = [play_run(agent, ToyRun(), cfg, np.random.default_rng(s), s, use_evaluator=True,
                      collect_grpo=False, greedy=True)[2] for s in seeds]
    together = [o[2] for o in drive(agent.policy, [
        play_run_steps(agent, ToyRun(), cfg, np.random.default_rng(s), s, use_evaluator=True,
                       collect_grpo=False, greedy=True) for s in seeds])]
    assert alone == together


def test_drive_batches_runs_and_forks_together():
    """런 2판 × 분기 4: 한 번의 정책 호출에 요청이 8개까지 모이고, 평균으로도 하나씩보다 훨씬 크다."""
    from sts2ai.train.grpo import drive
    from sts2ai.train.loop import Config, play_run_steps

    agent = Agent(ToyRun.vocab, 32)
    sizes = []
    real_act = agent.policy.act

    def counting_act(batch, greedy=False, run_vec=None):
        sizes.append(batch["types"].shape[0])
        return real_act(batch, greedy=greedy, run_vec=run_vec)

    agent.policy.act = counting_act
    cfg = Config(group_size=4)
    outs = drive(agent.policy, [play_run_steps(agent, ToyRun(), cfg, np.random.default_rng(s), s,
                                               use_evaluator=False) for s in (0, 1)])
    assert all(len(g) == 4 for _, groups, _ in outs for g in groups)
    assert max(sizes) == 2 * 4
    # 분기마다 전투 길이가 달라 끝으로 갈수록 배치가 줄고, 진짜 전투는 런마다 한 개씩이라 평균은 8 보다 작다
    assert sum(sizes) / len(sizes) > 2


# ---- 전투만 따로 학습 ----

def test_combat_score_order():
    """이기면 무조건 진 것보다 높고, HP 를 덜 잃을수록·포션을 덜 쓸수록·최대 HP 가 오를수록 높다."""
    from sts2ai.train.combat_train import CombatConfig, combat_score

    cfg = CombatConfig()
    worst_win = combat_score(True, 125, 1, 125, 125, 9, 1.0, cfg)
    best_loss = combat_score(False, 125, 0, 125, 125, 0, 1.0, cfg)
    assert worst_win > best_loss
    base = combat_score(True, 50, 40, 60, 60, 1, 1.0, cfg)
    assert combat_score(True, 50, 45, 60, 60, 1, 1.0, cfg) > base  # HP 덜 잃음
    assert combat_score(True, 50, 40, 60, 60, 0, 1.0, cfg) > base  # 포션 덜 씀
    assert combat_score(True, 50, 40, 60, 63, 1, 1.0, cfg) > base  # 최대 HP 상승
    assert combat_score(False, 50, 0, 60, 60, 0, 0.8, cfg) > combat_score(False, 50, 0, 60, 60, 0, 0.2, cfg)
    cfg.loss_ranking = False
    assert combat_score(False, 50, 0, 60, 60, 0, 0.8, cfg) == combat_score(False, 50, 0, 60, 60, 0, 0.2, cfg)


def test_combat_scenarios_cover_pool_within_ranges():
    from sts2ai.train.combat_train import CombatConfig, scenarios_for_round

    env, cfg = ToyRun(), CombatConfig()
    rng = np.random.default_rng(0)
    scs = [sc for _ in range(20) for sc in scenarios_for_round(env, rng, cfg)]
    assert {(sc.encounter, sc.act) for sc in scs[:18]} == set(env.combat_pool())
    for sc in scs:
        assert 10 <= len(sc.deck) <= 50 and 0 <= len(sc.potions) <= 9
        assert 1 <= sc.hp <= sc.max_hp <= 125
    assert min(sc.max_hp for sc in scs) <= 10 and max(sc.max_hp for sc in scs) >= 115
    assert min(len(sc.deck) for sc in scs) <= 12 and max(len(sc.deck) for sc in scs) >= 48
    assert {cid for sc in scs for cid, _ in sc.deck} == set(env.card_pool())


def test_combat_group_shares_seed():
    """GRPO 그룹의 시행들은 같은 전투(같은 셔플·적 HP)에서 출발한다."""
    from sts2ai.train.combat_train import CombatConfig, scenarios_for_round

    env = ToyRun()
    sc = scenarios_for_round(env, np.random.default_rng(1), CombatConfig())[0]
    a, b = (env.make_combat(sc.encounter, sc.act, sc.deck, sc.hp, sc.max_hp, sc.potions, sc.relics, sc.seed)
            for _ in range(2))
    oa, ob = a.observe(), b.observe()
    assert np.array_equal(oa.ids, ob.ids) and np.allclose(oa.feats, ob.feats)
    assert [e.hp for e in a.c.enemies] == [e.hp for e in b.c.enemies]
    assert [c.id for c in a.c.draw] == [c.id for c in b.c.draw]
    assert np.isclose(oa.feats[0, S.F_HP_ABS], sc.hp / 50)


def test_combat_training_resumes_exactly(tmp_path):
    import random

    from sts2ai.train.combat_train import CombatConfig, train_combat

    def run(out, iters, resume=False):
        cfg = CombatConfig(iterations=iters, group_size=2, deck_max=15, d_model=16, grpo_epochs=1,
                           eval_every=0, out_dir=str(out))
        return train_combat(ToyRun(), cfg, log=lambda *_: None, resume=resume)

    random.seed(0), np.random.seed(0)
    full = run(tmp_path / "full", 2)
    random.seed(0), np.random.seed(0)
    run(tmp_path / "split", 1)
    torch.manual_seed(5)
    resumed = run(tmp_path / "split", 2, resume=True)
    for k, v in full.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k]), k


def test_run_training_starts_from_combat_policy(tmp_path):
    from sts2ai.train.combat_train import CombatPolicy
    from sts2ai.encoders import SharedEmbeddings
    from sts2ai.train.loop import train

    pol = CombatPolicy(SharedEmbeddings(ToyRun.vocab, 16))
    torch.save(pol.state_dict(), tmp_path / "combat_policy.pt")
    cfg = _tiny_cfg(tmp_path / "run", 0)
    cfg.freeze_policy = True
    cfg.init_policy = str(tmp_path / "combat_policy.pt")
    agent = train(ToyRun(), cfg, log=lambda *_: None)
    for k, v in pol.state_dict().items():
        assert torch.equal(v, agent.policy.state_dict()[k]), k
    # 공유 임베딩이라 평가기 쪽 임베딩도 같이 바뀐다
    assert agent.emb is agent.policy.encoder.emb and torch.equal(agent.emb.card.weight, pol.encoder.emb.card.weight)


def test_focus_sampling_covers_all_and_favours_deadly():
    """모든 (조우, 막) 과 반드시 싸우고, 사망률이 높은 쪽을 더 많이 뽑는다."""
    from sts2ai.train.combat_train import CombatConfig, plan_iteration

    cfg = CombatConfig(focus_per_iter=2000)
    death = np.full(18, 0.05)
    death[[4, 9]] = [0.9, 0.45]
    idx = plan_iteration(18, death, np.random.default_rng(0), cfg)
    assert sorted(idx[:18]) == list(range(18))  # 고루 한 바퀴
    counts = np.bincount(idx[18:], minlength=18)
    assert counts[4] > 1.6 * counts[9] > 3 * counts[0]
    assert counts.min() > 0  # 사망률 바닥값 덕에 전부 가끔은 뽑힌다


def test_death_rates_track_losses():
    from sts2ai.train.combat_train import Scenario, update_death_rates
    from sts2ai.train.grpo import Trajectory

    pool = [(1, 0), (2, 0)]
    death = np.array([0.5, 0.5])
    sc = lambda e: Scenario(e, 0, [], 1, 1, [], 0)
    groups = [[Trajectory(won=False)] * 4, [Trajectory(won=True)] * 4]
    for _ in range(20):
        update_death_rates(death, pool, groups, [sc(1), sc(2)], 0.3)
    assert death[0] > 0.99 and death[1] < 0.01


def test_frozen_policy_does_not_change_while_evaluator_trains(tmp_path):
    """다른 모델(평가기)을 학습해도 전투 정책은 가중치도, 임베딩도 그대로다."""
    from sts2ai.encoders import SharedEmbeddings
    from sts2ai.train.combat_train import CombatPolicy
    from sts2ai.train.loop import train

    pol = CombatPolicy(SharedEmbeddings(ToyRun.vocab, 16))
    torch.save(pol.state_dict(), tmp_path / "p.pt")
    cfg = _tiny_cfg(tmp_path / "run", 3)
    cfg.freeze_policy, cfg.init_policy = True, str(tmp_path / "p.pt")
    agent = train(ToyRun(), cfg, log=lambda *_: None)
    for k, v in pol.state_dict().items():
        assert torch.equal(v, agent.policy.state_dict()[k]), k
    assert agent.embeddings_split and not agent.policy_uses_run_vec
    assert not torch.equal(agent.evaluator.encoder.emb.card.weight, pol.encoder.emb.card.weight)  # 평가기는 배움
    hist = __import__("json").loads((tmp_path / "run" / "history.json").read_text())
    assert all("grpo_steps" not in r for r in hist)  # GRPO 분기·업데이트 없음

    # 저장한 에이전트를 다시 불러도 나뉜 임베딩·RUN 토큰 설정이 유지된다 (scripts/evaluate.py 경로)
    fresh = Agent(ToyRun.vocab, 16)
    fresh.load_state_dict(torch.load(tmp_path / "run" / "agent.pt"))
    assert fresh.embeddings_split and not fresh.policy_uses_run_vec
    for k, v in agent.evaluator.state_dict().items():
        assert torch.equal(v, fresh.evaluator.state_dict()[k]), k
    for k, v in pol.state_dict().items():
        assert torch.equal(v, fresh.policy.state_dict()[k]), k


def test_freeze_requires_a_trained_policy(tmp_path):
    import pytest

    from sts2ai.train.loop import train

    cfg = _tiny_cfg(tmp_path, 1)
    cfg.freeze_policy = True
    with pytest.raises(ValueError, match="init-policy"):
        train(ToyRun(), cfg, log=lambda *_: None)
