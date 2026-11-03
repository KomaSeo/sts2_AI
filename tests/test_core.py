"""토이 환경으로 인코더/평가기/GRPO 의 핵심 성질을 확인한다."""

import numpy as np
import pytest
import torch

from sts2ai import schema as S
from sts2ai.agents.evaluator import IGNORE, survival_dp
from sts2ai.batching import collate_combat, collate_run
from sts2ai.envs.base import PHASE_COMBAT
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


def test_choice_policy_scores_every_option_kind():
    """선택 정책이 맵 노드·카드·덱 카드 선택지를 모두 점수 매기고, 기울기가 트렁크까지 흐른다."""
    from sts2ai.agents.choice_policy import collate_options
    from sts2ai.batching import collate_run
    from sts2ai.schema import Card, ChoiceOption
    from sts2ai.train.choice import map_options

    run = ToyRun()
    run.reset(0)
    agent = Agent(run.vocab, 32)
    state = run.run_state()
    opts_map = map_options(state)
    opts_other = [ChoiceOption(0, "card.take", card=Card(3)), ChoiceOption(1, "select.remove", deck_index=2),
                  ChoiceOption(2, "card.skip")]
    batch = collate_run([state, state], run.vocab)
    enc = agent.choice.trunk(agent.evaluator.encoder, batch, grad=True)
    logits = agent.choice(enc, collate_options([opts_map, opts_other], run.vocab))
    assert torch.isfinite(logits[0, : len(opts_map)]).all() and torch.isfinite(logits[1, :3]).all()
    if len(opts_map) < 3:
        assert (logits[0, len(opts_map):] < -1e8).all()  # 없는 선택지는 막힌다
    logits[1, :3].sum().backward()
    deck_grad = agent.evaluator.encoder.deck.body.layers[0].linear1.weight.grad
    assert deck_grad is not None and deck_grad.abs().sum() > 0  # 덱 카드 벡터를 거쳐 덱 인코더까지


def test_choice_rollout_compares_clones():
    """선택 복제 비교가 선택지마다 학습 샘플을 만들고, 원래 런은 건드리지 않는다."""
    from sts2ai.train.choice import rollout_choice_group, screen_options
    from sts2ai.train.loop import Config

    cfg = Config(d_model=32, choice_group_size=4, choice_horizon=1)
    run = ToyRun()
    run.reset(5)
    agent = Agent(run.vocab, 32)
    state = run.run_state()
    before = (run.floor, run.hp, run.current_node)
    opts = screen_options(run, state)
    samples = rollout_choice_group(run, agent, cfg, np.random.default_rng(0), state, opts, use_evaluator=False)
    assert (run.floor, run.hp, run.current_node) == before
    assert all(0 <= s.action < len(opts) for s in samples)
    assert all(s.advantage != 0.0 for s in samples)


def test_winged_boots_jumps_in_survival():
    """날개 달린 부츠: 남은 횟수만큼 간선 밖 다음 줄로 건너뛰는 길도 생존 계산에 들어가고, 횟수가 줄면 못 쓴다."""
    from sts2ai.agents.evaluator import log_survival_dp, log_survival_dp_jumps
    from sts2ai.batching import collate_maps

    # 0행: A(0)   1행: B(1) 위험, C(2) 안전 (A 와 간선 없음)   2행: 보스 D(3)  -- A→B→D, C→D
    m = S.MapGraph(node_types=[1, 2, 3, 6], rows=[0, 1, 1, 2], cols=[0, 0, 2, 1],
                   edges=[(0, 1), (1, 3), (2, 3)], current=0, n_rows=3, choices=[1, 2])
    b = collate_maps([m], [2], 3, [1])  # 마지막 막이라 가상 줄 없음, 부츠 1회 남음
    assert b["jump_adj"][0, 0, 2] == 1 and b["jump_adj"][0, 0, 1] == 0  # 간선이 있는 곳은 건너뛰기가 아니다
    assert b["jump_choice"][0].tolist() == [False, False, True, False]
    keep = torch.log(torch.tensor([[1.0, 0.1, 1.0, 0.9]]))  # B 에서 90% 사망
    mask = b["node_mask"].float()
    plain = log_survival_dp(keep, b["adj"], mask)
    jumps = log_survival_dp_jumps(keep, b["adj"], b["jump_adj"], mask, 1)
    assert torch.allclose(jumps[..., 0], plain)  # 횟수 0 = 예전 계산
    assert float(jumps[0, 0, 1].exp()) > 0.85 > float(plain[0, 0].exp())  # A 에서 1회 남으면 C 로 건너뛰어 0.9


def test_future_acts_are_chains_with_per_floor_labels():
    """미래 막은 N/A 노드 줄 + 보스 N/A 로 붙고, 미래 막에서 지나간 층마다 라벨이 붙는다."""
    run = ToyRun()
    run.reset(0)
    state = run.run_state()
    batch = collate_run([state], run.vocab)
    n_real, R = len(state.map), state.map.n_rows
    assert batch["node_type"].shape[1] == n_real + 2 * R  # 1막 실제 맵 + 2막 줄 + 3막 줄
    assert batch["node_type"][0, n_real + R - 1] == S.NODE_BOSS_NA
    assert batch["node_type"][0, n_real] == S.NODE_NA
    boss = [i for i, t in enumerate(state.map.node_types) if t == S.NODE_BOSS][0]
    assert batch["adj"][0, boss, n_real] == 1.0  # 1막 보스 → 2막 줄 첫 노드
    assert batch["adj"][0, n_real + R - 1, n_real + R] == 1.0  # 2막 보스 N/A → 3막 줄

    # 1막을 넘기고 2막 3층에서 죽은 판: 1막 스냅숏의 2막 줄 0~2층 = 0, 3층 = 1, 그 뒤 라벨 없음
    rec = RunRecorder(run.vocab.n_acts)
    rec.snapshot(state)
    rec.visit(0, 0, 0)
    rec.visit(0, boss, state.map.rows[boss], True)
    for r in range(4):
        rec.visit(1, 100 + r, r)
    snaps = rec.finish(True, 1, 103)
    node_l, _ = make_labels(snaps, batch["node_type"].shape[1], run.vocab.n_acts)
    chain = node_l[0, n_real:n_real + R].tolist()
    assert chain[:4] == [0.0, 0.0, 0.0, 1.0] and all(x == IGNORE for x in chain[4:])
    assert all(x == IGNORE for x in node_l[0, n_real + R:].tolist())  # 3막은 못 갔다

    agent = Agent(run.vocab, 32)
    ev = agent.evaluator(batch)
    # log V 는 가상 줄까지 이어진 생존 계산: 이번 막만 본 값보다 작거나 같다
    assert float(ev.log_run_value[0]) <= float(torch.log1p(-ev.act_death[0])) + 1e-5


def test_act_start_pool_restores_and_continues():
    """막 시작 상태를 저장해 되살리면 같은 막·층에서 이어지고, 원래 런과는 독립이다."""
    from sts2ai.train.loop import Config, play_run

    run = ToyRun()
    run.reset(11)
    run.choose_node(run.map_choices()[0])
    token = run.snapshot()
    other = ToyRun()
    other.restore(token, seed=99)
    assert (other.act, other.floor, other.hp, other.phase) == (run.act, run.floor, run.hp, run.phase)
    before = (run.combat.turn, len(run.combat.hand), run.combat.energy)
    for _ in range(5):  # 되살린 쪽을 진행해도 원래 런은 그대로
        if other.phase != PHASE_COMBAT:
            break
        other.combat_step(int(np.flatnonzero(other.combat_obs().legal)[0]))
    assert (run.combat.turn, len(run.combat.hand), run.combat.energy) == before

    # 2막 시작 상태처럼 만든 토큰에서 시작하면 결과에 시작 막이 남는다
    run.reset(12)
    run.act = 1
    token = run.snapshot()
    cfg = Config(d_model=32, group_size=2)
    agent = Agent(run.vocab, 32)
    _, _, res = play_run(agent, ToyRun(), cfg, np.random.default_rng(0), 5, use_evaluator=False, start=token)
    assert res["start_act"] == 1 and res["act"] >= 1


def test_parallel_collection_matches_serial():
    """같은 시드면 워커 프로세스로 나눠 돌려도 메인 프로세스와 결과가 같다."""
    from sts2ai.train.collect import RunCollector
    from sts2ai.train.loop import Config

    cfg = Config(group_size=2, d_model=32)
    agent = Agent(ToyRun().vocab, 32)
    seeds = [3, 4]
    with RunCollector(cfg, 0, make_run=ToyRun) as col:
        serial = col.collect(agent, seeds, use_evaluator=False)
    with RunCollector(cfg, 2, make_run=ToyRun) as col:
        parallel = col.collect(agent, seeds, use_evaluator=False)
    for (s1, g1, r1), (s2, g2, r2) in zip(serial, parallel):
        c1, c2 = r1.pop("choice_samples"), r2.pop("choice_samples")
        a1, a2 = r1.pop("act_starts"), r2.pop("act_starts")  # 토큰 안의 난수 객체는 값으로 비교되지 않는다
        assert [(a, t["act"], t["floor"]) for a, t in a1] == [(a, t["act"], t["floor"]) for a, t in a2]
        assert r1 == r2 and len(s1) == len(s2)
        # 고른 선택지는 같아야 한다. log 확률은 스레드 수에 따라 소수 7째 자리가 다를 수 있다.
        assert [c.action for c in c1] == [c.action for c in c2]
        assert all(abs(a.logp - b.logp) < 1e-5 for a, b in zip(c1, c2))
        assert [t.actions for g in g1 for t in g] == [t.actions for g in g2 for t in g]


def test_play_many_matches_policy_logprob():
    """분기를 묶어 진행해도 각 시행의 저장된 log 확률이 정책의 log_prob 와 같다 (학습 비율 계산이 맞다)."""
    from sts2ai.train.grpo import play_many

    run = _run_at_combat()
    agent = Agent(run.vocab, 32)
    trs = play_many(agent.policy, [run.fork_combat(k) for k in range(4)])
    assert len(trs) == 4 and all(t.actions for t in trs)
    for t in trs:
        lp, _ = agent.policy.log_prob(collate_combat(t.obs), torch.tensor(t.actions), torch.tensor(t.targets))
        assert torch.allclose(lp, torch.tensor(t.logp), atol=1e-4)


def test_server_dists_match_policy():
    """추론 서버의 (모든 행동에 대한 타깃 logit) 계산이 정책의 target_logits 와 같고, greedy 선택이 act 와 같다."""
    from sts2ai.train.infer_server import policy_dists, sample_from

    run = _run_at_combat()
    agent = Agent(run.vocab, 32)
    obs = []
    for k in range(3):
        f = run.fork_combat(k)
        for _ in range(k + 1):
            if not f.done:
                o = f.observe()
                obs.append(o)
                f.step(int(np.flatnonzero(o.legal)[0]) if hasattr(o, "legal") else 0, -1)
    obs = obs or [run.combat_obs()]
    batch = collate_combat(obs)
    with torch.no_grad():
        al, tl = policy_dists(agent.policy, batch, None)
        out = agent.policy(batch)
        for a in range(4):
            act = torch.full((len(obs),), a)
            ref = agent.policy.target_logits(out, batch, act)
            assert torch.allclose(tl[:, a], ref, atol=1e-5)
    a1, t1, lp1 = sample_from(al, tl, batch["needs_target"], batch["enemy_mask"], greedy=True)
    a2, t2, lp2 = agent.policy.act(batch, greedy=True)
    assert torch.equal(a1, a2) and torch.equal(t1, t2) and torch.allclose(lp1, lp2, atol=1e-5)


def test_infer_server_collection_runs():
    """추론 서버를 켜고 워커 2개로 수집해도 돌아가고, 서버가 실제로 호출을 받는다."""
    from sts2ai.train.collect import RunCollector
    from sts2ai.train.loop import Config

    cfg = Config(group_size=2, d_model=32, infer_server=True)
    agent = Agent(ToyRun().vocab, 32)
    with RunCollector(cfg, 2, make_run=ToyRun, server_device="cpu") as col:
        out = col.collect(agent, [3, 4], use_evaluator=False)
        st = col.server_stats()
    assert len(out) == 2 and all(g for _, g, _ in out)
    assert st["calls"] > 0 and st["rows"] >= st["calls"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 없음")
def test_graphed_policy_matches_and_follows_weights():
    """CUDA Graph 로 채워 계산한 결과가 그냥 계산과 같고, 가중치를 바꾸면 녹화한 그래프도 새 가중치를 쓴다."""
    from sts2ai.train.infer_server import GraphedPolicy, policy_dists

    run = _run_at_combat()
    agent = Agent(run.vocab, 32)
    obs = []
    for k in range(5):
        f = run.fork_combat(k)
        obs.append(f.observe())
        f.close()
    batch = collate_combat(obs)
    rv = torch.randn(len(obs), 32)
    dev = torch.device("cuda")
    pol = agent.policy.to(dev).eval()
    torch.backends.mha.set_fastpath_enabled(False)
    gp = GraphedPolicy(pol, dev)
    for _ in range(2):
        with torch.no_grad():
            ref_a, ref_t = policy_dists(pol, {k: v.to(dev) for k, v in batch.items()}, rv.to(dev))
        al, tl = gp(batch, rv)
        legal = batch["legal"].to(dev)
        assert torch.allclose(al[legal], ref_a[legal], atol=1e-4)
        # 가능한 행동만 비교 (불가능한 행동 칸은 토큰 수를 채운 만큼 잘리는 위치가 달라 값이 다르지만 쓰이지 않는다)
        em = batch["enemy_mask"].to(dev).unsqueeze(1) & legal.unsqueeze(2)
        assert torch.allclose(tl[em], ref_t[em], atol=1e-4)
        with torch.no_grad():  # 가중치 바꾸기: 두 번째 바퀴에서 그래프가 새 값을 써야 한다
            for p in pol.parameters():
                p.add_(0.01 * torch.randn_like(p))



def test_probe_helpers():
    """무작위 덱 바꾸기 범위, 적 고르기(다른 보스 제외), 채점, 보스 앞 상태와 보스 노드 라벨."""
    from sts2ai.train.evaluator_train import Snapshot, make_labels
    from sts2ai.train.grpo import Trajectory
    from sts2ai.train.probe import before_boss, modify_deck, pick_encounter, probe_score

    rng = np.random.default_rng(0)
    deck = [(i, False) for i in range(1, 13)]
    pool = list(range(100, 150))
    for _ in range(50):
        d = modify_deck(deck, pool, rng, 5, 5)
        kept = [c for c in d if c[0] < 100]
        added = [c for c in d if c[0] >= 100]
        assert 7 <= len(kept) <= 12 and len(added) <= 5 and set(kept) <= set(deck)
    w = {0: {(5, False): 3.0, (9, True): 4.0, (7, True): 2.0}}
    seen = {pick_encounter(w, 0, 7, rng) for _ in range(200)}
    assert seen == {(5, False), (7, True)}  # 이번 런 보스가 아닌 9 는 나오지 않는다
    assert pick_encounter({}, 1, 3, rng) == (3, True)
    won = Trajectory(won=True, hp=40)
    assert abs(probe_score(won, 80) - 1.5) < 1e-9

    run = _run_at_combat()
    st = run.run_state()
    b = before_boss(st, st.deck[:5], 33)
    assert b is not None and b.hp == 33 and len(b.deck) == 5
    boss = b.map.choices[0]
    assert b.map.node_types[boss] == S.NODE_BOSS and b.map.next_choices() == [boss]
    batch = collate_run([b], run.vocab)
    node, _ = make_labels([Snapshot(b, node_labels=[(boss, 0.75)])], batch["node_type"].shape[1], run.vocab.n_acts)
    labeled = (node[0] != -1).nonzero().flatten().tolist()
    assert labeled == [boss] and abs(float(node[0, boss]) - 0.75) < 1e-6
