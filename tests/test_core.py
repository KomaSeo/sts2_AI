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
                  eval_epochs=1, grpo_epochs=1, warmup_snapshots=3, val_every=2, out_dir=str(out_dir))


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
    for part in a:
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
