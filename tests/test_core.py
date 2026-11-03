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
