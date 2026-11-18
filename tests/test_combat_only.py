import numpy as np

from sts2ai.train.combat_only import CombatConfig, Scenario, sample_scenario, summarize
from sts2ai.train.grpo import Trajectory


def test_sample_scenario_ranges():
    cfg = CombatConfig()
    rng = np.random.default_rng(0)
    for _ in range(200):
        sc = sample_scenario(rng, cfg, [1, 2, 3], [7, 8], (5, 2, "Elite"))
        assert cfg.deck_min <= len(sc.deck) <= cfg.deck_max
        assert all(c in (1, 2, 3) for c, _ in sc.deck)
        assert len(sc.potions) <= cfg.potions_max
        assert cfg.max_hp_min <= sc.max_hp <= cfg.max_hp_max
        assert 1 <= sc.hp <= sc.max_hp
        assert (sc.encounter, sc.sim_act, sc.kind) == (5, 2, "Elite")


def test_summarize_by_kind():
    sc_w = Scenario(1, 1, "Weak", [], 50, 100, [], 0)
    sc_b = Scenario(2, 1, "Boss", [], 50, 100, [], 0)
    g_w = [Trajectory(won=True, hp=40, score=1.4), Trajectory(won=False, score=0.2)]
    g_b = [Trajectory(won=False, score=0.1)]
    s = summarize([g_w, g_b], [sc_w, sc_b], "")
    assert s["n"] == 3 and abs(s["win_Weak"] - 0.5) < 1e-9 and s["win_Boss"] == 0.0
    assert abs(s["hp_left_win"] - 0.4) < 1e-9
