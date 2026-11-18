import numpy as np

from sts2ai.train.combat_only import CombatConfig, Scenario, sample_potion_count, sample_scenario, summarize
from sts2ai.train.grpo import Trajectory


def test_sample_scenario_ranges():
    cfg = CombatConfig()
    rng = np.random.default_rng(0)
    for _ in range(200):
        sc = sample_scenario(rng, cfg, [1, 2, 3], [7, 8], (5, 2, "Elite"), relic_pool=list(range(100, 150)))
        assert cfg.deck_min <= len(sc.deck) <= cfg.deck_max
        assert all(c in (1, 2, 3) for c, _ in sc.deck)
        assert len(sc.potions) <= 9
        assert cfg.max_hp_min <= sc.max_hp <= cfg.max_hp_max
        assert 1 <= sc.hp <= sc.max_hp
        assert (sc.encounter, sc.sim_act, sc.kind) == (5, 2, "Elite")
        assert cfg.relics_min <= len(sc.relics) <= cfg.relics_max and len(set(sc.relics)) == len(sc.relics)
        assert all(100 <= r < 150 for r in sc.relics)


def test_summarize_by_kind():
    sc_w = Scenario(1, 1, "Weak", [], 50, 100, [], 0)
    sc_b = Scenario(2, 1, "Boss", [], 50, 100, [], 0)
    g_w = [Trajectory(won=True, hp=40, score=1.4), Trajectory(won=False, score=0.2)]
    g_b = [Trajectory(won=False, score=0.1)]
    s = summarize([g_w, g_b], [sc_w, sc_b], "")
    assert s["n"] == 3 and abs(s["win_Weak"] - 0.5) < 1e-9 and s["win_Boss"] == 0.0
    assert abs(s["hp_left_win"] - 0.4) < 1e-9


def test_potion_count_bands():
    rng = np.random.default_rng(0)
    n = np.array([sample_potion_count(rng, CombatConfig().potion_bands) for _ in range(20000)])
    assert n.min() == 0 and n.max() == 9
    assert abs((n <= 2).mean() - 0.7) < 0.02
    assert abs(((n >= 3) & (n <= 6)).mean() - 0.2) < 0.02
    assert abs((n >= 7).mean() - 0.1) < 0.02


def test_starter_cards_first():
    cfg = CombatConfig()
    rng = np.random.default_rng(1)
    starters = [(10, 0, 5), (11, 0, 5), (12, 0, 1)]
    for _ in range(300):
        sc = sample_scenario(rng, cfg, [1, 2, 3, 10, 11, 12], [7], (5, 2, "Elite"), starters=starters)
        ids = [c for c, _ in sc.deck]
        assert ids.count(10) <= 5 and ids.count(11) <= 5 and ids.count(12) <= 1
        assert cfg.deck_min <= len(ids) <= max(cfg.deck_max, 11)
        assert all(c in (1, 2, 3, 10, 11, 12) for c in ids)
