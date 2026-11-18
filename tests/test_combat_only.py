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


def test_fight_text_and_view_accept_new_intents():
    from sts2ai.train.combat_only import fight_text
    from sts2ai.train.fight_view import to_md

    step = {"turn": 1, "hp": 50, "block": 0, "energy": 3, "action": "턴 종료", "prob_pct": 40.0, "action_prob_pct": 40.0,
            "top": [["턴 종료", 40.0]], "hand": [["Bash", 2, True]], "select": None, "endturn": [-4.0, -8.0, 0.0, 1],
            "enemies": [{"slot": 0, "name": "Byrdonis", "hp": 40, "block": 0, "intent": ["Attack", 24, 3],
                         "next": ["Debuff", 1, 1]},
                        {"slot": 1, "name": "Old", "hp": 10, "block": 0, "intent": ["Attack", 6]}]}
    rec = {"fmt": 2, "enemy_names": ["Byrdonis"], "encounter": 1, "sim_act": 1, "kind": "Normal", "won": True,
           "hp_start": 50, "max_hp": 80, "hp_end": 40, "score": 1.5, "deck": ["Bash"], "potions": [], "relics": [],
           "steps": [step]}
    txt = fight_text(rec)
    assert "Attack 24 (8x3)" in txt and "+[Debuff 1]" in txt
    md = to_md(0, [rec])
    assert "(8×3)" in md


def test_act_relics_follow_ancients():
    from sts2ai.train.combat_only import sample_act_relics

    anc = {"neow": [1, 2], 2: [10, 11], 3: [20, 21]}
    pool = list(range(1, 200))
    rng = np.random.default_rng(0)
    for act in (1, 2, 3):
        for _ in range(200):
            r = sample_act_relics(rng, act, pool, anc, 12)
            assert len(set(r)) == len(r)
            assert sum(x in (1, 2) for x in r) == 1
            assert sum(x in (10, 11) for x in r) == (1 if act >= 2 else 0)
            assert sum(x in (20, 21) for x in r) == (1 if act >= 3 else 0)
            rest = [x for x in r if x not in (1, 2, 10, 11, 20, 21)]
            assert len(rest) <= 12 * act
