"""zhiyue/sts2-rl-agent 어댑터. 체크아웃이 있을 때만 돈다: ZHIYUE_PATH=../sts2-rl-agent pytest tests"""

import os
from pathlib import Path

import numpy as np
import pytest

ZHIYUE = Path(os.environ.get("ZHIYUE_PATH", "../sts2-rl-agent"))
pytestmark = pytest.mark.skipif(not (ZHIYUE / "sts2_env").is_dir(), reason="zhiyue/sts2-rl-agent 체크아웃 없음")


@pytest.fixture(scope="module")
def env():
    from sts2ai.envs.zhiyue_adapter import ZhiyueCombatEnv

    return ZhiyueCombatEnv(ZHIYUE)


def _scenario(env, enc, act, seed):
    from sts2ai.train.combat_train import CombatConfig, sample_scenario

    return sample_scenario(env, np.random.default_rng(seed), CombatConfig(), enc, act)


def _play_random(sim, rng, max_steps=400):
    for _ in range(max_steps):
        if sim.done:
            break
        o = sim.observe()
        legal = np.flatnonzero(o.legal)
        assert len(legal), "합법 행동이 하나도 없다"
        a = int(rng.choice(legal))
        t = int(rng.choice(o.enemy_slots)) if o.needs_target[a] and len(o.enemy_slots) else -1
        sim.step(a, t)


def test_every_encounter_plays_with_random_scenarios(env):
    """조우 80개 전부: 무작위 덱·HP·포션으로 무작위 합법 행동을 끝까지 넣어도 에러가 없다."""
    pool = env.combat_pool()
    assert len(pool) >= 70 and {a for _, a in pool} == {0, 1, 2, 3}
    rng = np.random.default_rng(0)
    for k, (enc, act) in enumerate(pool):
        sc = _scenario(env, enc, act, k)
        sim = env.make_combat(sc.encounter, sc.act, sc.deck, sc.hp, sc.max_hp, sc.potions, sc.relics, sc.seed)
        o = sim.observe()
        assert o.types[0] == 1 and len(o.enemy_tokens) > 0
        assert (o.ids < env.vocab.n_enemies)[o.types == 6].all()
        _play_random(sim, rng)


def test_same_seed_combats_stay_identical_when_interleaved(env):
    for enc, act in env.combat_pool()[::7]:
        sc = _scenario(env, enc, act, enc)
        a, b = (env.make_combat(sc.encounter, sc.act, sc.deck, sc.hp, sc.max_hp, sc.potions, sc.relics, sc.seed)
                for _ in range(2))
        rng = np.random.default_rng(enc)
        while not (a.done or b.done):
            oa, ob = a.observe(), b.observe()
            assert np.array_equal(oa.ids, ob.ids) and np.allclose(oa.feats, ob.feats)
            act_ = int(rng.choice(np.flatnonzero(oa.legal)))
            t = int(oa.enemy_slots[0]) if oa.needs_target[act_] and len(oa.enemy_slots) else -1
            a.step(act_, t)
            b.step(act_, t)
        assert a.done == b.done and a.hp() == b.hp()


def test_feed_raises_max_hp(env):
    """포식으로 적을 처치하면 최대 HP 가 오른다 (토이에는 없던 메커니즘)."""
    feed = env.s.card_ids[next(c for c in env.s.card_ids if c.name == "FEED")]
    weak = [(e, a) for e, a in env.combat_pool() if "weak" in env.encounter_name(e)]
    gains = []
    for k, (enc, act) in enumerate(weak[:4]):
        sim = env.make_combat(enc, act, [(feed, False)] * 10, 80, 80, [], [], k)
        _play_random(sim, np.random.default_rng(k))
        gains.append(sim.max_hp() - 80 if sim.won else None)
    assert any(g and g > 0 for g in gains), gains


def test_potions_fill_up_to_nine_slots(env):
    enc, act = env.combat_pool()[0]
    pots = env.potion_pool()[:9]
    sim = env.make_combat(enc, act, [(env.card_pool()[0], False)] * 10, 50, 50, pots, [], 0)
    assert sim.potions() == pots
    o = sim.observe()
    assert (o.types == 8).sum() == 9


def test_combat_training_runs_on_real_encounters(env, tmp_path):
    from sts2ai.train.combat_train import CombatConfig, train_combat

    cfg = CombatConfig(iterations=1, group_size=2, rounds_per_iter=1, focus_per_iter=4, deck_max=15, d_model=16,
                       grpo_epochs=1, eval_every=0, out_dir=str(tmp_path))
    pol = train_combat(env, cfg, log=lambda *_: None)
    hist = __import__("json").loads((tmp_path / "combat_history.json").read_text())
    assert hist[0]["grpo_trajs"] > 0 and len(hist[0]["death_rate"]) == len(env.combat_pool())
    assert pol is not None
