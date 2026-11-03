"""무작위 덱 모의전: AI 가 평소 만들지 않는 덱으로, AI 가 많이 죽는 적과 싸워 보게 한다.

맵 화면에서 가끔(probe_prob) 지금 덱을 바꾼 덱을 만든다.
- 직업 카드·무색 카드·저주·상태이상 카드 풀에서 0~probe_add_max 장 무작위로 넣고, 지금 덱에서 0~probe_remove_max 장
  무작위로 뺀다.
- 무작위 카드 0~probe_upgrade_max 장을 강화하고(저주·상태이상 제외), 강화돼 있던 카드 0~probe_downgrade_max 장의
  강화를 되돌린다.
그 덱으로 이번 막의 위험한 적(최근 사망 횟수에 비례해 고름. 보스는 이번 런의 보스만)과 전투를 바로 시작해
probe_group_size 번 분기해 싸운다.

쓰는 곳
- 전투 정책: 분기들을 GRPO 그룹 하나로 쓴다. 채점은 런 맥락 없이 전투 결과로만 한다
  (이기면 1 + 남은 HP 비율, 지면 적에게 준 피해 비율 → 다 져도 순위가 생긴다).
- 평가기: 보스 모의전이면 "이 덱·이 HP 로 보스 바로 앞에 서 있는" 상태를 만들고, 보스 노드에만
  라벨(분기 중 죽은 비율, 0~1)을 붙인다. 나머지 노드는 라벨 없음. 일반·엘리트 적은 맵에서 어떤 적이 나올지
  모르는 노드라 라벨을 붙이지 않는다 (그 종류 노드가 늘 위험하다고 잘못 배우게 된다).
"""

from __future__ import annotations

import numpy as np

from .. import schema as S
from ..schema import MapGraph, RunState
from .evaluator_train import Snapshot
from .grpo import Trajectory, advantages, play_many


def modify_deck(deck: list[tuple[int, bool]], pool: list[int], rng: np.random.Generator, add_max: int,
                remove_max: int, min_size: int = 5, upgrade_max: int = 0, downgrade_max: int = 0,
                upgradable=None) -> list[tuple[int, bool]]:
    """지금 덱을 무작위로 바꾼다. 순서: 0~remove_max 장 빼기 → 풀에서 0~add_max 장 넣기(강화 안 된 카드) →
    강화 안 된 카드 중 0~upgrade_max 장 강화 → (그 전부터) 강화돼 있던 카드 중 0~downgrade_max 장 강화 되돌리기.

    upgradable(card_id) 가 False 인 카드(저주·상태이상)는 강화하지 않는다.
    """
    out = list(deck)
    n_rm = int(rng.integers(0, remove_max + 1))
    n_rm = min(n_rm, max(0, len(out) - min_size))
    for i in sorted(rng.choice(len(out), n_rm, replace=False).tolist(), reverse=True) if n_rm else []:
        out.pop(i)
    n_add = int(rng.integers(0, add_max + 1))
    out += [(int(c), False) for c in rng.choice(pool, n_add)] if n_add else []
    was_up = [i for i, (_, up) in enumerate(out) if up]
    can_up = [i for i, (c, up) in enumerate(out) if not up and (upgradable is None or upgradable(c))]
    n_up = min(int(rng.integers(0, upgrade_max + 1)), len(can_up))
    for i in rng.choice(can_up, n_up, replace=False).tolist() if n_up else []:
        out[i] = (out[i][0], True)
    n_down = min(int(rng.integers(0, downgrade_max + 1)), len(was_up))
    for i in rng.choice(was_up, n_down, replace=False).tolist() if n_down else []:
        out[i] = (out[i][0], False)
    return out


def pick_encounter(weights: dict, act: int, boss_raw: int, rng: np.random.Generator) -> tuple[int, bool]:
    """weights[act] = {(encounter, 보스인가): 가중치}. 보스는 이번 런의 보스만 후보 (가중치가 없으면 1)."""
    table = dict(weights.get(act, {}))
    cands = [(e, b, w) for (e, b), w in table.items() if not b and w > 0]
    boss_w = sum(w for (e, b), w in table.items() if b and e == boss_raw)
    cands.append((boss_raw, True, max(boss_w, 1.0)))
    w = np.array([c[2] for c in cands], dtype=np.float64)
    i = int(rng.choice(len(cands), p=w / w.sum()))
    return cands[i][0], cands[i][1]


def probe_score(t: Trajectory, max_hp: int) -> float:
    """이기면 1 + 남은 HP 비율 (1~2), 지면 마지막 관측에서 적에게 준 피해 비율 (0~1)."""
    if t.won:
        return 1.0 + t.hp / max(1, max_hp)
    if not t.obs:
        return 0.0
    o = t.obs[-1]
    if not len(o.enemy_tokens):
        return 0.0
    frac = o.feats[o.enemy_tokens, S.F_ENEMY_HP_FRAC]
    return float(1.0 - np.clip(frac, 0, 1).mean())


def before_boss(state: RunState, deck, hp: int) -> RunState | None:
    """state 를 "덱 deck·HP hp 로 이번 막 보스 바로 앞에 서 있는" 상태로 바꾼 복사본. 보스가 맵에 없으면 None."""
    m = state.map
    if m is None:
        return None
    bosses = [i for i, t in enumerate(m.node_types) if t == S.NODE_BOSS]
    if not bosses:
        return None
    boss = bosses[0]
    preds = [a for a, b in m.edges if b == boss]
    if not preds:
        return None
    pred = preds[0]
    cur_row = m.rows[m.current] if m.current is not None else min(m.rows) - 1
    nm = MapGraph(node_types=list(m.node_types), rows=list(m.rows), cols=list(m.cols), edges=list(m.edges),
                  current=pred, n_rows=m.n_rows, choices=[boss])
    return RunState(act=state.act, floor=state.floor + max(0, m.rows[pred] - cur_row), hp=hp, max_hp=state.max_hp,
                    gold=state.gold, ascension=state.ascension, deck=deck, relics=state.relics, potions=state.potions,
                    map=nm, boss=state.boss, map_jumps=0)


def run_probe(agent, run, cfg, rng: np.random.Generator, state: RunState, weights: dict, combats_before: int):
    """모의전 하나. (GRPO 그룹, 평가기 스냅숏 또는 None, 기록) 을 돌려준다. 시작하지 못하면 None."""
    deck = modify_deck(run.deck_raw(), run.probe_card_pool(), rng, cfg.probe_add_max, cfg.probe_remove_max,
                       upgrade_max=cfg.probe_upgrade_max, downgrade_max=cfg.probe_downgrade_max,
                       upgradable=getattr(run, "card_upgradable", None))
    boss_raw = run.boss_raw()
    enc, is_boss = pick_encounter(weights, state.act, boss_raw, rng)
    max_hp = state.max_hp
    hp = int(round(max_hp * rng.uniform(cfg.probe_hp_min, 1.0)))
    p = run.start_probe(int(rng.integers(2**31)), deck, enc, max(1, hp), combats_before)
    if p is None:
        return None
    try:
        pstate = p.run_state()
        run_vec = agent.run_vec(pstate) if cfg.policy_sees_map else None
        forks = [p.fork_combat(int(rng.integers(2**31))) for _ in range(cfg.probe_group_size)]
        trajs = play_many(agent.combat_actor(), forks, run_vec=run_vec)
    finally:
        p.close()
    scores = [probe_score(t, max_hp) for t in trajs]
    for t, sc, a in zip(trajs, scores, advantages(scores, cfg.advantage_mode)):
        t.score, t.advantage, t.run_state = sc, a, pstate
    snap = None
    dead = float(np.mean([not t.won for t in trajs]))
    if is_boss:
        st = before_boss(state, pstate.deck, hp)
        if st is not None:
            boss_node = st.map.choices[0]
            snap = Snapshot(st, node_labels=[(boss_node, dead)])
    info = {"encounter": enc, "boss": is_boss, "deck_size": len(deck), "hp": hp, "dead_frac": dead,
            "win_frac": 1.0 - dead, "scores": [round(s, 3) for s in scores]}
    return trajs, snap, info
