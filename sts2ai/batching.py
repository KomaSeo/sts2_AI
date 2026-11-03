"""RunState / CombatObs 리스트 → 패딩된 텐서 배치."""

from __future__ import annotations

import numpy as np
import torch

from .schema import CombatObs, MapGraph, RunState, Vocab, MAX_ACTIONS, N_COMBAT_FEATURES


def _pad(rows: list[list], width: int, fill=0, dtype=torch.long) -> torch.Tensor:
    out = torch.full((len(rows), max(1, width)), fill, dtype=dtype)
    for i, row in enumerate(rows):
        if row:
            out[i, : len(row)] = torch.as_tensor(row, dtype=dtype)
    return out


def collate_run(states: list[RunState], vocab: Vocab) -> dict[str, torch.Tensor]:
    """런 상태 배치. 덱/유물/포션은 가변 길이 집합, 맵은 밀집 인접행렬."""
    B = len(states)
    n_deck = max(len(s.deck) for s in states)
    deck_ids = _pad([[c.id for c in s.deck] for s in states], n_deck)
    deck_upg = _pad([[int(c.upgraded) for c in s.deck] for s in states], n_deck)
    deck_ench = _pad([[min(c.enchant, vocab.n_enchants - 1) for c in s.deck] for s in states], n_deck)
    deck_cost = _pad([[c.cost for c in s.deck] for s in states], n_deck, 0.0, torch.float)
    deck_mask = _pad([[1] * len(s.deck) for s in states], n_deck).bool()

    n_rel = max(len(s.relics) for s in states)
    relic_ids = _pad([[r for r, _ in s.relics] for s in states], n_rel)
    relic_cnt = _pad([[c / 20.0 for _, c in s.relics] for s in states], n_rel, 0.0, torch.float)
    relic_mask = _pad([[1] * len(s.relics) for s in states], n_rel).bool()

    pots = [[p for p in s.potions if p] for s in states]
    n_pot = max(len(p) for p in pots)
    potion_ids = _pad(pots, n_pot)
    potion_mask = _pad([[1] * len(p) for p in pots], n_pot).bool()

    status = torch.zeros(B, 6 + vocab.n_acts)
    for i, s in enumerate(states):
        mh = max(1, s.max_hp)
        status[i, 0] = s.hp / mh
        status[i, 1] = s.hp / 100.0
        status[i, 2] = mh / 100.0
        status[i, 3] = s.gold / 300.0
        status[i, 4] = s.floor / 50.0
        status[i, 5] = s.ascension / 10.0
        status[i, 6 + min(s.act, vocab.n_acts - 1)] = 1.0
    deck_stats = torch.zeros(B, 3)
    for i, s in enumerate(states):
        n = max(1, len(s.deck))
        deck_stats[i, 0] = len(s.deck) / 40.0
        deck_stats[i, 1] = sum(c.upgraded for c in s.deck) / n
        deck_stats[i, 2] = sum(c.cost for c in s.deck) / n

    batch = {
        "deck_ids": deck_ids,
        "deck_upg": deck_upg,
        "deck_ench": deck_ench,
        "deck_cost": deck_cost,
        "deck_mask": deck_mask,
        "deck_stats": deck_stats,
        "relic_ids": relic_ids,
        "relic_cnt": relic_cnt,
        "relic_mask": relic_mask,
        "potion_ids": potion_ids,
        "potion_mask": potion_mask,
        "status": status,
        "act": torch.tensor([min(s.act, vocab.n_acts - 1) for s in states]),
        "boss": torch.tensor([s.boss for s in states]),
    }
    batch.update(collate_maps([s.map for s in states]))
    return batch


def collate_maps(maps: list[MapGraph | None]) -> dict[str, torch.Tensor]:
    B = len(maps)
    V = max([len(m) for m in maps if m is not None] + [1])
    node_type = torch.zeros(B, V, dtype=torch.long)
    node_feat = torch.zeros(B, V, 4)  # row 정규화, 현재 위치, 다음 선택지, 도달 가능
    node_mask = torch.zeros(B, V, dtype=torch.bool)
    choice_mask = torch.zeros(B, V, dtype=torch.bool)
    adj = torch.zeros(B, V, V)  # adj[b, from, to]
    for b, m in enumerate(maps):
        if m is None or len(m) == 0:
            continue
        n = len(m)
        node_type[b, :n] = torch.as_tensor(m.node_types)
        node_mask[b, :n] = True
        rows = torch.as_tensor(m.rows, dtype=torch.float)
        node_feat[b, :n, 0] = rows / max(1, m.n_rows - 1)
        if m.current is not None:
            node_feat[b, m.current, 1] = 1.0
        choices = m.next_choices()
        node_feat[b, choices, 2] = 1.0
        choice_mask[b, choices] = True
        node_feat[b, :n, 3] = torch.as_tensor(m.reachable(), dtype=torch.float)
        for a, c in m.edges:
            adj[b, a, c] = 1.0
    return {
        "node_type": node_type,
        "node_feat": node_feat,
        "node_mask": node_mask,
        "choice_mask": choice_mask,
        "adj": adj,
    }


def collate_combat(obs: list[CombatObs]) -> dict[str, torch.Tensor]:
    B = len(obs)
    T = max(len(o) for o in obs)
    E = max([len(o.enemy_tokens) for o in obs] + [1])
    types = torch.zeros(B, T, dtype=torch.long)
    ids = torch.zeros(B, T, dtype=torch.long)
    owners = torch.zeros(B, T, dtype=torch.long)
    feats = torch.zeros(B, T, N_COMBAT_FEATURES)
    legal = torch.zeros(B, MAX_ACTIONS, dtype=torch.bool)
    needs_target = torch.zeros(B, MAX_ACTIONS, dtype=torch.bool)
    enemy_tok = torch.zeros(B, E, dtype=torch.long)
    enemy_mask = torch.zeros(B, E, dtype=torch.bool)
    for b, o in enumerate(obs):
        n = len(o)
        types[b, :n] = torch.from_numpy(o.types)
        ids[b, :n] = torch.from_numpy(o.ids)
        owners[b, :n] = torch.from_numpy(o.owners)
        feats[b, :n] = torch.from_numpy(o.feats)
        # 토큰이 없는 행동은 고를 수 없다 (행동 i ↔ 토큰 1 + i).
        n_act = min(MAX_ACTIONS, n - 1)
        legal[b, :n_act] = torch.from_numpy(o.legal[:n_act])
        needs_target[b, :n_act] = torch.from_numpy(o.needs_target[:n_act])
        k = len(o.enemy_tokens)
        if k:
            enemy_tok[b, :k] = torch.from_numpy(o.enemy_tokens)
            alive = o.enemy_alive if len(o.enemy_alive) == k else np.ones(k, dtype=bool)
            enemy_mask[b, :k] = torch.from_numpy(alive)
    return {
        "types": types,
        "ids": ids,
        "owners": owners,
        "feats": feats,
        "tok_mask": types != 0,
        "legal": legal,
        "needs_target": needs_target,
        "enemy_tok": enemy_tok,
        "enemy_mask": enemy_mask,
    }
