"""RunState / CombatObs 리스트 → 패딩된 텐서 배치."""

from __future__ import annotations

import numpy as np
import torch

from .schema import (CombatObs, MapGraph, RunState, Vocab, MAX_ACTIONS, N_COMBAT_FEATURES, NODE_BOSS, NODE_BOSS_NA,
                     NODE_NA)


def to_device(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    """collate_* 결과(항상 CPU 에서 만든다)를 모델이 있는 장치로 옮긴다. 같은 장치면 그대로."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


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
    batch.update(collate_maps([s.map for s in states], [s.act for s in states], vocab.n_acts,
                              [getattr(s, "map_jumps", 0) for s in states]))
    return batch


def future_chain_start(m: MapGraph | None, act: int, k: int, n_acts: int) -> tuple[int, int] | None:
    """미래 막 k 의 가상 줄이 노드 배열에서 시작하는 위치와 줄 길이. 해당 없으면 None.

    배치의 노드 순서: [이번 막의 실제 노드 n 개] + [다음 막 줄 R 개] + [그다음 막 줄 R 개] ...
    맵이 없으면(막 시작 전 등) 이번 막도 가상 줄로 본다. 줄 길이 R = 이번 막 맵의 층수.
    """
    n, first = (len(m), act + 1) if (m is not None and len(m)) else (0, act)
    R = m.n_rows if (m is not None and len(m)) else 16
    if k < first or k >= n_acts:
        return None
    return n + (k - first) * R, R


def collate_maps(maps: list[MapGraph | None], acts: list[int] | None = None,
                 n_acts: int | None = None, jumps: list[int] | None = None) -> dict[str, torch.Tensor]:
    """맵 배치. acts·n_acts 를 주면 이번 막 맵 뒤에 남은 막마다 가상 줄(N/A 노드 R-1 개 + 보스 N/A)을 붙인다.

    이번 막 보스 → 다음 막 줄의 첫 노드 → ... → 다음 막 보스 N/A → 그다음 막 줄 ... 로 이어져서,
    평가기의 생존 계산이 마지막 막 끝까지 한 번에 흐른다.

    jumps (날개 달린 부츠의 남은 횟수) 가 있으면 이번 막 실제 맵에 "다음 줄 아무 노드로" 가는 간선(jump_adj)을
    따로 만든다. 그 간선을 한 번 쓸 때마다 횟수가 하나 준다 (간선을 따라가면 공짜). 지금 선택지 중
    간선으로 이어지지 않은 노드는 jump_choice 로 표시한다 (고르면 횟수를 하나 쓴다).
    """
    B = len(maps)
    acts = acts if acts is not None else [0] * B
    jumps = jumps if jumps is not None else [0] * B
    sizes = []
    for m, a in zip(maps, acts):
        n = len(m) if m is not None else 0
        tail = 0
        if n_acts is not None:
            for k in range(n_acts):
                s = future_chain_start(m, a, k, n_acts)
                if s is not None:
                    tail = s[0] + s[1] - n
        sizes.append(n + tail)
    V = max(sizes + [1])
    node_type = torch.zeros(B, V, dtype=torch.long)
    node_feat = torch.zeros(B, V, 4)  # row 정규화, 현재 위치, 다음 선택지, 도달 가능
    node_mask = torch.zeros(B, V, dtype=torch.bool)
    choice_mask = torch.zeros(B, V, dtype=torch.bool)
    chain_mask = torch.zeros(B, V, dtype=torch.bool)  # 미래 막 가상 노드
    node_act = torch.zeros(B, V, dtype=torch.long)
    adj = torch.zeros(B, V, V)  # adj[b, from, to]
    jump_adj = torch.zeros(B, V, V)  # 날개 달린 부츠: 다음 줄 아무 노드로 (간선이 없는 곳만)
    jump_choice = torch.zeros(B, V, dtype=torch.bool)
    jumps_t = torch.tensor([max(0, int(j)) for j in jumps], dtype=torch.long)
    for b, (m, a) in enumerate(zip(maps, acts)):
        n = len(m) if m is not None else 0
        a_idx = min(a, (n_acts or a + 1) - 1)
        if n:
            node_type[b, :n] = torch.as_tensor(m.node_types)
            node_mask[b, :n] = True
            node_act[b, :n] = a_idx
            rows = torch.as_tensor(m.rows, dtype=torch.float)
            node_feat[b, :n, 0] = rows / max(1, m.n_rows - 1)
            if m.current is not None:
                node_feat[b, m.current, 1] = 1.0
            choices = m.next_choices()
            node_feat[b, choices, 2] = 1.0
            choice_mask[b, choices] = True
            node_feat[b, :n, 3] = torch.as_tensor(m.reachable(), dtype=torch.float)
            for x, y in m.edges:
                adj[b, x, y] = 1.0
            if jumps_t[b] > 0:
                by_row: dict[int, list[int]] = {}
                for v, r in enumerate(m.rows):
                    by_row.setdefault(r, []).append(v)
                for v, r in enumerate(m.rows):
                    for w in by_row.get(r + 1, []):
                        if adj[b, v, w] == 0:
                            jump_adj[b, v, w] = 1.0
                if m.current is not None:
                    normal = set(m.successors()[m.current])
                    for c in choices:
                        if c not in normal:
                            jump_choice[b, c] = True
        if n_acts is None:
            continue
        # 미래 막 가상 줄
        prev_tail = [v for v in range(n) if not any(x == v for x, _ in (m.edges if n else []))] if n else []
        prev_tail = [v for v in prev_tail if m.node_types[v] == NODE_BOSS] or prev_tail  # 이번 막 보스
        first_chain = None
        for k in range(n_acts):
            s = future_chain_start(m, a, k, n_acts)
            if s is None:
                continue
            start, R = s
            idx = list(range(start, start + R))
            node_type[b, idx] = NODE_NA
            node_type[b, idx[-1]] = NODE_BOSS_NA
            node_mask[b, idx] = True
            chain_mask[b, idx] = True
            node_act[b, idx] = k
            node_feat[b, idx, 0] = torch.arange(R, dtype=torch.float) / max(1, R - 1)
            node_feat[b, idx, 3] = 1.0
            for v in prev_tail:
                adj[b, v, idx[0]] = 1.0
            for x, y in zip(idx, idx[1:]):
                adj[b, x, y] = 1.0
            prev_tail = [idx[-1]]
            first_chain = idx[0] if first_chain is None else first_chain
        if first_chain is not None and not choice_mask[b].any():
            # 이번 막에서 더 고를 노드가 없으면(보스를 이겼거나 맵이 없으면) 다음 막 줄에서 시작
            choice_mask[b, first_chain] = True
            node_feat[b, first_chain, 2] = 1.0
    return {
        "node_type": node_type,
        "node_feat": node_feat,
        "node_mask": node_mask,
        "choice_mask": choice_mask,
        "chain_mask": chain_mask,
        "node_act": node_act,
        "adj": adj,
        "jump_adj": jump_adj,
        "jump_choice": jump_choice,
        "jumps": jumps_t,
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
    enchants = torch.zeros(B, T, dtype=torch.long)
    for b, o in enumerate(obs):
        n = len(o)
        types[b, :n] = torch.from_numpy(o.types)
        ids[b, :n] = torch.from_numpy(o.ids)
        if getattr(o, "enchants", None) is not None:
            enchants[b, :n] = torch.from_numpy(np.asarray(o.enchants, dtype=np.int64))
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
        "enchants": enchants,
    }
