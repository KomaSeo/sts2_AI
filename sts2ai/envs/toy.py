"""토이 StS: 실제 시뮬레이터 없이 파이프라인 전체를 검증하기 위한 작은 덱빌딩 게임.

실제 게임을 흉내 낸 것이 아니라 '같은 모양의 문제'를 만든 것이다:
3막, 막마다 DAG 맵(일반/엘리트/휴식/이벤트/상점/보스), 에너지·손패·방어도·힘·취약,
적 의도, 유물·포션, 카드 보상. 관측은 CombatObs/RunState 규약을 그대로 따른다.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np

from .. import schema as S
from ..schema import Card, CombatObs, MapGraph, RunState, Vocab
from .base import PHASE_COMBAT, PHASE_DONE, PHASE_MAP, PHASE_OTHER


@dataclass(frozen=True)
class CardDef:
    name: str
    cost: int
    dmg: int = 0
    block: int = 0
    target: bool = False
    aoe: bool = False
    vuln: int = 0
    draw: int = 0
    strength: int = 0
    hits: int = 1


CARDS = {
    1: CardDef("Strike", 1, dmg=6, target=True),
    2: CardDef("Defend", 1, block=5),
    3: CardDef("Bash", 2, dmg=8, vuln=2, target=True),
    4: CardDef("Cleave", 1, dmg=8, aoe=True),
    5: CardDef("Iron Wave", 1, dmg=5, block=5, target=True),
    6: CardDef("Heavy Blade", 2, dmg=15, target=True),
    7: CardDef("Shrug", 1, block=8, draw=1),
    8: CardDef("Inflame", 1, strength=2),
    9: CardDef("Pommel", 1, dmg=9, draw=1, target=True),
    10: CardDef("Twin Strike", 1, dmg=5, hits=2, target=True),
    11: CardDef("Impervious", 2, block=25),
}
REWARD_POOL = list(range(4, 12))
STARTER = [1] * 5 + [2] * 4 + [3]

# 의도
I_ATTACK, I_BLOCK, I_BUFF, I_ATTACK_BLOCK = 1, 2, 3, 4
# 버프
B_STRENGTH, B_VULN = 1, 2
# 유물
R_BURNING, R_VAJRA, R_ANCHOR, R_LANTERN = 1, 2, 3, 4
# 포션
P_FIRE, P_BLOCK, P_STRENGTH = 1, 2, 3


@dataclass(frozen=True)
class EnemyDef:
    name: str
    hp: tuple[int, int]
    moves: tuple[tuple[int, int, int], ...]  # (의도, 수치, 반복 횟수/방어도)


ENEMIES = {
    1: EnemyDef("Slime", (12, 16), ((I_ATTACK, 6, 1), (I_ATTACK, 6, 1), (I_BLOCK, 6, 0))),
    2: EnemyDef("Cultist", (44, 50), ((I_BUFF, 3, 0), (I_ATTACK, 6, 1), (I_ATTACK, 6, 1))),
    3: EnemyDef("Jaw Worm", (40, 44), ((I_ATTACK, 11, 1), (I_ATTACK_BLOCK, 7, 6), (I_BUFF, 3, 0))),
    4: EnemyDef("Brute", (75, 80), ((I_ATTACK, 14, 1), (I_ATTACK, 7, 2), (I_BUFF, 2, 0))),
    5: EnemyDef("Guardian", (80, 90), ((I_BLOCK, 10, 0), (I_ATTACK, 6, 2), (I_ATTACK, 15, 1), (I_BUFF, 2, 0))),
}
# encounter id -> 적 목록
ENCOUNTERS = {
    1: [1, 1],  # slimes
    2: [2],  # cultist
    3: [3],  # jaw worm
    4: [4],  # elite brute
    5: [3, 3],  # elite twin worms
    6: [5],  # boss (막마다 강해짐)
}
NORMAL_ENC, ELITE_ENC, BOSS_ENC = (1, 2, 3), (4, 5), 6

N_ACTS = 3
ROWS = 9  # 막당 층 수 (마지막 = 보스)
COLS = 4
MAX_HAND = 10

VOCAB = Vocab(
    n_cards=max(CARDS) + 1,
    n_relics=5,
    n_potions=4,
    n_buffs=3,
    n_enemies=max(ENEMIES) + 1,
    n_intents=5,
    n_encounters=max(ENCOUNTERS) + 1,
    n_enchants=2,
    n_acts=N_ACTS,
)


@dataclass
class Enemy:
    def_id: int
    hp: int
    max_hp: int
    block: int = 0
    strength: int = 0
    vuln: int = 0
    move_i: int = 0

    def intent(self, act: int):
        kind, mag, extra = ENEMIES[self.def_id].moves[self.move_i % len(ENEMIES[self.def_id].moves)]
        if kind in (I_ATTACK, I_ATTACK_BLOCK):
            mag = int(mag * (1 + 0.3 * act)) + self.strength
        return kind, mag, extra


@dataclass
class Combat:
    """전투 상태. deepcopy 한 뒤 reseed 하면 분기 복사본이 된다."""

    act: int
    encounter: int
    hp: int
    max_hp: int
    deck: list[Card]
    relics: list[int]
    potions: list[int]
    seed: int
    enemies: list[Enemy] = field(default_factory=list)
    draw: list[Card] = field(default_factory=list)
    hand: list[Card] = field(default_factory=list)
    discard: list[Card] = field(default_factory=list)
    exhaust: list[Card] = field(default_factory=list)
    energy: int = 3
    block: int = 0
    strength: int = 0
    turn: int = 0
    done: bool = False
    won: bool = False
    steps: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        scale = 1 + 0.6 * self.act
        for d in ENCOUNTERS[self.encounter]:
            lo, hi = ENEMIES[d].hp
            hp = int(self.rng.randint(lo, hi) * scale)
            self.enemies.append(Enemy(d, hp, hp, move_i=self.rng.randrange(3) if d != 5 else 0))
        self.draw = [copy.copy(c) for c in self.deck]
        self.rng.shuffle(self.draw)
        if R_VAJRA in self.relics:
            self.strength += 1
        self._start_turn()
        if R_ANCHOR in self.relics:
            self.block += 10
        if R_LANTERN in self.relics:
            self.energy += 1

    # ---- 분기 ----
    def reseed(self, seed: int) -> None:
        """숨은 정보 재샘플: 뽑을 더미 순서와 이후 난수."""
        self.rng = random.Random(seed)
        self.rng.shuffle(self.draw)

    # ---- 규칙 ----
    def _draw(self, n: int) -> None:
        for _ in range(n):
            if not self.draw:
                if not self.discard:
                    return
                self.draw, self.discard = self.discard, []
                self.rng.shuffle(self.draw)
            if len(self.hand) < MAX_HAND:
                self.hand.append(self.draw.pop())

    def _start_turn(self) -> None:
        self.turn += 1
        self.block = 0
        self.energy = 3
        self._draw(5)

    def _damage_enemy(self, e: Enemy, base: int) -> None:
        dmg = base + self.strength
        if e.vuln > 0:
            dmg = int(dmg * 1.5)
        absorbed = min(e.block, dmg)
        e.block -= absorbed
        e.hp = max(0, e.hp - (dmg - absorbed))

    def _alive(self) -> list[int]:
        return [i for i, e in enumerate(self.enemies) if e.hp > 0]

    def _card_value(self, c: Card) -> CardDef:
        d = CARDS[c.id]
        if not c.upgraded:
            return d
        return CardDef(
            d.name, d.cost, d.dmg + (3 if d.dmg else 0), d.block + (3 if d.block else 0),
            d.target, d.aoe, d.vuln + (1 if d.vuln else 0), d.draw, d.strength + (1 if d.strength else 0), d.hits,
        )

    def legal(self) -> np.ndarray:
        m = np.zeros(S.MAX_ACTIONS, dtype=bool)
        if self.done:
            return m
        for i, c in enumerate(self.hand):
            m[i] = CARDS[c.id].cost <= self.energy
        m[len(self.hand)] = True
        for s, p in enumerate(self.potions):
            m[len(self.hand) + 1 + s] = p != 0
        return m

    def needs_target(self) -> np.ndarray:
        m = np.zeros(S.MAX_ACTIONS, dtype=bool)
        for i, c in enumerate(self.hand):
            m[i] = CARDS[c.id].target
        for s, p in enumerate(self.potions):
            m[len(self.hand) + 1 + s] = p == P_FIRE
        return m

    def _pick_target(self, slot: int) -> Enemy:
        alive = self._alive()
        if slot not in alive:
            slot = alive[0]
        return self.enemies[slot]

    def step(self, action: int, target_slot: int = -1) -> None:
        if self.done:
            return
        self.steps += 1
        if not self.legal()[action]:
            raise ValueError(f"illegal action {action}")
        n = len(self.hand)
        if action < n:
            card = self.hand.pop(action)
            d = self._card_value(card)
            self.energy -= d.cost
            if d.dmg:
                targets = self._alive() if d.aoe else [self.enemies.index(self._pick_target(target_slot))]
                for _ in range(d.hits):
                    for t in targets:
                        self._damage_enemy(self.enemies[t], d.dmg)
            if d.vuln and self._alive():
                self._pick_target(target_slot).vuln += d.vuln
            self.block += d.block
            self.strength += d.strength
            (self.exhaust if d.strength else self.discard).append(card)
            if d.draw:
                self._draw(d.draw)
        elif action == n:
            self._end_turn()
        else:
            s = action - n - 1
            p, self.potions[s] = self.potions[s], 0
            if p == P_FIRE:
                self._damage_enemy(self._pick_target(target_slot), 20 - self.strength)
            elif p == P_BLOCK:
                self.block += 12
            elif p == P_STRENGTH:
                self.strength += 2
        self._check_end()
        if self.steps > 400 and not self.done:  # 무한 루프 방지: 시간 초과 = 패배
            self.done, self.won = True, False

    def _end_turn(self) -> None:
        self.discard.extend(self.hand)
        self.hand = []
        for e in self.enemies:
            if e.hp <= 0:
                continue
            e.block = 0
            kind, mag, extra = e.intent(self.act)
            if kind in (I_ATTACK, I_ATTACK_BLOCK):
                for _ in range(max(1, extra if kind == I_ATTACK else 1)):
                    dmg = mag
                    absorbed = min(self.block, dmg)
                    self.block -= absorbed
                    self.hp -= dmg - absorbed
                if kind == I_ATTACK_BLOCK:
                    e.block += extra
            elif kind == I_BLOCK:
                e.block += mag
            elif kind == I_BUFF:
                e.strength += mag
            e.move_i += 1
            e.vuln = max(0, e.vuln - 1)
        if self.hp <= 0:
            self.hp = 0
            self.done, self.won = True, False
            return
        self._start_turn()

    def _check_end(self) -> None:
        if not self._alive():
            self.done, self.won = True, True

    # ---- 관측 ----
    def observe(self) -> CombatObs:
        types, ids, owners, feats = [], [], [], []

        def push(t, i=0, owner=0, **ch):
            row = np.zeros(S.N_COMBAT_FEATURES, dtype=np.float32)
            for k, v in ch.items():
                row[getattr(S, k)] = v
            types.append(t)
            ids.append(i)
            owners.append(owner)
            feats.append(row)

        push(S.TYPE_GLOBAL, self.encounter, F_HP=self.hp / self.max_hp, F_BLOCK=self.block / 50,
             F_ENERGY=self.energy / 10, F_DRAW_N=len(self.draw) / 20, F_DISCARD_N=len(self.discard) / 20,
             F_EXHAUST_N=len(self.exhaust) / 20)
        for c in self.hand:  # 행동 블록: 정렬하지 않는다
            push(S.TYPE_HAND, c.id, F_UPGRADED=float(c.upgraded), F_COST=CARDS[c.id].cost / 5)
        push(S.TYPE_END_TURN)
        if any(self.potions):
            for p in self.potions:
                push(S.TYPE_POTION, p, F_MAGNITUDE=float(p != 0))
        for pile, t in ((self.draw, S.TYPE_DRAW), (self.discard, S.TYPE_DISCARD), (self.exhaust, S.TYPE_EXHAUST)):
            for c in sorted(pile, key=lambda c: (c.id, c.upgraded)):  # 뽑을 더미 순서는 숨긴다
                push(t, c.id, F_UPGRADED=float(c.upgraded), F_COST=CARDS[c.id].cost / 5)
        for r in self.relics:
            push(S.TYPE_RELIC, r)
        if self.strength:
            push(S.TYPE_BUFF, B_STRENGTH, F_MAGNITUDE=self.strength / 20)
        enemy_tokens, enemy_slots = [], []
        for slot, e in enumerate(self.enemies):
            if e.hp <= 0:
                continue
            enemy_tokens.append(len(types))
            enemy_slots.append(slot)
            push(S.TYPE_ENEMY, e.def_id, owner=slot + 1, F_ENEMY_HP=e.hp / 50,
                 F_ENEMY_HP_FRAC=e.hp / e.max_hp, F_ENEMY_BLOCK=e.block / 50)
            kind, mag, _ = e.intent(self.act)
            push(S.TYPE_INTENT, kind, owner=slot + 1, F_INTENT_MAG=mag / 30)
            if e.strength:
                push(S.TYPE_ENEMY_BUFF, B_STRENGTH, owner=slot + 1, F_MAGNITUDE=e.strength / 20)
            if e.vuln:
                push(S.TYPE_ENEMY_BUFF, B_VULN, owner=slot + 1, F_MAGNITUDE=e.vuln / 20)
        return CombatObs(
            types=np.array(types, dtype=np.int64),
            ids=np.array(ids, dtype=np.int64),
            owners=np.array(owners, dtype=np.int64),
            feats=np.stack(feats),
            legal=self.legal(),
            needs_target=self.needs_target(),
            enemy_tokens=np.array(enemy_tokens, dtype=np.int64),
            enemy_slots=np.array(enemy_slots, dtype=np.int64),
            enemy_alive=np.ones(len(enemy_tokens), dtype=bool),
        )


class ToyCombatSim:
    """CombatSim 프로토콜 구현 (분기 복사본)."""

    def __init__(self, combat: Combat):
        self.c = combat

    @property
    def done(self) -> bool:
        return self.c.done

    @property
    def won(self) -> bool:
        return self.c.won

    def observe(self) -> CombatObs:
        return self.c.observe()

    def step(self, action: int, target_slot: int = -1) -> None:
        self.c.step(action, target_slot)

    def hp(self) -> int:
        hp = self.c.hp
        if self.c.won and R_BURNING in self.c.relics:
            hp = min(self.c.max_hp, hp + 6)
        return hp

    def potions(self) -> list[int]:
        return list(self.c.potions)

    def close(self) -> None:
        pass


def generate_map(rng: random.Random) -> MapGraph:
    rows_nodes: list[list[int]] = []
    types, rows, cols = [], [], []
    for r in range(ROWS):
        if r == ROWS - 1:
            row_cols = [COLS // 2]
        else:
            row_cols = sorted(rng.sample(range(COLS), rng.randint(2, COLS)))
        idx = []
        for c in row_cols:
            if r == 0:
                t = S.NODE_NORMAL
            elif r == ROWS - 1:
                t = S.NODE_BOSS
            elif r == ROWS - 2:
                t = S.NODE_REST
            else:
                t = rng.choices(
                    [S.NODE_NORMAL, S.NODE_ELITE, S.NODE_REST, S.NODE_EVENT, S.NODE_SHOP],
                    weights=[45, 17 if r >= 2 else 0, 12, 18, 8],
                )[0]
            idx.append(len(types))
            types.append(t)
            rows.append(r)
            cols.append(c)
        rows_nodes.append(idx)
    edges = set()
    for r in range(ROWS - 1):
        nxt = rows_nodes[r + 1]
        for v in rows_nodes[r]:
            near = sorted(nxt, key=lambda u: abs(cols[u] - cols[v]))
            for u in near[: rng.randint(1, 2)]:
                edges.add((v, u))
        for u in nxt:  # 모든 노드는 선행 노드가 있어야 한다
            if not any(b == u for _, b in edges):
                v = min(rows_nodes[r], key=lambda v: abs(cols[u] - cols[v]))
                edges.add((v, u))
    return MapGraph(types, rows, cols, sorted(edges), None, n_rows=ROWS)


class ToyRun:
    """RunSim 프로토콜 구현."""

    vocab = VOCAB

    def __init__(self, ascension: int = 0):
        self.ascension = ascension

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.act = 0
        self.floor = 0
        self.max_hp = 70
        self.hp = 70
        self.gold = 99
        self.deck = [Card(i, False, 0, CARDS[i].cost / 5) for i in STARTER]
        self.relics = [R_BURNING]
        self.potions = [0, 0, 0]
        self.won = self.died = False
        self._combat_counter = 0
        self._new_act()

    def _new_act(self) -> None:
        self.map = generate_map(self.rng)
        self.current_node = None
        self.combat: Combat | None = None
        self.pending: tuple | None = None
        self.phase = PHASE_MAP

    # ---- 상태 ----
    def run_state(self) -> RunState:
        m = copy.copy(self.map)
        m.current = self.current_node
        return RunState(
            act=self.act, floor=self.floor, hp=self.hp, max_hp=self.max_hp, gold=self.gold,
            ascension=self.ascension, deck=list(self.deck), relics=[(r, 0) for r in self.relics],
            potions=list(self.potions), map=m, boss=BOSS_ENC,
        )

    # ---- 맵 ----
    def choose_node(self, node: int) -> None:
        assert self.phase == PHASE_MAP and node in self.map_choices(), (node, self.map_choices())
        self.current_node = node
        self.floor += 1
        t = self.map.node_types[node]
        if t == S.NODE_NORMAL:
            self._start_combat(self.rng.choice(NORMAL_ENC))
        elif t == S.NODE_ELITE:
            self._start_combat(self.rng.choice(ELITE_ENC))
        elif t == S.NODE_BOSS:
            self._start_combat(BOSS_ENC)
        elif t == S.NODE_EVENT:
            roll = self.rng.random()
            if roll < 0.25:
                self._start_combat(self.rng.choice(NORMAL_ENC))
            elif roll < 0.6:
                self.hp = min(self.max_hp, self.hp + 12)
                self._after_room()
            else:
                self.hp = max(1, self.hp - 7)
                self.pending = ("card", [self.rng.choice(REWARD_POOL) for _ in range(3)])
                self.phase = PHASE_OTHER
        elif t == S.NODE_REST:
            self.pending = ("rest",)
            self.phase = PHASE_OTHER
        elif t == S.NODE_SHOP:
            self.pending = ("shop",)
            self.phase = PHASE_OTHER
        else:
            self._after_room()

    def map_choices(self) -> list[int]:
        m = copy.copy(self.map)
        m.current = self.current_node
        return m.next_choices()

    # ---- 전투 ----
    def _start_combat(self, encounter: int) -> None:
        self._combat_counter += 1
        self.combat = Combat(
            act=self.act, encounter=encounter, hp=self.hp, max_hp=self.max_hp, deck=self.deck,
            relics=self.relics, potions=list(self.potions), seed=self.rng.randrange(2**31),
        )
        self.phase = PHASE_COMBAT

    def combat_id(self) -> int:
        return self._combat_counter

    def combat_obs(self) -> CombatObs:
        return self.combat.observe()

    def fork_combat(self, seed: int) -> ToyCombatSim:
        c = copy.deepcopy(self.combat)
        c.reseed(seed)
        return ToyCombatSim(c)

    def combat_step(self, action: int, target_slot: int = -1) -> None:
        c = self.combat
        c.step(action, target_slot)
        if not c.done:
            return
        self.potions = list(c.potions)
        if not c.won:
            self.hp = 0
            self.died = True
            self.phase = PHASE_DONE
            return
        self.hp = ToyCombatSim(c).hp()
        enc = c.encounter
        self.combat = None
        if enc == BOSS_ENC:
            if self.act == N_ACTS - 1:
                self.won = True
                self.phase = PHASE_DONE
                return
            self.act += 1
            self.hp = self.hp + int((self.max_hp - self.hp) * 0.75)
            self._new_act()
            return
        self.gold += self.rng.randint(15, 25) if enc in NORMAL_ENC else 35
        if enc in ELITE_ENC:
            r = self.rng.choice([R_VAJRA, R_ANCHOR, R_LANTERN])
            if r not in self.relics:
                self.relics.append(r)
        if self.rng.random() < 0.4 and 0 in self.potions:
            self.potions[self.potions.index(0)] = self.rng.choice([P_FIRE, P_BLOCK, P_STRENGTH])
        self.pending = ("card", self.rng.sample(REWARD_POOL, 3))
        self.phase = PHASE_OTHER

    # ---- 그 밖의 화면: 간단한 규칙 ----
    def default_step(self, rng: np.random.Generator) -> None:
        kind = self.pending[0]
        if kind == "card":
            if rng.random() < 0.7:
                cid = int(rng.choice(self.pending[1]))
                self.deck.append(Card(cid, False, 0, CARDS[cid].cost / 5))
        elif kind == "rest":
            if self.hp < 0.6 * self.max_hp:
                self.hp = min(self.max_hp, self.hp + int(0.3 * self.max_hp))
            else:
                cands = [c for c in self.deck if not c.upgraded]
                if cands:
                    c = cands[int(rng.integers(len(cands)))]
                    i = self.deck.index(c)
                    self.deck[i] = Card(c.id, True, c.enchant, c.cost)
        elif kind == "shop":
            if self.gold >= 75:
                basics = [i for i, c in enumerate(self.deck) if c.id in (1, 2)]
                if basics:
                    self.deck.pop(basics[0])
                    self.gold -= 75
        self.pending = None
        self._after_room()

    def _after_room(self) -> None:
        self.phase = PHASE_MAP

    def close(self) -> None:
        pass
