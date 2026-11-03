"""시뮬레이터와 무관한 공통 데이터 구조.

모든 인코더와 에이전트는 이 구조만 읽는다. 실제 시뮬레이터(r33hab/sts2)든 테스트용
토이 환경이든, 어댑터가 자기 상태를 이 구조로 바꿔 주기만 하면 된다.

ID 규칙: 모든 id 는 "임베딩 인덱스"다. 0 은 패딩/알 수 없음으로 예약한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---- 맵 노드 종류 (r33hab/sts2 의 run_constants 와 같은 번호) ----
NODE_NONE = 0
NODE_NORMAL = 1
NODE_ELITE = 2
NODE_REST = 3
NODE_SHOP = 4
NODE_RELIC = 5  # 보물
NODE_BOSS = 6
NODE_EVENT = 7
NODE_ANCIENT = 8
N_NODE_TYPES = 9

# 전투가 벌어지는 방. 이벤트도 전투가 날 수 있지만 위험도는 평가기가 배운다.
COMBAT_NODES = (NODE_NORMAL, NODE_ELITE, NODE_BOSS)


@dataclass(frozen=True)
class Vocab:
    """임베딩 테이블 크기. 각 값은 '최대 id + 1' (0 = 패딩)."""

    n_cards: int
    n_relics: int
    n_potions: int
    n_buffs: int
    n_enemies: int
    n_intents: int
    n_encounters: int
    n_enchants: int = 16
    n_acts: int = 3  # StS2 는 3막 (1막은 Overgrowth/Underdocks 중 하나)


@dataclass
class Card:
    id: int
    upgraded: bool = False
    enchant: int = 0
    cost: float = 0.0  # 정규화된 비용 (대략 cost / 5)


@dataclass
class MapGraph:
    """현재 막의 맵. 노드는 리스트 인덱스로 가리킨다.

    edges 는 (from, to) — 아래층에서 위층 방향.
    """

    node_types: list[int]
    rows: list[int]
    cols: list[int]
    edges: list[tuple[int, int]]
    current: int | None  # 지금 서 있는 노드 (막 시작 전이면 None)
    n_rows: int = 16
    # 시뮬레이터가 직접 알려 주는 선택지 (날개 장화처럼 간선 밖으로 가는 경우). None 이면 간선으로 계산.
    choices: list[int] | None = None

    def __len__(self) -> int:
        return len(self.node_types)

    def successors(self) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in self.node_types]
        for a, b in self.edges:
            out[a].append(b)
        return out

    def next_choices(self) -> list[int]:
        """지금 고를 수 있는 다음 노드들."""
        if self.choices is not None:
            return list(self.choices)
        if self.current is None:
            low = min(self.rows) if self.rows else 0
            return [i for i, r in enumerate(self.rows) if r == low]
        return self.successors()[self.current]

    def reachable(self) -> np.ndarray:
        """현재 위치에서 앞으로 갈 수 있는 노드 표시 (현재 노드 제외)."""
        succ = self.successors()
        seen = np.zeros(len(self), dtype=bool)
        stack = list(self.next_choices())
        while stack:
            v = stack.pop()
            if seen[v]:
                continue
            seen[v] = True
            stack.extend(succ[v])
        return seen


@dataclass
class RunState:
    """런 수준 상태: 덱, 유물, 포션, 플레이어 수치, 맵."""

    act: int  # 0-based 막 번호 (0 = 1막)
    floor: int
    hp: int
    max_hp: int
    gold: int
    ascension: int
    deck: list[Card]
    relics: list[tuple[int, int]]  # (relic_id, counter)
    potions: list[int]  # potion id, 빈 칸은 0
    map: MapGraph | None
    boss: int = 0  # 이 막의 보스 encounter id (게임이 막 시작부터 알려 줌)

    def with_combat_result(self, hp: int, potions: list[int]) -> RunState:
        """전투 직후 상태 근사: HP 와 포션만 바뀐다고 본다 (보상은 아직 모름)."""
        return RunState(
            act=self.act,
            floor=self.floor,
            hp=max(0, hp),
            max_hp=self.max_hp,
            gold=self.gold,
            ascension=self.ascension,
            deck=self.deck,
            relics=self.relics,
            potions=potions,
            map=self.map,
            boss=self.boss,
        )


# ---- 전투 토큰 (r33hab/sts2 의 rl/combat_tokens.py 와 같은 규약) ----
TYPE_PAD = 0
TYPE_GLOBAL = 1
TYPE_HAND = 2
TYPE_DRAW = 3
TYPE_DISCARD = 4
TYPE_EXHAUST = 5
TYPE_ENEMY = 6
TYPE_BUFF = 7
TYPE_POTION = 8
TYPE_ENEMY_BUFF = 9
TYPE_END_TURN = 10
TYPE_SELECT = 11
TYPE_DECLINE = 12
TYPE_HAND_LOCKED = 13
TYPE_RELIC = 14
TYPE_INTENT = 15
TYPE_INTENT_NEXT = 16
TYPE_RUN = 17  # 런 맥락(맵 포함) 토큰. CombatEncoder 가 맨 뒤에 붙인다. 시뮬레이터 관측에는 없다.
N_TOKEN_TYPES = 18

# 토큰 종류별로 ids 가 어느 임베딩 테이블을 가리키는지
CARD_TOKEN_TYPES = (TYPE_HAND, TYPE_DRAW, TYPE_DISCARD, TYPE_EXHAUST, TYPE_SELECT, TYPE_HAND_LOCKED)
BUFF_TOKEN_TYPES = (TYPE_BUFF, TYPE_ENEMY_BUFF)
INTENT_TOKEN_TYPES = (TYPE_INTENT, TYPE_INTENT_NEXT)

N_COMBAT_FEATURES = 18
MAX_ACTIONS = 32

# 수치 특징 채널 (combat_tokens.py 와 같은 번호)
F_HP = 0
F_BLOCK = 1
F_ENERGY = 2
F_DRAW_N = 3
F_DISCARD_N = 4
F_EXHAUST_N = 5
F_UPGRADED = 6
F_COST = 7
F_ENCHANT = 8
F_ENEMY_HP = 9
F_ENEMY_HP_FRAC = 10
F_ENEMY_BLOCK = 11
F_INTENT_MAG = 12
F_MAGNITUDE = 13


@dataclass
class CombatObs:
    """전투 한 시점.

    행동 블록 규약: 토큰 1 + i 가 행동 i 다 (0 번 토큰은 GLOBAL).
    needs_target[i] 가 참인 행동은 적 하나를 골라야 한다.
    enemy_tokens 는 타깃이 될 수 있는 적 토큰의 위치, enemy_slots 는 그 적의 슬롯 번호
    (시뮬레이터의 target_enemy_index).
    """

    types: np.ndarray  # int64 [T]
    ids: np.ndarray  # int64 [T]
    owners: np.ndarray  # int64 [T]
    feats: np.ndarray  # float32 [T, F]
    legal: np.ndarray  # bool [MAX_ACTIONS]
    needs_target: np.ndarray  # bool [MAX_ACTIONS]
    enemy_tokens: np.ndarray  # int64 [E]
    enemy_slots: np.ndarray  # int64 [E]
    enemy_alive: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __len__(self) -> int:
        return len(self.types)
