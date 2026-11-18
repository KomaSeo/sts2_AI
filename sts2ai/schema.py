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
# 아직 맵이 공개되지 않은 미래 막의 가상 노드 (평가기 전용). 미래 막 하나 = N/A 노드 한 줄 + 끝의 보스 N/A.
NODE_NA = 9
NODE_BOSS_NA = 10
N_NODE_TYPES = 11

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
    # 간선으로 이어지지 않은 다음 줄 노드로 건너뛸 수 있는 남은 횟수 (날개 달린 부츠). 평가기의 생존 계산이
    # 이 횟수만큼 "다음 줄 아무 노드로" 가는 길도 본다.
    map_jumps: int = 0

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
            map_jumps=self.map_jumps,
        )


# ---- 전투·맵 밖 선택지 ----
# 선택지 종류. 실제 시뮬레이터의 선택지 종류(rl/run_train/features.py OPTION_KINDS)와 같은 이름을 쓰고,
# 맵 노드와 토이 게임에만 있는 것을 더했다. 목록에 없는 종류는 0 ("other") 으로 본다.
CHOICE_KINDS = (
    "other",
    "map.node",
    "neow.relic",
    "card.take", "card.skip", "card.sacrifice", "card.reroll",
    "reward.potion", "reward.relic", "reward.card", "reward.potion_swap", "reward.proceed",
    "shop.card", "shop.relic", "shop.potion", "shop.potion_swap", "shop.remove", "shop.leave",
    "rest.heal", "rest.smith", "rest.lift", "rest.dig", "rest.cook", "rest.hatch", "rest.kindle", "rest.clone",
    "rest.leave",
    "event.option",
    "select.offer", "select.offer_done", "select.upgrade", "select.remove", "select.transform",
    "select.transform_to", "select.enchant", "select.duplicate",
    "bundle.pick", "sphere.cell", "merchant.throw", "unknown",
)
CHOICE_KIND_INDEX = {k: i for i, k in enumerate(CHOICE_KINDS)}
N_EVENT_IDS = 128  # 이벤트 id 임베딩 크기 (넘는 id 는 나머지로 접는다)


@dataclass
class ChoiceOption:
    """선택 화면의 선택지 하나. 신경망 입력과, 고른 뒤 시뮬레이터에 보낼 행동 번호를 함께 담는다.

    모든 id 는 임베딩 인덱스(0 = 없음)다. node 는 RunState.map 의 노드 인덱스, deck_index 는
    RunState.deck 의 위치다 (-1 = 해당 없음).
    """

    action: int  # 시뮬레이터 행동 번호 (맵이면 노드 인덱스)
    kind: str = "other"
    card: Card | None = None  # 얻거나 사는 카드
    cards: list[Card] = field(default_factory=list)  # 묶음으로 얻는 카드 (bundle.pick)
    relic: int = 0
    potion: int = 0
    node: int = -1
    deck_index: int = -1  # 덱에서 고르는 카드 (제거·강화·변환 등)
    price: int = 0
    event: int = 0
    event_option: int = 0


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
# 어댑터가 붙이는 토큰 (combat_tokens 에는 없다)
TYPE_ORB = 18  # 오브 하나: ids = 오브 종류 + 1, F_MAGNITUDE = 패시브 값, F_INTENT_MAG = 발동 값
TYPE_RESOURCE = 19  # 캐릭터 자원 하나: ids = RESOURCE_* 종류, 값은 F_MAGNITUDE (오스티는 HP 칸)
# 카드 고르기 화면이 열려 있을 때 하나: ids = 시뮬레이터 CardSelectionKind (1 손패 소멸, 15 손패 강화 …).
# 무엇을 위해 고르는지(강화인지 소멸인지)를 정책이 알게 한다. combat_tokens 는 화면이 열렸는지만 쓴다.
TYPE_SELECTION = 20
N_TOKEN_TYPES = 21
N_SELECTION_KINDS = 32
RESOURCE_GOLD, RESOURCE_MAX_ENERGY, RESOURCE_ORB_SLOTS, RESOURCE_STARS, RESOURCE_OSTY = 1, 2, 3, 4, 5
N_RESOURCE_KINDS = 8
N_ORB_KINDS = 16

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
    # 카드 토큰의 인챈트 (임베딩 인덱스, 0 = 없음). None 이면 모두 0.
    enchants: np.ndarray | None = None  # int64 [T]

    def __len__(self) -> int:
        return len(self.types)
