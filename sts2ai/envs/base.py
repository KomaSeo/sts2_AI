"""AI 가 시뮬레이터에 요구하는 최소 인터페이스.

토이 환경(envs/toy.py)과 실제 시뮬레이터 어댑터(envs/sts2_adapter.py)가 모두 이걸 구현한다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..schema import CombatObs, RunState, Vocab

PHASE_COMBAT = "combat"
PHASE_MAP = "map"
PHASE_OTHER = "other"  # 카드 보상, 휴식, 상점, 이벤트 — 규칙 정책 또는 기대값 선택기가 처리
PHASE_DONE = "done"


class CombatSim(Protocol):
    """전투 하나. 분기(fork)된 복사본이라 마음대로 굴려도 원래 런에는 영향이 없다."""

    done: bool
    won: bool

    def observe(self) -> CombatObs: ...
    def step(self, action: int, target_slot: int = -1) -> None: ...
    def hp(self) -> int: ...
    def potions(self) -> list[int]: ...
    def close(self) -> None: ...


class RunSim(Protocol):
    vocab: Vocab
    phase: str
    act: int
    floor: int
    won: bool
    died: bool
    current_node: int | None  # 이번 막 맵에서의 노드 인덱스

    def reset(self, seed: int) -> None: ...
    def run_state(self) -> RunState: ...
    # 전투 (진짜 런을 진행시킨다)
    def combat_obs(self) -> CombatObs: ...
    def combat_step(self, action: int, target_slot: int = -1) -> None: ...
    def combat_id(self) -> int:
        """지금 전투를 구분하는 번호. 바뀌면 새 전투가 시작된 것."""
        ...
    # 숨은 정보(셔플, 적 난수)를 다시 뽑은 전투 복사본
    def fork_combat(self, seed: int) -> CombatSim: ...
    # 맵
    def choose_node(self, node: int) -> None: ...
    # 전투·맵 밖의 선택 화면 (카드 보상, 상점, 휴식, 이벤트, 덱에서 카드 고르기)
    def choice_actions(self) -> list[int]: ...
    def step_choice(self, action: int) -> None: ...
    def screen_kind(self) -> str: ...
    def rule_action(self, rng: np.random.Generator) -> int:
        """비교 기준이 되는 규칙 정책의 선택."""
        ...
    def default_step(self, rng: np.random.Generator) -> None:
        """규칙 정책으로 한 번 고른다."""
        ...
    def spawn(self) -> RunSim:
        """같은 설정(승천 등)의 독립된 새 런. reset 전 상태. 여러 런을 나란히 굴릴 때 쓴다."""
        ...
    def clone(self, seed: int) -> RunSim:
        """숨은 난수(무작위 변환·강화 결과, 앞으로의 보상 등)를 다시 뽑은 복사본."""
        ...
    def close(self) -> None: ...
