"""AI 가 시뮬레이터에 요구하는 최소 인터페이스.

토이 환경(envs/toy.py)과 실제 시뮬레이터 어댑터(envs/sts2_adapter.py)가 모두 이걸 구현한다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..schema import CombatObs, RunState, Vocab

PHASE_COMBAT = "combat"
PHASE_MAP = "map"
PHASE_OTHER = "other"  # 카드 보상, 휴식, 상점, 이벤트 — 이번 범위 밖이라 규칙 정책이 처리
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
    # 그 밖의 화면은 간단한 규칙으로
    def default_step(self, rng: np.random.Generator) -> None: ...
    def close(self) -> None: ...
