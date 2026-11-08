"""기대값 선택기: 전투·맵 밖의 선택(카드 보상, 상점, 휴식, 이벤트, 덱에서 카드 고르기)을 평가기로 고른다.

무작위 결과(무작위 변환, 줍는 순간 무작위 강화하는 유물)는 고르기 전에는 결과 상태를 알 수 없다.
그래서 시뮬레이터의 '숨은 난수를 다시 뽑은 복사본'으로 같은 선택을 K 번 굴려 기대값을 낸다.

    선택지 a 의 가치 = log E_k[ V(a 를 고른 뒤 상태_k) ]      (V = P(런 클리어))

- 평균은 log V 가 아니라 확률 V 로 낸다 (logsumexp - log K). log 로 평균하면 "가끔 대박" 인
  선택이 과소평가된다.
- 다음 화면이 또 선택이면 (상점에서 '카드 제거' → 뺄 카드 고르기) 그 화면에서도 같은 방식으로
  가장 좋은 선택을 고른다고 보고 max 를 취한다 (expectimax). 깊이 제한에 걸리면 나머지 화면은
  규칙 정책으로 끝까지 진행한 상태를 평가한다 — 돈만 내고 카드는 안 뺀 '거래 중' 상태를
  평가하지 않기 위해서다.
- 고른 직후 전투가 시작되면 (전투 이벤트) 그 방의 hazard 를 빼서 반영한다.
- 표본 두 개의 결과 상태가 똑같으면 결정적인 선택(제거, 강화 등)으로 보고 표본을 더 뽑지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..batching import collate_run, module_device, to_device
from ..envs.base import PHASE_COMBAT, PHASE_DONE, PHASE_OTHER
from .evaluator import Evaluator

DEATH_LOGV = -30.0  # 선택 직후 죽은 경우 (토이에는 없지만 실제 게임에는 HP 를 깎는 선택이 있다)


@dataclass
class _Node:
    leaf: int | None = None  # 평가할 상태의 인덱스
    children: dict[int, list[_Node]] = field(default_factory=dict)  # 행동 -> 표본들


def _signature(sim) -> tuple:
    """결과 상태 비교용 요약. 같으면 같은 상태로 본다."""
    s = sim.run_state()
    screen = sim.screen_kind() if sim.phase == PHASE_OTHER else ""
    return (sim.phase, screen, s.hp, s.max_hp, s.gold, tuple(s.potions), tuple(s.relics),
            tuple((c.id, c.upgraded, c.enchant) for c in s.deck))


def _logmeanexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs) / len(xs))


class ExpectedValueChooser:
    def __init__(self, evaluator: Evaluator, vocab, samples: int = 8, follow_samples: int = 2, max_depth: int = 2):
        self.evaluator = evaluator
        self.vocab = vocab
        self.samples = samples
        self.follow_samples = follow_samples
        self.max_depth = max_depth

    # ---- 트리 펼치기 ----
    def _expand(self, sim, depth: int, rng: np.random.Generator, leaves: list) -> _Node:
        if sim.phase != PHASE_OTHER or depth >= self.max_depth:
            while sim.phase == PHASE_OTHER:  # 깊이 제한: 남은 화면은 규칙 정책으로 마무리
                sim.default_step(rng)
            leaves.append(sim)
            return _Node(leaf=len(leaves) - 1)
        node = _Node()
        k = self.samples if depth == 0 else self.follow_samples
        for a in sim.choice_actions():
            outcomes = []
            for i in range(k):
                c = sim.clone(int(rng.integers(2**31)))
                self._created.append(c)
                c.step_choice(a)
                outcomes.append(c)
                if i == 1 and _signature(outcomes[0]) == _signature(outcomes[1]):
                    outcomes = outcomes[:1]  # 결정적인 선택: 한 번이면 충분
                    break
            node.children[a] = [self._expand(c, depth + 1, rng, leaves) for c in outcomes]
        return node

    # ---- 잎 평가 ----
    @torch.no_grad()
    def _leaf_values(self, sims: list) -> list[float]:
        values = [0.0] * len(sims)
        idx = []
        for i, s in enumerate(sims):
            if s.phase == PHASE_DONE:
                values[i] = 0.0 if s.won else DEATH_LOGV
            else:
                idx.append(i)
        if idx:
            states = [sims[i].run_state() for i in idx]
            batch = to_device(collate_run(states, self.vocab), module_device(self.evaluator))
            ev = self.evaluator(batch)
            logv = ev.log_run_value.double()
            for j, i in enumerate(idx):
                v = float(logv[j])
                cur = states[j].map.current if states[j].map is not None else None
                if sims[i].phase == PHASE_COMBAT and cur is not None:
                    v += math.log(max(1e-12, 1.0 - float(ev.hazard[j, cur])))  # 아직 안 끝난 이 방의 위험
                values[i] = v
        return values

    def _value(self, node: _Node, leaf_v: list[float]) -> float:
        if node.leaf is not None:
            return leaf_v[node.leaf]
        return max(self._action_value(node, a, leaf_v) for a in node.children)

    def _action_value(self, node: _Node, a: int, leaf_v: list[float]) -> float:
        return _logmeanexp([self._value(k, leaf_v) for k in node.children[a]])

    def evaluate(self, sim, rng: np.random.Generator) -> dict[int, float]:
        """지금 화면의 행동마다 log E[V]."""
        assert sim.phase == PHASE_OTHER
        leaves: list = []
        self._created: list = []
        try:
            root = self._expand(sim, 0, rng, leaves)
            leaf_v = self._leaf_values(leaves)
            return {a: self._action_value(root, a, leaf_v) for a in root.children}
        finally:
            for c in self._created:  # 실제 시뮬레이터에서는 복사본이 네이티브 핸들이다
                c.close()
            self._created = []

    def choose(self, sim, rng: np.random.Generator) -> int:
        values = self.evaluate(sim, rng)
        return max(values, key=values.get)
