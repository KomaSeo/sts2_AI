"""평가 에이전트 학습 데이터 만들기와 학습.

맵에서 다음 노드를 고르는 순간마다 스냅숏을 남기고, 런이 끝나면 라벨을 붙인다.
- 노드 라벨: 스냅숏 이후 이 막에서 실제로 지나간 노드들. 죽은 노드 = 1, 살아서 지나간 노드 = 0.
  죽은 뒤의 노드는 라벨 없음(중도 절단).
- 막 라벨: 스냅숏보다 뒤의 막 k 에 실제로 도착했다면 그 막에서 죽었나(1/0). 못 갔으면 라벨 없음.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from ..agents import Evaluator
from ..agents.evaluator import IGNORE
from ..batching import collate_run
from ..schema import RunState, Vocab


@dataclass
class Snapshot:
    state: RunState
    future_nodes: list[int] = field(default_factory=list)
    died_node: int | None = None
    act_label: list[float] = field(default_factory=list)


class RunRecorder:
    """한 런 동안 스냅숏과 지나간 노드를 기록한다."""

    def __init__(self, n_acts: int):
        self.n_acts = n_acts
        self.snaps: list[Snapshot] = []
        self.visits: list[tuple[int, int]] = []  # (막, 노드)
        self._marks: list[int] = []  # 스냅숏 시점의 visits 길이

    def snapshot(self, state: RunState) -> None:
        self.snaps.append(Snapshot(state))
        self._marks.append(len(self.visits))

    def visit(self, act: int, node: int) -> None:
        self.visits.append((act, node))

    def finish(self, died: bool, death_act: int, death_node: int | None) -> list[Snapshot]:
        for snap, mark in zip(self.snaps, self._marks):
            a = snap.state.act
            snap.future_nodes = [v for act, v in self.visits[mark:] if act == a]
            if died and death_act == a:
                snap.died_node = death_node
            labels = []
            for k in range(self.n_acts):
                if k <= a or k > death_act:  # 현재/과거 막, 또는 도착 못 한 막
                    labels.append(IGNORE)
                else:
                    labels.append(1.0 if (died and k == death_act) else 0.0)
            snap.act_label = labels
        return self.snaps


def make_labels(snaps: list[Snapshot], n_nodes: int, n_acts: int):
    node = torch.full((len(snaps), n_nodes), IGNORE)
    act = torch.full((len(snaps), n_acts), IGNORE)
    for i, s in enumerate(snaps):
        for v in s.future_nodes:
            node[i, v] = 0.0
        if s.died_node is not None and s.died_node in s.future_nodes:
            node[i, s.died_node] = 1.0
        act[i] = torch.tensor(s.act_label)
    return node, act


@torch.no_grad()
def eval_bce(evaluator: Evaluator, snaps: list[Snapshot], vocab: Vocab, batch_size: int = 128) -> float:
    """검증 세트 손실 (노드 + 막 BCE)."""
    if not snaps:
        return float("nan")
    was = evaluator.training
    evaluator.eval()
    total, n = 0.0, 0
    for start in range(0, len(snaps), batch_size):
        chunk = snaps[start : start + batch_size]
        batch = collate_run([s.state for s in chunk], vocab)
        node_l, act_l = make_labels(chunk, batch["node_type"].shape[1], vocab.n_acts)
        loss, _ = evaluator.loss(batch, node_l, act_l)
        total += float(loss) * len(chunk)
        n += len(chunk)
    evaluator.train(was)
    return total / n


def train_evaluator(
    evaluator: Evaluator,
    optim: torch.optim.Optimizer,
    buffer: list[Snapshot],
    vocab: Vocab,
    epochs: int = 3,
    batch_size: int = 64,
    val: list[Snapshot] | None = None,
) -> dict:
    """val 이 있으면 에폭마다 검증 손실을 보고, 나빠지면 직전 가중치로 되돌리고 멈춘다."""
    if not buffer:
        return {}
    stats = {"eval_loss": 0.0, "node_bce": 0.0, "act_bce": 0.0}
    n = 0
    best = eval_bce(evaluator, val, vocab) if val else None
    best_state = {k: v.clone() for k, v in evaluator.state_dict().items()} if val else None
    epochs_run = 0
    for _ in range(epochs):
        order = list(range(len(buffer)))
        random.shuffle(order)
        for start in range(0, len(order), batch_size):
            snaps = [buffer[i] for i in order[start : start + batch_size]]
            batch = collate_run([s.state for s in snaps], vocab)
            node_l, act_l = make_labels(snaps, batch["node_type"].shape[1], vocab.n_acts)
            loss, info = evaluator.loss(batch, node_l, act_l)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
            optim.step()
            stats["eval_loss"] += float(loss)
            stats["node_bce"] += info["node_bce"]
            stats["act_bce"] += info["act_bce"]
            n += 1
        epochs_run += 1
        if val:
            v = eval_bce(evaluator, val, vocab)
            if v > best:
                evaluator.load_state_dict(best_state)  # 과적합 시작 → 되돌리고 중단
                break
            best, best_state = v, {k: t.clone() for k, t in evaluator.state_dict().items()}
    out = {k: v / max(1, n) for k, v in stats.items()}
    if val:
        out["val_bce"] = best
        out["eval_epochs_run"] = epochs_run
    return out
