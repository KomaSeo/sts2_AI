"""정책을 고정한 채 런을 모으고, 평가기를 처음부터 학습하는 도구 (실험용).

RL 루프 안에서는 정책이 계속 바뀌고 같은 버퍼를 반복 학습해서 평가기가 과적합했다.
여기서는 데이터가 고정돼 있고, 검증 손실이 나빠지면 조기 종료한다.
"""

from __future__ import annotations

import random
import time

import torch

from ..batching import collate_run
from .evaluator_train import Snapshot, eval_bce, make_labels
from .loop import Agent, Config, play_run


def collect_runs(agent: Agent, run, seeds: list[int], rng, cfg: Config | None = None,
                 greedy: bool = False, log=None) -> list[list[Snapshot]]:
    cfg = cfg or Config()
    out = []
    t = time.time()
    for i, s in enumerate(seeds):
        snaps, _, _ = play_run(agent, run, cfg, rng, s, use_evaluator=True, collect_grpo=False, greedy=greedy)
        out.append(snaps)
        if log and (i + 1) % 100 == 0:
            log(f"  수집 {i + 1}/{len(seeds)}판 ({time.time() - t:.0f}초)")
    return out


def train_fresh_evaluator(vocab, train: list[Snapshot], val: list[Snapshot], d: int = 64,
                          max_epochs: int = 30, patience: int = 3, seed: int = 0):
    """새 평가기를 처음부터 학습. 반환: (평가기, 에폭 수, 최고 검증 손실)."""
    torch.manual_seed(seed)
    random.seed(seed)
    ev = Agent(vocab, d).evaluator
    opt = torch.optim.AdamW(ev.parameters(), lr=1e-3, weight_decay=0.01)
    best, best_state, bad, epochs = float("inf"), None, 0, 0
    for _ in range(max_epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        for start in range(0, len(order), 64):
            chunk = [train[i] for i in order[start : start + 64]]
            batch = collate_run([s.state for s in chunk], vocab)
            nl, al = make_labels(chunk, batch["node_type"].shape[1], vocab.n_acts)
            loss, _ = ev.loss(batch, nl, al)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ev.parameters(), 1.0)
            opt.step()
        epochs += 1
        v = eval_bce(ev, val, vocab)
        if v < best:
            best, best_state, bad = v, {k: t.clone() for k, t in ev.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    ev.load_state_dict(best_state)
    ev.eval()
    return ev, epochs, best
