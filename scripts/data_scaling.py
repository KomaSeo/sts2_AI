"""평가기 과적합이 데이터 부족 때문인지 확인하는 실험.

전투 정책(과 맵 선택)을 체크포인트 상태로 고정하고 런을 모은 뒤, 학습 런 수만 바꿔 가며
평가기를 처음부터 다시 학습한다. 모두 같은 테스트 런으로 보정을 잰다.
- 데이터가 늘수록 확신 과잉(예측 0.7 이상인데 실제는 낮음)과 BCE 가 좋아지면 → 데이터 부족이 주원인
- 별로 안 좋아지면 → 모델/학습 방식(외우기)이 주원인

    python scripts/data_scaling.py --ckpt runs/toy_v3/agent.pt --sizes 50 100 200 400 800
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.agents.evaluator import IGNORE  # noqa: E402
from sts2ai.batching import collate_run  # noqa: E402
from sts2ai.envs.toy import ToyRun  # noqa: E402
from sts2ai.train.evaluator_train import Snapshot, eval_bce, make_labels  # noqa: E402
from sts2ai.train.loop import Agent, Config, play_run  # noqa: E402

EPS = 1e-6


def collect(agent: Agent, run, seeds: list[int], rng) -> list[list[Snapshot]]:
    out = []
    for s in seeds:
        snaps, _, _ = play_run(agent, run, Config(), rng, s, use_evaluator=True, collect_grpo=False, greedy=False)
        out.append(snaps)
    return out


@torch.no_grad()
def event_preds(evaluator, runs: list[list[Snapshot]], vocab):
    """사건 단위(들어가기 직전 마지막 예측) 노드/막 예측과 라벨."""
    evaluator.eval()
    node, act = {}, {}
    for r, snaps in enumerate(runs):
        if not snaps:
            continue
        batch = collate_run([s.state for s in snaps], vocab)
        ev = evaluator(batch)
        nl, al = make_labels(snaps, batch["node_type"].shape[1], vocab.n_acts)
        for b, s in enumerate(snaps):
            for v in (nl[b] != IGNORE).nonzero().flatten().tolist():
                node[(r, s.state.act, v)] = (float(ev.hazard[b, v]), float(nl[b, v]))
            for k in (al[b] != IGNORE).nonzero().flatten().tolist():
                act[(r, k)] = (float(ev.p_act[b, k]), float(al[b, k]))
    evaluator.train()
    to_np = lambda d: (np.array([p for p, _ in d.values()]), np.array([y for _, y in d.values()]))  # noqa: E731
    return to_np(node), to_np(act)


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {}
    bce = float(-(y * np.log(pred + EPS) + (1 - y) * np.log(1 - pred + EPS)).mean())
    base = y.mean()
    bce0 = float(-(base * np.log(base + EPS) + (1 - base) * np.log(1 - base + EPS)))
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0001]
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.any():
            ece += m.mean() * abs(pred[m].mean() - y[m].mean())
    hi = pred >= 0.7
    return {
        "events": int(len(y)),
        "bce": bce,
        "bce_const": bce0,
        "ece": float(ece),
        "hi_n": int(hi.sum()),
        "hi_pred": float(pred[hi].mean()) if hi.any() else float("nan"),
        "hi_actual": float(y[hi].mean()) if hi.any() else float("nan"),
    }


def train_fresh(vocab, train: list[Snapshot], val: list[Snapshot], d: int, max_epochs: int, patience: int, seed: int):
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
    return ev, epochs, best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    p.add_argument("--val-runs", type=int, default=80)
    p.add_argument("--test-runs", type=int, default=150)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--out", default="runs/data_scaling.json")
    p.add_argument("--threads", type=int, default=1, help="PyTorch 스레드 수")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    run = ToyRun()
    collector = Agent(run.vocab, args.d_model)
    collector.load_state_dict(torch.load(args.ckpt))
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    n_train = max(args.sizes)
    train_runs = collect(collector, run, [300_000 + i for i in range(n_train)], rng)
    val_runs = collect(collector, run, [400_000 + i for i in range(args.val_runs)], rng)
    test_runs = collect(collector, run, [500_000 + i for i in range(args.test_runs)], rng)
    print(f"수집: 학습 {n_train}판, 검증 {args.val_runs}판, 테스트 {args.test_runs}판 ({time.time() - t0:.0f}초)", flush=True)
    val = [s for r in val_runs for s in r]

    results = []
    for n in args.sizes:
        t = time.time()
        train = [s for r in train_runs[:n] for s in r]
        deaths = sum(1 for r in train_runs[:n] if r and r[-1].died_node is not None)
        ev, epochs, val_loss = train_fresh(run.vocab, train, val, args.d_model, args.max_epochs, args.patience, seed=0)
        (tn, ty), (ta, tay) = event_preds(ev, train_runs[:n], run.vocab)
        (nn_, ny), (na, nay) = event_preds(ev, test_runs, run.vocab)
        row = {
            "train_runs": n,
            "train_snapshots": len(train),
            "train_node_deaths": deaths,
            "epochs": epochs,
            "val_loss": val_loss,
            "node_train": metrics(tn, ty),
            "node_test": metrics(nn_, ny),
            "act_train": metrics(ta, tay),
            "act_test": metrics(na, nay),
            "sec": round(time.time() - t, 1),
        }
        results.append(row)
        nt, at = row["node_test"], row["act_test"]
        print(
            f"n={n:4d} snaps={len(train):5d} 학습셋사망={deaths:3d} ep={epochs:2d} | "
            f"노드 test BCE {nt['bce']:.3f} (상수 {nt['bce_const']:.3f}) ECE {nt['ece']:.3f} "
            f"0.7↑ {nt['hi_n']}개 예측 {nt['hi_pred']:.2f}/실제 {nt['hi_actual']:.2f} "
            f"train BCE {row['node_train']['bce']:.3f} | "
            f"막 test BCE {at.get('bce', float('nan')):.3f} (상수 {at.get('bce_const', float('nan')):.3f}) "
            f"ECE {at.get('ece', float('nan')):.3f} | {row['sec']}초",
            flush=True,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
