"""평가 에이전트 보정(calibration) 확인: 예측한 사망 확률 vs 실제 사망률.

학습에 쓰지 않은 시드로 런을 돌려, 맵 스냅숏마다
- 노드 평가기: 실제로 지나간 노드의 hazard 예측 vs 그 노드에서 실제로 죽었나
- 미래 막 평가기: 실제로 도착한 미래 막의 p_act 예측 vs 그 막에서 실제로 죽었나
를 모아 구간별로 비교한다. 예측 0.3 구간의 실제 사망률이 0.3 근처면 잘 보정된 것이다.

같은 사건(예: 어느 런의 1막 보스전)은 그 막의 스냅숏마다 한 번씩 예측되므로, 쌍을 그대로
세면 표본이 부풀려진다. 그래서 두 가지 표를 낸다.
- 모든 쌍: 여러 시점에서 내린 예측 전부 (먼 미래 예측 포함)
- 사건 단위: 사건마다 가장 마지막 스냅숏(그 노드/막에 들어가기 직전)의 예측 하나만.
  실제 사망률 옆에 Wilson 95% 신뢰구간을 붙인다.

    python scripts/calibration.py --ckpt runs/toy/agent.pt --runs 50 --sample
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.agents.evaluator import IGNORE  # noqa: E402
from sts2ai.batching import collate_run  # noqa: E402
from sts2ai.train.evaluator_train import make_labels  # noqa: E402
from sts2ai.train.loop import Agent, Config, play_run  # noqa: E402

BINS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0001]
NODE_NAMES = {1: "일반", 2: "엘리트", 3: "휴식", 4: "상점", 5: "보물", 6: "보스", 7: "이벤트", 8: "고대"}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def bce(pred: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    loss = -(label * np.log(pred + eps) + (1 - label) * np.log(1 - pred + eps)).mean()
    base = label.mean()
    loss0 = -(base * np.log(base + eps) + (1 - base) * np.log(1 - base + eps))
    return float(loss), float(loss0)


def table(name: str, pred: np.ndarray, label: np.ndarray, ci: bool) -> None:
    print(f"\n[{name}] 표본 {len(pred)}개, 평균 예측 {pred.mean():.3f} / 실제 {label.mean():.3f}")
    print("  예측 구간        개수   평균 예측   실제 사망률" + ("   95% 신뢰구간" if ci else ""))
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (pred >= lo) & (pred < hi)
        n = int(m.sum())
        if n:
            line = f"  [{lo:.2f}, {min(hi, 1):.2f})  {n:6d}   {pred[m].mean():9.3f}   {label[m].mean():11.3f}"
            if ci:
                a, b = wilson(label[m].sum(), n)
                line += f"   [{a:.2f}, {b:.2f}]"
            print(line)
    loss, loss0 = bce(pred, label)
    print(f"  BCE {loss:.4f}  (항상 평균값만 말하는 예측기의 BCE {loss0:.4f})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["toy", "sts2"], default="toy")
    p.add_argument("--sts2-repo", default="../sts2")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--seed-base", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0, help="정책 샘플링·맵 탐색 난수 (같은 값이면 같은 결과)")
    p.add_argument("--sample", action="store_true", help="학습 때처럼 정책을 샘플링 (기본은 greedy)")
    args = p.parse_args()
    if args.env == "toy":
        from sts2ai.envs.toy import ToyRun

        run = ToyRun()
    else:
        from sts2ai.envs.sts2_adapter import Sts2Run

        run = Sts2Run(args.sts2_repo)
    agent = Agent(run.vocab, args.d_model)
    agent.load_state_dict(torch.load(args.ckpt))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # (런, 스냅숏 순번, 사건 키, 예측, 라벨, 노드 종류)
    node_rows: list[tuple] = []
    act_rows: list[tuple] = []
    for i in range(args.runs):
        snaps, _, _ = play_run(agent, run, Config(), rng, args.seed_base + i, use_evaluator=True,
                               collect_grpo=False, greedy=not args.sample)
        if not snaps:
            continue
        with torch.no_grad():
            batch = collate_run([s.state for s in snaps], run.vocab)
            ev = agent.evaluator(batch)
            nl, al = make_labels(snaps, batch["node_type"].shape[1], run.vocab.n_acts)
        for b, s in enumerate(snaps):
            for v in (nl[b] != IGNORE).nonzero().flatten().tolist():
                key = (i, s.state.act, v)
                node_rows.append((i, b, key, float(ev.hazard[b, v]), float(nl[b, v]), s.state.map.node_types[v]))
            for k in (al[b] != IGNORE).nonzero().flatten().tolist():
                act_rows.append((i, b, (i, k), float(ev.p_act[b, k]), float(al[b, k]), None))

    def last_per_event(rows):
        latest: dict = {}
        for r in rows:
            if r[2] not in latest or r[1] > latest[r[2]][1]:
                latest[r[2]] = r
        return list(latest.values())

    for name, rows in (("노드 평가기", node_rows), ("미래 막 평가기", act_rows)):
        if not rows:
            continue
        ev_rows = last_per_event(rows)
        pred_all = np.array([r[3] for r in rows])
        y_all = np.array([r[4] for r in rows])
        pred_ev = np.array([r[3] for r in ev_rows])
        y_ev = np.array([r[4] for r in ev_rows])
        table(f"{name} — 모든 쌍", pred_all, y_all, ci=False)
        table(f"{name} — 사건 단위 (들어가기 직전 예측)", pred_ev, y_ev, ci=True)
        if name == "노드 평가기":
            print("  노드 종류별 (사건 단위):")
            for t in sorted({r[5] for r in ev_rows}):
                sub = [r for r in ev_rows if r[5] == t]
                k, n = sum(r[4] for r in sub), len(sub)
                a, b = wilson(k, n)
                print(f"    {NODE_NAMES.get(t, t):>4}: 사건 {n:4d}  평균 예측 {np.mean([r[3] for r in sub]):.3f}"
                      f"  실제 {k / n:.3f} [{a:.2f}, {b:.2f}]")


if __name__ == "__main__":
    main()
