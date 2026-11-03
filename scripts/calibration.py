"""평가 에이전트 보정(calibration) 확인: 예측한 사망 확률 vs 실제 사망률.

학습에 쓰지 않은 시드로 런을 돌려, 맵 스냅숏마다
- 노드 평가기: 실제로 지나간 노드의 hazard 예측 vs 그 노드에서 실제로 죽었나
- 미래 막 평가기: 실제로 도착한 미래 막의 p_act 예측 vs 그 막에서 실제로 죽었나
를 모아 구간별로 비교한다. 예측 0.3 구간의 실제 사망률이 0.3 근처면 잘 보정된 것이다.

    python scripts/calibration.py --ckpt runs/toy/agent.pt --runs 50
"""

from __future__ import annotations

import argparse
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


def table(name: str, pred: np.ndarray, label: np.ndarray) -> None:
    print(f"\n[{name}] 표본 {len(pred)}개, 평균 예측 {pred.mean():.3f} / 실제 {label.mean():.3f}")
    print("  예측 구간        개수   평균 예측   실제 사망률")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum():
            print(f"  [{lo:.2f}, {min(hi, 1):.2f})  {m.sum():6d}   {pred[m].mean():9.3f}   {label[m].mean():11.3f}")
    eps = 1e-6
    bce = -(label * np.log(pred + eps) + (1 - label) * np.log(1 - pred + eps)).mean()
    base = label.mean()
    bce0 = -(base * np.log(base + eps) + (1 - base) * np.log(1 - base + eps))
    print(f"  BCE {bce:.4f}  (항상 평균값만 말하는 예측기의 BCE {bce0:.4f})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["toy", "sts2"], default="toy")
    p.add_argument("--sts2-repo", default="../sts2")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--seed-base", type=int, default=50_000)
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
    rng = np.random.default_rng(0)
    snaps = []
    for i in range(args.runs):
        s, _, _ = play_run(agent, run, Config(), rng, args.seed_base + i, use_evaluator=True,
                           collect_grpo=False, greedy=not args.sample)
        snaps.extend(s)
    node_p, node_y, act_p, act_y = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(snaps), 64):
            chunk = snaps[start : start + 64]
            batch = collate_run([s.state for s in chunk], run.vocab)
            ev = agent.evaluator(batch)
            nl, al = make_labels(chunk, batch["node_type"].shape[1], run.vocab.n_acts)
            m = nl != IGNORE
            node_p.append(ev.hazard[m].numpy())
            node_y.append(nl[m].numpy())
            m = al != IGNORE
            act_p.append(ev.p_act[m].numpy())
            act_y.append(al[m].numpy())
    table("노드 평가기: 지나간 노드에서 죽을 확률", np.concatenate(node_p), np.concatenate(node_y))
    if sum(len(a) for a in act_y):
        table("미래 막 평가기: 도착한 미래 막에서 죽을 확률", np.concatenate(act_p), np.concatenate(act_y))


if __name__ == "__main__":
    main()
