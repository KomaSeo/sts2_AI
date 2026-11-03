"""학습된 에이전트를 고정 시드로 평가 (탐색 없이 greedy).

    python scripts/evaluate.py --ckpt runs/toy1/agent.pt --runs 50
    python scripts/evaluate.py --random --runs 50      # 학습 안 한 에이전트 기준선
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.train.loop import Agent, Config, play_run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["toy", "sts2"], default="toy")
    p.add_argument("--sts2-repo", default="../sts2")
    p.add_argument("--ckpt")
    p.add_argument("--random", action="store_true", help="학습 안 한 에이전트")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--sample", action="store_true", help="greedy 대신 정책 확률대로 샘플링")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--seed-base", type=int, default=10_000)
    args = p.parse_args()
    if args.env == "toy":
        from sts2ai.envs.toy import ToyRun

        run = ToyRun()
    else:
        from sts2ai.envs.sts2_adapter import Sts2Run

        run = Sts2Run(args.sts2_repo)
    torch.manual_seed(0)
    agent = Agent(run.vocab, args.d_model)
    if not args.random:
        agent.load_state_dict(torch.load(args.ckpt))
    cfg = Config()
    rng = np.random.default_rng(0)
    res = []
    for i in range(args.runs):
        _, _, r = play_run(agent, run, cfg, rng, args.seed_base + i, use_evaluator=True,
                           collect_grpo=False, greedy=not args.sample)
        res.append(r)
    acts = np.bincount([r["act"] for r in res if r["died"]], minlength=run.vocab.n_acts)
    print(f"runs={len(res)} win_rate={np.mean([r['won'] for r in res]):.3f} "
          f"mean_floor={np.mean([r['floor'] for r in res]):.2f} deaths_by_act={acts.tolist()}")


if __name__ == "__main__":
    main()
