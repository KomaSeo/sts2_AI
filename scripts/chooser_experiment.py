"""기대값 선택기가 '좋은 평가기' 를 쓰면 규칙 정책보다 나은지 확인하는 실험.

1) 체크포인트의 전투 정책·맵 선택을 고정하고 (전투·맵 밖은 규칙 정책) 런을 모은다.
2) 그 데이터로 선택기 전용 평가기를 처음부터 학습한다 (조기 종료).
3) 같은 테스트 시드에서 세 가지를 비교한다.
   - rules        : 규칙 정책
   - ev[원래]     : RL 루프에서 학습된 평가기로 기대값 선택
   - ev[새로 학습] : 2) 의 평가기로 기대값 선택
   전투 정책은 원래 평가기의 run_vec 으로 학습됐으므로 그대로 두고, 선택기만 평가기를 바꾼다.

    python scripts/chooser_experiment.py --ckpt runs/toy_v4/agent.pt --train-runs 400 --test-runs 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.envs.toy import ToyRun  # noqa: E402
from sts2ai.train.loop import Agent, Config, play_run  # noqa: E402
from sts2ai.train.offline import collect_runs, train_fresh_evaluator  # noqa: E402


def evaluate(agent: Agent, run, cfg: Config, seeds: list[int]) -> dict:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    res = [play_run(agent, run, cfg, rng, s, use_evaluator=True, collect_grpo=False, greedy=True)[2] for s in seeds]
    wins = np.array([r["won"] for r in res], dtype=float)
    n = len(wins)
    se = float(np.sqrt(wins.mean() * (1 - wins.mean()) / n))
    return {"runs": n, "win_rate": float(wins.mean()), "win_se": se,
            "mean_floor": float(np.mean([r["floor"] for r in res]))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train-runs", type=int, default=400)
    p.add_argument("--val-runs", type=int, default=60)
    p.add_argument("--test-runs", type=int, default=100)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="runs/chooser_experiment.json")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    log = lambda m: print(m, flush=True)  # noqa: E731

    run = ToyRun()
    agent = Agent(run.vocab, args.d_model)
    agent.load_state_dict(torch.load(args.ckpt))
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t = time.time()
    train = collect_runs(agent, run, [700_000 + i for i in range(args.train_runs)], rng, log=log)
    val = collect_runs(agent, run, [800_000 + i for i in range(args.val_runs)], rng)
    log(f"수집 끝 ({time.time() - t:.0f}초)")
    fresh, epochs, vloss = train_fresh_evaluator(
        run.vocab, [s for r in train for s in r], [s for r in val for s in r], args.d_model)
    log(f"새 평가기: {epochs} 에폭, 검증 손실 {vloss:.3f}")

    seeds = [900_000 + i for i in range(args.test_runs)]
    out = {}
    out["rules"] = evaluate(agent, run, Config(chooser="rules"), seeds)
    log(f"rules        {out['rules']}")
    out["ev_original"] = evaluate(agent, run, Config(chooser="ev"), seeds)
    log(f"ev[원래]     {out['ev_original']}")
    agent.chooser_evaluator = fresh
    out["ev_fresh"] = evaluate(agent, run, Config(chooser="ev"), seeds)
    log(f"ev[새로 학습] {out['ev_fresh']}")
    out["fresh_epochs"], out["fresh_val_loss"] = epochs, vloss
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
