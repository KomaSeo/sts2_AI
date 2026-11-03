"""이 컴퓨터에서 학습이 얼마나 빠른지 잰다. 다른 컴퓨터와 비교하려고 만든 스크립트.

    python scripts/benchmark.py            # 결과를 화면에 출력
    python scripts/benchmark.py --json out.json

각 항목은 학습 루프에서 시간을 실제로 쓰는 부분이다.
1) 런 플레이 (전투 정책 + 맵 선택, GRPO 없음)        — 평가기 데이터 수집 속도
2) GRPO 분기 전투 롤아웃                              — 전투 정책 학습 데이터 수집 속도
3) 평가기 학습 스텝 (배치 64)                          — CPU 와 GPU(있으면) 둘 다
4) 전투 정책 GRPO 업데이트 스텝 (배치 256)             — CPU 와 GPU(있으면) 둘 다
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.batching import collate_combat, collate_run  # noqa: E402
from sts2ai.envs.toy import ToyRun  # noqa: E402
from sts2ai.train.evaluator_train import make_labels  # noqa: E402
from sts2ai.train.grpo import play  # noqa: E402
from sts2ai.train.loop import Agent, Config, play_run  # noqa: E402


def timed(fn, repeat: int) -> float:
    fn()  # 워밍업
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t) / repeat


def devices() -> list[str]:
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        out.append("mps")
    return out


def to(batch: dict, dev: str) -> dict:
    return {k: v.to(dev) for k, v in batch.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json")
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    run = ToyRun()
    agent = Agent(run.vocab, 64)
    cfg = Config()
    res: dict = {
        "machine": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "devices": devices(),
    }

    # 1) 런 플레이
    snaps_all = []
    t = time.perf_counter()
    for i in range(args.runs):
        s, _, _ = play_run(agent, run, cfg, rng, 1000 + i, use_evaluator=True, collect_grpo=False)
        snaps_all.extend(s)
    dt = time.perf_counter() - t
    res["runs_per_sec"] = args.runs / dt

    # 2) GRPO 분기 롤아웃
    run.reset(7)
    run.choose_node(run.map_choices()[0])
    n_fork, steps = 16, 0
    t = time.perf_counter()
    for k in range(n_fork):
        steps += len(play(agent.policy, run.fork_combat(k)).actions)
    dt = time.perf_counter() - t
    res["fork_rollouts_per_sec"] = n_fork / dt
    res["policy_steps_per_sec"] = steps / dt

    # 3), 4) 학습 스텝
    snaps = snaps_all[:64]
    obs = []
    for k in range(4):
        tr = play(agent.policy, run.fork_combat(100 + k))
        obs.extend(tr.obs)
    obs = (obs * (256 // max(1, len(obs)) + 1))[:256]
    for dev in res["devices"]:
        ag = Agent(run.vocab, 64)
        ag.evaluator.to(dev)
        ag.policy.to(dev)
        opt_e = torch.optim.AdamW(ag.evaluator.parameters(), lr=1e-3)
        opt_p = torch.optim.Adam(ag.policy.parameters(), lr=3e-4)
        rb = collate_run([s.state for s in snaps], run.vocab)
        nl, al = make_labels(snaps, rb["node_type"].shape[1], run.vocab.n_acts)
        rb, nl, al = to(rb, dev), nl.to(dev), al.to(dev)
        cbatch = to(collate_combat(obs), dev)
        a = torch.zeros(256, dtype=torch.long, device=dev)
        tg = torch.zeros(256, dtype=torch.long, device=dev)

        def eval_step():
            loss, _ = ag.evaluator.loss(rb, nl, al)
            opt_e.zero_grad()
            loss.backward()
            opt_e.step()
            if dev == "cuda":
                torch.cuda.synchronize()

        def pol_step():
            lp, ent = ag.policy.log_prob(cbatch, a, tg)
            loss = -(lp.mean() + 0.01 * ent.mean())
            opt_p.zero_grad()
            loss.backward()
            opt_p.step()
            if dev == "cuda":
                torch.cuda.synchronize()

        res[f"evaluator_step_ms[{dev}]"] = timed(eval_step, 10) * 1000
        res[f"policy_step_ms[{dev}]"] = timed(pol_step, 10) * 1000

    width = max(len(k) for k in res)
    for k, v in res.items():
        print(f"{k:<{width}}  {v:.2f}" if isinstance(v, float) else f"{k:<{width}}  {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
