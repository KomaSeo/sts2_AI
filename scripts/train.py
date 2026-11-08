"""학습 실행.

    python scripts/train.py --env toy --iterations 30
    python scripts/train.py --env sts2 --sts2-repo ../sts2 --iterations 200   # 실제 시뮬레이터 (빌드 필요)
    python scripts/train.py --env toy --iterations 200 --out-dir runs/a --resume  # 끊긴 곳부터 이어서
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.train.loop import Config, train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["toy", "sts2"], default="toy")
    p.add_argument("--sts2-repo", default="../sts2", help="r33hab/sts2 체크아웃 경로 (빌드된 out/ 포함)")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="--out-dir 에 체크포인트가 있으면 이어서 학습 (없으면 처음부터)")
    for name, field in Config.__dataclass_fields__.items():
        p.add_argument(f"--{name.replace('_', '-')}", type=type(field.default), default=field.default)
    args = p.parse_args()
    cfg = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
    if args.env == "toy":
        from sts2ai.envs.toy import ToyRun

        run = ToyRun(ascension=args.ascension)
    else:
        from sts2ai.envs.sts2_adapter import Sts2Run

        run = Sts2Run(args.sts2_repo, ascension=args.ascension)
    train(run, cfg, resume=args.resume)


if __name__ == "__main__":
    main()
