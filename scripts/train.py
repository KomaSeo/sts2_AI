"""학습 실행.

    python scripts/train.py --env toy --iterations 30
    python scripts/train.py --env toy --workers 8 --device cuda   # 수집 워커 8개, 학습은 GPU
    python scripts/train.py --env sts2 --sts2-repo ../sts2 --iterations 200   # 실제 시뮬레이터 (빌드 필요)
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.train.loop import Config, train  # noqa: E402


def arg_type(default):
    """bool 옵션은 'False'/'0' 도 거짓으로 읽는다 (type=bool 은 빈 문자열이 아니면 모두 참)."""
    if isinstance(default, bool):
        return lambda s: s.strip().lower() in ("1", "true", "yes", "y", "on")
    return type(default)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["toy", "sts2"], default="toy")
    p.add_argument("--sts2-repo", default="../sts2", help="r33hab/sts2 체크아웃 경로 (빌드된 out/ 포함)")
    p.add_argument("--ascension", type=int, default=0)
    for name, field in Config.__dataclass_fields__.items():
        p.add_argument(f"--{name.replace('_', '-')}", type=arg_type(field.default), default=field.default)
    args = p.parse_args()
    cfg = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
    # 워커 프로세스가 각자 환경을 만들 수 있게 pickle 가능한 생성 함수로 넘긴다.
    if args.env == "toy":
        from sts2ai.envs.toy import ToyRun

        make_run = partial(ToyRun, ascension=args.ascension)
    else:
        from sts2ai.envs.sts2_adapter import Sts2Run

        make_run = partial(Sts2Run, args.sts2_repo, ascension=args.ascension)
    train(None, cfg, make_run=make_run)


if __name__ == "__main__":
    main()
