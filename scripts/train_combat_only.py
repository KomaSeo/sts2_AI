r"""전투 에이전트만 따로 학습 (r33hab 시뮬레이터, 무작위 덱, 전투 결과 점수, 처음부터).

    .venv\Scripts\python scripts\train_combat_only.py --out-dir runs\combat_only_v1
    .venv\Scripts\python scripts\train_combat_only.py --iterations 5 --workers 4 --out-dir runs\combat_only_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.train.combat_only import CombatConfig, train_combat_only  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sts2-repo", default="../sts2")
    for name, f in CombatConfig.__dataclass_fields__.items():
        if isinstance(f.default, bool):
            p.add_argument(f"--{name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(f"--{name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = p.parse_args()
    cfg = CombatConfig(**{k: getattr(args, k) for k in CombatConfig.__dataclass_fields__})
    train_combat_only(args.sts2_repo, cfg)


if __name__ == "__main__":
    main()
