"""전투 에이전트만 따로 학습 (무작위 덱·HP·포션으로 모든 적과 싸우는 GRPO).

    python scripts/train_combat.py --iterations 200 --out-dir runs/combat
    python scripts/train_combat.py --iterations 400 --out-dir runs/combat --resume   # 이어서
    python scripts/train.py --env toy --init-policy runs/combat/combat_policy.pt     # 런 학습에 이어 붙이기
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.train.cli import add_config_args, config_from_args  # noqa: E402
from sts2ai.train.combat_train import CombatConfig, train_combat  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["toy"], default="toy",
                   help="실제 시뮬레이터는 임의 덱·HP 로 전투를 시작하는 API 가 확인되면 추가")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="--out-dir 에 체크포인트가 있으면 이어서 학습")
    add_config_args(p, CombatConfig)
    args = p.parse_args()
    cfg = config_from_args(args, CombatConfig)
    from sts2ai.envs.toy import ToyRun

    train_combat(ToyRun(ascension=args.ascension), cfg, resume=args.resume)


if __name__ == "__main__":
    main()
