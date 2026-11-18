r"""전투만 학습한 폴더의 전투 기록을 HTML 한 장(+ Markdown)으로 만든다. 학습 중에는 학습이 반복마다 fights.html 을 직접 다시 쓴다.

    .venv\Scripts\python scriptsights_html.py runs\combat_only_v3          # → runs\combat_only_v3ights.html
    .venv\Scripts\python scriptsights_html.py runs\combat_only_v3 --md     # 반복마다 fights\iter_NNNN.md 도
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.train.fight_view import write_all  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--md", action="store_true", help="반복마다 fights/iter_NNNN.md 도 쓴다")
    p.add_argument("--artifact", default="", help="아티팩트로 올릴 HTML 을 이 경로에 쓴다")
    args = p.parse_args()
    n = write_all(Path(args.folder), md=args.md, artifact=args.artifact)
    print(f"{Path(args.folder) / 'fights.html'} ({n}개 반복)")


if __name__ == "__main__":
    main()
