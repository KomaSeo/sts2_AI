"""dataclass 설정 → argparse 인자."""

from __future__ import annotations

import argparse
from dataclasses import fields


def add_config_args(p: argparse.ArgumentParser, cls) -> None:
    """필드마다 --이름-형식 인자를 만든다. bool 은 --x / --no-x (type=bool 은 "False" 도 True 로 읽는다)."""
    for f in fields(cls):
        flag = f"--{f.name.replace('_', '-')}"
        if isinstance(f.default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(flag, type=type(f.default), default=f.default)


def config_from_args(args: argparse.Namespace, cls):
    return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})
