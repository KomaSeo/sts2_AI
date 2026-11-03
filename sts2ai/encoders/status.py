"""StatusEncoder: HP, 골드, 층, 승천, 막 같은 스칼라 수치."""

from __future__ import annotations

import torch
from torch import nn


class StatusEncoder(nn.Module):
    def __init__(self, n_acts: int, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6 + n_acts, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(batch["status"])
