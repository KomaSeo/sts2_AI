"""RunEncoder: 덱 + 유물/포션 + 상태 + 보스 → 런 맥락 벡터, 그리고 맵 노드 벡터.

두 평가 에이전트(노드 위험도, 미래 막 위험도)가 이 트렁크를 공유한다.
"""

from __future__ import annotations

import torch
from torch import nn

from .deck import DeckEncoder
from .embeddings import SharedEmbeddings
from .items import ItemEncoder
from .map import MapEncoder
from .status import StatusEncoder


class RunEncoder(nn.Module):
    def __init__(self, emb: SharedEmbeddings):
        super().__init__()
        d = emb.d
        self.emb = emb
        self.deck = DeckEncoder(emb)
        self.items = ItemEncoder(emb)
        self.status = StatusEncoder(emb.vocab.n_acts, d)
        self.fuse = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.LayerNorm(d))
        self.map = MapEncoder(emb)

    def context(self, batch: dict) -> torch.Tensor:
        deck, _ = self.deck(batch)
        boss = self.emb.encounter(batch["boss"]) + self.emb.act(batch["act"])
        return self.fuse(torch.cat([deck, self.items(batch), self.status(batch), boss], dim=-1))

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.context(batch)
        return ctx, self.map(batch, ctx)
