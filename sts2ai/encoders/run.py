"""RunEncoder: 덱 + 유물/포션 + 상태 + 보스 → 런 맥락 ctx, 맵 노드 벡터, 맵까지 합친 run_vec.

- ctx     [B,d]  : 덱·유물·상태·보스. 맵을 읽을 때 FiLM 조건으로 쓴다.
- nodes   [B,V,d]: 맵 노드별 벡터 (ctx 로 조건화된 메시지 전달 결과).
- run_vec [B,d]  : ctx + 맵 요약(다음 선택지, 남은 길). 미래 막 평가기와 전투 정책이 읽는다.
두 평가 에이전트가 이 트렁크를 공유한다.
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
        self.with_map = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.LayerNorm(d))

    def context(self, batch: dict) -> torch.Tensor:
        deck, _ = self.deck(batch)
        boss = self.emb.encounter(batch["boss"]) + self.emb.act(batch["act"])
        return self.fuse(torch.cat([deck, self.items(batch), self.status(batch), boss], dim=-1))

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.context(batch)
        nodes = self.map(batch, ctx)
        run_vec = self.with_map(torch.cat([ctx, self.map.summary(nodes, batch)], dim=-1))
        return ctx, nodes, run_vec
