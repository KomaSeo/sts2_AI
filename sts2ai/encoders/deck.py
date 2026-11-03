"""DeckEncoder: 덱 = 순서 없는 카드 집합.

출력
- pooled [B, d]: 덱 전체 요약 ("이 덱이 얼마나 강한가")
- tokens [B, N, d]: 카드별 표현 (카드 제거/강화/변환 선택 시 포인터 대상)
"""

from __future__ import annotations

import torch
from torch import nn

from .embeddings import SharedEmbeddings, set_transformer, with_cls


class DeckEncoder(nn.Module):
    def __init__(self, emb: SharedEmbeddings, layers: int = 2, heads: int = 4):
        super().__init__()
        d = emb.d
        self.emb = emb
        self.cost = nn.Linear(1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.stats = nn.Linear(3, d)  # 덱 크기, 강화 비율, 평균 비용 (초반 학습 가속용)
        self.body = set_transformer(d, layers, heads)
        self.out = nn.LayerNorm(d)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb.card_vec(batch["deck_ids"], batch["deck_upg"], batch["deck_ench"])
        x = x + self.cost(batch["deck_cost"].unsqueeze(-1))
        x, mask = with_cls(self.cls + self.stats(batch["deck_stats"]).unsqueeze(1), x, batch["deck_mask"])
        h = self.body(x, src_key_padding_mask=~mask)
        h = self.out(h)
        return h[:, 0], h[:, 1:]
