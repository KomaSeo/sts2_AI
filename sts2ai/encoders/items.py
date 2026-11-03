"""ItemEncoder: 유물 + 포션.

유물은 효과가 제각각이라 숫자 몇 개로 요약할 수 없다. 토큰 하나씩 넣고, 카운터
(몇 번 발동했는지)는 다음 발동 여부를 결정하는 살아 있는 상태라서 같이 넣는다.
"""

from __future__ import annotations

import torch
from torch import nn

from .embeddings import SharedEmbeddings, set_transformer, with_cls


class ItemEncoder(nn.Module):
    def __init__(self, emb: SharedEmbeddings, layers: int = 1, heads: int = 4):
        super().__init__()
        d = emb.d
        self.emb = emb
        self.counter = nn.Linear(1, d)
        self.kind = nn.Embedding(2, d)  # 0 유물, 1 포션
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.body = set_transformer(d, layers, heads)
        self.out = nn.LayerNorm(d)

    def forward(self, batch: dict) -> torch.Tensor:
        relics = self.emb.relic(batch["relic_ids"]) + self.counter(batch["relic_cnt"].unsqueeze(-1))
        relics = relics + self.kind.weight[0]
        potions = self.emb.potion(batch["potion_ids"]) + self.kind.weight[1]
        x = torch.cat([relics, potions], dim=1)
        mask = torch.cat([batch["relic_mask"], batch["potion_mask"]], dim=1)
        x, mask = with_cls(self.cls, x, mask)
        h = self.out(self.body(x, src_key_padding_mask=~mask))
        return h[:, 0]
