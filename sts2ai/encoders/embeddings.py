"""모든 인코더가 공유하는 임베딩 테이블.

핵심 원칙: '강타'는 손패에 있든, 덱에 있든, 카드 보상으로 나왔든 같은 벡터다.
그래야 전투에서 배운 카드의 가치가 덱 평가와 경로 평가로 그대로 흘러간다.
"""

from __future__ import annotations

import torch
from torch import nn

from ..schema import N_NODE_TYPES, Vocab


class SharedEmbeddings(nn.Module):
    def __init__(self, vocab: Vocab, d: int):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.card = nn.Embedding(vocab.n_cards, d, padding_idx=0)
        self.card_upgraded = nn.Embedding(2, d)
        self.enchant = nn.Embedding(vocab.n_enchants, d, padding_idx=0)
        self.relic = nn.Embedding(vocab.n_relics, d, padding_idx=0)
        self.potion = nn.Embedding(vocab.n_potions, d, padding_idx=0)
        self.buff = nn.Embedding(vocab.n_buffs, d, padding_idx=0)
        self.enemy = nn.Embedding(vocab.n_enemies, d, padding_idx=0)
        self.intent = nn.Embedding(vocab.n_intents, d, padding_idx=0)
        self.encounter = nn.Embedding(vocab.n_encounters, d, padding_idx=0)
        self.node_type = nn.Embedding(N_NODE_TYPES, d)
        self.act = nn.Embedding(vocab.n_acts, d)
        for emb in (self.card, self.relic, self.potion, self.buff, self.enemy, self.intent, self.encounter):
            nn.init.normal_(emb.weight, std=0.02)

    def card_vec(self, ids, upgraded=None, enchant=None) -> torch.Tensor:
        v = self.card(ids)
        if upgraded is not None:
            v = v + self.card_upgraded(upgraded.long().clamp(0, 1))
        if enchant is not None:
            v = v + self.enchant(enchant)
        return v


def set_transformer(d: int, layers: int, heads: int = 4) -> nn.TransformerEncoder:
    """위치 인코딩 없는 Transformer = 순서를 모르는 집합 인코더."""
    layer = nn.TransformerEncoderLayer(
        d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def with_cls(cls: nn.Parameter, tokens: torch.Tensor, mask: torch.Tensor):
    """맨 앞에 [CLS] 토큰을 붙인다. 빈 집합이어도 attention 이 NaN 이 되지 않는다."""
    B = tokens.shape[0]
    tokens = torch.cat([cls.expand(B, 1, -1), tokens], dim=1)
    mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
    return tokens, mask
