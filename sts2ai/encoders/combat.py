"""CombatEncoder: 전투 한 시점을 토큰 집합으로 읽는 Transformer.

토큰 하나 = 손패 카드 / 더미의 카드 / 적 / 적 의도 / 버프 / 유물 / 포션 / 행동(턴 종료 등).
각 토큰 = 종류 임베딩 + 소유자 임베딩(0 플레이어, 1..N 적 슬롯) + 수치 특징 + id 임베딩.
id 임베딩은 토큰 종류에 따라 SharedEmbeddings 의 다른 테이블에서 가져온다.
위치 인코딩이 없어서 뽑을 더미의 순서를 표현할 수조차 없다 (정보 누출 방지).
"""

from __future__ import annotations

import torch
from torch import nn

from .. import schema as S
from .embeddings import SharedEmbeddings, set_transformer

MAX_OWNERS = 9


class CombatEncoder(nn.Module):
    def __init__(self, emb: SharedEmbeddings, layers: int = 3, heads: int = 4, ff_mult: int = 2):
        super().__init__()
        d = emb.d
        self.emb = emb
        self.token_type = nn.Embedding(S.N_TOKEN_TYPES, d)
        self.owner = nn.Embedding(MAX_OWNERS, d)
        # 공유 임베딩에 없는 것: 오브 종류, 캐릭터 자원 종류 (전투에서만 쓰인다)
        self.orb = nn.Embedding(S.N_ORB_KINDS, d, padding_idx=0)
        self.resource = nn.Embedding(S.N_RESOURCE_KINDS, d, padding_idx=0)
        self.selection = nn.Embedding(S.N_SELECTION_KINDS, d, padding_idx=0)  # 카드 고르기 화면 종류
        self.feat = nn.Linear(S.N_COMBAT_FEATURES, d)
        self.run_proj = nn.Linear(d, d)
        self.body = set_transformer(d, layers, heads, ff_mult)
        self.out = nn.LayerNorm(d)

    def _id_embedding(self, types: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        e = self.emb

        def table(module: nn.Embedding, kinds: tuple[int, ...]) -> torch.Tensor:
            sel = torch.zeros_like(types, dtype=torch.bool)
            for k in kinds:
                sel |= types == k
            safe = torch.where(sel, ids, torch.zeros_like(ids)).clamp(0, module.num_embeddings - 1)
            return module(safe) * sel.unsqueeze(-1)

        return (
            table(e.card, S.CARD_TOKEN_TYPES)
            + table(e.buff, S.BUFF_TOKEN_TYPES)
            + table(e.intent, S.INTENT_TOKEN_TYPES)
            + table(e.enemy, (S.TYPE_ENEMY,))
            + table(e.relic, (S.TYPE_RELIC,))
            + table(e.potion, (S.TYPE_POTION,))
            + table(e.encounter, (S.TYPE_GLOBAL,))
            + table(self.orb, (S.TYPE_ORB,))
            + table(self.resource, (S.TYPE_RESOURCE,))
            + table(self.selection, (S.TYPE_SELECTION,))
        )

    def forward(self, batch: dict, run_vec: torch.Tensor | None = None) -> torch.Tensor:
        """run_vec [B,d] 이 있으면 RUN 토큰 하나를 맨 뒤에 붙인다.

        위치 인코딩이 없어서 어디에 붙여도 같고, 맨 뒤라서 "토큰 1+i = 행동 i" 규약이 그대로다.
        모든 토큰이 attention 으로 RUN 토큰(다음 노드가 휴식인지 등)을 읽는다.
        """
        types = batch["types"]
        x = (
            self.token_type(types)
            + self.owner(batch["owners"].clamp(0, MAX_OWNERS - 1))
            + self.feat(batch["feats"])
            + self._id_embedding(types, batch["ids"])
        )
        if "enchants" in batch:  # 카드 토큰의 인챈트 (공유 임베딩: 덱 인코더와 같은 벡터)
            is_card = torch.zeros_like(types, dtype=torch.bool)
            for k in S.CARD_TOKEN_TYPES:
                is_card |= types == k
            ench = batch["enchants"].clamp(0, self.emb.enchant.num_embeddings - 1)
            x = x + self.emb.enchant(ench) * is_card.unsqueeze(-1)
        mask = batch["tok_mask"]
        T = x.shape[1]
        if run_vec is not None:
            run_tok = self.run_proj(run_vec) + self.token_type.weight[S.TYPE_RUN]
            x = torch.cat([x, run_tok.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        h = self.body(x, src_key_padding_mask=~mask)
        return self.out(h)[:, :T]  # [B, T, d]; 토큰 0 은 GLOBAL. RUN 토큰은 출력에서 뺀다
