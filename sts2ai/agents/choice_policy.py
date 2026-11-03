"""선택 정책: 전투 밖의 모든 선택(맵 노드, 카드 보상, 상점, 휴식, 이벤트, 덱에서 카드 고르기).

선택지 하나 = 토큰 하나.
  종류 임베딩 + 얻는 카드(공유 카드 임베딩) + 유물/포션 임베딩 + 가격
  + 맵 노드면 MapEncoder 의 그 노드 벡터
  + 덱에서 고르는 카드면 DeckEncoder 의 그 카드 벡터 (덱 안에서의 맥락이 담긴 표현)
  + 이벤트 id / 이벤트 선택지 번호 임베딩
선택지 토큰끼리 집합 Transformer 1층으로 서로 비교한 뒤, 런 맥락(run_vec)과 합쳐 점수를 낸다.

런 맥락·노드 벡터·덱 카드 벡터는 평가기 트렁크(RunEncoder)에서 가져온다. trunk_grad 가 참이면
선택 정책의 손실이 트렁크까지 흘러가서 평가기 인코더도 함께 학습된다.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from ..encoders import SharedEmbeddings
from ..encoders.embeddings import set_transformer
from ..schema import CHOICE_KIND_INDEX, CHOICE_KINDS, N_EVENT_IDS, ChoiceOption

NEG = -1e9
MAX_BUNDLE = 3


def collate_options(option_lists: list[list[ChoiceOption]], vocab) -> dict[str, torch.Tensor]:
    B = len(option_lists)
    O = max(1, max(len(o) for o in option_lists))

    def z(dtype=torch.long):
        return torch.zeros(B, O, dtype=dtype)

    out = {k: z() for k in ("kind", "card", "card_upg", "card_ench", "relic", "potion", "event", "event_opt")}
    out["node"] = torch.full((B, O), -1, dtype=torch.long)
    out["deck"] = torch.full((B, O), -1, dtype=torch.long)
    out["price"] = z(torch.float)
    out["bundle"] = torch.zeros(B, O, MAX_BUNDLE, dtype=torch.long)
    out["mask"] = z(torch.bool)
    n_ench = vocab.n_enchants
    for b, opts in enumerate(option_lists):
        for i, o in enumerate(opts):
            out["mask"][b, i] = True
            out["kind"][b, i] = CHOICE_KIND_INDEX.get(o.kind, 0)
            if o.card is not None:
                out["card"][b, i] = o.card.id
                out["card_upg"][b, i] = int(o.card.upgraded)
                out["card_ench"][b, i] = min(o.card.enchant, n_ench - 1)
            for j, c in enumerate(o.cards[:MAX_BUNDLE]):
                out["bundle"][b, i, j] = c.id
            out["relic"][b, i] = o.relic
            out["potion"][b, i] = o.potion
            out["node"][b, i] = o.node
            out["deck"][b, i] = o.deck_index
            out["price"][b, i] = o.price / 100.0
            out["event"][b, i] = o.event % N_EVENT_IDS
            out["event_opt"][b, i] = min(max(o.event_option, 0), 15)
    return out


def _gather(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """table [B,N,d] 에서 idx [B,O] (-1 = 없음) 위치를 뽑는다. 없으면 0 벡터."""
    if table.shape[1] == 0:
        return torch.zeros(*idx.shape, table.shape[-1], device=table.device)
    safe = idx.clamp(min=0, max=table.shape[1] - 1)
    out = torch.gather(table, 1, safe.unsqueeze(-1).expand(-1, -1, table.shape[-1]))
    return out * (idx >= 0).unsqueeze(-1)


class ChoicePolicy(nn.Module):
    def __init__(self, emb: SharedEmbeddings, layers: int = 1):
        super().__init__()
        d = emb.d
        self.emb = emb
        self.kind = nn.Embedding(len(CHOICE_KINDS), d)
        self.price = nn.Linear(1, d)
        self.event = nn.Embedding(N_EVENT_IDS, d)
        self.event_opt = nn.Embedding(16, d)
        self.node_proj = nn.Linear(d, d)
        self.deck_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.body = set_transformer(d, layers)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def trunk(self, encoder, run_batch: dict, grad: bool) -> dict[str, torch.Tensor]:
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            return encoder.encode_all(run_batch)

    def forward(self, enc: dict[str, torch.Tensor], opt: dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.emb
        has_card = (opt["card"] > 0).unsqueeze(-1)
        tok = self.kind(opt["kind"])
        tok = tok + e.card_vec(opt["card"], opt["card_upg"], opt["card_ench"]) * has_card
        tok = tok + (e.card(opt["bundle"]) * (opt["bundle"] > 0).unsqueeze(-1)).sum(2)
        tok = tok + e.relic(opt["relic"]) + e.potion(opt["potion"])
        tok = tok + self.node_proj(_gather(enc["nodes"], opt["node"])) * (opt["node"] >= 0).unsqueeze(-1)
        tok = tok + self.deck_proj(_gather(enc["deck_tokens"], opt["deck"])) * (opt["deck"] >= 0).unsqueeze(-1)
        tok = tok + self.price(opt["price"].unsqueeze(-1)) + self.event(opt["event"]) + self.event_opt(opt["event_opt"])
        tok = self.norm(tok)
        mask = opt["mask"]
        tok = self.body(tok, src_key_padding_mask=~mask)
        ctx = enc["run_vec"].unsqueeze(1).expand(-1, tok.shape[1], -1)
        logits = self.head(torch.cat([tok, ctx], dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask, NEG)
