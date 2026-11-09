"""전투 에이전트: CombatEncoder + 행동 포인터 헤드 + 타깃 포인터 헤드.

- 행동: 토큰 1+i 의 점수 = 행동 i 의 logit (카드 i 사용 / 턴 종료 / 포션 / 선택지).
  고정 크기 출력층을 쓰지 않으므로 손패 수가 바뀌어도 의미가 흔들리지 않는다.
- 타깃: 고른 행동 토큰과 각 적 토큰의 쌍선형 점수. 타깃이 필요한 행동에서만 쓴다.
log π(a, t) = log π(a) + [타깃 필요] · log π(t | a)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..encoders import CombatEncoder, SharedEmbeddings
from ..schema import MAX_ACTIONS

NEG = -1e9


@dataclass
class PolicyOut:
    action_logits: torch.Tensor  # [B, MAX_ACTIONS]
    tokens: torch.Tensor  # [B, T, d]


class CombatPolicy(nn.Module):
    def __init__(self, emb: SharedEmbeddings, layers: int = 3, heads: int = 4):
        super().__init__()
        d = emb.d
        self.encoder = CombatEncoder(emb, layers=layers, heads=heads)
        self.action = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.tq = nn.Linear(d, d)
        self.tk = nn.Linear(d, d)

    def forward(self, batch: dict, run_vec: torch.Tensor | None = None) -> PolicyOut:
        h = self.encoder(batch, run_vec)
        B, T, d = h.shape
        n = min(MAX_ACTIONS, T - 1)
        acts = h[:, 1 : 1 + n]
        g = h[:, :1].expand(-1, n, -1)
        logits = torch.full((B, MAX_ACTIONS), NEG, device=h.device)
        logits[:, :n] = self.action(torch.cat([acts, g], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~batch["legal"], NEG)
        return PolicyOut(logits, h)

    def target_logits(self, out: PolicyOut, batch: dict, action: torch.Tensor) -> torch.Tensor:
        h = out.tokens
        b = torch.arange(h.shape[0], device=h.device)
        q = self.tq(h[b, 1 + action.clamp(max=h.shape[1] - 2)])  # [B, d]
        enemies = h[b.unsqueeze(1), batch["enemy_tok"]]  # [B, E, d]
        logits = (self.tk(enemies) * q.unsqueeze(1)).sum(-1) / h.shape[-1] ** 0.5
        return logits.masked_fill(~batch["enemy_mask"], NEG)

    def log_prob(self, batch: dict, action: torch.Tensor, target: torch.Tensor, run_vec: torch.Tensor | None = None):
        """주어진 (행동, 타깃 인덱스) 의 log 확률과 엔트로피. target 은 enemy_tok 안의 인덱스."""
        out = self(batch, run_vec)
        logp_a = torch.log_softmax(out.action_logits, -1)
        lp = logp_a.gather(1, action.unsqueeze(1)).squeeze(1)
        p_a = logp_a.exp()
        ent = -(p_a * logp_a.clamp(min=-30)).sum(-1)
        needs = batch["needs_target"].gather(1, action.unsqueeze(1)).squeeze(1) & batch["enemy_mask"].any(-1)
        if needs.any():
            logp_t = torch.log_softmax(self.target_logits(out, batch, action), -1)
            lt = logp_t.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1)
            lp = lp + torch.where(needs, lt, torch.zeros_like(lt))
        return lp, ent

    @torch.no_grad()
    def act(self, batch: dict, greedy: bool = False, run_vec: torch.Tensor | None = None):
        """배치 하나에서 (행동, 타깃 인덱스, log 확률) 을 뽑는다. 타깃 불필요 시 -1."""
        out = self(batch, run_vec)
        dist = torch.distributions.Categorical(logits=out.action_logits)
        a = out.action_logits.argmax(-1) if greedy else dist.sample()
        lp = dist.log_prob(a)
        t = torch.full_like(a, -1)
        needs = batch["needs_target"].gather(1, a.unsqueeze(1)).squeeze(1) & batch["enemy_mask"].any(-1)
        if needs.any():
            tl = self.target_logits(out, batch, a)
            td = torch.distributions.Categorical(logits=tl)
            tt = tl.argmax(-1) if greedy else td.sample()
            t = torch.where(needs, tt, t)
            lp = lp + torch.where(needs, td.log_prob(tt), torch.zeros_like(lp))
        return a, t, lp
