"""평가 에이전트: 노드별 사망 확률 하나로 이번 막과 미래 막을 함께 평가한다.

맵 = 이번 막의 실제 맵 + 남은 막마다 가상 줄 (N/A 노드 R-1 개 + 보스 N/A, R = 이번 막 층수).
이번 막 보스 → 다음 막 줄 → ... 로 이어진다 (batching.collate_maps).

- hazard[v]   : 노드 v 에 가면 그 방에서 죽을 확률. 가상 노드는 "그 막의 그 층에서 죽을 확률".
- survive[v]  : v 로 간 뒤 남은 길을 가장 안전하게 골랐을 때 런을 끝까지 살아남을 확률
                survive[v] = (1 - hazard[v]) * max_{u ∈ 다음(v)} survive[u]   (가상 줄까지 이어짐)
- 런 가치 V    = max_{c ∈ 지금 고를 수 있는 노드} survive[c] = P(런 클리어)
                GRPO 점수로는 log V 를 쓴다 (전부 log 공간에서 계산해 0 근처에서도 비교 가능).
- act_death   : 이번 막만 따로 본 사망 확률 (가상 줄을 뗀 맵으로 같은 계산)
- p_act[k]    : 미래 막 k 의 가상 줄 전체를 지날 때의 사망 확률 1 - Π(1 - hazard)
- future_death: 남은 막 전체의 사망 확률

학습(생존 분석 방식): 실제로 지나간 노드는 "거기서 죽었나(1)/살았나(0)", 죽은 뒤는 라벨 없음.
미래 막도 같다: 실제로 지나간 층마다 0, 죽은 층 1 (train/evaluator_train.py). 그래서 "다음 막 몇 층까지
버티나"가 학습된다 (예전에는 미래 막 하나당 0/1 라벨 하나였다).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ..encoders import RunEncoder, SharedEmbeddings

IGNORE = -1.0


class NodeRiskHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ctx: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        ctx = ctx.unsqueeze(1).expand(-1, nodes.shape[1], -1)
        return self.net(torch.cat([nodes, ctx], dim=-1)).squeeze(-1)  # hazard logit [B, V]


class FutureActHead(nn.Module):
    def __init__(self, emb: SharedEmbeddings):
        super().__init__()
        d = emb.d
        self.emb = emb
        self.gap = nn.Embedding(emb.vocab.n_acts, d)  # 지금으로부터 몇 막 뒤인가
        self.net = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ctx: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        n = self.emb.vocab.n_acts
        ks = torch.arange(n, device=ctx.device)
        gap = (ks.unsqueeze(0) - act.unsqueeze(1)).clamp(0, n - 1)  # [B, n]
        x = torch.cat(
            [
                ctx.unsqueeze(1).expand(-1, n, -1),
                self.emb.act(ks).unsqueeze(0).expand(ctx.shape[0], -1, -1),
                self.gap(gap),
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)  # [B, n] logit


def log_survival_dp(log_keep: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """log 공간 DP: log S[v] = log(1-h[v]) + max_{u∈succ(v)} log S[u].

    확률을 직접 곱하면 위험한 길에서 값이 0 으로 뭉개져(언더플로) 서로 비교할 수 없게 된다.
    log 로 계산하면 "둘 다 거의 죽는다" 끼리도 몇 배 차이인지가 남는다.
    """
    log_keep = log_keep * node_mask
    has_succ = adj.sum(-1) > 0
    neg = torch.finfo(log_keep.dtype).min / 4
    surv = log_keep.clone()
    for _ in range(adj.shape[-1]):
        cand = torch.where(adj > 0, surv.unsqueeze(1), torch.full_like(adj, neg))  # [B, v, u]
        best = cand.max(-1).values
        new = log_keep + torch.where(has_succ, best, torch.zeros_like(best))
        if torch.allclose(new, surv):
            break
        surv = new
    return surv


def log_survival_dp_jumps(log_keep: torch.Tensor, adj: torch.Tensor, jump_adj: torch.Tensor,
                          node_mask: torch.Tensor, max_jumps: int) -> torch.Tensor:
    """날개 달린 부츠까지 본 log 공간 DP. 반환 [B, V, J+1]: 남은 건너뛰기 횟수 j 일 때의 log S.

    log S[v, j] = log(1-h[v]) + max( max_{u∈succ(v)} log S[u, j],          (간선: 횟수 그대로)
                                     max_{u∈jump(v)} log S[u, j-1] if j>0 ) (건너뛰기: 하나 씀)
    갈 곳이 없으면(보스·마지막 노드) log S[v, j] = log(1-h[v]).
    """
    log_keep = log_keep * node_mask
    neg = torch.finfo(log_keep.dtype).min / 4
    J = max(0, int(max_jumps))
    surv = log_keep.unsqueeze(-1).repeat(1, 1, J + 1)  # [B, V, J+1]
    has_succ = adj.sum(-1) > 0  # [B, V]
    has_jump = jump_adj.sum(-1) > 0
    fill = torch.full_like(adj, neg)
    for _ in range(adj.shape[-1]):
        cols = []
        for j in range(J + 1):
            best = torch.where(adj > 0, surv[..., j].unsqueeze(1), fill).max(-1).values
            can = has_succ
            if j > 0:
                bj = torch.where(jump_adj > 0, surv[..., j - 1].unsqueeze(1), fill).max(-1).values
                best = torch.maximum(best, bj)
                can = has_succ | has_jump
            cols.append(log_keep + torch.where(can, best, torch.zeros_like(best)))
        new = torch.stack(cols, -1)
        if torch.allclose(new, surv):
            break
        surv = new
    return surv


def survival_dp(hazard: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """확률 버전: survive[v] = (1-h[v]) * max_{u∈succ(v)} survive[u]."""
    return log_survival_dp(torch.log1p(-hazard.clamp(max=1 - 1e-7)), adj, node_mask).exp() * node_mask


@dataclass
class Evaluation:
    hazard: torch.Tensor  # [B, V]
    survive: torch.Tensor  # [B, V]
    log_survive: torch.Tensor  # [B, V]
    act_death: torch.Tensor  # [B]
    p_act: torch.Tensor  # [B, n_acts]
    future_mask: torch.Tensor  # [B, n_acts]  (k > 현재 막)
    future_death: torch.Tensor  # [B]
    log_run_value: torch.Tensor  # [B]  log P(런 클리어) — GRPO 점수

    @property
    def run_value(self) -> torch.Tensor:
        return self.log_run_value.exp()


class Evaluator(nn.Module):
    """노드 평가 헤드 하나가 이번 막 실제 노드와 미래 막 가상 노드를 함께 평가한다."""

    def __init__(self, emb: SharedEmbeddings):
        super().__init__()
        self.encoder = RunEncoder(emb)
        self.node_head = NodeRiskHead(emb.d)
        self.n_acts = emb.vocab.n_acts

    def logits(self, batch: dict) -> torch.Tensor:
        ctx, nodes, _ = self.encoder(batch)
        return self.node_head(ctx, nodes)  # [B, V] (실제 노드 + 가상 노드)

    def run_vector(self, batch: dict) -> torch.Tensor:
        """전투 정책에 넘겨줄 런 맥락 (덱·유물·상태·보스·맵) [B, d]."""
        return self.encoder(batch)[2]

    def forward(self, batch: dict) -> Evaluation:
        node_logit = self.logits(batch)
        mask = batch["node_mask"].float()
        chain = batch.get("chain_mask", torch.zeros_like(batch["node_mask"]))
        hazard = torch.sigmoid(node_logit) * mask
        # log(1 - sigmoid(x)) = logsigmoid(-x): h 가 1 에 아주 가까워도 정확하다.
        log_keep = F.logsigmoid(-node_logit)
        choice = batch["choice_mask"]
        jumps = batch.get("jumps")
        if jumps is None:
            jumps = torch.zeros(node_logit.shape[0], dtype=torch.long, device=node_logit.device)
        jump_adj = batch.get("jump_adj", torch.zeros_like(batch["adj"]))
        jump_choice = batch.get("jump_choice", torch.zeros_like(choice))
        J = int(jumps.max()) if jumps.numel() else 0
        neg = torch.finfo(log_keep.dtype).min / 4

        def surv_with_jumps(adj_, jadj_, mask_):
            """[B,V] 두 가지: 남은 횟수 그대로일 때, 하나 쓴 뒤일 때 (지금 선택지가 건너뛰기면 후자)."""
            if J == 0:
                s = log_survival_dp(log_keep, adj_, mask_)
                return s, s
            s = log_survival_dp_jumps(log_keep, adj_, jadj_, mask_, J)  # [B,V,J+1]
            idx_now = jumps.view(-1, 1, 1).expand(-1, s.shape[1], 1)
            idx_used = (jumps - 1).clamp(min=0).view(-1, 1, 1).expand(-1, s.shape[1], 1)
            return s.gather(-1, idx_now).squeeze(-1), s.gather(-1, idx_used).squeeze(-1)

        def best_of(now, used, ch):
            val = torch.where(jump_choice, used, now)
            b = torch.where(ch, val, torch.full_like(val, neg)).max(-1).values
            return torch.where(ch.any(-1), b, torch.zeros_like(b))  # 고를 노드가 없으면 더 죽을 일이 없다

        log_surv, log_surv_used = surv_with_jumps(batch["adj"], jump_adj, mask)
        log_v = best_of(log_surv, log_surv_used, choice)
        # 이번 막만: 가상 줄을 떼어 낸 맵으로 같은 계산
        real = mask * (~chain).float()
        act_now, act_used = surv_with_jumps(batch["adj"] * real.unsqueeze(1), jump_adj * real.unsqueeze(1), real)
        log_act = best_of(act_now, act_used, choice & ~chain)
        # 미래 막 k: 그 막 가상 줄 전체를 지날 때의 생존
        ks = torch.arange(self.n_acts, device=node_logit.device)
        in_k = chain.unsqueeze(1) & (batch["node_act"].unsqueeze(1) == ks.view(1, -1, 1))  # [B, n_acts, V]
        log_keep_k = (log_keep.unsqueeze(1) * in_k).sum(-1)  # [B, n_acts]
        future_mask = in_k.any(-1)
        # 노드별 표시용 log S: 지금 선택지가 건너뛰기라면 그 노드는 횟수를 하나 쓴 값
        log_surv = torch.where(jump_choice, log_surv_used, log_surv)
        return Evaluation(
            hazard=hazard,
            survive=log_surv.exp() * mask,
            log_survive=log_surv,
            act_death=1 - log_act.exp(),
            p_act=1 - log_keep_k.exp(),
            future_mask=future_mask,
            future_death=1 - (log_keep_k * future_mask).sum(-1).exp(),
            log_run_value=log_v,
        )

    @staticmethod
    def path_death(ev: Evaluation, b: int, path: list[int]) -> float:
        """경로(노드 인덱스 리스트)를 그대로 따라갈 때 이 막에서 죽을 확률."""
        h = ev.hazard[b, path]
        return float(1 - (1 - h).prod())

    @staticmethod
    def best_path(ev: Evaluation, batch: dict, b: int) -> list[int]:
        """지금 위치에서 생존 확률이 가장 높은 길."""
        adj = batch["adj"][b]
        surv = ev.log_survive[b]
        choices = batch["choice_mask"][b].nonzero().flatten().tolist()
        if not choices:
            return []
        v = max(choices, key=lambda c: float(surv[c]))
        path = [v]
        while True:
            succ = adj[v].nonzero().flatten().tolist()
            if not succ:
                return path
            v = max(succ, key=lambda u: float(surv[u]))
            path.append(v)

    def loss(self, batch: dict, node_label: torch.Tensor, act_label: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """node_label [B,V]: 1 사망, 0 생존, -1 라벨 없음 (실제 노드와 미래 막 가상 노드 모두).

        act_label 은 예전 막 단위 라벨로, 더 쓰지 않는다 (호출 호환용).
        기록: node_bce = 이번 막 실제 노드, act_bce = 미래 막 가상 노드의 BCE.
        """
        node_logit = self.logits(batch)
        lab = node_label != IGNORE
        chain = batch.get("chain_mask", torch.zeros_like(lab))
        zero = node_logit.sum() * 0
        loss = F.binary_cross_entropy_with_logits(node_logit[lab], node_label[lab]) if lab.any() else zero

        def part(m):
            m = lab & m
            if not m.any():
                return 0.0
            return float(F.binary_cross_entropy_with_logits(node_logit[m].detach(), node_label[m]))

        return loss, {"node_bce": part(~chain), "act_bce": part(chain), "n_node": int((lab & ~chain).sum()),
                      "n_act": int((lab & chain).sum())}
