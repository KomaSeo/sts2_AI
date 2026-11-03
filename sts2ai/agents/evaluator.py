"""평가 에이전트 두 개 + 둘을 합친 런 가치.

1) NodeRiskHead — 현재 막(맵이 공개된 막)
   - hazard[v]   : 지금 상태로 출발해 노드 v 에 도착했을 때, 그 방에서 죽을 확률
   - survive[v]  : v 로 간 뒤 남은 길을 가장 안전하게 골랐을 때 이 막을 살아서 끝낼 확률
                   survive[v] = (1 - hazard[v]) * max_{u ∈ 다음(v)} survive[u]
   - act_death   : 1 - max_{c ∈ 지금 고를 수 있는 노드} survive[c]
   길(경로) 하나의 사망 확률은 1 - Π(1 - hazard) 로 바로 계산된다 (path_death).

2) FutureActHead — 아직 맵이 공개되지 않은 다음 막들
   - p_act[k]    : 막 k 에 도착한다면 막 k 안에서 죽을 확률 (k > 현재 막만 유효)
   - future_death: 1 - Π_{k > 현재}(1 - p_act[k])
   마지막 막(3막)에서는 남은 막이 없으므로 출력이 비고 future_death = 0 이다.

3) 런 가치 V(s) = P(런 클리어) = (1 - act_death) * (1 - future_death)
   전투 에이전트의 GRPO 점수로는 log V 를 쓴다 (전부 log 공간에서 계산해 0 근처에서도 비교 가능).

학습(생존 분석 방식): 실제 런에서 지나간 노드는 "거기서 죽었나(1)/살았나(0)"로 라벨이
붙고, 죽은 뒤의 노드는 라벨이 없다(중도 절단). 미래 막도 그 막에 실제로 도착한 경우에만
라벨이 붙는다.
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
    """두 평가 헤드가 RunEncoder 트렁크를 공유한다."""

    def __init__(self, emb: SharedEmbeddings):
        super().__init__()
        self.encoder = RunEncoder(emb)
        self.node_head = NodeRiskHead(emb.d)
        self.future_head = FutureActHead(emb)
        self.n_acts = emb.vocab.n_acts

    def logits(self, batch: dict):
        ctx, nodes = self.encoder(batch)
        return self.node_head(ctx, nodes), self.future_head(ctx, batch["act"])

    def forward(self, batch: dict) -> Evaluation:
        node_logit, act_logit = self.logits(batch)
        mask = batch["node_mask"].float()
        hazard = torch.sigmoid(node_logit) * mask
        # log(1 - sigmoid(x)) = logsigmoid(-x): h 가 1 에 아주 가까워도 정확하다.
        log_surv = log_survival_dp(F.logsigmoid(-node_logit), batch["adj"], mask)
        choice = batch["choice_mask"]
        neg = torch.finfo(log_surv.dtype).min / 4
        best = torch.where(choice, log_surv, torch.full_like(log_surv, neg)).max(-1).values
        # 고를 노드가 없으면 (보스를 이겼거나 맵이 없으면) 이 막에서 더 죽을 일은 없다.
        log_act = torch.where(choice.any(-1), best, torch.zeros_like(best))
        ks = torch.arange(self.n_acts, device=act_logit.device)
        future_mask = ks.unsqueeze(0) > batch["act"].unsqueeze(1)
        log_future = torch.where(future_mask, F.logsigmoid(-act_logit), torch.zeros_like(act_logit)).sum(-1)
        return Evaluation(
            hazard=hazard,
            survive=log_surv.exp() * mask,
            log_survive=log_surv,
            act_death=1 - log_act.exp(),
            p_act=torch.sigmoid(act_logit),
            future_mask=future_mask,
            future_death=1 - log_future.exp(),
            log_run_value=log_act + log_future,
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

    def loss(self, batch: dict, node_label: torch.Tensor, act_label: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """node_label [B,V], act_label [B,n_acts]: 1 사망, 0 생존, -1 라벨 없음."""
        node_logit, act_logit = self.logits(batch)
        nm = node_label != IGNORE
        am = act_label != IGNORE
        zero = node_logit.sum() * 0
        l_node = F.binary_cross_entropy_with_logits(node_logit[nm], node_label[nm]) if nm.any() else zero
        l_act = F.binary_cross_entropy_with_logits(act_logit[am], act_label[am]) if am.any() else zero
        return l_node + l_act, {"node_bce": float(l_node.detach()), "act_bce": float(l_act.detach()), "n_node": int(nm.sum()), "n_act": int(am.sum())}
