"""MapEncoder: 막 하나의 맵(DAG)을 노드별 벡터로 만든다.

- 노드 입력: 방 종류 임베딩 + [층 위치, 현재 위치, 다음 선택지, 도달 가능]
- 런 맥락(덱/HP 등)을 FiLM 으로 주입한다. 같은 엘리트 방이라도 덱이 약하면 위험하니까.
- 메시지 전달은 두 방향:
    위쪽(후속 노드 → 나): "이 노드 다음에 무엇이 기다리나" (보스 쪽에서 내려오는 정보)
    아래쪽(선행 노드 → 나): "여기 오기까지 무엇을 거쳐 오나" (누적 소모)
"""

from __future__ import annotations

import torch
from torch import nn

from .embeddings import SharedEmbeddings


class _MPLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.mix = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, adj, mask):
        # adj[b, from, to]. 후속 노드 평균 = adj @ h, 선행 노드 평균 = adj^T @ h
        out_deg = adj.sum(-1, keepdim=True).clamp(min=1)
        in_deg = adj.sum(-2).unsqueeze(-1).clamp(min=1)
        succ = adj @ h / out_deg
        pred = adj.transpose(1, 2) @ h / in_deg
        h = self.norm(h + self.mix(torch.cat([h, succ, pred], dim=-1)))
        return h * mask.unsqueeze(-1)


class MapEncoder(nn.Module):
    def __init__(self, emb: SharedEmbeddings, layers: int = 4):
        super().__init__()
        d = emb.d
        self.emb = emb
        self.feat = nn.Linear(4, d)
        self.film = nn.Linear(d, 2 * d)
        # 맵 깊이(≈16층)를 다 건너가려면 층 수가 깊이보다 작아도 되도록 가중치를 공유해 반복한다.
        self.layers = nn.ModuleList([_MPLayer(d) for _ in range(layers)])
        self.repeats = 4

    def forward(self, batch: dict, ctx: torch.Tensor) -> torch.Tensor:
        mask = batch["node_mask"]
        h = self.emb.node_type(batch["node_type"]) + self.feat(batch["node_feat"])
        gamma, beta = self.film(ctx).unsqueeze(1).chunk(2, dim=-1)
        h = (h * (1 + gamma) + beta) * mask.unsqueeze(-1)
        for _ in range(self.repeats):
            for layer in self.layers:
                h = layer(h, batch["adj"], mask)
        return h  # [B, V, d]

    @staticmethod
    def summary(h: torch.Tensor, batch: dict) -> torch.Tensor:
        """맵 요약 [B, 2d] = [다음 선택지 노드 평균, 앞으로 갈 수 있는 노드 전체 평균].

        "다음이 휴식인가 전투인가" 는 앞쪽에, "남은 길에 엘리트·휴식이 얼마나 있나" 는 뒤쪽에 담긴다.
        고를 노드가 없으면 (보스를 이겼거나 맵이 없으면) 0 벡터.
        """

        def masked_mean(m: torch.Tensor) -> torch.Tensor:
            w = m.float().unsqueeze(-1)
            return (h * w).sum(1) / w.sum(1).clamp(min=1)

        reachable = batch["node_feat"][..., 3] > 0
        return torch.cat([masked_mean(batch["choice_mask"]), masked_mean(reachable)], dim=-1)
