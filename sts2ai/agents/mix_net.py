"""β·γ 신경망: 상황에 따라 덱 빌딩 모듈의 비중 β 와 강화 비중 γ 를 고친다.

실제 런에서 덱이 바뀌는 화면은 logit = 단기 + β·덱 빌딩 log 확률, 휴식 장소는 강화 logit 에 γ·강화 이득을 더한다.
막마다 하나씩인 기본값(log β, log γ, 진화 전략으로 학습)에 이 신경망의 보정값을 더한다:

    log β = 막별 기본값 + Δβ(상황),  log γ = 막별 기본값 + Δγ(상황)

입력: 평가기 run 벡터(기울기 끊음) + 막 one-hot + 휴식 장소인가. 출력: (Δβ, Δγ), 2·tanh 로 ±2 안에 묶는다.
마지막 층을 0 으로 시작하므로 처음에는 막별 기본값과 똑같이 고른다.
"""
from __future__ import annotations

import torch
from torch import nn

MAX_DELTA = 2.0


class MixNet(nn.Module):
    def __init__(self, d: int, n_acts: int, hidden: int = 64):
        super().__init__()
        self.n_acts = n_acts
        self.net = nn.Sequential(nn.Linear(d + n_acts + 1, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, run_vec: torch.Tensor, act: int, is_rest: bool) -> torch.Tensor:
        """[d + n_acts + 1] 입력 벡터 (CPU)."""
        extra = torch.zeros(self.n_acts + 1)
        extra[min(max(int(act), 0), self.n_acts - 1)] = 1.0
        extra[-1] = 1.0 if is_rest else 0.0
        return torch.cat([run_vec.detach().float().cpu().view(-1), extra])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, d + n_acts + 1] → [B, 2] = (Δlog β, Δlog γ)."""
        return MAX_DELTA * torch.tanh(self.net(x))
