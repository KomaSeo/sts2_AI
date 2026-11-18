"""적별 상성 신경망 M: (덱, 유물, 들어갈 때 HP, 적) → 그 적과 싸웠을 때의 결과 예측.

출력 세 가지
- win: 이길 확률 (logit)
- hp_after: 이겼을 때 남는 HP / 최대 HP
- dmg: 적에게 준 피해 비율 (이기면 1, 지면 마지막에 남은 적 HP 로 계산). 질 게 뻔한 상대(1막 덱 대 3막 보스)도
  덱마다 차이가 나도록 매끄러운 값을 함께 배운다.
입력 상태는 RunState 로 받는다: boss 칸에 "싸울 적"의 encounter id, act 는 그 적이 나오는 막, hp 는 들어갈 때 HP.
맵은 쓰지 않는다 (RunEncoder.context 만 쓴다). 자기 임베딩을 따로 가진다: 덱 빌딩 모듈의 보상을 계산하는
심판이라, 다른 모듈의 손실이 이 신경망으로 흘러들면 안 된다 (train/matchup.py 에서 실제 전투 결과로만 학습).
"""

from __future__ import annotations

import torch
from torch import nn

from ..batching import collate_run, module_device, to_device
from ..encoders import RunEncoder, SharedEmbeddings
from ..schema import RunState


class MatchupNet(nn.Module):
    def __init__(self, vocab, d: int):
        super().__init__()
        self.vocab = vocab
        self.emb = SharedEmbeddings(vocab, d)
        self.encoder = RunEncoder(self.emb)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, states: list[RunState]) -> torch.Tensor:
        """[B, 3] = (이길 확률 logit, 남는 HP 비율 logit, 준 피해 비율 logit)."""
        batch = to_device(collate_run(states, self.vocab), module_device(self))
        return self.head(self.encoder.context(batch))

    @torch.no_grad()
    def predict(self, states: list[RunState]) -> dict[str, torch.Tensor]:
        out = self(states)
        return {"win": torch.sigmoid(out[:, 0]), "hp_after": torch.sigmoid(out[:, 1]), "dmg": torch.sigmoid(out[:, 2])}
