"""덱 빌딩 모듈: 치명적인 적 하나를 대비해 덱을 짜는 별도 신경망.

평가기·선택 정책과 가중치를 공유하지 않는다 (자기 임베딩·인코더를 따로 가진다).
입력 상태는 RunState 로 받되, 맵은 없고(None) boss 칸에 "대비하려는 적"의 encounter id 를 넣는다.
- 선택 머리: 선택 정책(ChoicePolicy)과 같은 구조. 선택지 토큰 + 덱 카드 벡터 → 점수.
- 가치 머리 D: 이 덱·유물로 그 적과 싸우면 받을 점수의 예측 (이기면 1 + 남은 HP 비율, 지면 준 피해 비율).
  휴식 장소에서 "강화가 얼마나 도움이 되나"를 잴 때와, 덱 빌딩 재생의 가치 학습에 쓴다.
학습: train/deckbuild.py (기록한 덱 빌딩 화면을 순서대로 재생하는 GRPO).
"""

from __future__ import annotations

import torch
from torch import nn

from ..batching import collate_run, module_device, to_device
from ..encoders import RunEncoder, SharedEmbeddings
from ..schema import Card, ChoiceOption, RunState
from .choice_policy import ChoicePolicy, collate_options


MAX_HORIZON = 63


def builder_state(deck: list[Card], relics: list[int], gold: int, max_hp: int, act: int, floor: int,
                  target: int, horizon: int = 0, hp: int | None = None) -> RunState:
    """덱 빌딩 모듈의 입력 상태. relics 는 유물 임베딩 인덱스, target 은 대비하는 적의 encounter 인덱스.

    horizon = 대상 적을 만나기까지 남은 층수. 같은 덱이라도 남은 성장 기회에 따라 가치가 다르다.
    hp = 그 적과 싸울 때의 HP (없으면 최대 HP). 가치 머리는 "남은 층수 뒤 이 HP 로 그 적과 싸우면"을 예측한다.
    """
    st = RunState(act=act, floor=floor, hp=max_hp if hp is None else int(hp), max_hp=max_hp, gold=gold, ascension=0, deck=list(deck),
                  relics=[(r, 0) for r in relics], potions=[], map=None, boss=target)
    st.horizon = int(max(0, min(MAX_HORIZON, horizon)))  # RunState 에 없는 칸이라 속성으로 붙인다
    return st


class DeckBuilder(nn.Module):
    def __init__(self, vocab, d: int):
        super().__init__()
        self.vocab = vocab
        self.emb = SharedEmbeddings(vocab, d)
        self.encoder = RunEncoder(self.emb)
        self.policy = ChoicePolicy(self.emb)
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # 대상 적까지 남은 층수: ctx 와 run_vec 에 더한다 (예전 가중치에는 없어 0 근처에서 시작)
        self.horizon = nn.Embedding(MAX_HORIZON + 1, d)
        nn.init.normal_(self.horizon.weight, std=0.02)

    def _enc(self, states: list[RunState]):
        dev = module_device(self)
        enc = self.encoder.encode_all(to_device(collate_run(states, self.vocab), dev))
        h = self.horizon(torch.tensor([getattr(s, "horizon", 0) for s in states], device=dev))
        enc["ctx"] = enc["ctx"] + h
        enc["run_vec"] = enc["run_vec"] + h
        return enc

    def forward(self, states: list[RunState], option_lists: list[list[ChoiceOption]]):
        """(선택지 logit [B,O], 가치 [B])."""
        enc = self._enc(states)
        opt = to_device(collate_options(option_lists, self.vocab), module_device(self))
        return self.policy(enc, opt), self.value(enc["ctx"]).squeeze(-1)

    def values(self, states: list[RunState]) -> torch.Tensor:
        return self.value(self._enc(states)["ctx"]).squeeze(-1)
