"""적별 상성 신경망 M 의 데이터·학습과 적 등장 확률표.

- 적 목록: 시뮬레이터 생성 표(Generated/EncounterTags.g.cs)의 막별 Weak/Normal/Elite/Boss 목록.
  시뮬레이터 막 번호 1 과성장, 2 지하 선착장 = 1막, 3 군락 = 2막, 4 영광 = 3막.
- 적 등장 확률 P(적 | 막, 노드 종류): 목록 안에서 고르게. 일반 전투 노드는 그 막의 처음 몇 번은 약한 적
  목록에서 나온다 (weak_first 개). 다음 막 보스는 막이 시작돼야 알 수 있으므로 후보에 고르게.
- 학습: MatchSample(상태, 이겼나, 남은 HP 비율, 준 피해 비율) 로 BCE (세 출력 모두 0~1 목표).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..schema import RunState

SIM_ACT_TO_ORDINAL = {1: 0, 2: 0, 3: 1, 4: 2}
REGION_TO_SIM_ACT = {"overgrowth": 1, "underdocks": 2, "hive": 3, "glory": 4}
KINDS = ("Weak", "Normal", "Elite", "Boss")


def load_pools(sts2_repo: str | Path) -> dict[int, dict[str, list[int]]]:
    """{시뮬레이터 막 번호: {종류: [encounter id]}}."""
    text = (Path(sts2_repo) / "src" / "Sts2Emulator" / "Generated" / "EncounterTags.g.cs").read_text(encoding="utf-8")
    out: dict[int, dict[str, list[int]]] = {}
    for a, kind, ids in re.findall(r'\((\d+), "(\w+)"\) => \[([^\]]*)\]', text):
        out.setdefault(int(a), {})[kind] = [int(x) for x in ids.replace("\n", "").split(",") if x.strip()]
    return out


# 막마다 처음 몇 번의 일반 전투가 약한 적 목록에서 나오나 (RunConstants.ActRoomCounts: 1막 3, 군락·영광 2)
WEAK_FIRST = {1: 3, 2: 3, 3: 2, 4: 2}


def encounter_distribution(pools, sim_act: int, kind: str, n_normal_seen: int = 0,
                           seen: set[int] | frozenset = frozenset()) -> dict[int, float]:
    """P(적 | 막, 노드 종류). kind: "Normal" | "Elite" | "Boss".

    게임은 목록을 주머니에 넣고 하나씩 꺼내며, 비면 다시 채운다 (RunMapGenerator). 그래서 이번 막에서 이미 만난
    적(seen)은 주머니가 빌 때까지 다시 나오지 않는다. n_normal_seen = 이번 막에서 이미 치른 일반 전투 수.
    """
    p = pools.get(sim_act, {})
    if kind == "Normal" and n_normal_seen < WEAK_FIRST.get(sim_act, 3) and p.get("Weak"):
        ids = p["Weak"]
    else:
        ids = p.get(kind, [])
    left = [e for e in ids if e not in seen] or list(ids)  # 다 만났으면 주머니를 다시 채운다
    return {e: 1.0 / len(left) for e in left} if left else {}


@dataclass
class MatchSample:
    state: RunState  # boss = 싸운 적의 encounter 인덱스, act = 그 적의 막, hp = 들어갈 때 HP
    won: bool
    hp_after: float  # 남은 HP / 최대 HP (지면 0)
    dmg: float  # 적에게 준 피해 비율 (이기면 1)
    synthetic: bool = False  # 막을 넘나드는 모의 전투인가


def matchup_loss(net, samples: list[MatchSample]) -> tuple[torch.Tensor, dict]:
    out = net([s.state for s in samples])
    dev = out.device
    tgt = torch.tensor([[float(s.won), s.hp_after, s.dmg] for s in samples], dtype=torch.float, device=dev)
    per = F.binary_cross_entropy_with_logits(out, tgt, reduction="none")  # [B,3]
    won = tgt[:, 0] > 0.5
    # 남는 HP 는 이긴 전투에서만 의미가 있다
    loss = per[:, 0].mean() + per[:, 2].mean() + (per[won, 1].mean() if won.any() else out.sum() * 0)
    return loss, {"win_bce": float(per[:, 0].mean()), "dmg_bce": float(per[:, 2].mean())}


def train_matchup(net, optim, samples: list[MatchSample], epochs: int = 3, batch_size: int = 128) -> dict:
    if not samples:
        return {}
    stats, n = {"m_loss": 0.0, "m_win_bce": 0.0, "m_dmg_bce": 0.0}, 0
    for _ in range(epochs):
        order = list(range(len(samples)))
        random.shuffle(order)
        for i in range(0, len(order), batch_size):
            loss, info = matchup_loss(net, [samples[j] for j in order[i: i + batch_size]])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
            stats["m_loss"] += float(loss.detach())
            stats["m_win_bce"] += info["win_bce"]
            stats["m_dmg_bce"] += info["dmg_bce"]
            n += 1
    return {k: v / max(1, n) for k, v in stats.items()}
