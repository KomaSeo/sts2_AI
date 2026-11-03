"""전투 에이전트 학습: GRPO (Group Relative Policy Optimization).

같은 전투를 G 번 분기해서 굴린다(각 분기는 셔플과 적 난수가 다르다). 각 시행이 끝난
뒤의 런 상태를 평가 에이전트가 채점한다 (점수 = log P(런 클리어), 전투에서 죽으면 최하점).
그룹 안에서 상대 평가를 해서, 평균보다 좋은 시행의 행동은 강화하고 나쁜 시행의 행동은
약화한다. 별도의 가치망(critic) 없이 그룹 평균이 기준선 역할을 한다.

advantage_mode
- "zscore"       : A = (점수 - 그룹 평균) / 그룹 표준편차   ← 표준 GRPO
- "best_vs_rest" : 1등 +1, 나머지 -1/(G-1)               ← "제일 좋은 방식에 보상, 나머지에 벌"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..agents import CombatPolicy, Evaluator
from ..batching import collate_combat, collate_run
from ..envs.base import CombatSim, RunSim
from ..schema import CombatObs, RunState


@dataclass
class Trajectory:
    obs: list[CombatObs] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)  # enemy_tok 안의 인덱스, 없으면 -1
    logp: list[float] = field(default_factory=list)
    won: bool = False
    hp: int = 0
    potions: list[int] = field(default_factory=list)
    score: float = 0.0
    advantage: float = 0.0


def play(policy: CombatPolicy, sim: CombatSim, greedy: bool = False, max_steps: int = 300) -> Trajectory:
    tr = Trajectory()
    while not sim.done and len(tr.actions) < max_steps:
        o = sim.observe()
        a, t, lp = policy.act(collate_combat([o]), greedy=greedy)
        a, t = int(a), int(t)
        slot = int(o.enemy_slots[t]) if t >= 0 else -1
        tr.obs.append(o)
        tr.actions.append(a)
        tr.targets.append(t)
        tr.logp.append(float(lp))
        sim.step(a, slot)
    tr.won = sim.won and sim.done
    tr.hp = sim.hp()
    tr.potions = sim.potions()
    sim.close()
    return tr


DEATH_MARGIN = 3.0  # 전투에서 죽은 시행의 점수 = (살아남은 시행 중 최저 log V) - 3 nat


@torch.no_grad()
def score_outcomes(
    evaluator: Evaluator | None, base: RunState, trajs: list[Trajectory], vocab
) -> list[float]:
    """전투 직후 런 상태의 점수 (클수록 좋음).

    evaluator 가 있으면 log P(런 클리어). 확률 자체를 쓰면, 아직 약한 에이전트에게는 모든 상태가
    'P ≈ 0' 이라 시행끼리 구분이 안 된다. log 로 보면 "클리어 확률이 2배" 같은 비율 차이가 남는다.
    evaluator 가 None 이면 (학습 전 워밍업) 남은 HP 비율.
    """
    alive = [i for i, t in enumerate(trajs) if t.won]
    if evaluator is None:
        return [0.5 + 0.5 * t.hp / max(1, base.max_hp) if t.won else 0.0 for t in trajs]
    if not alive:
        return [0.0] * len(trajs)
    posts = [base.with_combat_result(trajs[i].hp, trajs[i].potions) for i in alive]
    logv = evaluator(collate_run(posts, vocab)).log_run_value.double().tolist()
    death = min(logv) - DEATH_MARGIN
    scores = [death] * len(trajs)
    for i, val in zip(alive, logv):
        scores[i] = val
    return scores


def advantages(scores: list[float], mode: str = "zscore") -> list[float]:
    s = np.asarray(scores, dtype=np.float64)
    if s.std() < 1e-6:
        return [0.0] * len(s)  # 전부 같으면 비교할 게 없다 → 학습 신호 없음
    if mode == "zscore":
        return list((s - s.mean()) / (s.std() + 1e-8))
    if mode == "best_vs_rest":
        a = np.full(len(s), -1.0 / (len(s) - 1))
        a[int(s.argmax())] = 1.0
        return list(a)
    raise ValueError(mode)


def rollout_group(
    run: RunSim,
    policy: CombatPolicy,
    evaluator: Evaluator | None,
    group_size: int,
    rng: np.random.Generator,
    mode: str = "zscore",
) -> list[Trajectory]:
    """지금 런이 서 있는 전투를 group_size 번 분기해서 굴리고 점수와 이점을 매긴다."""
    base = run.run_state()
    trajs = [play(policy, run.fork_combat(int(rng.integers(2**31)))) for _ in range(group_size)]
    for t, sc in zip(trajs, score_outcomes(evaluator, base, trajs, run.vocab)):
        t.score = sc
    for t, a in zip(trajs, advantages([t.score for t in trajs], mode)):
        t.advantage = a
    return trajs


def grpo_update(
    policy: CombatPolicy,
    optim: torch.optim.Optimizer,
    groups: list[list[Trajectory]],
    epochs: int = 2,
    clip: float = 0.2,
    ent_coef: float = 0.01,
    minibatch: int = 256,
) -> dict:
    """PPO 식 clipped objective. 각 시행 안에서는 스텝 평균, 그다음 시행들 평균 (GRPO)."""
    steps = []
    n_traj = 0
    for g in groups:
        for t in g:
            if t.advantage == 0.0 or not t.actions:
                continue
            n_traj += 1
            w = 1.0 / len(t.actions)
            for o, a, tg, lp in zip(t.obs, t.actions, t.targets, t.logp):
                steps.append((o, a, tg, lp, t.advantage, w))
    if not steps:
        return {"grpo_steps": 0}
    stats = {"loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    n_batches = 0
    for _ in range(epochs):
        perm = np.random.permutation(len(steps))
        for start in range(0, len(steps), minibatch):
            chunk = [steps[i] for i in perm[start : start + minibatch]]
            batch = collate_combat([c[0] for c in chunk])
            a = torch.tensor([c[1] for c in chunk])
            tg = torch.tensor([c[2] for c in chunk])
            old = torch.tensor([c[3] for c in chunk])
            adv = torch.tensor([c[4] for c in chunk], dtype=torch.float)
            w = torch.tensor([c[5] for c in chunk], dtype=torch.float)
            lp, ent = policy.log_prob(batch, a, tg)
            ratio = (lp - old).exp()
            obj = torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            loss = -(obj * w).sum() / n_traj - ent_coef * (ent * w).sum() / n_traj
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            stats["loss"] += float(loss)
            stats["entropy"] += float(ent.mean())
            stats["clip_frac"] += float(((ratio - 1).abs() > clip).float().mean())
            n_batches += 1
    out = {k: v / max(1, n_batches) for k, v in stats.items()}
    out["grpo_steps"] = len(steps)
    out["grpo_trajs"] = n_traj
    return out
