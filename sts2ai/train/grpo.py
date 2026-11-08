"""전투 에이전트 학습: GRPO (Group Relative Policy Optimization).

같은 전투를 G 번 분기해서 굴린다(각 분기는 셔플과 적 난수가 다르다). 각 시행이 끝난
뒤의 런 상태를 평가 에이전트가 채점한다 (점수 = log P(런 클리어), 전투에서 죽으면 최하점).
그룹 안에서 상대 평가를 해서, 평균보다 좋은 시행의 행동은 강화하고 나쁜 시행의 행동은
약화한다. 별도의 가치망(critic) 없이 그룹 평균이 기준선 역할을 한다.

advantage_mode
- "zscore"       : A = (점수 - 그룹 평균) / 그룹 표준편차   ← 표준 GRPO
- "best_vs_rest" : 1등 +1, 나머지 -1/(G-1)               ← "제일 좋은 방식에 보상, 나머지에 벌"

배치 롤아웃
-----------
전투를 굴리는 코드는 제너레이터로 쓴다. 정책이 필요할 때마다 `yield [ActRequest, ...]` 로 행동을
요청하고, `send` 로 [(행동, 타깃, log 확률), ...] 를 돌려받는다. `drive` 가 여러 제너레이터를 같이
돌리면서 한 라운드의 요청을 전부 모아 정책 forward 한 번으로 처리한다. 그래서 그룹 하나의 분기 G 개,
나아가 런 여러 판의 분기와 진짜 전투가 한 배치로 묶인다 (GPU 를 쓸 때 중요).
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np
import torch

from ..agents import CombatPolicy, Evaluator
from ..batching import collate_combat, collate_run, module_device, to_device
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
    run_vec: torch.Tensor | None = None  # [d] 전투 시작 시점의 런 맥락 (맵 포함). 전투 내내 같다.


@dataclass
class ActRequest:
    obs: CombatObs
    run_vec: torch.Tensor | None = None  # [d]
    greedy: bool = False


Action = tuple[int, int, float]  # (행동, enemy_tok 안의 타깃 인덱스 또는 -1, log 확률)
# 정책에게 행동을 물어보며 진행하는 코루틴. 끝나면 return 값이 결과.
Steps = Generator[list[ActRequest], list[Action], object]


@torch.no_grad()
def act_batch(policy: CombatPolicy, reqs: list[ActRequest]) -> list[Action]:
    """요청 여러 개를 정책 forward 로 처리한다. greedy 여부가 섞여 있으면 둘로 나눈다."""
    out: list[Action | None] = [None] * len(reqs)
    dev = module_device(policy)
    for greedy in (False, True):
        idx = [i for i, r in enumerate(reqs) if r.greedy == greedy]
        if not idx:
            continue
        rvs = [reqs[i].run_vec for i in idx]
        if all(v is None for v in rvs):
            rv = None
        elif any(v is None for v in rvs):
            raise ValueError("run_vec 이 있는 요청과 없는 요청을 한 배치에 섞을 수 없습니다")
        else:
            rv = torch.stack(rvs).to(dev)
        batch = to_device(collate_combat([reqs[i].obs for i in idx]), dev)
        a, t, lp = policy.act(batch, greedy=greedy, run_vec=rv)
        for i, ai, ti, li in zip(idx, a.tolist(), t.tolist(), lp.tolist()):
            out[i] = (ai, ti, li)
    return out  # type: ignore[return-value]


def drive(policy: CombatPolicy, gens: list[Steps]) -> list:
    """제너레이터들을 같이 끝까지 돌리고 각자의 return 값을 순서대로 돌려준다.

    한 라운드 = 아직 안 끝난 제너레이터 전부의 요청을 모아 act_batch 한 번.
    """
    results: list = [None] * len(gens)
    pending: dict[int, list[ActRequest]] = {}

    def advance(i: int, answer: list[Action] | None) -> None:
        try:
            req = next(gens[i]) if answer is None else gens[i].send(answer)
            while not req:  # 빈 요청은 바로 빈 답으로 넘긴다
                req = gens[i].send([])
            pending[i] = req
        except StopIteration as stop:
            results[i] = stop.value
            pending.pop(i, None)

    for i in range(len(gens)):
        advance(i, None)
    while pending:
        order = list(pending)
        flat = [r for i in order for r in pending[i]]
        answers = act_batch(policy, flat)
        k = 0
        for i in order:
            n = len(pending[i])
            advance(i, answers[k : k + n])
            k += n
    return results


def play_steps(sims: list[CombatSim], greedy: bool = False, max_steps: int = 300,
               run_vec: torch.Tensor | None = None) -> Steps:
    """전투 여러 개를 한 스텝씩 같이 진행한다. return: 전투마다 Trajectory."""
    trajs = [Trajectory(run_vec=run_vec) for _ in sims]
    while True:
        live = [i for i, sim in enumerate(sims) if not sim.done and len(trajs[i].actions) < max_steps]
        if not live:
            break
        obs = [sims[i].observe() for i in live]
        answers = yield [ActRequest(o, run_vec, greedy) for o in obs]
        for i, o, (a, t, lp) in zip(live, obs, answers):
            slot = int(o.enemy_slots[t]) if t >= 0 else -1
            tr = trajs[i]
            tr.obs.append(o)
            tr.actions.append(a)
            tr.targets.append(t)
            tr.logp.append(lp)
            sims[i].step(a, slot)
    for tr, sim in zip(trajs, sims):
        tr.won = sim.won and sim.done
        tr.hp = sim.hp()
        tr.potions = sim.potions()
        sim.close()
    return trajs


def play(policy: CombatPolicy, sim: CombatSim, greedy: bool = False, max_steps: int = 300,
         run_vec: torch.Tensor | None = None) -> Trajectory:
    """전투 하나를 끝까지 굴린다."""
    return drive(policy, [play_steps([sim], greedy, max_steps, run_vec)])[0][0]


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
    batch = to_device(collate_run(posts, vocab), module_device(evaluator))
    logv = evaluator(batch).log_run_value.double().tolist()
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


def rollout_group_steps(
    run: RunSim,
    evaluator: Evaluator | None,
    group_size: int,
    rng: np.random.Generator,
    mode: str = "zscore",
    run_vec: torch.Tensor | None = None,
) -> Steps:
    """지금 런이 서 있는 전투를 group_size 번 분기해서 같이 굴리고 점수와 이점을 매긴다."""
    base = run.run_state()
    sims = [run.fork_combat(int(rng.integers(2**31))) for _ in range(group_size)]
    trajs = yield from play_steps(sims, run_vec=run_vec)
    for t, sc in zip(trajs, score_outcomes(evaluator, base, trajs, run.vocab)):
        t.score = sc
    for t, a in zip(trajs, advantages([t.score for t in trajs], mode)):
        t.advantage = a
    return trajs


def rollout_group(
    run: RunSim,
    policy: CombatPolicy,
    evaluator: Evaluator | None,
    group_size: int,
    rng: np.random.Generator,
    mode: str = "zscore",
    run_vec: torch.Tensor | None = None,
) -> list[Trajectory]:
    return drive(policy, [rollout_group_steps(run, evaluator, group_size, rng, mode, run_vec)])[0]


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
                steps.append((o, a, tg, lp, t.advantage, w, t.run_vec))
    if not steps:
        return {"grpo_steps": 0}
    stats = {"loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    n_batches = 0
    dev = module_device(policy)
    for _ in range(epochs):
        perm = np.random.permutation(len(steps))
        for start in range(0, len(steps), minibatch):
            chunk = [steps[i] for i in perm[start : start + minibatch]]
            batch = to_device(collate_combat([c[0] for c in chunk]), dev)
            a = torch.tensor([c[1] for c in chunk], device=dev)
            tg = torch.tensor([c[2] for c in chunk], device=dev)
            old = torch.tensor([c[3] for c in chunk], device=dev)
            adv = torch.tensor([c[4] for c in chunk], dtype=torch.float, device=dev)
            w = torch.tensor([c[5] for c in chunk], dtype=torch.float, device=dev)
            rvs = [c[6] for c in chunk]
            rv = torch.stack(rvs).to(dev) if rvs[0] is not None else None
            lp, ent = policy.log_prob(batch, a, tg, rv)
            ratio = (lp - old).exp()
            obj = torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            loss = -(obj * w).sum() / n_traj - ent_coef * (ent * w).sum() / n_traj
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            stats["loss"] += float(loss.detach())
            stats["entropy"] += float(ent.detach().mean())
            stats["clip_frac"] += float(((ratio - 1).abs() > clip).float().mean())
            n_batches += 1
    out = {k: v / max(1, n_batches) for k, v in stats.items()}
    out["grpo_steps"] = len(steps)
    out["grpo_trajs"] = n_traj
    return out
