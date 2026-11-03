"""선택 정책 학습: 전투 밖 선택도 GRPO 로.

선택 화면(맵 노드, 카드 보상, 상점, 휴식, 이벤트, 덱 카드 고르기)에서 choice_fork_prob 확률로
런 전체를 choice_group_size 번 복제한다(숨은 난수는 복제마다 새로 뽑음). 복제마다 선택 정책에서
선택지를 하나씩 뽑아 실행하고, 맵 화면이 choice_horizon 번 더 나올 때까지(대략 다음 방 1~2개)
지금 정책들로 이어서 플레이한다. 그 시점의 평가기 log V 로 채점해 그룹 안에서 비교한다.

- 죽은 복제: 살아남은 복제 중 최저 점수 - DEATH_MARGIN. 모두 죽거나 모두 같으면 신호 없음.
- 런을 클리어한 복제: log V = 0 (확률 1).
- 워밍업(평가기 데이터가 모이기 전): 남은 HP 비율.
전투 GRPO 처럼 critic 없이 그룹 평균이 기준선이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..agents.choice_policy import collate_options
from ..batching import collate_run, module_device, to_device
from ..envs.base import PHASE_COMBAT, PHASE_DONE, PHASE_MAP, RunSim
from ..schema import ChoiceOption, RunState
from .grpo import DEATH_MARGIN, advantages


@dataclass
class ChoiceSample:
    state: RunState
    options: list[ChoiceOption]
    action: int  # options 안의 인덱스
    logp: float
    advantage: float = 0.0
    score: float = 0.0


def map_options(state: RunState) -> list[ChoiceOption]:
    return [ChoiceOption(action=n, kind="map.node", node=n) for n in state.map.next_choices()]


def screen_options(run: RunSim, state: RunState) -> list[ChoiceOption]:
    """지금 화면의 선택지. 맵이면 다음 노드들, 그 밖의 화면이면 환경이 주는 목록(자동 진행이면 빈 목록)."""
    if run.phase == PHASE_MAP:
        return map_options(state)
    return run.choice_options()


@torch.no_grad()
def choice_logits(agent, states: list[RunState], option_lists: list[list[ChoiceOption]]) -> torch.Tensor:
    dev = module_device(agent.choice)
    run_batch = to_device(collate_run(states, agent.vocab), dev)
    opt = to_device(collate_options(option_lists, agent.vocab), dev)
    enc = agent.evaluator.encoder.encode_all(run_batch)
    return agent.choice(enc, opt)


def pick(agent, state: RunState, options: list[ChoiceOption], greedy: bool) -> tuple[int, float]:
    """(선택지 인덱스, log 확률). 샘플링은 torch 전역 난수를 쓴다 (런마다 시드가 고정돼 있다)."""
    logits = choice_logits(agent, [state], [options])[0, : len(options)]
    dist = torch.distributions.Categorical(logits=logits)
    i = int(logits.argmax()) if greedy else int(dist.sample())
    return i, float(dist.log_prob(torch.tensor(i)))


def apply_option(run: RunSim, option: ChoiceOption) -> None:
    if run.phase == PHASE_MAP:
        run.choose_node(option.node)
    else:
        run.step_choice(option.action)


def play_until(run: RunSim, agent, cfg, rng: np.random.Generator, horizon: int, max_steps: int = 5000) -> None:
    """맵 화면을 horizon 번 더 만나거나 런이 끝날 때까지, 지금 정책들로(샘플링) 진행한다. 기록은 남기지 않는다."""
    maps_seen, steps, last_combat, run_vec = 0, 0, None, None
    while run.phase != PHASE_DONE and steps < max_steps:
        steps += 1
        if run.phase == PHASE_COMBAT:
            cid = run.combat_id()
            if cid != last_combat:
                last_combat = cid
                run_vec = agent.run_vec(run.run_state()) if cfg.policy_sees_map else None
            o = run.combat_obs()
            rv = run_vec.unsqueeze(0) if run_vec is not None else None
            a, t, _ = agent.combat_actor()([o], rv, False)
            run.combat_step(int(a[0]), int(o.enemy_slots[int(t[0])]) if int(t[0]) >= 0 else -1)
            continue
        if run.phase == PHASE_MAP:
            maps_seen += 1
            if maps_seen > horizon:
                return
        state = run.run_state()
        opts = screen_options(run, state)
        if not opts:
            run.default_step(rng)
            continue
        i, _ = pick(agent, state, opts, greedy=False)
        apply_option(run, opts[i])


@torch.no_grad()
def _score(run: RunSim, agent, use_evaluator: bool) -> float | None:
    """None = 죽음. 클리어면 0 (log 1)."""
    if run.died:
        return None
    if run.won:
        return 0.0
    state = run.run_state()
    if not use_evaluator:
        return 0.5 + 0.5 * state.hp / max(1, state.max_hp)
    batch = collate_run([state], agent.vocab)
    return float(agent.evaluator(batch).log_run_value[0])


def rollout_choice_group(run: RunSim, agent, cfg, rng: np.random.Generator, state: RunState,
                         options: list[ChoiceOption], use_evaluator: bool) -> list[ChoiceSample]:
    """지금 선택 화면에서 런을 G 번 복제해 서로 다른 선택을 비교한다."""
    G = cfg.choice_group_size
    logits = choice_logits(agent, [state], [options])[0, : len(options)]
    dist = torch.distributions.Categorical(logits=logits)
    picks = [int(dist.sample()) for _ in range(G)]
    samples, raw = [], []
    for i in picks:
        c = run.clone(int(rng.integers(2**31)))
        try:
            apply_option(c, options[i])
            play_until(c, agent, cfg, rng, cfg.choice_horizon)
            raw.append(_score(c, agent, use_evaluator))
        finally:
            c.close()
        samples.append(ChoiceSample(state, options, i, float(dist.log_prob(torch.tensor(i)))))
    alive = [s for s in raw if s is not None]
    if not alive:
        return []
    death = min(alive) - DEATH_MARGIN if use_evaluator else 0.0
    scores = [s if s is not None else death for s in raw]
    for smp, sc, adv in zip(samples, scores, advantages(scores, cfg.advantage_mode)):
        smp.score, smp.advantage = float(sc), float(adv)
    return [s for s in samples if s.advantage != 0.0]


def choice_update(agent, optim: torch.optim.Optimizer, samples: list[ChoiceSample], epochs: int = 2,
                  clip: float = 0.2, ent_coef: float = 0.01, minibatch: int = 128, trunk_grad: bool = True,
                  params=None) -> dict:
    """PPO 식 clipped objective (선택 한 번 = 한 스텝)."""
    if not samples:
        return {"choice_samples": 0}
    dev = module_device(agent.choice)
    params = list(params) if params is not None else list(agent.choice.parameters())
    stats = {"choice_loss": 0.0, "choice_entropy": 0.0, "choice_clip_frac": 0.0}
    n_batches = 0
    for _ in range(epochs):
        perm = np.random.permutation(len(samples))
        for start in range(0, len(samples), minibatch):
            chunk = [samples[i] for i in perm[start : start + minibatch]]
            run_batch = to_device(collate_run([s.state for s in chunk], agent.vocab), dev)
            opt = to_device(collate_options([s.options for s in chunk], agent.vocab), dev)
            enc = agent.choice.trunk(agent.evaluator.encoder, run_batch, grad=trunk_grad)
            logits = agent.choice(enc, opt)
            logp_all = torch.log_softmax(logits, -1)
            a = torch.tensor([s.action for s in chunk], device=dev)
            lp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
            ent = -(logp_all.exp() * logp_all.clamp(min=-30)).sum(-1)
            old = torch.tensor([s.logp for s in chunk], device=dev)
            adv = torch.tensor([s.advantage for s in chunk], dtype=torch.float, device=dev)
            ratio = (lp - old).exp()
            obj = torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            loss = -obj.mean() - ent_coef * ent.mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            stats["choice_loss"] += float(loss.detach())
            stats["choice_entropy"] += float(ent.mean().detach())
            stats["choice_clip_frac"] += float(((ratio.detach() - 1).abs() > clip).float().mean())
            n_batches += 1
    out = {k: v / max(1, n_batches) for k, v in stats.items()}
    out["choice_samples"] = len(samples)
    return out
