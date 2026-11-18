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

from .. import schema as S
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
    # [d] 전투 시작 시점의 런 맥락 (맵 포함). 전투 내내 같다.
    # 워커 프로세스에서 돌려받을 때는 numpy 배열일 수 있다 (grpo_update 가 둘 다 받는다).
    run_vec: torch.Tensor | np.ndarray | None = None
    # 전투 시작 시점의 런 상태. trunk_grad 로 학습할 때 run_vec 을 기울기와 함께 다시 계산하는 데 쓴다.
    run_state: RunState | None = None
    # 전투가 끝난 뒤의 실제 런 상태 (런 전체를 복제한 분기만 준다). 없으면 채점은 HP·포션만 바꿔 넣는 근사.
    post_state: RunState | None = None


class LocalActor:
    """전투 정책을 이 프로세스에서 바로 돌리는 actor. actor(관측 목록, run_vec [B,d] 또는 None, greedy) → (행동, 타깃, log 확률)."""

    def __init__(self, policy: CombatPolicy):
        self.policy = policy

    def __call__(self, obs: list[CombatObs], run_vec: torch.Tensor | None, greedy: bool = False):
        dev = module_device(self.policy)
        batch = to_device(collate_combat(obs), dev)
        rv = run_vec.to(dev) if run_vec is not None else None
        a, t, lp = self.policy.act(batch, greedy=greedy, run_vec=rv)
        return a.cpu(), t.cpu(), lp.cpu()


def as_actor(policy_or_actor):
    return LocalActor(policy_or_actor) if isinstance(policy_or_actor, CombatPolicy) else policy_or_actor


def play_many(policy_or_actor, sims: list[CombatSim], greedy: bool = False, max_steps: int = 300,
              run_vec: torch.Tensor | None = None) -> list[Trajectory]:
    """전투 여러 개를 한 스텝씩 같이 진행한다. 아직 안 끝난 전투들의 관측을 한 배치로 묶어 정책을 한 번만 부른다.

    배치 1 을 여러 번 부르는 것보다 관측당 계산이 몇 배 빠르다 (배치 8 에서 CPU 약 3배).
    run_vec 은 [d] (모든 전투가 같은 런 맥락) 또는 [len(sims), d] (전투마다 다른 런 맥락).
    """
    actor = as_actor(policy_or_actor)
    per_sim = run_vec is not None and run_vec.dim() == 2
    trs = [Trajectory(run_vec=(run_vec[i] if per_sim else run_vec)) for i in range(len(sims))]
    live = [i for i, s in enumerate(sims) if not s.done]
    while live:
        obs = [sims[i].observe() for i in live]
        if run_vec is None:
            rv = None
        elif per_sim:
            rv = run_vec[live]
        else:
            rv = run_vec.unsqueeze(0).expand(len(live), -1)
        a, t, lp = actor(obs, rv, greedy)
        nxt = []
        for k, i in enumerate(live):
            ai, ti = int(a[k]), int(t[k])
            o, tr = obs[k], trs[i]
            tr.obs.append(o)
            tr.actions.append(ai)
            tr.targets.append(ti)
            tr.logp.append(float(lp[k]))
            sims[i].step(ai, int(o.enemy_slots[ti]) if ti >= 0 else -1)
            if not sims[i].done and len(tr.actions) < max_steps:
                nxt.append(i)
        live = nxt
    for tr, sim in zip(trs, sims):
        tr.won = sim.won and sim.done
        tr.hp = sim.hp()
        tr.potions = sim.potions()
        if hasattr(sim, "post_state"):
            tr.post_state = sim.post_state()
        sim.close()
    return trs


def play(policy_or_actor, sim: CombatSim, greedy: bool = False, max_steps: int = 300,
         run_vec: torch.Tensor | None = None) -> Trajectory:
    return play_many(policy_or_actor, [sim], greedy, max_steps, run_vec)[0]


DEATH_MARGIN = 3.0  # 전투에서 죽은 시행의 점수 = (살아남은 시행 중 최저 log V) - 3 nat


@torch.no_grad()
def damage_dealt(t: Trajectory) -> float:
    """마지막 관측에서 적에게 준 피해 비율 (0~1): 1 − 살아 있는 적들의 HP 비율 평균."""
    if not t.obs:
        return 0.0
    o = t.obs[-1]
    if not len(o.enemy_tokens):
        return 1.0 if t.won else 0.0
    frac = o.feats[o.enemy_tokens, S.F_ENEMY_HP_FRAC]
    return float(1.0 - np.clip(frac, 0, 1).mean())


def score_outcomes(
    evaluator: Evaluator | None, base: RunState, trajs: list[Trajectory], vocab
) -> list[float]:
    """전투 직후 런 상태의 점수 (클수록 좋음).

    evaluator 가 있으면 이긴 분기는 log P(런 클리어). 확률 자체를 쓰면, 아직 약한 에이전트에게는 모든 상태가
    'P ≈ 0' 이라 시행끼리 구분이 안 된다. log 로 보면 "클리어 확률이 2배" 같은 비율 차이가 남는다.
    진 분기는 전투 전 상태의 log V − DEATH_MARGIN + 적에게 준 피해 비율 (0~1): 그룹과 상관없는 같은 척도라,
    모두 진 그룹도 이긴 그룹보다 점수가 높아지지 않고, 진 분기끼리는 적을 많이 깎을수록 높다.
    이긴 분기가 있으면 진 분기는 이긴 분기 최저 점수 − 1 을 넘지 않게 자른다 (이기면 늘 더 좋다).
    evaluator 가 None 이면 (학습 전 워밍업) 이기면 0.5 + 0.5 × 남은 HP 비율, 지면 0.5 × 피해 비율.
    """
    if evaluator is None:
        return [0.5 + 0.5 * t.hp / max(1, base.max_hp) if t.won else 0.5 * damage_dealt(t) for t in trajs]
    alive = [i for i, t in enumerate(trajs) if t.won]
    # 런 전체를 복제한 분기면 전투 뒤 실제 런 상태(덱·골드·최대 HP·유물 카운터 변화 포함), 아니면 HP·포션만 바꾼 근사
    posts = [trajs[i].post_state or base.with_combat_result(trajs[i].hp, trajs[i].potions) for i in alive]
    batch = to_device(collate_run([base] + posts, vocab), module_device(evaluator))  # 메인 프로세스(GPU)에서도 돌게
    logv = evaluator(batch).log_run_value.double().tolist()
    base_v, post_v = logv[0], logv[1:]
    cap = min(post_v) - 1.0 if post_v else float("inf")
    scores = [min(base_v - DEATH_MARGIN + damage_dealt(t), cap) for t in trajs]
    for i, val in zip(alive, post_v):
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
    policy,  # CombatPolicy 또는 actor
    evaluator: Evaluator | None,
    group_size: int,
    rng: np.random.Generator,
    mode: str = "zscore",
    run_vec: torch.Tensor | None = None,
    crn: bool = False,
    adapt_max: int = 0,
    adapt_eps: float = 0.05,
    seed: int | None = None,
) -> list[Trajectory]:
    """지금 런이 서 있는 전투를 group_size 번 분기해서 굴리고 점수와 이점을 매긴다.

    crn: 공통 난수. 분기들이 같은 시드로 숨은 난수(뽑기 순서·적 행동 등)를 뽑아, 결과 차이가 주로 행동에서 나오게 한다.
    adapt_max > group_size 면 적응 그룹: (가장 좋은 점수 − 평균) 이 adapt_eps 보다 크면 group_size 개씩 더 싸워 보고,
    새로 싸운 묶음이 가장 좋은 점수를 adapt_eps 넘게 올리지 못하면 멈춘다 (최대 adapt_max 개).
    """
    base = run.run_state()
    seed = int(rng.integers(2**31)) if seed is None else int(seed)

    def batch(n):
        forks = [run.fork_combat(seed if crn else int(rng.integers(2**31))) for _ in range(n)]
        out = play_many(policy, forks, run_vec=run_vec)  # 분기들을 같이 진행 (정책 호출을 묶는다)
        return out, score_outcomes(evaluator, base, out, run.vocab)

    trajs, scores = batch(group_size)
    while adapt_max > len(trajs):
        if max(scores) - float(np.mean(scores)) <= adapt_eps:
            break
        more, ms = batch(group_size)
        prev = max(scores)
        trajs, scores = trajs + more, scores + ms
        if max(ms) <= prev + adapt_eps:
            break
    for t, sc in zip(trajs, scores):
        t.score = sc
        t.run_state = base
    for t, a in zip(trajs, advantages(scores, mode)):
        t.advantage = a
    return trajs


def group_gap(group: list[Trajectory], n: int | None = None) -> float:
    """개선 여지: 그룹(앞 n 개)에서 찾은 가장 좋은 점수 − 평균 점수. 0 에 가까우면 그 상황은 충분히 배운 것."""
    s = [t.score for t in (group[:n] if n else group)]
    return float(max(s) - np.mean(s)) if s else 0.0


def action_tokens(t: Trajectory) -> list[tuple[int, int, int]]:
    """궤적의 행동마다 (행동 토큰 종류, 토큰 id, 타깃 적 슬롯 또는 -1). 행동 i 는 토큰 1 + i 다."""
    out = []
    for o, a, ti in zip(t.obs, t.actions, t.targets):
        k = 1 + int(a)
        ty = int(o.types[k]) if k < len(o.types) else 0
        vid = int(o.ids[k]) if k < len(o.ids) else 0
        slot = int(o.enemy_slots[ti]) if ti is not None and 0 <= ti < len(o.enemy_slots) else -1
        out.append((ty, vid, slot))
    return out


def _trace_group(group: list[Trajectory]) -> list[dict]:
    return [{"score": float(t.score), "won": bool(t.won), "hp": int(t.hp), "actions": action_tokens(t)} for t in group]


def focus_train(policy, optim, run, token: dict, evaluator, cfg, rng: np.random.Generator, other_groups,
                run_vec_fn=None, params=None, run_vec_of=None, trace: bool = False,
                other_tokens: list | None = None) -> dict:
    """집중 학습: 한 전투 상태에서, 학습용 시드로 싸워 업데이트하고 검증용 시드 점수가 더 오르지 않으면 멈춘다.

    학습용 시드 focus_train_seeds 개 × 분기 group_size 개 (시드마다 공통 난수 그룹),
    검증용 시드 focus_val_seeds 개 × group_size 개 (업데이트에 쓰지 않고 평균 점수만 잰다).
    검증 점수가 focus_eps 넘게 오르지 않으면 멈춘다 (학습 점수만 오르면 그 판의 난수를 외우는 과적합).
    업데이트에는 이번 반복의 다른 전투 그룹을 같은 수만큼 섞어, 다른 전투에서 배운 것이 흐트러지지 않게 한다.
    other_tokens: 다른 전투 상태들. 시드를 고정해 같은 난수로 처음과 매 라운드 다시 싸워 보고, 평균 점수가
    focus_other_tol 넘게 떨어진 라운드는 받아들이지 않는다 (그 전투만 맞추느라 다른 전투가 나빠지는 것을 막는다).
    멈출 때는 받아들인 라운드 중 가장 좋았던 가중치로 되돌린다.
    """
    G = cfg.group_size
    train_seeds = [int(rng.integers(2**31)) for _ in range(cfg.focus_train_seeds)]
    val_seeds = [int(rng.integers(2**31)) for _ in range(cfg.focus_val_seeds)]

    def at_start(tok=None):
        run.restore(tok or token, int(rng.integers(2**31)))

    others = [(tok, int(rng.integers(2**31))) for tok in (other_tokens or [])]

    def other_score():
        if not others:
            return 0.0
        vals = []
        for tok, s in others:
            at_start(tok)
            rv = run_vec_of(run.run_state()) if run_vec_of is not None else None
            g = rollout_group(run, policy, evaluator, G, rng, "zscore", run_vec=rv, crn=True, seed=s)
            vals.append(float(np.mean([t.score for t in g])))
        return float(np.mean(vals))

    def groups_for(seeds):
        out = []
        for s in seeds:
            at_start()
            rv = run_vec_of(run.run_state()) if run_vec_of is not None else None
            out.append(rollout_group(run, policy, evaluator, G, rng, "zscore", run_vec=rv, crn=True, seed=s))
        return out

    rounds_log = []  # trace: 라운드마다 검증 점수와, 첫 검증 시드(같은 난수)에서 싸운 분기들

    def val_score():
        gs = groups_for(val_seeds)
        v = float(np.mean([t.score for g in gs for t in g]))
        if trace:
            rounds_log.append({"val": v, "same_seed": _trace_group(gs[0])})
        return v

    # 다른 전투 점수의 흔들림(정책이 확률로 고르는 데서 옴)을 두 번 재서, 허용 폭을 그 두 배 이상으로 잡는다
    if others:
        oa, ob = other_score(), other_score()
        o0, tol = (oa + ob) / 2, max(cfg.focus_other_tol, 2 * abs(oa - ob))
    else:
        o0, tol = 0.0, float("inf")
    o_last = o0
    v0 = best = val_score()
    # 검증 점수가 가장 좋았던 때의 가중치. 멈출 때 이걸로 되돌린다 (마지막 업데이트가 검증 점수를 낮췄을 수 있다)
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    tr0 = None
    rounds, stale = 0, 0
    for _ in range(cfg.focus_rounds):
        tg = groups_for(train_seeds)
        tr = float(np.mean([t.score for g in tg for t in g]))
        tr0 = tr if tr0 is None else tr0
        if trace and rounds_log:
            rounds_log[-1]["train"] = tr  # 이 라운드 업데이트에 쓴 학습용 시드 점수
        mix = [other_groups[int(i)] for i in rng.choice(len(other_groups), min(len(tg), len(other_groups)),
                                                        replace=False)] if other_groups else []
        grpo_update(policy, optim, tg + mix, epochs=1, run_vec_fn=run_vec_fn, params=params)
        rounds += 1
        v = val_score()
        o_now = other_score() if others else o0
        ok_other = o_now >= o0 - tol
        if trace and rounds_log:
            rounds_log[-1]["other"] = o_now
        if v > best + cfg.focus_eps and ok_other:
            best, stale, o_last = v, 0, o_now
            best_state = {k: x.detach().clone() for k, x in policy.state_dict().items()}
        else:
            stale += 1
            if stale >= 1:
                break
    policy.load_state_dict(best_state)  # 가장 좋았던 라운드로 되돌린다 (나빠진 업데이트는 버림)
    out = {"rounds": rounds, "val_gain": best - v0, "other_delta": o_last - o0, "others": len(others), "train_first": tr0 if tr0 is not None else 0.0}
    if trace:
        out["trace"] = {"val_seed": val_seeds[0], "rounds": rounds_log}
    return out


def grpo_update(
    policy: CombatPolicy,
    optim: torch.optim.Optimizer,
    groups: list[list[Trajectory]],
    epochs: int = 2,
    clip: float = 0.2,
    ent_coef: float = 0.01,
    minibatch: int = 256,
    run_vec_fn=None,
    params=None,
    ent_floor: float = 0.0,
    ent_floor_coef: float = 1.0,
    amp: bool = False,
) -> dict:
    """PPO 식 clipped objective. 각 시행 안에서는 스텝 평균, 그다음 시행들 평균 (GRPO).

    엔트로피 하한 (ent_floor > 0): 할 수 있는 행동이 2 개 이상인 스텝마다 H / log(행동 수) 가 ent_floor 아래면
    ent_floor_coef × (ent_floor − H/log K) 를 손실에 더한다 (덱 빌딩 모듈의 deck_ent_floor 와 같은 방식).

    run_vec_fn(states) 를 주면 저장해 둔 run_vec 대신 전투 시작 상태에서 run_vec 을 기울기와 함께
    다시 계산한다 (정책 손실이 평가기 트렁크까지 흘러간다). params 는 기울기를 자를 파라미터 목록.
    """
    steps = []
    n_traj = 0
    for g in groups:
        for t in g:
            if t.advantage == 0.0 or not t.actions:
                continue
            n_traj += 1
            w = 1.0 / len(t.actions)
            for o, a, tg, lp in zip(t.obs, t.actions, t.targets, t.logp):
                steps.append((o, a, tg, lp, t.advantage, w, t.run_vec, t.run_state))
    if not steps:
        return {"grpo_steps": 0}
    stats = {"loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "ent_norm": 0.0, "below_floor": 0.0}
    n_batches = 0
    dev = module_device(policy)
    stats["oom_splits"] = 0

    def run_chunk(chunk) -> None:
        """미니배치 하나로 한 번 업데이트. GPU 메모리가 모자라면 반으로 나눠 다시 한다 (각 반쪽이 한 번씩 업데이트)."""
        try:
            update_chunk(chunk)
        except Exception as exc:  # noqa: BLE001 — torch 버전마다 OOM 예외 종류가 다르다
            if "out of memory" not in str(exc) or len(chunk) <= 1 or dev.type != "cuda":
                raise
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            stats["oom_splits"] += 1
            half = len(chunk) // 2
            run_chunk(chunk[:half])
            run_chunk(chunk[half:])

    def update_chunk(chunk) -> None:
        nonlocal n_batches
        if True:
            batch = to_device(collate_combat([c[0] for c in chunk]), dev)
            a = torch.tensor([c[1] for c in chunk], device=dev)
            tg = torch.tensor([c[2] for c in chunk], device=dev)
            old = torch.tensor([c[3] for c in chunk], device=dev)
            adv = torch.tensor([c[4] for c in chunk], dtype=torch.float, device=dev)
            w = torch.tensor([c[5] for c in chunk], dtype=torch.float, device=dev)
            rvs = [c[6] for c in chunk]
            if run_vec_fn is not None and rvs[0] is not None and chunk[0][7] is not None:
                # 같은 전투의 스텝들은 같은 시작 상태를 공유하므로 상태별로 한 번만 계산한다
                uniq: dict[int, int] = {}
                states = []
                for c in chunk:
                    if id(c[7]) not in uniq:
                        uniq[id(c[7])] = len(states)
                        states.append(c[7])
                vecs = run_vec_fn(states)
                rv = vecs[torch.tensor([uniq[id(c[7])] for c in chunk], device=vecs.device)]
            else:
                rv = torch.stack([torch.as_tensor(r) for r in rvs]).to(dev) if rvs[0] is not None else None
            # amp: GPU 에서 bf16 으로 계산 (큰 모델용, 손실·확률은 float 로 되돌린다)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
                lp, ent = policy.log_prob(batch, a, tg, rv)
            lp, ent = lp.float(), ent.float()
            ratio = (lp - old).exp()
            obj = torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            loss = -(obj * w).sum() / n_traj - ent_coef * (ent * w).sum() / n_traj
            k = batch["legal"].sum(-1).float()
            multi = k > 1
            hn = torch.where(multi, ent / torch.log(k.clamp(min=2)), torch.ones_like(ent))
            if ent_floor > 0 and multi.any():
                loss = loss + ent_floor_coef * torch.relu(ent_floor - hn[multi]).mean()
            if multi.any():
                stats["ent_norm"] += float(hn[multi].mean().detach())
                stats["below_floor"] += float((hn[multi] < ent_floor).float().mean()) if ent_floor > 0 else 0.0
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(params) if params is not None else policy.parameters(), 1.0)
            optim.step()
            stats["loss"] += float(loss.detach())
            stats["entropy"] += float(ent.mean().detach())
            stats["clip_frac"] += float(((ratio - 1).abs() > clip).float().mean())
            n_batches += 1

    for _ in range(epochs):
        perm = np.random.permutation(len(steps))
        for start in range(0, len(steps), minibatch):
            run_chunk([steps[i] for i in perm[start : start + minibatch]])
    oom_splits = stats.pop("oom_splits")
    out = {k: v / max(1, n_batches) for k, v in stats.items()}
    out["oom_splits"] = oom_splits
    if dev.type == "cuda":
        torch.cuda.empty_cache()  # 다음 수집 때 추론 서버가 쓸 GPU 메모리를 돌려준다
    out["grpo_steps"] = len(steps)
    out["grpo_trajs"] = n_traj
    return out
