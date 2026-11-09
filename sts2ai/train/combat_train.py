"""전투 에이전트만 따로 학습한다 (런 없이, 무작위 상황에서 GRPO).

런 안에서만 배우면 전투 에이전트는 그 런의 카드 선택이 고른 카드만 많이 보게 된다. 그러면 "강한 카드"가
아니라 "전투 에이전트가 잘 쓰는 카드"가 계속 뽑히는 쪽으로 굳는다. 여기서는 상황을 무작위로 만들어
모든 카드와 모든 적을 고르게 보여 준다.

시나리오 하나 = (조우, 막) + 무작위 덱(10~50장) + 무작위 포션(0~9개) + 무작위 최대 HP(1~125)
               + 무작위 HP(1~최대 HP) + 시드.
- (조우, 막) 고르기 = 고루 + 집중
  - 고루: 반복마다 env.combat_pool() 을 rounds_per_iter 바퀴 돈다. 모든 적과 반드시 싸운다.
  - 집중: 추가로 focus_per_iter 개를 '만났을 때 사망률' 에 비례해 뽑는다. 사망률은 학습 중 GRPO 시행의
    결과로 (조우, 막) 마다 지수이동평균을 낸다 (처음엔 모두 0.5). 잘 지는 적과 더 많이 싸운다.
- GRPO 그룹: 같은 시나리오·같은 시드로 G 번 싸운다. 셔플과 적 난수가 같으므로 시행끼리의 차이는
  정책의 선택에서만 나온다.

점수 (클수록 좋음, 이긴 시행은 항상 진 시행보다 높다)
- 이김: 2 + clip(q / 125, -1, 1),  q = ΔHP + max_hp_weight·Δ최대HP − potion_cost_hp·쓴 포션 수
        → HP 를 덜 잃을수록, 최대 HP 가 오를수록 높다. 범위 [1, 3].
        포션은 기본으로 마음껏 쓴다 (potion_cost_hp = 0). 아껴 쓰게 하려면 0 보다 크게.
- 짐  : −1 + (적에게 준 피해 비율)   (loss_ranking=False 면 −1 고정). 범위 [−1, 0].
그룹 안에서 z-score 로 이점을 매기는 것은 런 학습의 GRPO 와 같다 (grpo.advantages, grpo_update).

전투만 학습하므로 런 맥락(RUN 토큰)은 넣지 않는다. 런 학습에서 `--init-policy` 로 불러와 이어서 배우면
그때부터 RUN 토큰 쓰는 법을 배운다.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..agents import CombatPolicy
from ..encoders import SharedEmbeddings
from .grpo import Steps, Trajectory, advantages, drive, grpo_update, play_steps
from .loop import _atomic_save, _atomic_write_text, _check_resume_config, resolve_device

CHECKPOINT = "combat_checkpoint.pt"


@dataclass
class CombatConfig:
    iterations: int = 200
    rounds_per_iter: int = 1  # 고루: 반복마다 combat_pool 을 몇 바퀴 돌지 (토이: 한 바퀴 = 조우 6 × 막 3 = 18)
    focus_per_iter: int = 18  # 집중: 사망률에 비례해 더 뽑을 시나리오 수
    focus_power: float = 1.0  # 뽑을 확률 ∝ max(사망률, focus_floor) ** focus_power. 크면 더 치우친다
    focus_floor: float = 0.05  # 사망률이 0 이 된 적도 가끔은 집중 쪽에서 뽑히게
    death_rate_ema: float = 0.3  # 사망률 지수이동평균의 새 값 비중
    group_size: int = 8
    deck_min: int = 10
    deck_max: int = 50
    upgrade_prob: float = 0.2  # 덱의 카드 한 장이 강화돼 있을 확률
    potions_min: int = 0
    potions_max: int = 9
    max_hp_min: int = 1
    max_hp_max: int = 125
    hp_min: int = 1  # HP 는 [hp_min, 최대 HP] 에서 뽑는다
    potion_cost_hp: float = 0.0  # 포션 1개를 HP 몇과 같게 볼지. 0 = 포션은 마음껏 (쓴다고 점수가 깎이지 않음)
    max_hp_weight: float = 1.0  # 최대 HP 1 을 HP 몇과 같게 볼지
    loss_ranking: bool = True  # 진 시행끼리는 적에게 준 피해가 큰 쪽을 높게
    advantage_mode: str = "zscore"
    max_steps: int = 300
    d_model: int = 64
    lr_policy: float = 3e-4
    grpo_epochs: int = 2
    eval_every: int = 10  # 반복 N 번마다 고정된 검증 시나리오를 greedy 로 평가 (0 이면 끔)
    eval_rounds: int = 3
    seed: int = 0
    torch_threads: int = 1
    device: str = "cpu"
    out_dir: str = "runs/combat"


@dataclass
class Scenario:
    encounter: int
    act: int
    deck: list[tuple[int, bool]]
    hp: int
    max_hp: int
    potions: list[int]
    seed: int
    relics: list[int] = field(default_factory=list)


def sample_scenario(env, rng: np.random.Generator, cfg: CombatConfig, encounter: int, act: int) -> Scenario:
    cards, pots = env.card_pool(), env.potion_pool()
    n_deck = int(rng.integers(cfg.deck_min, cfg.deck_max + 1))
    deck = [(int(rng.choice(cards)), bool(rng.random() < cfg.upgrade_prob)) for _ in range(n_deck)]
    n_pot = int(rng.integers(cfg.potions_min, cfg.potions_max + 1))
    potions = [int(rng.choice(pots)) for _ in range(n_pot)]
    max_hp = int(rng.integers(max(1, cfg.max_hp_min), cfg.max_hp_max + 1))
    hp = int(rng.integers(min(max(1, cfg.hp_min), max_hp), max_hp + 1))
    return Scenario(encounter, act, deck, hp, max_hp, potions, int(rng.integers(2**31)))


def scenarios_for_round(env, rng: np.random.Generator, cfg: CombatConfig) -> list[Scenario]:
    """combat_pool 한 바퀴. 순서는 섞는다."""
    pool = list(env.combat_pool())
    order = rng.permutation(len(pool))
    return [sample_scenario(env, rng, cfg, *pool[i]) for i in order]


def plan_iteration(n_pool: int, death: np.ndarray, rng: np.random.Generator, cfg: CombatConfig) -> list[int]:
    """이번 반복에 싸울 combat_pool 인덱스들. 고루(전부 rounds_per_iter 번) + 집중(사망률 비례)."""
    idx = [int(i) for _ in range(cfg.rounds_per_iter) for i in rng.permutation(n_pool)]
    if cfg.focus_per_iter > 0:
        w = np.maximum(death, cfg.focus_floor) ** cfg.focus_power
        idx += [int(i) for i in rng.choice(n_pool, size=cfg.focus_per_iter, p=w / w.sum())]
    return idx


def update_death_rates(death: np.ndarray, pool: list, groups: list[list[Trajectory]], scenarios: list[Scenario],
                       beta: float) -> None:
    """이번 반복 시행 결과로 (조우, 막) 별 사망률 지수이동평균을 갱신한다 (제자리)."""
    where = {key: i for i, key in enumerate(pool)}
    lost: dict[int, list[float]] = {}
    for g, sc in zip(groups, scenarios):
        lost.setdefault(where[(sc.encounter, sc.act)], []).extend(0.0 if t.won else 1.0 for t in g)
    for i, v in lost.items():
        death[i] = (1 - beta) * death[i] + beta * float(np.mean(v))


def combat_score(won: bool, hp0: int, hp1: int, max0: int, max1: int, pots_used: int, damage_frac: float,
                 cfg: CombatConfig) -> float:
    if won:
        q = (hp1 - hp0) - cfg.potion_cost_hp * pots_used + cfg.max_hp_weight * (max1 - max0)
        return 2.0 + float(np.clip(q / 125.0, -1.0, 1.0))
    return -1.0 + (float(np.clip(damage_frac, 0.0, 1.0)) if cfg.loss_ranking else 0.0)


def group_steps(env, sc: Scenario, cfg: CombatConfig, greedy: bool = False) -> Steps:
    """같은 시나리오·같은 시드로 G 번 (greedy 평가면 1 번) 싸우고 점수와 이점을 매긴다."""
    n = 1 if greedy else cfg.group_size
    sims = [env.make_combat(sc.encounter, sc.act, sc.deck, sc.hp, sc.max_hp, sc.potions, sc.relics, sc.seed)
            for _ in range(n)]
    # play_steps 가 끝에 close 하므로 점수에 필요한 값은 마지막 스텝 직후에 읽어야 한다 → 미리 감싼다
    info = [{} for _ in sims]
    wrapped = [_Recording(s, rec, sc.max_hp) for s, rec in zip(sims, info)]
    trajs = yield from play_steps(wrapped, greedy=greedy, max_steps=cfg.max_steps)
    n_pot0 = sum(1 for p in sc.potions if p)
    for t, rec in zip(trajs, info):
        used = max(0, n_pot0 - sum(1 for p in t.potions if p))
        t.score = combat_score(t.won, sc.hp, t.hp, sc.max_hp, rec["max_hp"], used, rec["damage"], cfg)
        rec["potions_used"] = used
        t.info = rec
    for t, a in zip(trajs, advantages([t.score for t in trajs], cfg.advantage_mode)):
        t.advantage = a
    return trajs


class _Recording:
    """CombatSim 을 감싸서 close 직전에 최대 HP 와 적에게 준 피해를 기록한다."""

    def __init__(self, sim, rec: dict, max_hp0: int):
        self.sim, self.rec, self.max_hp0 = sim, rec, max_hp0

    @property
    def done(self) -> bool:
        return self.sim.done

    @property
    def won(self) -> bool:
        return self.sim.won

    def observe(self):
        return self.sim.observe()

    def step(self, action: int, target_slot: int = -1) -> None:
        self.sim.step(action, target_slot)

    def hp(self) -> int:
        return self.sim.hp()

    def potions(self) -> list[int]:
        return self.sim.potions()

    def close(self) -> None:
        self.rec["max_hp"] = self.sim.max_hp() if hasattr(self.sim, "max_hp") else self.max_hp0
        self.rec["damage"] = self.sim.enemy_damage_frac() if hasattr(self.sim, "enemy_damage_frac") else 0.0
        self.sim.close()


def _summary(groups: list[list[Trajectory]], scenarios: list[Scenario], prefix: str) -> dict:
    trajs = [t for g in groups for t in g]
    wins = [t for t in trajs if t.won]
    hp_lost = [sc.hp - t.hp for g, sc in zip(groups, scenarios) for t in g if t.won]
    out = {
        f"{prefix}win_rate": float(np.mean([t.won for t in trajs])),
        f"{prefix}score": float(np.mean([t.score for t in trajs])),
        f"{prefix}hp_lost_on_win": float(np.mean(hp_lost)) if hp_lost else float("nan"),
        f"{prefix}potions_used": float(np.mean([t.info["potions_used"] for t in trajs])),
    }
    if wins:
        out[f"{prefix}win_steps"] = float(np.mean([len(t.actions) for t in wins]))
    return out


def _per_encounter(groups, scenarios) -> dict[str, float]:
    by: dict[str, list[bool]] = {}
    for g, sc in zip(groups, scenarios):
        by.setdefault(f"{sc.encounter}@{sc.act}", []).extend(t.won for t in g)
    return {k: float(np.mean(v)) for k, v in sorted(by.items())}


def evaluate_policy(policy: CombatPolicy, env, scenarios: list[Scenario], cfg: CombatConfig) -> dict:
    groups = drive(policy, [group_steps(env, sc, cfg, greedy=True) for sc in scenarios])
    out = _summary(groups, scenarios, "eval_")
    out["eval_by_encounter"] = _per_encounter(groups, scenarios)
    return out


def train_combat(env, cfg: CombatConfig, log=print, resume: bool = False) -> CombatPolicy:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    rng = np.random.default_rng(cfg.seed)
    device = resolve_device(cfg.device)
    policy = CombatPolicy(SharedEmbeddings(env.vocab, cfg.d_model)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr_policy)
    # 검증 시나리오는 학습 난수와 따로 고정
    eval_rng = np.random.default_rng(10**6 + cfg.seed)
    eval_set = [sc for _ in range(cfg.eval_rounds) for sc in scenarios_for_round(env, eval_rng, cfg)]
    history: list[dict] = []
    start = 0
    pool = list(env.combat_pool())
    death = np.full(len(pool), 0.5)  # (조우, 막) 별 사망률 추정
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / CHECKPOINT
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _check_resume_config(ck["config"], cfg)
        policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["opt"])
        history, start = ck["history"], ck["next_iter"]
        death = np.asarray(ck["death_rate"], dtype=np.float64)
        rng.bit_generator.state = ck["rng"]["numpy_generator"]
        np.random.set_state(ck["rng"]["numpy_global"])
        random.setstate(ck["rng"]["python"])
        torch.set_rng_state(ck["rng"]["torch"])
        if device.type == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        log(f"resume: {ckpt_path} 에서 반복 {start} 부터 이어서 학습합니다 (목표 {cfg.iterations})")
    (out / "combat_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log(f"device={device} 시나리오/반복={cfg.rounds_per_iter * len(pool)}(고루)+{cfg.focus_per_iter}(집중) "
        f"group={cfg.group_size}")
    for it in range(start, cfg.iterations):
        t0 = time.time()
        scenarios = [sample_scenario(env, rng, cfg, *pool[i]) for i in plan_iteration(len(pool), death, rng, cfg)]
        groups = drive(policy, [group_steps(env, sc, cfg) for sc in scenarios])
        update_death_rates(death, pool, groups, scenarios, cfg.death_rate_ema)
        row = {"iter": it, **_summary(groups, scenarios, "")}
        row.update(grpo_update(policy, opt, groups, epochs=cfg.grpo_epochs))
        if cfg.eval_every and ((it + 1) % cfg.eval_every == 0 or it + 1 == cfg.iterations):
            policy.eval()
            row.update(evaluate_policy(policy, env, eval_set, cfg))
            policy.train()
        row["death_rate"] = {f"{e}@{a}": round(float(d), 3) for (e, a), d in zip(pool, death)}
        row["sec"] = round(time.time() - t0, 1)
        history.append(row)
        log(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                     if k not in ("eval_by_encounter", "death_rate")))
        top = sorted(row["death_rate"].items(), key=lambda kv: -kv[1])[:4]
        log("  사망률 높은 조우@막: " + " ".join(f"{k}={v:.2f}" for k, v in top))
        if "eval_by_encounter" in row:
            log("  eval 조우@막 승률: " + " ".join(f"{k}={v:.2f}" for k, v in row["eval_by_encounter"].items()))
        _atomic_save(policy.state_dict(), out / "combat_policy.pt")
        _atomic_write_text(json.dumps(history, indent=2), out / "combat_history.json")
        _atomic_save({
            "next_iter": it + 1,
            "config": asdict(cfg),
            "policy": policy.state_dict(),
            "opt": opt.state_dict(),
            "history": history,
            "death_rate": death.tolist(),
            "rng": {
                "numpy_generator": rng.bit_generator.state,
                "numpy_global": np.random.get_state(),
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }, ckpt_path)
    return policy
