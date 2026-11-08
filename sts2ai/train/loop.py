"""전체 학습 루프.

반복마다:
  1) 런을 여러 판 플레이한다.
     - 전투가 시작되면: 그 전투를 G 번 분기해 굴리고(GRPO 그룹), 그다음 진짜 전투를 정책으로 플레이.
     - 맵에서는: 평가 에이전트가 계산한 '이 노드로 가면 막을 살아서 끝낼 확률'이 가장 높은 노드
       (ε 확률로 무작위 — 다양한 길의 데이터를 모으려고).
     - 나머지 화면은 규칙 정책.
  2) 모은 스냅숏으로 평가 에이전트를 학습한다 (노드 위험도 + 미래 막 위험도).
  3) 모은 GRPO 그룹으로 전투 에이전트를 학습한다.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..agents import CombatPolicy, Evaluator
from ..agents.chooser import ExpectedValueChooser
from ..batching import collate_combat, collate_run
from ..encoders import SharedEmbeddings
from ..envs.base import PHASE_COMBAT, PHASE_DONE, PHASE_MAP, RunSim
from .evaluator_train import RunRecorder, Snapshot, train_evaluator
from .grpo import Trajectory, grpo_update, rollout_group


@dataclass
class Config:
    iterations: int = 20
    runs_per_iter: int = 8
    group_size: int = 8  # GRPO 그룹 크기 G
    grpo_every_n_combats: int = 1  # 전투 몇 번에 한 번 그룹을 굴릴지 (속도 조절)
    advantage_mode: str = "zscore"  # 또는 "best_vs_rest"
    map_epsilon: float = 0.2
    d_model: int = 64
    lr_policy: float = 3e-4
    lr_eval: float = 1e-3
    eval_epochs: int = 3
    grpo_epochs: int = 2
    buffer_size: int = 1500  # 최근 데이터만: 정책이 좋아지면 옛 사망 기록은 틀린 라벨이 된다
    val_every: int = 10  # 런 N 판마다 1 판은 평가기 검증용으로만 쓴다
    val_size: int = 400
    warmup_snapshots: int = 200  # 이만큼 모이기 전에는 GRPO 점수로 HP 비율 사용
    chooser: str = "rules"  # 전투·맵 밖 선택: "rules" 규칙 정책, "ev" 기대값 선택기 (평가기 사용)
    chooser_samples: int = 8  # 기대값 선택기가 선택지마다 굴리는 복사본 수
    policy_sees_map: bool = True
    # PyTorch 스레드 수. 배치가 작은 계산(롤아웃 한 스텝, 선택기)이 대부분이라 1 이 보통 더 빠르다.
    # 다른 작업과 CPU 를 나눠 쓸 때 4 스레드는 한 번 계산이 50배 넘게 느려진 적이 있다.
    torch_threads: int = 1  # 전투 정책에 RUN 토큰(덱·유물·상태·보스·맵 요약)을 넣을지. 끄면 전투 토큰만 본다
    seed: int = 0
    out_dir: str = "runs/latest"


class Agent:
    """공유 임베딩 + 평가 에이전트 + 전투 에이전트."""

    def __init__(self, vocab, d: int):
        self.vocab = vocab
        self.emb = SharedEmbeddings(vocab, d)
        self.evaluator = Evaluator(self.emb)
        self.policy = CombatPolicy(self.emb)
        # 기대값 선택기가 쓸 평가기. None 이면 self.evaluator. 전투 정책은 self.evaluator 의 run_vec 으로
        # 학습됐으므로, 선택기 쪽만 다른 평가기로 바꿔 실험할 때 쓴다.
        self.chooser_evaluator: Evaluator | None = None

    @torch.no_grad()
    def run_vec(self, state) -> torch.Tensor:
        """전투 정책에 줄 런 맥락 [d]. 평가기 트렁크에서 계산하고 기울기는 끊는다.

        끊는 이유: 평가기는 생존 라벨로만 배우게 두고, 전투 정책 손실이 평가기 표현을 흔들지 않게.
        정책은 자기 쪽 run_proj 로 이 벡터를 어떻게 쓸지 배운다.
        """
        return self.evaluator.run_vector(collate_run([state], self.vocab))[0]

    def state_dict(self):
        return {"evaluator": self.evaluator.state_dict(), "policy": self.policy.state_dict()}

    def load_state_dict(self, sd):
        self.evaluator.load_state_dict(sd["evaluator"])
        self.policy.load_state_dict(sd["policy"])


@torch.no_grad()
def choose_node(agent: Agent, run: RunSim, rng: np.random.Generator, epsilon: float) -> int:
    state = run.run_state()
    choices = state.map.next_choices()
    if rng.random() < epsilon:
        return int(rng.choice(choices))
    batch = collate_run([state], agent.vocab)
    ev = agent.evaluator(batch)
    return max(choices, key=lambda c: float(ev.log_survive[0, c]))


def play_run(agent: Agent, run: RunSim, cfg: Config, rng: np.random.Generator, seed: int,
             use_evaluator: bool, collect_grpo: bool = True, greedy: bool = False):
    run.reset(seed)
    rec = RunRecorder(agent.vocab.n_acts)
    groups: list[list[Trajectory]] = []
    last_combat, n_combats = None, 0
    run_vec = None
    steps = 0
    while run.phase != PHASE_DONE and steps < 20000:
        steps += 1
        if run.phase == PHASE_COMBAT:
            cid = run.combat_id()
            if cid != last_combat:
                last_combat = cid
                n_combats += 1
                # 전투가 시작될 때 한 번: 이 전투가 어느 노드에서 벌어지고 다음에 뭐가 오는지 (맵 포함)
                run_vec = agent.run_vec(run.run_state()) if cfg.policy_sees_map else None
                if collect_grpo and n_combats % cfg.grpo_every_n_combats == 0:
                    groups.append(rollout_group(
                        run, agent.policy, agent.evaluator if use_evaluator else None,
                        cfg.group_size, rng, cfg.advantage_mode, run_vec=run_vec))
            o = run.combat_obs()
            rv = run_vec.unsqueeze(0) if run_vec is not None else None
            a, t, _ = agent.policy.act(collate_combat([o]), greedy=greedy, run_vec=rv)
            run.combat_step(int(a), int(o.enemy_slots[int(t)]) if int(t) >= 0 else -1)
        elif run.phase == PHASE_MAP:
            rec.snapshot(run.run_state())
            node = choose_node(agent, run, rng, 0.0 if greedy else cfg.map_epsilon)
            rec.visit(run.act, node)
            run.choose_node(node)
        else:
            if cfg.chooser == "ev" and use_evaluator:
                chooser = ExpectedValueChooser(agent.chooser_evaluator or agent.evaluator, agent.vocab,
                                               samples=cfg.chooser_samples)
                run.step_choice(chooser.choose(run, rng))
            else:
                run.default_step(rng)
    death_act = run.act if run.died else agent.vocab.n_acts - 1
    snaps = rec.finish(run.died, death_act, run.current_node)
    result = {"won": run.won, "died": run.died, "act": run.act, "floor": run.floor, "combats": n_combats}
    return snaps, groups, result


CHECKPOINT = "checkpoint.pt"
# 이어서 학습할 때 바뀌어도 되는 설정. 나머지가 다르면 다른 실험이므로 거부한다.
RESUMABLE_CHANGES = {"iterations", "out_dir", "torch_threads"}


def _atomic_save(obj, path: Path) -> None:
    """쓰는 도중 인스턴스가 꺼져도 이전 파일이 깨지지 않게 임시 파일에 쓰고 바꿔 끼운다."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _check_resume_config(saved: dict, cfg: Config) -> None:
    now = asdict(cfg)
    diff = {k: (saved.get(k), v) for k, v in now.items() if k not in RESUMABLE_CHANGES and saved.get(k) != v}
    if diff:
        lines = ", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in diff.items())
        raise ValueError(f"체크포인트와 설정이 다릅니다 ({lines}). 다른 --out-dir 를 쓰거나 설정을 맞추세요.")


def train(run: RunSim, cfg: Config, log=print, resume: bool = False) -> Agent:
    """resume=True 이고 out_dir 에 체크포인트가 있으면 마지막으로 끝난 반복 다음부터 이어서 학습한다.

    체크포인트에는 가중치뿐 아니라 옵티마이저, 평가기 버퍼, 난수 상태, 기록까지 들어 있어서
    중간에 끊겼다가 이어도 끊기지 않고 돈 것과 같은 결과가 나온다.
    """
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    rng = np.random.default_rng(cfg.seed)
    agent = Agent(run.vocab, cfg.d_model)
    opt_eval = torch.optim.AdamW(agent.evaluator.parameters(), lr=cfg.lr_eval, weight_decay=0.01)
    opt_pol = torch.optim.Adam(agent.policy.parameters(), lr=cfg.lr_policy)
    buffer: list[Snapshot] = []
    val_buffer: list[Snapshot] = []
    n_runs = 0
    history = []
    start = 0
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / CHECKPOINT
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        _check_resume_config(ck["config"], cfg)
        agent.load_state_dict(ck["agent"])
        opt_eval.load_state_dict(ck["opt_eval"])
        opt_pol.load_state_dict(ck["opt_pol"])
        buffer, val_buffer = ck["buffer"], ck["val_buffer"]
        n_runs, history, start = ck["n_runs"], ck["history"], ck["next_iter"]
        rng.bit_generator.state = ck["rng"]["numpy_generator"]
        np.random.set_state(ck["rng"]["numpy_global"])
        random.setstate(ck["rng"]["python"])
        torch.set_rng_state(ck["rng"]["torch"])
        if torch.cuda.is_available() and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        log(f"resume: {ckpt_path} 에서 반복 {start} 부터 이어서 학습합니다 (목표 {cfg.iterations})")
    elif resume:
        log(f"resume: {ckpt_path} 가 없어서 처음부터 학습합니다")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    for it in range(start, cfg.iterations):
        t0 = time.time()
        use_eval = len(buffer) >= cfg.warmup_snapshots
        results, groups = [], []
        for r in range(cfg.runs_per_iter):
            snaps, g, res = play_run(agent, run, cfg, rng, seed=int(rng.integers(2**31)), use_evaluator=use_eval)
            n_runs += 1
            (val_buffer if n_runs % cfg.val_every == 0 else buffer).extend(snaps)
            groups.extend(g)
            results.append(res)
        buffer = buffer[-cfg.buffer_size:]
        val_buffer = val_buffer[-cfg.val_size:]
        ev_stats = train_evaluator(agent.evaluator, opt_eval, buffer, agent.vocab, epochs=cfg.eval_epochs,
                                   val=val_buffer or None)
        pg_stats = grpo_update(agent.policy, opt_pol, groups, epochs=cfg.grpo_epochs)
        group_win = float(np.mean([t.won for g in groups for t in g])) if groups else 0.0
        row = {
            "iter": it,
            "win_rate": float(np.mean([r["won"] for r in results])),
            "mean_floor": float(np.mean([r["floor"] for r in results])),
            "mean_act": float(np.mean([r["act"] for r in results])) + 1,
            "fork_combat_win": group_win,
            "scorer": "evaluator" if use_eval else "hp-warmup",
            "buffer": len(buffer),
            "sec": round(time.time() - t0, 1),
            **ev_stats,
            **pg_stats,
        }
        history.append(row)
        log(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        _atomic_save(agent.state_dict(), out / "agent.pt")
        _atomic_write_text(json.dumps(history, indent=2), out / "history.json")
        _atomic_save({
            "next_iter": it + 1,
            "config": asdict(cfg),
            "agent": agent.state_dict(),
            "opt_eval": opt_eval.state_dict(),
            "opt_pol": opt_pol.state_dict(),
            "buffer": buffer,
            "val_buffer": val_buffer,
            "n_runs": n_runs,
            "history": history,
            "rng": {
                "numpy_generator": rng.bit_generator.state,
                "numpy_global": np.random.get_state(),
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }, ckpt_path)
    _atomic_write_text(json.dumps(history, indent=2), out / "history.json")
    return agent
