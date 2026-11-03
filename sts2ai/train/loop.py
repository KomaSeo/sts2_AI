"""전체 학습 루프.

반복마다:
  1) 런을 여러 판 플레이한다.
     - 전투가 시작되면: 그 전투를 G 번 분기해 굴리고(GRPO 그룹), 그다음 진짜 전투를 정책으로 플레이.
     - 맵·전투 밖 선택(learn_choices): 선택 정책이 고른다. choice_fork_prob 확률로 런을 복제해
       선택지끼리 비교한다(train/choice.py).
       learn_choices 를 끄면 예전처럼 맵은 평가기의 생존 확률 최대(ε 무작위), 나머지는 규칙 정책.
  2) 모은 스냅숏으로 평가 에이전트를 학습한다 (노드 위험도 + 미래 막 위험도).
  3) 모은 GRPO 그룹으로 전투 에이전트를, 선택 비교 샘플로 선택 에이전트를 학습한다.
     trunk_grad 면 두 정책의 손실이 평가기 트렁크(덱·유물·상태·맵 인코더)까지 흘러간다.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..agents import ChoicePolicy, CombatPolicy, Evaluator
from ..agents.chooser import ExpectedValueChooser
from ..batching import collate_run
from ..encoders import SharedEmbeddings
from ..envs.base import PHASE_COMBAT, PHASE_DONE, PHASE_MAP, RunSim
from ..schema import NODE_BOSS
from .choice import ChoiceSample, choice_update, pick, rollout_choice_group, screen_options
from .evaluator_train import RunRecorder, Snapshot, train_evaluator
from .grpo import LocalActor, Trajectory, grpo_update, rollout_group


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
    # 맵 노드와 전투 밖 선택(카드 보상·상점·휴식·이벤트·덱 카드 고르기)을 선택 정책이 고르고 학습할지.
    # 끄면 예전처럼 맵은 평가기의 생존 확률 최대(map_epsilon 탐험), 나머지는 chooser 로 고른다.
    learn_choices: bool = True
    choice_group_size: int = 4  # 선택 하나를 비교할 때 런을 몇 번 복제하나
    choice_fork_prob: float = 0.25  # 선택 화면마다 이 확률로 복제 비교를 한다 (비용 조절)
    choice_horizon: int = 2  # 복제를 맵 화면이 몇 번 더 나올 때까지 이어서 플레이할지 (≈ 다음 방 수)
    lr_choice: float = 3e-4
    # 전투·선택 정책의 손실을 평가기 트렁크(덱·유물·상태·맵 인코더)까지 흘려보낼지.
    # 끄면 정책은 run_vec 을 기울기 없이 받는다 (평가기는 생존 라벨로만 학습).
    trunk_grad: bool = True
    chooser: str = "rules"  # learn_choices=False 일 때 전투 밖 선택: "rules" 규칙 정책, "ev" 기대값 선택기
    chooser_samples: int = 8  # 기대값 선택기가 선택지마다 굴리는 복사본 수
    policy_sees_map: bool = True  # 전투 정책에 RUN 토큰(덱·유물·상태·보스·맵 요약)을 넣을지. 끄면 전투 토큰만 본다
    # 메인 프로세스의 PyTorch 스레드 수. -1 = 자동: 수집을 메인에서 하면(workers=0) 1,
    # 워커에 맡기면 메인은 학습만 하므로 PyTorch 기본값(물리 코어 수). 0 = PyTorch 기본값.
    # 배치가 작은 계산(롤아웃 한 스텝, 선택기)은 1 스레드가 빠르다. 다른 작업과 CPU 를 나눠 쓸 때
    # 4 스레드는 한 번 계산이 50배 넘게 느려진 적이 있다. 워커는 항상 1 스레드.
    torch_threads: int = -1
    # 런 수집 워커 프로세스 수. -1 = 자동(논리 코어 절반과 runs_per_iter 중 작은 값), 0 = 메인 프로세스에서.
    workers: int = -1
    # 학습(평가기·GRPO 업데이트)을 할 장치. "auto" = CUDA 가 있으면 cuda, 없으면 cpu.
    # 수집(배치 1 짜리 롤아웃)은 항상 CPU 에서 한다.
    device: str = "auto"
    # 수집 워커들의 전투 정책 호출을 학습 장치(device)의 추론 서버 프로세스 하나가 모아서 배치로 계산할지.
    # 워커가 있을 때만 쓴다. 평가기(run_vec, 채점)와 선택 정책은 계속 워커 CPU 에서 계산한다.
    infer_server: bool = False
    # 무작위 덱 모의전 (train/probe.py): 맵 화면마다 이 확률로, 지금 덱을 바꾼 덱으로 이번 막의 위험한 적과 싸워 본다.
    # 0 = 끔. 시뮬레이터에 덱을 지정해 전투를 시작하는 기능(start_probe)이 있는 환경에서만 돈다.
    probe_prob: float = 0.0
    probe_group_size: int = 8  # 모의전 하나를 몇 번 분기해 싸우나 (GRPO 그룹)
    probe_add_max: int = 5  # 직업·무색 카드 풀에서 넣는 장수 0~이 값
    probe_remove_max: int = 5  # 지금 덱에서 빼는 장수 0~이 값
    probe_hp_min: float = 0.5  # 모의전 HP = 최대 HP × U(이 값, 1)
    probe_eval_frac: float = 0.3  # 평가기 학습 데이터 중 보스 모의전 스냅숏 비율
    probe_buffer_size: int = 600
    probe_decay: float = 0.8  # 적 고르기 가중치(사망 횟수)를 반복마다 이만큼 줄여 최근 사망을 더 본다
    seed: int = 0
    out_dir: str = "runs/latest"
    # 이전 학습의 agent.pt 에서 이어서 시작. 가중치만 불러오고 옵티마이저 상태·평가기 버퍼는 새로 시작한다.
    init_ckpt: str = ""
    # 막 시작 상태 풀: 런이 다음 막의 첫 맵 화면에 도착하면 그 상태를 저장하고, 반복마다 일부 런을
    # 1막이 아니라 저장된 막 시작 상태에서(숨은 난수는 새로 뽑아) 시작한다. 뒤쪽 막도 학습 데이터를 받게.
    act_pool_frac: float = 0.5  # 반복당 런 중 풀에서 시작하는 비율 (풀이 비어 있으면 0). 0 = 끔
    act_pool_size: int = 200  # 막마다 보관하는 시작 상태 수 (오래된 것부터 버림)
    # 이전 학습이 저장한 막 시작 상태 풀(<out_dir>/act_pool.pkl)을 이어서 쓴다. 풀은 반복마다 저장된다.
    init_pool: str = ""


class Agent:
    """공유 임베딩 + 평가 에이전트 + 전투 에이전트 + 선택 에이전트."""

    def __init__(self, vocab, d: int):
        self.vocab = vocab
        self.emb = SharedEmbeddings(vocab, d)
        self.evaluator = Evaluator(self.emb)
        self.policy = CombatPolicy(self.emb)
        self.choice = ChoicePolicy(self.emb)
        # 기대값 선택기가 쓸 평가기. None 이면 self.evaluator. 전투 정책은 self.evaluator 의 run_vec 으로
        # 학습됐으므로, 선택기 쪽만 다른 평가기로 바꿔 실험할 때 쓴다.
        self.chooser_evaluator: Evaluator | None = None
        # 전투 정책 호출을 다른 프로세스(GPU 추론 서버)에 맡길 때 넣는 actor. None 이면 self.policy 를 바로 쓴다.
        self.remote_actor = None

    def combat_actor(self):
        """actor(관측 목록, run_vec [B,d] 또는 None, greedy) → (행동, 타깃, log 확률)."""
        return self.remote_actor if self.remote_actor is not None else LocalActor(self.policy)

    @torch.no_grad()
    def run_vec(self, state) -> torch.Tensor:
        """전투 정책에 줄 런 맥락 [d]. 평가기 트렁크에서 계산하고 기울기는 끊는다.

        끊는 이유: 평가기는 생존 라벨로만 배우게 두고, 전투 정책 손실이 평가기 표현을 흔들지 않게.
        정책은 자기 쪽 run_proj 로 이 벡터를 어떻게 쓸지 배운다.
        """
        return self.evaluator.run_vector(collate_run([state], self.vocab))[0]

    def state_dict(self):
        return {"evaluator": self.evaluator.state_dict(), "policy": self.policy.state_dict(),
                "choice": self.choice.state_dict()}

    def to(self, device) -> "Agent":
        self.evaluator.to(device)
        self.policy.to(device)  # 공유 임베딩은 세 모듈에 같이 들어 있다
        self.choice.to(device)
        return self

    def run_vec_with_grad(self, states) -> torch.Tensor:
        """trunk_grad 학습용: 기울기가 평가기 트렁크까지 흐르는 run_vec [B,d]."""
        from ..batching import module_device, to_device  # noqa: PLC0415

        batch = to_device(collate_run(states, self.vocab), module_device(self.evaluator))
        return self.evaluator.run_vector(batch)

    def cpu_state_dict(self):
        """장치와 상관없이 CPU 텐서로 된 가중치 (저장·워커 전송용)."""
        return {k: {n: t.detach().cpu() for n, t in sd.items()} for k, sd in self.state_dict().items()}

    def load_state_dict(self, sd):
        """예전 구조의 가중치도 받는다: 없어진 모듈(미래 막 헤드)은 버리고, 커진 임베딩(노드 종류 9 → 11)은
        있던 줄만 복사하고 새 줄은 초기값을 둔다."""
        for name, module in (("evaluator", self.evaluator), ("policy", self.policy), ("choice", self.choice)):
            if name in sd:  # 선택 정책이 생기기 전에 저장한 가중치에는 "choice" 가 없다
                module.load_state_dict(_adapt_state(module.state_dict(), sd[name]))


def _adapt_state(current: dict, loaded: dict) -> dict:
    out = {}
    for k, cur in current.items():
        old = loaded.get(k)
        if old is None:
            out[k] = cur  # 새로 생긴 파라미터: 초기값
        elif old.shape == cur.shape:
            out[k] = old
        elif old.dim() == cur.dim() and old.dim() >= 1 and old.shape[1:] == cur.shape[1:] and old.shape[0] < cur.shape[0]:
            t = cur.clone()
            t[: old.shape[0]] = old  # 임베딩이 커진 경우: 있던 줄만 복사
            out[k] = t
        else:
            raise ValueError(f"가중치 모양이 맞지 않는다: {k} {tuple(old.shape)} → {tuple(cur.shape)}")
    return out


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
             use_evaluator: bool, collect_grpo: bool = True, greedy: bool = False, start: dict | None = None,
             probe_weights: dict | None = None):
    """start 가 있으면 그 막 시작 상태에서(숨은 난수는 seed 로 새로 뽑아) 시작한다.

    probe_weights: 무작위 덱 모의전에서 적을 고를 가중치 {막: {(encounter, 보스인가): 가중치}} (train/probe.py).
    """
    from .probe import run_probe  # noqa: PLC0415

    do_probe = collect_grpo and cfg.probe_prob > 0 and hasattr(run, "start_probe")
    probe_snaps: list[Snapshot] = []
    probe_info: list[dict] = []
    last_enc, last_combat_floor = None, None
    if start is None:
        run.reset(seed)
    else:
        run.restore(start, seed)
    start_act, start_floor = run.act, run.floor
    seen_act = run.act
    act_starts: list[tuple[int, dict]] = []
    rec = RunRecorder(agent.vocab.n_acts)
    groups: list[list[Trajectory]] = []
    choice_samples: list[ChoiceSample] = []
    n_choice_groups = 0
    last_combat, n_combats = None, 0
    run_vec = None
    steps = 0
    while run.phase != PHASE_DONE and steps < 20000:
        steps += 1
        if run.phase == PHASE_MAP and run.act > seen_act:
            # 다음 막의 첫 맵 화면: 막 시작 상태로 저장
            seen_act = run.act
            if collect_grpo and cfg.act_pool_frac > 0:
                act_starts.append((run.act, run.snapshot()))
        if run.phase == PHASE_COMBAT:
            cid = run.combat_id()
            if cid != last_combat:
                last_combat = cid
                n_combats += 1
                if hasattr(run, "encounter_raw"):
                    is_boss = (run.current_node is not None and run.run_state().map.node_types[run.current_node]
                               == NODE_BOSS)
                    last_enc, last_combat_floor = (run.act, run.encounter_raw(), bool(is_boss)), run.floor
                # 전투가 시작될 때 한 번: 이 전투가 어느 노드에서 벌어지고 다음에 뭐가 오는지 (맵 포함)
                run_vec = agent.run_vec(run.run_state()) if cfg.policy_sees_map else None
                if collect_grpo and n_combats % cfg.grpo_every_n_combats == 0:
                    groups.append(rollout_group(
                        run, agent.combat_actor(), agent.evaluator if use_evaluator else None,
                        cfg.group_size, rng, cfg.advantage_mode, run_vec=run_vec))
            o = run.combat_obs()
            rv = run_vec.unsqueeze(0) if run_vec is not None else None
            a, t, _ = agent.combat_actor()([o], rv, greedy)
            run.combat_step(int(a[0]), int(o.enemy_slots[int(t[0])]) if int(t[0]) >= 0 else -1)
        elif cfg.learn_choices:
            # 맵 노드와 전투 밖 선택 모두 선택 정책이 고른다
            state = run.run_state()
            is_map = run.phase == PHASE_MAP
            if is_map:
                rec.snapshot(state)
                if do_probe and rng.random() < cfg.probe_prob:
                    out = run_probe(agent, run, cfg, rng, state, probe_weights or {}, n_combats)
                    if out is not None:
                        trajs, snap, info = out
                        groups.append(trajs)
                        probe_info.append(info)
                        if snap is not None:
                            probe_snaps.append(snap)
            opts = screen_options(run, state)
            if not opts:  # 고를 게 없는 화면: 자동 진행
                run.default_step(rng)
                continue
            if collect_grpo and len(opts) > 1 and rng.random() < cfg.choice_fork_prob:
                choice_samples.extend(rollout_choice_group(run, agent, cfg, rng, state, opts, use_evaluator))
                n_choice_groups += 1
            i, _ = pick(agent, state, opts, greedy)
            if is_map:
                n = opts[i].node
                rec.visit(run.act, n, state.map.rows[n], state.map.node_types[n] == NODE_BOSS)
                run.choose_node(n)
            else:
                run.step_choice(opts[i].action)
        elif run.phase == PHASE_MAP:
            st = run.run_state()
            rec.snapshot(st)
            node = choose_node(agent, run, rng, 0.0 if greedy else cfg.map_epsilon)
            rec.visit(run.act, node, st.map.rows[node], st.map.node_types[node] == NODE_BOSS)
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
    result = {"won": run.won, "died": run.died, "act": run.act, "floor": run.floor, "combats": n_combats,
              "choice_groups": n_choice_groups, "start_act": start_act, "start_floor": start_floor,
              "act_starts": act_starts,  # 학습 루프가 꺼내 풀에 넣는다
              # 선택 정책 학습 데이터. 반환 모양(snaps, groups, result)을 바꾸지 않으려고 결과 안에 담는다.
              "choice_samples": choice_samples,
              # 전투에서 죽었으면 (막, encounter, 보스인가). 모의전 적 고르기 가중치에 쓴다.
              "death_encounter": last_enc if (run.died and last_enc is not None and run.floor == last_combat_floor)
              else None,
              "probe_snaps": probe_snaps, "probe_info": probe_info}
    return snaps, groups, result


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(run: RunSim | None, cfg: Config, log=print, make_run=None) -> Agent:
    """run: 메인 프로세스에서 수집할 때 쓰는 환경. make_run: 워커가 환경을 만드는 함수 (pickle 가능).

    make_run 없이 run 만 주면 워커를 쓸 수 없으므로 메인 프로세스에서 수집한다.
    """
    from .collect import RunCollector, default_workers

    workers = cfg.workers if cfg.workers >= 0 else default_workers(cfg.runs_per_iter)
    if workers > 0 and make_run is None:
        log("make_run 이 없어서 워커 없이 메인 프로세스에서 수집합니다")
        workers = 0
    threads = cfg.torch_threads if cfg.torch_threads >= 0 else (1 if workers == 0 else 0)
    if threads > 0:
        torch.set_num_threads(threads)
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if run is None and workers == 0:
        run = make_run()
    vocab = run.vocab if run is not None else make_run().vocab
    agent = Agent(vocab, cfg.d_model).to(device)
    if cfg.init_ckpt:
        agent.load_state_dict(torch.load(cfg.init_ckpt, map_location="cpu"))
        log(f"init_ckpt={cfg.init_ckpt}")
    # 수집용 CPU 복사본. 학습을 CPU 에서 하고 수집도 메인에서 하면 따로 둘 필요가 없다.
    actor = agent if (device.type == "cpu" and workers == 0) else Agent(vocab, cfg.d_model)
    if actor is not agent:
        actor.load_state_dict(agent.cpu_state_dict())
    log(f"device={device} workers={workers} torch_threads={torch.get_num_threads()}")
    opt_eval = torch.optim.AdamW(agent.evaluator.parameters(), lr=cfg.lr_eval, weight_decay=0.01)

    def params_with_trunk(module):
        """정책 파라미터 + (trunk_grad 면) 평가기 트렁크 파라미터. 공유 임베딩은 한 번만."""
        seen, out = set(), []
        mods = [module] + ([agent.evaluator.encoder] if cfg.trunk_grad else [])
        for m in mods:
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    pol_params = params_with_trunk(agent.policy)
    choice_params = params_with_trunk(agent.choice)
    opt_pol = torch.optim.Adam(pol_params, lr=cfg.lr_policy)
    opt_choice = torch.optim.Adam(choice_params, lr=cfg.lr_choice)
    buffer: list[Snapshot] = []
    val_buffer: list[Snapshot] = []
    probe_buffer: list[Snapshot] = []  # 보스 모의전 스냅숏 (보스 노드에만 라벨)
    probe_weights: dict[int, dict[tuple[int, bool], float]] = {}  # 막 -> (encounter, 보스인가) -> 최근 사망 횟수
    n_runs = 0
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    history = []
    pools: dict[int, list[dict]] = {}  # 막 번호 -> 막 시작 상태 토큰들
    if cfg.init_pool:
        with open(cfg.init_pool, "rb") as f:
            pools = pickle.load(f)
        log("init_pool=" + "/".join(f"{a + 1}막:{len(p)}" for a, p in sorted(pools.items())))
    with RunCollector(cfg, workers, run=run, make_run=make_run, vocab=vocab, server_device=str(device)) as collector:
        for it in range(cfg.iterations):
            t0 = time.time()
            use_eval = len(buffer) >= cfg.warmup_snapshots
            seeds = [int(rng.integers(2**31)) for _ in range(cfg.runs_per_iter)]
            starts: list[dict | None] = [None] * len(seeds)
            acts = sorted(a for a, p in pools.items() if p)
            if acts and cfg.act_pool_frac > 0:
                n_pool = int(round(cfg.act_pool_frac * len(seeds)))
                # 풀 런을 앞에 둔다: 다시 재현하는 데 시간이 더 들어서, 늦게 시작하면 반복 끝에 워커가 논다
                for i in range(n_pool):
                    pool = pools[acts[int(rng.integers(len(acts)))]]  # 막을 고르게 고른 뒤 그 막의 상태 하나
                    starts[i] = pool[int(rng.integers(len(pool)))]
            results, groups, choice_samples, probe_info = [], [], [], []
            for snaps, g, res in collector.collect(actor, seeds, use_evaluator=use_eval, starts=starts,
                                                   probe_weights=probe_weights):
                n_runs += 1
                (val_buffer if n_runs % cfg.val_every == 0 else buffer).extend(snaps)
                groups.extend(g)
                choice_samples.extend(res.pop("choice_samples", []))
                probe_buffer.extend(res.pop("probe_snaps", []))
                probe_info.extend(res.pop("probe_info", []))
                death = res.pop("death_encounter", None)
                if death is not None:
                    a, enc, is_boss = death
                    tbl = probe_weights.setdefault(a, {})
                    tbl[(enc, is_boss)] = tbl.get((enc, is_boss), 0.0) + 1.0
                for a, tok in res.pop("act_starts", []):
                    pools.setdefault(a, []).append(tok)
                    pools[a] = pools[a][-cfg.act_pool_size:]
                results.append(res)
            fresh = [r for r in results if r["start_act"] == 0] or results
            pooled = [r for r in results if r["start_act"] > 0]
            t_collect = time.time() - t0
            srv = collector.server_stats()
            srv_stats = ({"infer_calls": srv["calls"], "infer_batch": srv["rows"] / max(1, srv["calls"]),
                          "infer_busy_sec": srv["busy"]} if srv else {})
            buffer = buffer[-cfg.buffer_size:]
            val_buffer = val_buffer[-cfg.val_size:]
            probe_buffer = probe_buffer[-cfg.probe_buffer_size:]
            for tbl in probe_weights.values():  # 오래된 사망은 덜 본다
                for k in list(tbl):
                    tbl[k] *= cfg.probe_decay
            # 평가기 데이터에 보스 모의전 스냅숏을 probe_eval_frac 비율만큼 섞는다 (반복마다 새로 뽑음)
            mixed = buffer
            if probe_buffer and cfg.probe_eval_frac > 0:
                want = int(round(cfg.probe_eval_frac / max(1e-6, 1 - cfg.probe_eval_frac) * len(buffer)))
                k = min(len(probe_buffer), max(1, want))
                pick_i = rng.choice(len(probe_buffer), k, replace=False)
                mixed = buffer + [probe_buffer[int(i)] for i in pick_i]
            ev_stats = train_evaluator(agent.evaluator, opt_eval, mixed, agent.vocab, epochs=cfg.eval_epochs,
                                       val=val_buffer or None)
            pg_stats = grpo_update(agent.policy, opt_pol, groups, epochs=cfg.grpo_epochs,
                                   run_vec_fn=agent.run_vec_with_grad if cfg.trunk_grad else None,
                                   params=pol_params)
            ch_stats = (choice_update(agent, opt_choice, choice_samples, epochs=cfg.grpo_epochs,
                                      trunk_grad=cfg.trunk_grad, params=choice_params)
                        if cfg.learn_choices else {})
            if actor is not agent:
                actor.load_state_dict(agent.cpu_state_dict())
            collector.sync(actor)
            group_win = float(np.mean([t.won for g in groups for t in g])) if groups else 0.0
            row = {
                "iter": it,
                # 1막부터 시작한 런만 (예전 기록과 비교 가능하게). 풀에서 시작한 런은 아래 pool_* 로 따로.
                "win_rate": float(np.mean([r["won"] for r in fresh])),
                "mean_floor": float(np.mean([r["floor"] for r in fresh])),
                "mean_act": float(np.mean([r["act"] for r in fresh])) + 1,
                "pool_runs": len(pooled),
                "pool_mean_floor": float(np.mean([r["floor"] for r in pooled])) if pooled else 0.0,
                "pool_win_rate": float(np.mean([r["won"] for r in pooled])) if pooled else 0.0,
                "pool_sizes": "/".join(f"{a + 1}막:{len(p)}" for a, p in sorted(pools.items())) or "-",
                "fork_combat_win": group_win,
                "scorer": "evaluator" if use_eval else "hp-warmup",
                "buffer": len(buffer),
                "sec": round(time.time() - t0, 1),
                "collect_sec": round(t_collect, 1),
                "choice_groups": int(sum(r.get("choice_groups", 0) for r in results)),
                **srv_stats,
                **({"probes": len(probe_info),
                    "probe_boss": sum(p["boss"] for p in probe_info),
                    "probe_win": float(np.mean([p["win_frac"] for p in probe_info])),
                    "probe_boss_win": float(np.mean([p["win_frac"] for p in probe_info if p["boss"]] or [0.0])),
                    "probe_buffer": len(probe_buffer)} if probe_info else {}),
                **ev_stats,
                **pg_stats,
                **ch_stats,
            }
            history.append(row)
            log(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            torch.save(agent.cpu_state_dict(), out / "agent.pt")
            if pools:
                with open(out / "act_pool.pkl", "wb") as f:
                    pickle.dump(pools, f)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return agent
