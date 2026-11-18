"""전투 에이전트만 따로 학습한다 (런 없이, 무작위 덱으로 모든 적과 싸우는 GRPO).

목적: 전투 에이전트의 구조(인코더·정책)가 전투 자체를 배울 수 있는지, 런·평가기·덱 빌딩의 영향 없이 본다.
설계는 원격 브랜치 ccr-efec8c2b-ct4hfn 의 train/combat_train.py (토이·zhiyue 용)를 r33hab 시뮬레이터(Sts2Run)에
옮긴 것이다.

시나리오 하나 = (적 조우, 시뮬레이터 막) + 완전 무작위 덱 + 무작위 포션 + 무작위 최대 HP·HP + 시드.
- 덱: 직업(아이언클래드)·무색 카드 풀에서 deck_min~deck_max 장을 고르게 뽑고, 장마다 upgrade_prob 로 강화.
  시작 덱(타격·수비)도 따로 넣지 않는다. 저주·상태이상은 넣지 않는다. 유물은 없다.
- 상대: 반복마다 모든 (조우, 막) 을 한 바퀴(고루) + 같은 수를 '사망률' 에 비례해 더 뽑는다(집중).
- GRPO 그룹: 같은 시나리오·같은 시드로 G 번 싸운다(공통 난수). 시행 사이 차이는 정책의 선택에서만 나온다.
- 점수(전투 결과만, 평가기 없음): 이기면 1 + 남은 HP/최대 HP (1~2), 지면 적에게 준 피해 비율 (0~1).
  probe.probe_score 와 같다.
- 런 맥락(RUN 토큰)은 넣지 않는다 (run_vec = None).

병렬: 워커 프로세스마다 Sts2Run 하나와 CPU 정책 복사본. 워커는 시나리오 묶음을 받아 그 안의 모든 분기를 한 배치로
같이 진행한다(play_many). 학습(grpo_update)은 메인 프로세스(GPU 가능)에서 한다.
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..agents import CombatPolicy
from ..encoders import SharedEmbeddings
from .grpo import Trajectory, advantages, grpo_update, play_many
from .probe import probe_score


@dataclass
class CombatConfig:
    iterations: int = 200
    rounds_per_iter: int = 1  # 고루: 반복마다 (조우, 막) 전체를 몇 바퀴
    focus_per_iter: int = -1  # 집중: 사망률 비례로 더 뽑을 수. -1 = (조우, 막) 수만큼
    focus_floor: float = 0.05
    death_rate_ema: float = 0.3
    group_size: int = 8
    deck_min: int = 10
    deck_max: int = 40
    upgrade_prob: float = 0.2
    potions_max: int = 9  # 포션 0~이 값 개. 쓸 수 있는 칸을 포션 수만큼 늘린다 (시뮬레이터 칸 9)
    relics_min: int = 0
    relics_max: int = 40  # 유물 0~이 값 개, Sts2Run.relic_pool() 에서 겹치지 않게
    max_hp_min: int = 50
    max_hp_max: int = 100
    hp_frac_min: float = 0.3  # HP = 최대 HP × U(이 값, 1)
    max_steps: int = 300
    d_model: int = 128
    layers: int = 3
    lr_policy: float = 3e-4
    grpo_epochs: int = 2
    advantage_mode: str = "zscore"
    eval_every: int = 10  # 반복 N 번마다(그리고 시작 전) 고정 검증 시나리오를 greedy 로
    eval_rounds: int = 2  # 검증 시나리오 = (조우, 막) 전체 × 이 값
    workers: int = 12
    scenarios_per_task: int = 4  # 워커에 한 번에 보내는 시나리오 수 (그 안의 분기를 한 배치로 진행)
    device: str = "auto"
    snapshot_every: int = 10  # 반복 N 번마다 정책 가중치를 따로 남긴다 (policy_it0010.pt …)
    resume: bool = False  # out_dir 에 저장된 것이 있으면 이어서 학습
    seed: int = 0
    out_dir: str = "runs/combat_only"


@dataclass
class Scenario:
    encounter: int
    sim_act: int
    kind: str
    deck: list[tuple[int, bool]]
    hp: int
    max_hp: int
    potions: list[int]
    seed: int
    relics: list[int] = field(default_factory=list)


def combat_pool(run) -> list[tuple[int, int, str]]:
    """(encounter id, 시뮬레이터 막, 종류) 전부. 같은 적이 여러 목록에 있으면 처음 것만."""
    from .matchup import KINDS, load_pools  # noqa: PLC0415

    out, seen = [], set()
    for act, pools in sorted(load_pools(run.lib.repo).items()):
        for kind in KINDS:
            for e in pools.get(kind, []):
                if (e, act) not in seen:
                    seen.add((e, act))
                    out.append((int(e), int(act), kind))
    return out


def card_pool(run) -> list[int]:
    return [c for c in run.probe_card_pool() if run.card_upgradable(c)]


def potion_pool(run) -> list[int]:
    return sorted(run.lib.potion_index)


def sample_scenario(rng: np.random.Generator, cfg: CombatConfig, cards, pots, enc: tuple[int, int, str],
                    relic_pool=()) -> Scenario:
    n = int(rng.integers(cfg.deck_min, cfg.deck_max + 1))
    deck = [(int(c), bool(rng.random() < cfg.upgrade_prob)) for c in rng.choice(cards, n)]
    n_pot = int(rng.integers(0, cfg.potions_max + 1))
    potions = [int(p) for p in rng.choice(pots, n_pot)] if n_pot else []
    max_hp = int(rng.integers(cfg.max_hp_min, cfg.max_hp_max + 1))
    hp = max(1, int(round(max_hp * rng.uniform(cfg.hp_frac_min, 1.0))))
    seed = int(rng.integers(2**31))
    relics: list[int] = []
    if cfg.relics_max > 0 and len(relic_pool):
        n_rel = min(int(rng.integers(cfg.relics_min, cfg.relics_max + 1)), len(relic_pool))
        relics = sorted(int(r) for r in rng.choice(relic_pool, n_rel, replace=False)) if n_rel else []
    return Scenario(enc[0], enc[1], enc[2], deck, hp, max_hp, potions, seed, relics)


def start_combat(run, sc: Scenario):
    """시나리오의 전투를 시작한 분기 (Sts2RunCombatFork). 시작하지 못하면 None."""
    from ..envs.sts2_adapter import Sts2RunCombatFork  # noqa: PLC0415

    p = run.start_probe(sc.seed, sc.deck, sc.encounter, sc.hp, 0, relics=sc.relics, max_hp=sc.max_hp,
                        potions=sc.potions)
    return None if p is None else Sts2RunCombatFork(p)


def play_scenarios(policy, run, scs: list[Scenario], group_size: int, greedy: bool, max_steps: int):
    """시나리오마다 같은 시드로 group_size 번(greedy 면 1 번) 싸운다. 모든 분기를 한 배치로 진행한다.

    돌려주는 것: 시나리오마다 시행 목록 (시작하지 못한 시나리오는 빈 목록). 점수·이점까지 매긴다.
    """
    n = 1 if greedy else group_size
    sims, owner = [], []
    for k, sc in enumerate(scs):
        for _ in range(n):
            s = start_combat(run, sc)
            if s is None:
                break
            sims.append(s)
            owner.append(k)
    trajs = play_many(policy, sims, greedy=greedy, max_steps=max_steps) if sims else []
    groups: list[list[Trajectory]] = [[] for _ in scs]
    for t, k in zip(trajs, owner):
        groups[k].append(t)
    for g, sc in zip(groups, scs):
        if not g:
            continue
        scores = [probe_score(t, sc.max_hp) for t in g]
        for t, s, a in zip(g, scores, advantages(scores, "zscore") if len(g) > 1 else [0.0] * len(g)):
            t.score, t.advantage = s, a
            t.post_state = None
            t.run_state = None
    return groups


# ---- 전투 기록 (반복마다 이긴 전투 하나, 진 전투 하나) ----
@torch.no_grad()
def fight_record(policy, run, t: Trajectory, sc: Scenario, top: int = 3) -> dict:
    """행동마다 고른 행동의 확률(%)과 확률이 높은 행동 top 개(%)를 남긴다. policy 는 그 전투를 싸운 가중치."""
    from .. import schema as S  # noqa: PLC0415
    from ..batching import collate_combat, module_device, to_device  # noqa: PLC0415

    def name(o, i):
        return run.action_token_name(int(o.types[1 + i]), int(o.ids[1 + i]))

    enemy_names, intent_names = _name_tables(run)

    def hand(o):
        """손패: [이름(+ 강화), 지금 비용, 낼 수 있나]. 비용 칸은 비용/5."""
        out = []
        for ti in np.flatnonzero((o.types == S.TYPE_HAND) | (o.types == S.TYPE_HAND_LOCKED)):
            nm = run.action_token_name(S.TYPE_HAND, int(o.ids[ti])) + ("+" if o.feats[ti, S.F_UPGRADED] > 0.5 else "")
            playable = bool(o.types[ti] == S.TYPE_HAND and 1 <= ti <= S.MAX_ACTIONS and o.legal[ti - 1])
            out.append([nm, int(round(float(o.feats[ti, S.F_COST]) * 5)), playable])
        return out

    def enemies(o):
        """적마다 이름·HP·방어도·이번 턴 행동(의도)·다음 턴 행동. 관측 특징의 배율: HP·방어도 /50, 의도 수치 /30."""
        out = []
        for ti in o.enemy_tokens:
            slot = int(o.owners[ti])
            e = {"slot": slot - 1, "name": enemy_names.get(int(o.ids[ti]), f"적{int(o.ids[ti])}"),
                 "hp": int(round(float(o.feats[ti, S.F_ENEMY_HP]) * 50)),
                 "block": int(round(float(o.feats[ti, S.F_ENEMY_BLOCK]) * 50))}
            for key, ty in (("intent", S.TYPE_INTENT), ("next", S.TYPE_INTENT_NEXT)):
                k = np.flatnonzero((o.types == ty) & (o.owners == slot))
                if len(k):
                    j = int(k[0])
                    e[key] = [intent_names.get(int(o.ids[j]), "?"), int(round(float(o.feats[j, S.F_INTENT_MAG]) * 30))]
            if e["hp"] > 0:
                out.append(e)
        return out

    probs = []
    for k in range(0, len(t.obs), 64):
        out = policy(to_device(collate_combat(t.obs[k : k + 64]), module_device(policy)), None)
        probs.append(torch.softmax(out.action_logits, -1).cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, S.MAX_ACTIONS))
    steps, turn = [], 1
    for o, a, tg, lp, pr in zip(t.obs, t.actions, t.targets, t.logp, probs):
        g = o.feats[0]
        legal = np.flatnonzero(o.legal)
        alts = sorted(((float(pr[i]) * 100, name(o, i)) for i in legal), reverse=True)[:top]
        steps.append({
            "turn": turn,
            "hp": int(round(float(g[S.F_HP]) * sc.max_hp)),  # 전역 토큰의 HP 칸 = HP / 최대 HP
            "energy": int(round(float(g[S.F_ENERGY]) * 10)),  # 에너지 / 10
            "block": int(round(float(g[S.F_BLOCK]) * 50)),
            "enemies": enemies(o),
            "hand": hand(o),
            "action": name(o, a) + (f" → 적{int(o.enemy_slots[tg])}" if tg >= 0 else ""),
            "prob_pct": round(float(np.exp(lp)) * 100, 1),  # 고른 (행동, 타깃) 의 확률
            "action_prob_pct": round(float(pr[a]) * 100, 1),  # 행동만의 확률
            "top": [[n, round(p, 1)] for p, n in alts],
        })
        if int(o.types[1 + a]) == S.TYPE_END_TURN:
            turn += 1
    first = enemies(t.obs[0]) if t.obs else []
    return {"fmt": 2, "enemy_names": [e["name"] for e in first], "encounter": sc.encounter, "sim_act": sc.sim_act, "kind": sc.kind, "won": bool(t.won),
            "hp_start": sc.hp, "max_hp": sc.max_hp, "hp_end": int(t.hp), "score": round(float(t.score), 3),
            "deck": [run.lib.names.spaced(run.lib.cards[c].name) + ("+" if up else "") if c in run.lib.cards
                     else str(c) for c, up in sc.deck],
            "potions": [run.lib.names.spaced(run.lib.names.potions().get(p, str(p))) for p in sc.potions],
            "relics": [run.lib.names.spaced(run.lib.names.relics().get(r, str(r))) for r in sc.relics],
            "steps": steps}


_NAMES: dict = {}


def _name_tables(run) -> tuple[dict[int, str], dict[int, str]]:
    """적 임베딩 인덱스 → 적 이름, 의도 임베딩 인덱스 → 의도 종류 이름 (Attack, Defend, Buff, Debuff, Unknown)."""
    if not _NAMES:
        v, names = run.lib.ct.VOCAB, run.lib.names
        en, it = names.enemies(), names.intents()
        _NAMES["enemy"] = {i: names.spaced(en.get(d, str(d))) for d, i in v.enemy.items()}
        _NAMES["intent"] = {i: it.get(d, str(d)) for d, i in v.intent.items()}
    return _NAMES["enemy"], _NAMES["intent"]


def enemy_text(e: dict) -> str:
    def intent(x):
        if not x:
            return "?"
        kind, mag = x
        return f"{kind} {mag}" if mag else kind

    s = f"{e['name']} HP{e['hp']}" + (f" 방어{e['block']}" if e.get("block") else "") + f" [{intent(e.get('intent'))}]"
    return s + (f" 다음[{intent(e['next'])}]" if e.get("next") else "")


def fight_text(rec: dict) -> str:
    head = (f"{'승리' if rec['won'] else '패배'} · {', '.join(rec.get('enemy_names', [])) or rec['encounter']} "
            f"(적 {rec['encounter']}, 막{rec['sim_act']}, {rec['kind']}) · "
            f"HP {rec['hp_start']}/{rec['max_hp']} → {rec['hp_end']} · 점수 {rec['score']} · "
            f"덱 {len(rec['deck'])}장")
    lines = [head, "  덱: " + ", ".join(rec["deck"]), "  포션: " + (", ".join(rec["potions"]) or "없음"),
             "  유물: " + (", ".join(rec.get("relics", [])) or "없음")]
    for st in rec["steps"]:
        alts = " / ".join(f"{n} {p}%" for n, p in st["top"])
        lines.append(f"  턴{st['turn']:>2} HP{st['hp']:>3} 방어{st.get('block', 0)} 에너지{st['energy']} | "
                     f"{' / '.join(enemy_text(e) for e in st.get('enemies', []))} | "
                     f"손패 {', '.join(f'{n}({c})' for n, c, _ in st.get('hand', []))} | "
                     f"{st['action']} ({st['prob_pct']}%) | 상위: {alts}")
    return "\n".join(lines)


# ---- 워커 ----
_W: dict = {}


def _init_worker(repo: str, cfg: CombatConfig) -> None:
    from ..envs.sts2_adapter import Sts2Run  # noqa: PLC0415

    torch.set_num_threads(1)
    run = Sts2Run(repo)
    run.reset(12345)
    policy = CombatPolicy(SharedEmbeddings(run.vocab, cfg.d_model), layers=cfg.layers)
    policy.eval()
    _W.update(run=run, policy=policy, cfg=cfg, version=-1)


def _work(task):
    version, blob, scs, greedy, seed = task
    if _W["version"] != version:
        _W["policy"].load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu"))
        _W["version"] = version
    torch.manual_seed(seed)
    cfg = _W["cfg"]
    with torch.no_grad():
        return play_scenarios(_W["policy"], _W["run"], scs, cfg.group_size, greedy, cfg.max_steps)


def _blob(policy) -> bytes:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, buf)
    return buf.getvalue()


class Runner:
    def __init__(self, repo: str, cfg: CombatConfig, run):
        self.cfg, self.run, self.pool, self.version, self.blob = cfg, run, None, 0, b""
        if cfg.workers > 0:
            self.pool = mp.get_context("spawn").Pool(cfg.workers, initializer=_init_worker,
                                                     initargs=(repo, cfg))

    def sync(self, policy) -> None:
        self.version += 1
        self.blob = _blob(policy)

    def play(self, policy, scs: list[Scenario], greedy: bool, rng: np.random.Generator) -> list[list[Trajectory]]:
        k = self.cfg.scenarios_per_task
        chunks = [scs[i : i + k] for i in range(0, len(scs), k)]
        seeds = [int(rng.integers(2**31)) for _ in chunks]
        if self.pool is None:
            out = []
            for ch, s in zip(chunks, seeds):
                torch.manual_seed(s)
                with torch.no_grad():
                    out += play_scenarios(policy, self.run, ch, self.cfg.group_size, greedy, self.cfg.max_steps)
            return out
        res = self.pool.map(_work, [(self.version, self.blob, ch, greedy, s) for ch, s in zip(chunks, seeds)],
                            chunksize=1)
        return [g for r in res for g in r]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _save(obj, path: Path) -> None:
    """임시 파일에 쓴 뒤 바꿔 넣는다 (쓰는 도중 꺼져도 예전 파일이 남는다)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def summarize(groups, scs, prefix: str) -> dict:
    trajs = [t for g in groups for t in g]
    if not trajs:
        return {f"{prefix}n": 0}
    out = {f"{prefix}n": len(trajs), f"{prefix}win": float(np.mean([t.won for t in trajs])),
           f"{prefix}score": float(np.mean([t.score for t in trajs]))}
    for kind in ("Weak", "Normal", "Elite", "Boss"):
        w = [t.won for g, sc in zip(groups, scs) if sc.kind == kind for t in g]
        if w:
            out[f"{prefix}win_{kind}"] = float(np.mean(w))
    hp = [t.hp / sc.max_hp for g, sc in zip(groups, scs) for t in g if t.won]
    out[f"{prefix}hp_left_win"] = float(np.mean(hp)) if hp else float("nan")
    out[f"{prefix}steps"] = float(np.mean([len(t.actions) for t in trajs]))
    return out


def train_combat_only(repo: str, cfg: CombatConfig, log=print) -> CombatPolicy:
    from ..envs.sts2_adapter import Sts2Run  # noqa: PLC0415
    from .loop import resolve_device  # noqa: PLC0415

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg.seed)
    run = Sts2Run(repo)
    run.reset(12345)
    pool = combat_pool(run)
    cards, pots, relic_pool = card_pool(run), potion_pool(run), run.relic_pool()
    device = resolve_device(cfg.device)
    policy = CombatPolicy(SharedEmbeddings(run.vocab, cfg.d_model), layers=cfg.layers).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr_policy)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log(f"device={device} workers={cfg.workers} 적(조우,막)={len(pool)} 카드풀={len(cards)} 포션풀={len(pots)} "
        f"유물풀={len(relic_pool)} 파라미터={sum(p.numel() for p in policy.parameters()):,}")
    eval_rng = np.random.default_rng(10**6 + cfg.seed)
    eval_set = [sample_scenario(eval_rng, cfg, cards, pots, e, relic_pool) for _ in range(cfg.eval_rounds) for e in pool]
    death = np.full(len(pool), 0.5)
    history: list[dict] = []
    start = 0
    ck = out / "checkpoint.pt"
    if cfg.resume and (ck.exists() or (out / "combat_policy.pt").exists()):
        if ck.exists():
            c = torch.load(ck, map_location="cpu", weights_only=False)
            policy.load_state_dict(c["policy"])
            opt.load_state_dict(c["opt"])
            history, death, start = c["history"], np.asarray(c["death"]), c["next_iter"]
            rng.bit_generator.state = c["rng"]
        else:  # 체크포인트를 만들기 전 코드로 저장된 폴더: 가중치·옵티마이저·기록·사망률만 있다
            policy.load_state_dict(torch.load(out / "combat_policy.pt", map_location="cpu"))
            opt.load_state_dict(torch.load(out / "optim.pt", map_location="cpu"))
            history = json.loads((out / "history.json").read_text())
            dr = json.loads((out / "death_rate.json").read_text())
            death = np.array([dr[f"{e}@{a}:{k}"] for e, a, k in pool])
            start = max(r["iter"] for r in history) + 1
            rng = np.random.default_rng(cfg.seed + 7919 * start)
        log(f"이어서 학습: 반복 {start} 부터")
    (out / "fights").mkdir(exist_ok=True)
    runner = Runner(repo, cfg, run)
    cpu_policy = policy if device.type == "cpu" else CombatPolicy(SharedEmbeddings(run.vocab, cfg.d_model),
                                                                  layers=cfg.layers)

    def actor():
        if cpu_policy is not policy:
            cpu_policy.load_state_dict({k: v.cpu() for k, v in policy.state_dict().items()})
        return cpu_policy

    def evaluate(row):
        a = actor()
        a.eval()
        g = runner.play(a, eval_set, True, np.random.default_rng(7))
        a.train()
        row.update(summarize(g, eval_set, "eval_"))
        by = {}
        for gg, sc in zip(g, eval_set):
            by.setdefault(f"{sc.encounter}@{sc.sim_act}:{sc.kind}", []).extend(t.won for t in gg)
        row["eval_by_encounter"] = {k: float(np.mean(v)) for k, v in by.items()}

    try:
        runner.sync(policy)
        if start == 0:
            row0 = {"iter": -1}
            t0 = time.time()
            evaluate(row0)
            row0["sec"] = round(time.time() - t0, 1)
            history.append(row0)
            log("시작 전 " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row0.items()
                                   if k != "eval_by_encounter"))
        for it in range(start, cfg.iterations):
            t0 = time.time()
            idx = [int(i) for _ in range(cfg.rounds_per_iter) for i in rng.permutation(len(pool))]
            nf = len(pool) if cfg.focus_per_iter < 0 else cfg.focus_per_iter
            if nf:
                w = np.maximum(death, cfg.focus_floor)
                idx += [int(i) for i in rng.choice(len(pool), nf, p=w / w.sum())]
            scs = [sample_scenario(rng, cfg, cards, pots, pool[i], relic_pool) for i in idx]
            a = actor()
            groups = runner.play(a, scs, False, rng)
            t_play = time.time() - t0
            lost: dict[int, list[float]] = {}
            for i, g in zip(idx, groups):
                lost.setdefault(i, []).extend(0.0 if t.won else 1.0 for t in g)
            for i, v in lost.items():
                death[i] = (1 - cfg.death_rate_ema) * death[i] + cfg.death_rate_ema * float(np.mean(v))
            row = {"iter": it, **summarize(groups, scs, ""),
                   "gap": float(np.mean([max(t.score for t in g) - np.mean([t.score for t in g])
                                         for g in groups if g])),
                   "failed": sum(1 for g in groups if not g)}
            # 이긴 전투 하나, 진 전투 하나. 확률은 업데이트 전 가중치(그 전투를 싸운 가중치)로 잰다.
            recs = []
            for want in (True, False):
                cand = [(t, sc) for g, sc in zip(groups, scs) for t in g if t.won == want]
                if cand:
                    t, sc = cand[int(rng.integers(len(cand)))]
                    recs.append(fight_record(a, run, t, sc))
            (out / "fights" / f"iter_{it:04d}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")
            (out / "fights" / f"iter_{it:04d}.txt").write_text("\n\n".join(fight_text(r) for r in recs),
                                                                encoding="utf-8")
            policy.train()
            row.update(grpo_update(policy, opt, [g for g in groups if len(g) > 1], epochs=cfg.grpo_epochs))
            runner.sync(policy)
            if cfg.eval_every and ((it + 1) % cfg.eval_every == 0 or it + 1 == cfg.iterations):
                evaluate(row)
            row["play_sec"] = round(t_play, 1)
            row["sec"] = round(time.time() - t0, 1)
            history.append(row)
            log(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                         if k != "eval_by_encounter"))
            sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
            _save(sd, out / "combat_policy.pt")
            _save(opt.state_dict(), out / "optim.pt")
            if cfg.snapshot_every and (it + 1) % cfg.snapshot_every == 0:
                _save(sd, out / f"policy_it{it + 1:04d}.pt")
            _save({"policy": sd, "opt": opt.state_dict(), "history": history, "death": death.tolist(),
                   "next_iter": it + 1, "rng": rng.bit_generator.state, "config": asdict(cfg)}, ck)
            (out / "history.json").write_text(json.dumps(history, indent=1))
            (out / "death_rate.json").write_text(json.dumps(
                {f"{e}@{a}:{k}": round(float(d), 3) for (e, a, k), d in zip(pool, death)}, indent=1))
    finally:
        runner.close()
    return policy
