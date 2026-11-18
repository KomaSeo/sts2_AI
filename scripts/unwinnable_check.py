r"""upper_bound.py 에서 한 번도 못 이긴 검증 전투를, 정책과 다르게 싸워서 정말 못 이기는지 본다.

    .venv\Scripts\python scripts\unwinnable_check.py runs\combat_only_v7 --tries 500 --workers 8

- 대상: <폴더>/upper_bound.json 에서 best_of 최대값이 False 인 전투 (검증 전투를 같은 난수로 다시 만든다).
- 싸우는 방법 (전투마다 같은 시드):
  - temp: 지금 정책의 행동·대상 점수를 온도 --temp 로 나눠 뽑는다 (더 고르게 = 정책이 안 하던 수도 해 본다).
  - random: 할 수 있는 행동 중 아무거나, 대상도 살아 있는 적 중 아무나.
- 결과: <폴더>/unwinnable_check.json
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.agents import CombatPolicy  # noqa: E402
from sts2ai.batching import collate_combat  # noqa: E402
from sts2ai.encoders import SharedEmbeddings  # noqa: E402
from sts2ai.train import combat_only as co  # noqa: E402

NEG = -1e9


class TempActor:
    """정책 점수를 온도로 나눠 뽑는다. play_many 가 부르는 actor(관측, run_vec, greedy) 규약."""

    def __init__(self, policy: CombatPolicy, temp: float):
        self.policy, self.temp = policy, temp

    @torch.no_grad()
    def __call__(self, obs, run_vec, greedy=False):
        batch = collate_combat(obs)
        out = self.policy(batch, None)
        dist = torch.distributions.Categorical(logits=out.action_logits / self.temp)
        a = dist.sample()
        lp = dist.log_prob(a)
        t = torch.full_like(a, -1)
        needs = batch["needs_target"].gather(1, a.unsqueeze(1)).squeeze(1) & batch["enemy_mask"].any(-1)
        if needs.any():
            tl = self.policy.target_logits(out, batch, a) / self.temp
            td = torch.distributions.Categorical(logits=tl)
            tt = td.sample()
            t = torch.where(needs, tt, t)
            lp = lp + torch.where(needs, td.log_prob(tt), torch.zeros_like(lp))
        return a, t, lp


class RandomActor:
    """할 수 있는 행동·살아 있는 대상 중 아무거나."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, run_vec, greedy=False):
        a, t = [], []
        for o in obs:
            ai = int(self.rng.choice(np.flatnonzero(o.legal)))
            alive = np.flatnonzero(o.enemy_alive) if len(o.enemy_alive) else np.arange(len(o.enemy_tokens))
            ti = int(self.rng.choice(alive)) if o.needs_target[ai] and len(alive) else -1
            a.append(ai)
            t.append(ti)
        return torch.tensor(a), torch.tensor(t), torch.zeros(len(a))


_W: dict = {}


def _init(repo, cfg, blob):
    from sts2ai.envs.sts2_adapter import Sts2Run  # noqa: PLC0415

    torch.set_num_threads(1)
    run = Sts2Run(repo)
    run.reset(12345)
    pol = co.make_policy(run.vocab, cfg)
    pol.load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu"))
    pol.eval()
    _W.update(run=run, policy=pol)


def _work(task):
    idx, sc, mode, n, temp, seed = task
    torch.manual_seed(seed)
    actor = TempActor(_W["policy"], temp) if mode == "temp" else RandomActor(seed)
    g = co.play_scenarios(actor, _W["run"], [sc], n, False, 300)[0]
    return idx, mode, sum(1 for t in g if t.won), len(g)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--tries", type=int, default=500, help="방법마다 전투 하나를 몇 번 싸우나")
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sts2-repo", default="../sts2")
    args = p.parse_args()
    folder = Path(args.folder)
    saved = json.loads((folder / "config.json").read_text())
    cfg = co.CombatConfig(**{f.name: saved[f.name] for f in fields(co.CombatConfig) if f.name in saved})
    ub = json.loads((folder / "upper_bound.json").read_text())

    from sts2ai.envs.sts2_adapter import Sts2Run  # noqa: PLC0415

    run = Sts2Run(args.sts2_repo)
    run.reset(12345)
    pool = co.combat_pool(run)
    P = co.scenario_pools(run, cfg)
    eval_rng = np.random.default_rng(10**6 + cfg.seed)  # 학습의 검증 전투와 같은 난수
    eval_set = [co.scenario_from(eval_rng, cfg, P, e) for _ in range(cfg.eval_rounds) for e in pool]
    rows = ub["scenarios"]
    assert len(rows) == len(eval_set)
    targets = [i for i, r in enumerate(rows) if not any(r["best_of"].values())]
    print("never-won fights:", len(targets))

    blob = io.BytesIO()
    torch.save(torch.load(folder / "combat_policy.pt", map_location="cpu"), blob)
    tasks, s = [], 0
    for i in targets:
        for mode in ("temp", "random"):
            for c in range(0, args.tries, args.chunk):
                s += 1
                tasks.append((i, eval_set[i], mode, min(args.chunk, args.tries - c), args.temp, 1000 + s))
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                      initargs=(args.sts2_repo, cfg, blob.getvalue())) as pl:
        res = pl.map(_work, tasks, chunksize=1)
    wins: dict = {}
    for i, mode, w, n in res:
        a = wins.setdefault(i, {}).setdefault(mode, [0, 0])
        a[0] += w
        a[1] += n
    out = []
    for i in targets:
        r = rows[i]
        out.append({"index": i, "kind": r["kind"], "encounter": r["encounter"], "sim_act": r["sim_act"],
                    "hp": r["hp"], "max_hp": r["max_hp"], "deck": r["deck"], "relics": r["relics"],
                    "potions": r["potions"], **{m: {"wins": v[0], "tries": v[1]} for m, v in wins[i].items()}})
        print(f"{r['kind']:6s} enc {r['encounter']:4d}@{r['sim_act']} HP {r['hp']}/{r['max_hp']} "
              f"deck {r['deck']} relics {r['relics']} | temp {wins[i]['temp'][0]}/{wins[i]['temp'][1]} "
              f"random {wins[i]['random'][0]}/{wins[i]['random'][1]}")
    ever = sum(1 for o in out if o["temp"]["wins"] + o["random"]["wins"] > 0)
    print(f"won at least once: {ever}/{len(out)}  sec {time.time() - t0:.0f}")
    (folder / "unwinnable_check.json").write_text(json.dumps({"temp": args.temp, "tries": args.tries, "fights": out},
                                                             indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
