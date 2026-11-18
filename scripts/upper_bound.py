r"""검증 전투의 이길 수 있는 상한: 같은 시드로 N 번 싸워 한 번이라도 이기면 이길 수 있는 전투로 센다.

    .venv\Scripts\python scripts\upper_bound.py runs\combat_only_v7 --n 128 --workers 8

- 검증 전투는 학습(train/combat_only.py)과 똑같이 만든다 (같은 설정, 같은 난수).
- 정책은 <폴더>/combat_policy.pt (지금 가중치). 행동은 확률대로 뽑는다. 같은 시드라 셔플·적 행동은 같고
  행동만 다르다. 그래서 "나눠 줄 카드를 미리 아는" 상한이다 (실제 실력의 상한보다 높게 나온다).
- 처음 8·32·…·N 번 중 한 번이라도 이긴 비율(best-of-k), 확률대로 한 번 싸운 평균 승률, greedy 승률을 낸다.
- 결과: <폴더>/upper_bound.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2ai.agents import CombatPolicy  # noqa: E402
from sts2ai.encoders import SharedEmbeddings  # noqa: E402
from sts2ai.train import combat_only as co  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sts2-repo", default="../sts2")
    args = p.parse_args()
    folder = Path(args.folder)
    saved = json.loads((folder / "config.json").read_text())
    cfg = co.CombatConfig(**{f.name: saved[f.name] for f in fields(co.CombatConfig) if f.name in saved})

    from sts2ai.envs.sts2_adapter import Sts2Run  # noqa: PLC0415

    run = Sts2Run(args.sts2_repo)
    run.reset(12345)
    pool = co.combat_pool(run)
    cards, pots, relic_pool = co.card_pool(run), co.potion_pool(run), run.relic_pool()
    starters = co.parse_starters(cfg.starter_cards, {i.name: c for c, i in run.lib.cards.items()})
    ancients = run.ancient_relic_pools() if getattr(cfg, "relic_scheme", "uniform") == "acts" else None
    eval_rng = np.random.default_rng(10**6 + cfg.seed)  # 학습의 검증 전투와 같은 난수
    eval_set = [co.sample_scenario(eval_rng, cfg, cards, pots, e, relic_pool, starters, ancients)
                for _ in range(cfg.eval_rounds) for e in pool]

    policy = co.make_policy(run.vocab, cfg)
    policy.load_state_dict(torch.load(folder / "combat_policy.pt", map_location="cpu"))
    policy.eval()

    rcfg = replace(cfg, group_size=args.n, scenarios_per_task=1, workers=args.workers)
    runner = co.Runner(args.sts2_repo, rcfg, run)
    t0 = time.time()
    try:
        runner.sync(policy)
        greedy = runner.play(policy, eval_set, True, np.random.default_rng(7))
        sampled = runner.play(policy, eval_set, False, np.random.default_rng(11))
    finally:
        runner.close()

    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= args.n]
    rows = []
    for sc, g, s in zip(eval_set, greedy, sampled):
        wins = [bool(t.won) for t in s]
        rows.append({"encounter": sc.encounter, "sim_act": sc.sim_act, "kind": sc.kind, "deck": len(sc.deck),
                     "hp": sc.hp, "max_hp": sc.max_hp, "potions": len(sc.potions), "relics": len(sc.relics),
                     "greedy_win": bool(g[0].won) if g else None, "n": len(wins),
                     "sample_win_rate": float(np.mean(wins)) if wins else None,
                     "best_of": {k: bool(any(wins[:k])) for k in ks}})

    def summary(sel):
        out = {"n": len(sel), "greedy": float(np.mean([r["greedy_win"] for r in sel])),
               "sampled_once": float(np.mean([r["sample_win_rate"] for r in sel]))}
        for k in ks:
            out[f"best_of_{k}"] = float(np.mean([r["best_of"][k] for r in sel]))
        return out

    result = {"policy": str(folder / "combat_policy.pt"), "n": args.n, "sec": round(time.time() - t0, 1),
              "all": summary(rows)}
    for kind in ("Weak", "Normal", "Elite", "Boss"):
        sel = [r for r in rows if r["kind"] == kind]
        if sel:
            result[kind] = summary(sel)
    result["scenarios"] = rows
    (folder / "upper_bound.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for key in ("all", "Weak", "Normal", "Elite", "Boss"):
        if key in result:
            s = result[key]
            print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    print("sec", result["sec"])


if __name__ == "__main__":
    main()
