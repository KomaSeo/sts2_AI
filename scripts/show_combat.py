"""학습한 전투 정책이 전투 하나를 치르는 과정을 글로 보여 준다 (zhiyue StS2 전투).

    python scripts/show_combat.py --zhiyue-path ../sts2-rl-agent --ckpt runs/combat_sts2/combat_policy.pt \
        --encounter byrdonis --deck-size 20 --hp 70 --max-hp 80 --potions 3

턴마다 상태(HP, 방어도, 에너지, 적 의도, 손패)를 찍고, 정책이 고른 행동과 그 행동의 확률을 찍는다.
덱·포션은 --seed 로 무작위로 만든다 (전투 학습과 같은 방식). 기본은 greedy, --sample 이면 확률대로.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sts2ai.agents import CombatPolicy  # noqa: E402
from sts2ai.batching import collate_combat  # noqa: E402
from sts2ai.encoders import SharedEmbeddings  # noqa: E402
from sts2ai.train.combat_train import CombatConfig, sample_scenario  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zhiyue-path", default="../sts2-rl-agent")
    p.add_argument("--ckpt", required=True, help="train_combat.py 의 combat_policy.pt")
    p.add_argument("--encounter", default="", help="조우 이름 일부 (예: byrdonis, vantom). 비우면 seed 로 고름")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deck-size", type=int, default=0, help="0 이면 학습과 같은 범위에서 무작위")
    p.add_argument("--hp", type=int, default=0)
    p.add_argument("--max-hp", type=int, default=0)
    p.add_argument("--potions", type=int, default=-1, help="-1 이면 무작위")
    p.add_argument("--relics", type=int, default=0, help="무작위 유물 개수")
    p.add_argument("--sample", action="store_true", help="greedy 대신 정책 확률대로 뽑기")
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()

    from sts2ai.envs.zhiyue_adapter import ZhiyueCombatEnv

    env = ZhiyueCombatEnv(args.zhiyue_path)
    cfg_path = Path(args.ckpt).with_name("combat_config.json")
    d_model = json.loads(cfg_path.read_text())["d_model"] if cfg_path.exists() else 64
    policy = CombatPolicy(SharedEmbeddings(env.vocab, d_model))
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    policy.eval()
    torch.manual_seed(args.seed)

    pool = env.combat_pool()
    rng = np.random.default_rng(args.seed)
    if args.encounter:
        cands = [(e, a) for e, a in pool if args.encounter.lower() in env.encounter_name(e).lower()]
        if not cands:
            raise SystemExit(f"'{args.encounter}' 조우가 없습니다. 예: " + ", ".join(env.encounter_name(e) for e, _ in pool[:8]))
        enc, act = cands[0]
    else:
        enc, act = pool[int(rng.integers(len(pool)))]
    cfg = CombatConfig()
    if args.deck_size:
        cfg.deck_min = cfg.deck_max = args.deck_size
    if args.potions >= 0:
        cfg.potions_min = cfg.potions_max = args.potions
    cfg.relics_min = cfg.relics_max = args.relics
    sc = sample_scenario(env, rng, cfg, enc, act)
    if args.max_hp:
        sc.max_hp = args.max_hp
    if args.hp:
        sc.hp = min(args.hp, sc.max_hp)
    sim = env.make_combat(sc.encounter, sc.act, sc.deck, sc.hp, sc.max_hp, sc.potions, sc.relics, sc.seed)

    names = sorted(env.s.cards_by_id[cid].name.lower() + ("+" if up else "") for cid, up in sc.deck)
    print(f"=== {env.encounter_name(enc)} | 덱 {len(sc.deck)}장 | HP {sc.hp}/{sc.max_hp} | 포션 {len(sc.potions)}개 | seed {sc.seed}")
    print("덱: " + ", ".join(names))
    if sc.relics:
        print("유물: " + ", ".join(sorted(env.relic_name(r) for r in sc.relics)))
    turn, steps = -1, 0
    while not sim.done and steps < args.max_steps:
        o = sim.observe()
        if sim.turn != turn:
            turn = sim.turn
            print(f"\n--- 턴 {turn} ---")
            print(sim.describe_state())
        batch = collate_combat([o])
        with torch.no_grad():
            out = policy(batch)
            probs = torch.softmax(out.action_logits[0], -1)
            a, t, _ = policy.act(batch, greedy=not args.sample)
        a, t = int(a), int(t)
        slot = int(o.enemy_slots[t]) if t >= 0 else -1
        chosen = sim.describe_action(a, slot)
        # 같은 카드가 손패에 여러 장이면 확률을 합쳐서 보여 준다
        by_name: dict[str, float] = {}
        for i in np.flatnonzero(o.legal):
            name = sim.describe_action(int(i)).split(" → ")[0]
            by_name[name] = by_name.get(name, 0.0) + float(probs[int(i)])
        p_chosen = by_name.get(chosen.split(" → ")[0], float(probs[a]))
        alts = [f"{n.split(': ', 1)[-1]} {q:.0%}" for n, q in sorted(by_name.items(), key=lambda kv: -kv[1])
                if n != chosen.split(" → ")[0] and q > 0.02][:2]
        before = sim.snapshot()
        sim.step(a, slot)
        change = sim.describe_change(before, sim.snapshot()) if not sim.done or sim.c is not None else ""
        print(f"  ▶ {chosen}  ({p_chosen:.0%})" + (f"  ⇒ {change}" if change else "")
              + (f"   [다른 후보: {', '.join(alts)}]" if alts else ""))
        steps += 1
    result = "승리" if sim.won else ("패배" if sim.done else f"{args.max_steps}스텝에서 중단")
    print(f"\n=== {result} | 남은 HP {sim.hp()}/{sim.max_hp()} | 남은 포션 {sum(1 for x in sim.potions() if x)}개 | {turn}턴, 행동 {steps}번")


if __name__ == "__main__":
    main()
