"""전투 에이전트만 따로 학습한다 (런 없이, 무작위 덱으로 모든 적과 싸우는 GRPO).

목적: 전투 에이전트의 구조(인코더·정책)가 전투 자체를 배울 수 있는지, 런·평가기·덱 빌딩의 영향 없이 본다.
설계는 원격 브랜치 ccr-efec8c2b-ct4hfn 의 train/combat_train.py (토이·zhiyue 용)를 r33hab 시뮬레이터(Sts2Run)에
옮긴 것이다.

시나리오 하나 = (적 조우, 시뮬레이터 막) + 완전 무작위 덱 + 무작위 포션 + 무작위 최대 HP·HP + 시드.
- 덱: deck_min~deck_max 장. 먼저 시작 카드(starter_cards: 타격 0~5, 수비 0~5, 강타 0~1)를 뽑고, 나머지를
  직업(아이언클래드)·무색 카드 풀(시작 카드 제외)에서 고르게 채운다. 장마다 upgrade_prob 로 강화.
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
    # 먼저 넣을 시작 카드 "시뮬레이터 카드 이름:최소-최대". 나머지는 이 카드들을 뺀 풀에서 채운다
    starter_cards: str = "StrikeIronclad:0-5,DefendIronclad:0-5,Bash:0-1"
    # 덱 장수 방식. "uniform" = deck_min~deck_max 장 (시작 카드 포함). "acts" = 시작 카드 + 적의 막 × 0~extra_per_act 장
    #   (1 막 0~9, 2 막 0~18, 3 막 0~27) + 저주 curses_min~curses_max 장.
    deck_scheme: str = "uniform"
    extra_per_act: int = 9
    curses_min: int = 0
    curses_max: int = 6
    min_playable: int = 5  # "acts" 에서 저주 아닌 카드가 이보다 적으면 풀에서 채운다 (빈 덱 방지)
    # 특수 카드(어느 카드 목록에도 없는 이벤트·고대 등급 카드) 1 장을 넣을 확률 ("acts" 에서만). 이벤트로 얻는 카드를 흉내 낸다
    special_prob: float = 0.0
    # special_max > 0 이면 확률 대신 special_min~special_max 장을 고르게 (겹침 허용)
    special_min: int = 0
    special_max: int = 0
    # 고대의 존재 유물을 고르면 그 유물이 주는 카드를 덱에 넣고, 그 카드들은 무작위 풀에서 뺀다 (relic_cards.py)
    relic_cards: bool = False
    # 포션 개수: "최소-최대:확률" 구간을 확률로 고른 뒤 구간 안에서 고르게. 쓸 수 있는 칸을 포션 수만큼 늘린다 (시뮬레이터 칸 9)
    potion_bands: str = "0-2:0.7,3-6:0.2,7-9:0.1"
    relics_min: int = 0
    relics_max: int = 40  # relic_scheme="uniform": 유물 0~이 값 개, Sts2Run.relic_pool() 에서 겹치지 않게
    # 유물 고르는 방식. "uniform" = 위의 개수만큼 고르게. "acts" = 적이 나오는 막까지 런을 따라가듯:
    #   1 막: 니오우 유물 1 + 무작위 0~relics_per_act. 2 막: 그 위에 2 막 고대의 존재 유물 1 + 무작위 0~relics_per_act.
    #   3 막: 그 위에 3 막 고대의 존재 유물 1 + 무작위 0~relics_per_act. 무작위 유물은 니오우·고대의 존재 유물을 뺀 풀에서.
    relic_scheme: str = "uniform"
    relics_per_act: int = 12
    max_hp_min: int = 50
    max_hp_max: int = 100
    hp_frac_min: float = 0.3  # HP = 최대 HP × U(이 값, 1)
    max_steps: int = 300
    d_model: int = 128
    layers: int = 3
    heads: int = 4  # attention 머리 수 (d_model 을 나눠떨어지게)
    ff_mult: int = 2  # Transformer 피드포워드 폭 = ff_mult × d_model
    infer_server: bool = False  # 워커의 정책 호출을 GPU 추론 서버 하나에 모은다 (큰 모델용)
    server_max_batch: int = 256  # 추론 서버가 한 번에 모으는 관측 수 (클수록 GPU 메모리를 더 쓴다)
    amp: bool = False  # 학습 업데이트를 GPU 에서 bf16 으로 (큰 모델용, 약 2 배 빠르고 메모리 40% 적게)
    lr_policy: float = 3e-4
    grpo_epochs: int = 2
    # 엔트로피 하한: 행동이 2 개 이상인 스텝마다 H/log(행동 수) 가 이 값 아래로 가면 벌점 (0 = 끔)
    ent_floor: float = 0.0
    ent_floor_coef: float = 1.0
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


def special_pool(run) -> list[int]:
    """시뮬레이터 카드 목록(CardPools.g.cs) 어디에도 없는 이벤트·고대 등급 카드 (멀티 전용 제외).
    퀘스트 카드와 전투 중에 생기는 토큰 카드는 뺀다."""
    import re  # noqa: PLC0415

    text = (run.lib.repo / "src" / "Sts2Emulator" / "Generated" / "CardPools.g.cs").read_text(encoding="utf-8")
    listed = {int(x) for b in re.findall(r"\w+ =>\s*\[([^\]]*)\]", text) for x in re.split(r"[,\s]+", b)
              if x.strip()}
    return sorted(c for c, info in run.lib.cards.items()
                  if c not in listed and info.rarity in ("Event", "Ancient") and c not in run.lib.multiplayer_only
                  and info.card_type not in ("Curse", "Status"))


def curse_pool(run) -> list[int]:
    """저주 카드 (멀티 전용 제외). 상태이상은 넣지 않는다."""
    return sorted(c for c in run.probe_card_pool()
                  if (info := run.lib.cards.get(c)) is not None and info.card_type == "Curse")


def potion_pool(run) -> list[int]:
    return sorted(run.lib.potion_index)


def parse_bands(spec: str) -> list[tuple[int, int, float]]:
    """"0-2:0.7,3-6:0.2,7-9:0.1" → [(0, 2, 0.7), (3, 6, 0.2), (7, 9, 0.1)]."""
    out = []
    for part in spec.split(","):
        rng_s, w = part.split(":")
        lo, hi = (int(x) for x in rng_s.split("-"))
        out.append((lo, hi, float(w)))
    return out


def sample_potion_count(rng: np.random.Generator, spec: str) -> int:
    bands = parse_bands(spec)
    w = np.array([b[2] for b in bands])
    lo, hi, _ = bands[int(rng.choice(len(bands), p=w / w.sum()))]
    return int(rng.integers(lo, hi + 1))


def parse_starters(spec: str, name_to_id: dict[str, int]) -> list[tuple[int, int, int]]:
    """"StrikeIronclad:0-5,Bash:0-1" → [(카드 id, 최소, 최대)]."""
    out = []
    for part in filter(None, spec.split(",")):
        name, rng_s = part.split(":")
        lo, hi = (int(x) for x in rng_s.split("-"))
        out.append((int(name_to_id[name]), lo, hi))
    return out


SIM_ACT_TO_ACT = {1: 1, 2: 1, 3: 2, 4: 3}  # 시뮬레이터 막 번호(과성장·지하 선착장·군락·영광) → 1~3 막


def sample_act_relics(rng: np.random.Generator, act: int, relic_pool, ancients: dict, per_act: int) -> list[int]:
    """1 막: 니오우 1 + 무작위 0~per_act. 막이 오를 때마다 그 막 고대의 존재 1 + 무작위 0~per_act 를 더한다."""
    special = set(ancients["neow"]) | set(ancients.get(2, [])) | set(ancients.get(3, []))
    fill = [r for r in relic_pool if r not in special]
    chosen: list[int] = []
    for a in range(1, act + 1):
        src = ancients["neow"] if a == 1 else ancients.get(a, [])
        left = [r for r in src if r not in chosen]
        if left:
            chosen.append(int(rng.choice(left)))
        n = int(rng.integers(0, per_act + 1))
        left = [r for r in fill if r not in chosen]
        n = min(n, len(left))
        if n:
            chosen += [int(r) for r in rng.choice(left, n, replace=False)]
    return sorted(chosen)


def sample_scenario(rng: np.random.Generator, cfg: CombatConfig, cards, pots, enc: tuple[int, int, str],
                    relic_pool=(), starters: list[tuple[int, int, int]] = (), ancients: dict | None = None,
                    curses=(), specials=(), relic_cards=None) -> Scenario:
    """starters 를 주면 그 카드들을 먼저 넣고, 나머지는 cards 에서 starters 를 뺀 풀로 채운다."""
    ids = [cid for cid, lo, hi in starters for _ in range(int(rng.integers(lo, hi + 1)))]
    fill_pool = [c for c in cards if c not in {s[0] for s in starters}] if starters else list(cards)
    curse_ids: list[int] = []
    if getattr(cfg, "deck_scheme", "uniform") == "acts":
        act = SIM_ACT_TO_ACT.get(enc[1], 1)
        rest = int(rng.integers(0, cfg.extra_per_act * act + 1))
        rest = max(rest, cfg.min_playable - len(ids))
        ids += [int(c) for c in rng.choice(fill_pool, rest)] if rest > 0 else []
        if len(specials) and getattr(cfg, "special_max", 0) > 0:
            n_sp = int(rng.integers(cfg.special_min, cfg.special_max + 1))
            ids += [int(c) for c in rng.choice(specials, n_sp)] if n_sp else []
        elif len(specials) and rng.random() < getattr(cfg, "special_prob", 0.0):
            ids.append(int(rng.choice(specials)))
        n_curse = int(rng.integers(cfg.curses_min, cfg.curses_max + 1))
        curse_ids = [int(c) for c in rng.choice(curses, n_curse)] if n_curse and len(curses) else []
    else:
        n = int(rng.integers(cfg.deck_min, cfg.deck_max + 1))
        rest = max(0, n - len(ids))
        ids += [int(c) for c in rng.choice(fill_pool, rest)] if rest else []
    deck = [(int(c), bool(rng.random() < cfg.upgrade_prob)) for c in ids] + [(c, False) for c in curse_ids]
    n_pot = sample_potion_count(rng, cfg.potion_bands)
    potions = [int(p) for p in rng.choice(pots, n_pot)] if n_pot else []
    max_hp = int(rng.integers(cfg.max_hp_min, cfg.max_hp_max + 1))
    hp = max(1, int(round(max_hp * rng.uniform(cfg.hp_frac_min, 1.0))))
    seed = int(rng.integers(2**31))
    relics: list[int] = []
    if getattr(cfg, "relic_scheme", "uniform") == "acts" and ancients is not None:
        relics = sample_act_relics(rng, SIM_ACT_TO_ACT.get(enc[1], 1), relic_pool, ancients, cfg.relics_per_act)
    elif cfg.relics_max > 0 and len(relic_pool):
        n_rel = min(int(rng.integers(cfg.relics_min, cfg.relics_max + 1)), len(relic_pool))
        relics = sorted(int(r) for r in rng.choice(relic_pool, n_rel, replace=False)) if n_rel else []
    if relic_cards is not None and relics:
        deck, max_hp, hp, _ = relic_cards.apply(deck, relics, max_hp, hp, rng)
    return Scenario(enc[0], enc[1], enc[2], deck, hp, max_hp, potions, seed, relics)


def scenario_pools(run, cfg) -> dict:
    """시나리오를 만드는 데 쓰는 풀 전부 (학습·분석 스크립트가 같이 쓴다). sample_scenario 에 **풀 로 넘긴다."""
    cards, curses, specials = card_pool(run), curse_pool(run), special_pool(run)
    rc = None
    if getattr(cfg, "relic_cards", False):
        from .relic_cards import RelicCards  # noqa: PLC0415

        rc = RelicCards(run)
        cards = [c for c in cards if c not in rc.reserved]
        curses = [c for c in curses if c not in rc.reserved]
        specials = [c for c in specials if c not in rc.reserved]
    return {"cards": cards, "pots": potion_pool(run), "relic_pool": run.relic_pool(),
            "starters": parse_starters(cfg.starter_cards, {i.name: c for c, i in run.lib.cards.items()}),
            "ancients": run.ancient_relic_pools() if getattr(cfg, "relic_scheme", "uniform") == "acts" else None,
            "curses": curses, "specials": specials, "relic_cards": rc}


def scenario_from(rng, cfg, P: dict, enc) -> Scenario:
    return sample_scenario(rng, cfg, P["cards"], P["pots"], enc, P["relic_pool"], P["starters"], P["ancients"],
                           P["curses"], P["specials"], P["relic_cards"])


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
    sel_names = run.lib.names.card_selection_kinds()

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
            # intent = 주 의도, next = 같은 행동에 함께 붙은 의도 (예: 공격 + 약화). 이름은 예전 기록과 맞춘다.
            # [종류, 수치(공격이면 총 피해), 때리는 횟수]
            for key, ty in (("intent", S.TYPE_INTENT), ("next", S.TYPE_INTENT_NEXT)):
                k = np.flatnonzero((o.types == ty) & (o.owners == slot))
                if len(k):
                    j = int(k[0])
                    e[key] = [intent_names.get(int(o.ids[j]), "?"), int(round(float(o.feats[j, S.F_INTENT_MAG]) * 30)),
                              int(round(float(o.feats[j, S.F_MAGNITUDE]) * 5))]
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
            "endturn": [round(float(g[c]) * 15, 1) for c in (14, 15)] + [round(float(g[16]), 2), int(g[17] > 0.5)],
            "select": sel_names.get(int(o.ids[np.flatnonzero(o.types == S.TYPE_SELECTION)[0]]), "?")
            if (o.types == S.TYPE_SELECTION).any() else None,  # 카드 고르기 화면 종류 (CardSelectionKind)
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
        kind, mag = x[0], x[1]
        hits = x[2] if len(x) > 2 else 1
        if not mag:
            return kind
        return f"{kind} {mag}" + (f" ({mag // hits}x{hits})" if kind == "Attack" and hits > 1 else "")

    s = f"{e['name']} HP{e['hp']}" + (f" 방어{e['block']}" if e.get("block") else "") + f" [{intent(e.get('intent'))}]"
    return s + (f" +[{intent(e['next'])}]" if e.get("next") else "")


def fight_text(rec: dict) -> str:
    head = (f"{'승리' if rec['won'] else '패배'} · {', '.join(rec.get('enemy_names', [])) or rec['encounter']} "
            f"(적 {rec['encounter']}, {SIM_ACT_TO_ACT.get(rec['sim_act'], rec['sim_act'])}막·지역{rec['sim_act']}, {rec['kind']}) · "
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


def make_policy(vocab, cfg) -> CombatPolicy:
    """설정 크기의 전투 정책. 예전 설정 파일(heads·ff_mult 없음)은 예전 크기(4, 2)로."""
    return CombatPolicy(SharedEmbeddings(vocab, cfg.d_model), layers=cfg.layers,
                        heads=getattr(cfg, "heads", 4), ff_mult=getattr(cfg, "ff_mult", 2))


def _init_worker(repo: str, cfg: CombatConfig, server_args=None) -> None:
    from ..envs.sts2_adapter import Sts2Run  # noqa: PLC0415

    torch.set_num_threads(1)
    run = Sts2Run(repo)
    run.reset(12345)
    actor = None
    if server_args is not None:  # 정책 계산은 추론 서버가 한다 (가중치도 서버만 가진다)
        from .infer_server import RemoteActor  # noqa: PLC0415

        req_q, resp_qs, id_q = server_args
        wid = id_q.get()
        actor = RemoteActor(wid, req_q, resp_qs[wid])
        policy = None
    else:
        policy = make_policy(run.vocab, cfg)
        policy.eval()
    _W.update(run=run, policy=policy, actor=actor, cfg=cfg, version=-1)


def _work(task):
    version, blob, scs, greedy, seed = task
    if _W["actor"] is None and _W["version"] != version:
        _W["policy"].load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu"))
        _W["version"] = version
    torch.manual_seed(seed)
    cfg = _W["cfg"]
    with torch.no_grad():
        return play_scenarios(_W["actor"] or _W["policy"], _W["run"], scs, cfg.group_size, greedy, cfg.max_steps)


def _blob(policy) -> bytes:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, buf)
    return buf.getvalue()


class Runner:
    def __init__(self, repo: str, cfg: CombatConfig, run, device: str = "cuda"):
        self.cfg, self.run, self.pool, self.version, self.blob = cfg, run, None, 0, b""
        self.server = None
        if cfg.workers > 0:
            server_args = None
            if getattr(cfg, "infer_server", False):
                from .infer_server import InferenceServer  # noqa: PLC0415

                self.server = InferenceServer(run.vocab, cfg.d_model, device, cfg.workers,
                                              max_batch=getattr(cfg, "server_max_batch", 256),
                                              policy_kwargs={"layers": cfg.layers, "heads": cfg.heads,
                                                             "ff_mult": cfg.ff_mult})
                server_args = self.server.worker_args()
            self.pool = mp.get_context("spawn").Pool(cfg.workers, initializer=_init_worker,
                                                     initargs=(repo, cfg, server_args))

    def sync(self, policy) -> None:
        self.version += 1
        self.blob = _blob(policy)
        if self.server is not None:
            self.server.load(self.blob)

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
        # 추론 서버를 쓰면 워커는 가중치가 필요 없다. 큰 모델(11M ≈ 48 MB)을 작업마다 보내면
        # 반복당 수백 번 복사돼 Windows 에서 시스템 리소스가 바닥났다 (WinError 1450).
        blob = b"" if self.server is not None else self.blob
        res = self.pool.map(_work, [(self.version, blob, ch, greedy, s) for ch, s in zip(chunks, seeds)],
                            chunksize=1)
        return [g for r in res for g in r]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
        if self.server is not None:
            self.server.close()


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
    P = scenario_pools(run, cfg)
    cards, pots, relic_pool = P["cards"], P["pots"], P["relic_pool"]
    device = resolve_device(cfg.device)
    policy = make_policy(run.vocab, cfg).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr_policy)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log(f"device={device} workers={cfg.workers} 적(조우,막)={len(pool)} 카드풀={len(cards)} 포션풀={len(pots)} "
        f"유물풀={len(relic_pool)} 파라미터={sum(p.numel() for p in policy.parameters()):,}")
    eval_rng = np.random.default_rng(10**6 + cfg.seed)
    eval_set = [scenario_from(eval_rng, cfg, P, e) for _ in range(cfg.eval_rounds) for e in pool]
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
    runner = Runner(repo, cfg, run, device=str(device))
    cpu_policy = policy if device.type == "cpu" else make_policy(run.vocab, cfg)

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
            scs = [scenario_from(rng, cfg, P, pool[i]) for i in idx]
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
            try:  # 보기 페이지(fights.html, 반복마다 .md)도 다시 쓴다. 실패해도 학습은 계속한다
                from .fight_view import to_md, write_all  # noqa: PLC0415

                (out / "fights" / f"iter_{it:04d}.md").write_text(to_md(it, recs), encoding="utf-8")
                write_all(out)
            except Exception as exc:  # noqa: BLE001
                log(f"  fights.html 을 쓰지 못했다: {exc!r}")
            policy.train()
            row.update(grpo_update(policy, opt, [g for g in groups if len(g) > 1], epochs=cfg.grpo_epochs,
                                   ent_floor=cfg.ent_floor, ent_floor_coef=cfg.ent_floor_coef, amp=cfg.amp))
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
            try:  # 학습 추이 그래프(progress.html)도 다시 쓴다. 실패해도 학습은 계속한다
                from .progress_view import write_progress  # noqa: PLC0415

                write_progress(out)
            except Exception as exc:  # noqa: BLE001
                log(f"  progress.html 을 쓰지 못했다: {exc!r}")
            (out / "death_rate.json").write_text(json.dumps(
                {f"{e}@{a}:{k}": round(float(d), 3) for (e, a, k), d in zip(pool, death)}, indent=1))
    finally:
        runner.close()
    return policy
