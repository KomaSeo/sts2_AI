"""덱 빌딩 재생 GRPO: 치명적인 적을 대비하는 덱 빌딩 모듈(agents/deck_builder.py) 학습과, 실제 런에서 합치기.

1) 기록 (DeckRecorder): 실제 런에서 런 시작부터 덱이 바뀌는 화면(카드 보상·상점·휴식·이벤트·고대인)을 순서대로 남긴다.
   선택지마다 충실한 복제(clone_exact)로 미리 골라 보고 효과(덱 ±카드, 유물, 골드, 최대 HP, 뒤따르는 "덱에서
   카드 고르기")를 잰다. 기록한 화면 사이에 생긴 변화(전투 보상 골드·유물, 저주 등)는 "외부 변화"로 남긴다.
2) 재생 (run_deck_group): 치명적인 적과의 전투가 시작되면 그때까지의 기록을 G 번 재생한다. 런 시작 덱에서 출발해
   화면을 순서대로 보여 주고 덱 빌딩 모듈이 고른다 (골드는 재생 안에서 계산, 못 사는 상품은 못 고름).
   휴식 장소는 쉬기/강화를 50:50 무작위 (강화면 어떤 카드를 강화할지는 모듈이 고름).
   최종 덱으로 그 적과 K 번 싸운다 (HP 는 그룹마다 무작위, 그룹 안에서는 같음, 시드도 그룹 안에서 공통).
   K 번 평균 점수로 그룹 안에서 이점을 매기고, 각 재생의 모든 선택에 그 이점을 준다. 가치 머리는 최종 점수를 맞힌다.
3) 합치기 (combined_logits): 실제 런의 덱이 바뀌는 화면에서
     점수 = 단기 모듈 logit + β × (덱 빌딩 모듈 log 확률, 대비할 적들의 사망 가중 평균)
   휴식 장소는 강화에 γ × [D(가장 좋은 강화 뒤) − D(지금)] 를 더한다. β, γ 는 막마다 런 결과로 학습 (train/loop.py).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
import torch

from ..agents.deck_builder import builder_state
from ..envs.base import PHASE_OTHER
from ..schema import ChoiceOption, RunState
from .grpo import advantages, play_many
from .probe import probe_score

SELECT_FOLLOW = {"select.remove": "remove", "select.upgrade": "upgrade", "select.transform": "transform",
                 "select.transform_to": "transform", "select.duplicate": "duplicate"}


def screen_group(opts: list[ChoiceOption]) -> str | None:
    """덱 빌딩 화면 종류. None 이면 기록하지 않는다 (맵·보상 목록·뒤따르는 선택 화면 등)."""
    kinds = {o.kind for o in opts}
    if any(k.startswith("select.") for k in kinds):
        return None  # 앞 선택의 뒤따르는 화면 (재생에서는 그 시점 덱으로 다시 고른다)
    if any(k.startswith("card.") for k in kinds):
        return "card"
    if any(k.startswith("shop.") for k in kinds):
        return "shop"
    if "rest.heal" in kinds or "rest.smith" in kinds:
        return "rest"
    if "event.option" in kinds or "neow.relic" in kinds or "bundle.pick" in kinds:
        return "event"
    return None


@dataclass
class Delta:
    add: list = field(default_factory=list)  # [(카드 id, 강화)]
    remove: list = field(default_factory=list)
    relics: list = field(default_factory=list)  # 유물 id
    gold: int = 0
    max_hp: int = 0


def diff(before: dict, after: dict) -> Delta:
    b, a = collections.Counter(before["deck"]), collections.Counter(after["deck"])
    rb, ra = collections.Counter(before["relics"]), collections.Counter(after["relics"])
    return Delta(add=list((a - b).elements()), remove=list((b - a).elements()), relics=list((ra - rb).elements()),
                 gold=after["gold"] - before["gold"], max_hp=after["max_hp"] - before["max_hp"])


@dataclass
class ROpt:
    opt: ChoiceOption
    delta: Delta
    follow: tuple | None = None  # ("remove"|"upgrade"|"transform"|"duplicate", 1) 또는 ("offer", [(id, 강화)])


@dataclass
class RScreen:
    kind: str
    options: list[ROpt]
    ext: Delta  # 이 화면 전에 생긴 외부 변화
    act: int
    floor: int
    shop_picks: int = 1  # 상점: 실제로 산 횟수 (재생에서 최대 이만큼 + 1 번 고를 수 있음)


@dataclass
class Episode:
    base: dict
    screens: list[RScreen]
    ext_final: Delta
    act: int
    floor: int
    target_raw: int
    is_boss: bool


def measure(run, opt: ChoiceOption, rng: np.random.Generator, max_event_steps: int = 6) -> ROpt:
    """충실한 복제로 이 선택지를 골라 보고 효과를 잰다. 이벤트는 이벤트가 끝날 때까지 규칙 정책으로 이어 간다."""
    c = run.clone_exact()
    try:
        before = c.raw_state()
        c.step_choice(opt.action)
        follow = None
        for _ in range(max_event_steps):
            nxt = c.choice_options() if c.phase == PHASE_OTHER else []
            kinds = {o.kind for o in nxt}
            sel = next((SELECT_FOLLOW[k] for k in kinds if k in SELECT_FOLLOW), None)
            if sel is not None:
                follow = (sel, 1)
                break
            if "select.offer" in kinds:
                follow = ("offer", [(c.card_raw(o.card.id), o.card.upgraded) for o in nxt
                                    if o.kind == "select.offer" and o.card is not None])
                break
            if c.screen_kind() != "PHASE_EVENT" or not nxt:
                break
            c.step_choice(c.rule_action(rng))
        return ROpt(opt=opt, delta=diff(before, c.raw_state()), follow=follow)
    finally:
        c.close()


class DeckRecorder:
    """실제 런 하나의 덱 빌딩 화면 기록."""

    def __init__(self, run):
        self.base = run.raw_state()
        self.expect = dict(self.base)  # 마지막 실제 선택(과 뒤따르는 화면)까지 반영한 상태
        self.screens: list[RScreen] = []
        self._pending = False
        self._shop_open = None

    def observe(self, run, opts: list[ChoiceOption] | None) -> None:
        """매 스텝, 행동하기 전에 부른다. 실제 선택의 뒤따르는 화면이 끝나면 그 시점 상태를 기대 상태로 잡는다."""
        if not self._pending:
            return
        if opts and any(o.kind.startswith("select.") for o in opts):
            return  # 아직 실제 선택의 뒤따르는 화면
        self.expect = run.raw_state()
        self._pending = False

    def screen(self, run, opts: list[ChoiceOption], rng: np.random.Generator) -> None:
        """덱 빌딩 화면이면 선택지 효과를 재서 기록한다. 실제 선택을 하기 직전에 부른다."""
        kind = screen_group(opts)
        if kind is None:
            if self._shop_open is not None and run.screen_kind() != "PHASE_SHOP":
                self._shop_open = None
            return
        if kind == "shop" and self._shop_open is not None and self._shop_open.floor == run.floor:
            self._shop_open.shop_picks += 1  # 같은 상점 방문의 다음 구매
            self._pending = True
            return
        now = run.raw_state()
        sc = RScreen(kind=kind, options=[measure(run, o, rng) for o in opts], ext=diff(self.expect, now),
                     act=run.act, floor=run.floor)
        self.screens.append(sc)
        self._shop_open = sc if kind == "shop" else None
        self._pending = True

    def episode(self, run, target_raw: int, is_boss: bool) -> Episode:
        now = run.raw_state()
        ext = diff(self.expect, now) if not self._pending else Delta()
        return Episode(base=dict(self.base), screens=list(self.screens), ext_final=ext, act=run.act,
                       floor=run.floor, target_raw=target_raw, is_boss=is_boss)


# ---- 재생 ----

@dataclass
class DeckSample:
    state: RunState
    options: list[ChoiceOption]
    action: int  # -1 = 가치 학습만 (휴식 장소의 무작위 선택 등)
    logp: float = 0.0
    advantage: float = 0.0
    target: float = 0.0  # 가치 머리의 목표 (그 재생의 최종 점수)
    after: RunState | None = None  # 이 선택의 효과를 적용한 직후 상태 (선택별 이점 = D(after) − D(state))


class _Replay:
    def __init__(self, ep: Episode):
        self.deck = list(ep.base["deck"])
        self.relics = list(ep.base["relics"])
        self.gold = ep.base["gold"]
        self.max_hp = ep.base["max_hp"]
        self.samples: list[DeckSample] = []

    def state(self, run, ep: Episode, target: int, floor: int) -> RunState:
        return builder_state([run.raw_card(c, u) for c, u in self.deck], [run.relic_vocab(r) for r in self.relics],
                             self.gold, self.max_hp, ep.act, floor, target)

    def apply(self, d: Delta) -> None:
        for c in d.remove:
            if c in self.deck:
                self.deck.remove(c)
        self.deck += list(d.add)
        self.relics += list(d.relics)
        self.gold = max(0, self.gold + d.gold)
        self.max_hp = max(1, self.max_hp + d.max_hp)


def _decide(builder, run, rep: _Replay, ep: Episode, target: int, options: list[ChoiceOption], floor: int,
            greedy: bool = False) -> int:
    st = builder_state([run.raw_card(c, u) for c, u in rep.deck], [run.relic_vocab(r) for r in rep.relics],
                       rep.gold, rep.max_hp, ep.act, floor, target)
    with torch.no_grad():
        logits, _ = builder([st], [options])
    logits = logits[0, : len(options)]
    dist = torch.distributions.Categorical(logits=logits)
    i = int(logits.argmax()) if greedy else int(dist.sample())
    rep.samples.append(DeckSample(st, options, i, float(dist.log_prob(torch.tensor(i)))))
    return i


def _value_only(run, rep: _Replay, ep: Episode, target: int, floor: int) -> None:
    st = builder_state([run.raw_card(c, u) for c, u in rep.deck], [run.relic_vocab(r) for r in rep.relics],
                       rep.gold, rep.max_hp, ep.act, floor, target)
    rep.samples.append(DeckSample(st, [], -1))


def _follow(builder, run, rep: _Replay, ep: Episode, target: int, follow, floor: int, rng) -> None:
    kind, arg = follow
    if kind == "offer":
        cards = list(arg)
        if not cards:
            return
        opts = [ChoiceOption(action=i, kind="select.offer", card=run.raw_card(c, u)) for i, (c, u) in enumerate(cards)]
        rep.deck.append(cards[_decide(builder, run, rep, ep, target, opts, floor)])
        rep.samples[-1].after = rep.state(run, ep, target, floor)
        return
    if kind == "upgrade":
        idx = [i for i, (c, u) in enumerate(rep.deck) if not u and run.card_upgradable(c)]
    else:
        idx = list(range(len(rep.deck)))
    if not idx:
        return
    opts = [ChoiceOption(action=i, kind=f"select.{kind}", card=run.raw_card(*rep.deck[i]), deck_index=i) for i in idx]
    i = idx[_decide(builder, run, rep, ep, target, opts, floor)]
    smp = rep.samples[-1]
    c, u = rep.deck[i]
    if kind == "remove":
        rep.deck.pop(i)
    elif kind == "upgrade":
        rep.deck[i] = (c, True)
    elif kind == "transform":
        pool = run.transform_pool()
        rep.deck[i] = (int(rng.choice(pool)), False)
    elif kind == "duplicate":
        rep.deck.append((c, u))
    smp.after = rep.state(run, ep, target, floor)


def _pick_and_apply(builder, run, rep, ep, target, cands: list[ROpt], floor, rng) -> ROpt:
    i = _decide(builder, run, rep, ep, target, [o.opt for o in cands], floor)
    smp = rep.samples[-1]
    o = cands[i]
    rep.apply(o.delta)
    smp.after = rep.state(run, ep, target, floor)
    if o.follow is not None:
        _follow(builder, run, rep, ep, target, o.follow, floor, rng)
    return o


def replay(builder, run, ep: Episode, target: int, rng: np.random.Generator) -> _Replay:
    rep = _Replay(ep)
    for sc in ep.screens:
        rep.apply(sc.ext)
        if sc.kind == "rest":
            heal = next((o for o in sc.options if o.opt.kind == "rest.heal"), None)
            smith = next((o for o in sc.options if o.opt.kind == "rest.smith"), None)
            _value_only(run, rep, ep, target, sc.floor)
            if smith is not None and (heal is None or rng.random() < 0.5):
                _follow(builder, run, rep, ep, target, ("upgrade", 1), sc.floor, rng)
            continue
        affordable = [o for o in sc.options if -o.delta.gold <= rep.gold or o.delta.gold >= 0]
        if sc.kind == "shop":
            left = list(sc.options)
            for _ in range(sc.shop_picks + 1):
                cands = [o for o in left if -o.delta.gold <= rep.gold or o.delta.gold >= 0]
                if not cands:
                    break
                o = _pick_and_apply(builder, run, rep, ep, target, cands, sc.floor, rng)
                if o.opt.kind == "shop.leave":
                    break
                if o in left and o.opt.kind != "shop.remove":
                    left.remove(o)
                elif o.opt.kind == "shop.remove":
                    left = [x for x in left if x.opt.kind != "shop.remove"]  # 제거는 한 번만
            continue
        if affordable:
            _pick_and_apply(builder, run, rep, ep, target, affordable, sc.floor, rng)
    rep.apply(ep.ext_final)
    return rep


def run_deck_group(agent, run, cfg, rng: np.random.Generator, ep: Episode, combats_before: int):
    """에피소드 하나를 G 번 재생하고 최종 덱마다 K 번 싸운다. (DeckSample 목록, 기록)."""
    from ..envs.sts2_adapter import Sts2RunCombatFork  # noqa: PLC0415

    builder = agent.deck
    target = run.encounter_vocab(ep.target_raw)
    reps = [replay(builder, run, ep, target, rng) for _ in range(cfg.deck_group_size)]
    hp_frac = float(rng.uniform(cfg.deck_hp_min, 1.0))
    seeds = [int(rng.integers(2**31)) for _ in range(cfg.deck_fights)]
    sims, owners, rvs = [], [], []
    for g, rep in enumerate(reps):
        hp = max(1, int(round(rep.max_hp * hp_frac)))
        for s in seeds:
            p = run.start_probe(s, rep.deck, ep.target_raw, hp, combats_before, relics=rep.relics, max_hp=rep.max_hp)
            if p is None:
                continue
            if cfg.policy_sees_map:
                rvs.append(agent.run_vec(p.run_state()))
            sims.append(Sts2RunCombatFork(p))
            owners.append(g)
    if not sims:
        return [], None
    rv = torch.stack(rvs) if rvs else None
    trajs = play_many(agent.combat_actor(), sims, run_vec=rv)
    per = collections.defaultdict(list)
    wins = collections.defaultdict(list)
    for g, t in zip(owners, trajs):
        per[g].append(probe_score(t, reps[g].max_hp))
        wins[g].append(t.won)
    scores = [float(np.mean(per[g])) if per[g] else 0.0 for g in range(len(reps))]
    adv = advantages(scores, cfg.advantage_mode)
    samples = []
    for g, rep in enumerate(reps):
        for s in rep.samples:
            s.advantage, s.target = (float(adv[g]) if cfg.deck_adv != "td" else 0.0), scores[g]
            samples.append(s)
    if cfg.deck_adv in ("td", "mix"):
        # 선택별 이점: 그 선택이 덱 가치 D 를 얼마나 올렸나. 결정 20여 개가 최종 점수 하나를 나눠 갖지 않게 한다.
        pol = [s for s in samples if s.action >= 0 and s.after is not None]
        if pol:
            with torch.no_grad():
                v = builder.values([s.state for s in pol] + [s.after for s in pol]).float().cpu().numpy()
            td = v[len(pol):] - v[: len(pol)]
            td = td / (td.std() + 1e-6)
            for s, a in zip(pol, td):
                s.advantage += float(a)
    info = {"target": ep.target_raw, "boss": ep.is_boss, "screens": len(ep.screens), "decisions": len(samples),
            "score": float(np.mean(scores)), "win": float(np.mean([w for g in wins for w in wins[g]])),
            "score_spread": float(np.std(scores))}
    return samples, info


def deck_update(builder, optim, samples: list[DeckSample], epochs: int = 2, clip: float = 0.2,
                ent_coef: float = 0.01, value_coef: float = 0.5, minibatch: int = 128) -> dict:
    """선택은 PPO 클리핑(이점 0 인 표본은 빼고), 가치 머리는 모든 표본의 최종 점수 회귀."""
    if not samples:
        return {}
    from ..batching import module_device  # noqa: PLC0415

    dev = module_device(builder)
    stats = {"deck_loss": 0.0, "deck_value_loss": 0.0, "deck_entropy": 0.0}
    n = 0
    for _ in range(epochs):
        perm = np.random.permutation(len(samples))
        for start in range(0, len(samples), minibatch):
            chunk = [samples[i] for i in perm[start: start + minibatch]]
            opts = [s.options if s.options else [ChoiceOption(action=0)] for s in chunk]
            logits, value = builder([s.state for s in chunk], opts)
            tgt = torch.tensor([s.target for s in chunk], dtype=torch.float, device=dev)
            v_loss = ((value - tgt) ** 2).mean()
            act = [i for i, s in enumerate(chunk) if s.action >= 0 and s.advantage != 0.0]
            p_loss = torch.zeros((), device=dev)
            ent = torch.zeros((), device=dev)
            if act:
                idx = torch.tensor(act, device=dev)
                lp_all = torch.log_softmax(logits[idx], -1)
                a = torch.tensor([chunk[i].action for i in act], device=dev)
                lp = lp_all.gather(1, a.unsqueeze(1)).squeeze(1)
                old = torch.tensor([chunk[i].logp for i in act], device=dev)
                adv = torch.tensor([chunk[i].advantage for i in act], dtype=torch.float, device=dev)
                ratio = (lp - old).exp()
                p_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                ent = -(lp_all.exp() * lp_all.clamp(min=-30)).sum(-1).mean()
            loss = p_loss - ent_coef * ent + value_coef * v_loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(builder.parameters(), 1.0)
            optim.step()
            stats["deck_loss"] += float(p_loss.detach())
            stats["deck_value_loss"] += float(v_loss.detach())
            stats["deck_entropy"] += float(ent.detach())
            n += 1
    out = {k: v / max(1, n) for k, v in stats.items()}
    out["deck_samples"] = len(samples)
    return out


# ---- 실제 런에서 합치기 ----

def deck_targets(run, weights: dict, act: int, top: int = 2) -> list[tuple[int, float]]:
    """대비할 적들 [(encounter 인덱스, 가중치)]: 이번 런 보스(항상) + 이 막의 사망 상위 일반·엘리트 적 top 개."""
    table = weights.get(act, {})
    boss_raw = run.boss_raw()
    boss_w = max(1.0, sum(w for (e, b), w in table.items() if b and e == boss_raw))
    others = sorted(((w, e) for (e, b), w in table.items() if not b and w > 0), reverse=True)[:top]
    out = [(run.encounter_vocab(boss_raw), boss_w)] + [(run.encounter_vocab(e), w) for w, e in others]
    tot = sum(w for _, w in out)
    return [(t, w / tot) for t, w in out]


@torch.no_grad()
def deck_logprob(builder, run, state: RunState, opts: list[ChoiceOption], targets) -> torch.Tensor:
    """대비할 적들에 대한 덱 빌딩 모듈 log 확률의 가중 평균 [len(opts)]."""
    sts = [builder_state(state.deck, [r for r, _ in state.relics], state.gold, state.max_hp, state.act, state.floor, t)
           for t, _ in targets]
    logits, _ = builder(sts, [opts] * len(sts))
    lp = torch.log_softmax(logits[:, : len(opts)], -1)
    w = torch.tensor([w for _, w in targets], dtype=lp.dtype, device=lp.device).unsqueeze(1)
    return (lp * w).sum(0).cpu()


@torch.no_grad()
def upgrade_gain(builder, run, state: RunState, targets) -> float:
    """D(가장 좋은 강화 뒤 덱) − D(지금 덱), 대비할 적들 가중 평균."""
    raw = run.deck_raw()
    idx = [i for i, (c, u) in enumerate(raw) if not u and run.card_upgradable(c)]
    if not idx or len(raw) != len(state.deck):
        return 0.0
    gain = 0.0
    for t, w in targets:
        decks = [state.deck] + [[run.raw_card(c, True) if j == i else state.deck[j] for j, (c, _) in enumerate(raw)]
                                for i in idx]
        sts = [builder_state(d, [r for r, _ in state.relics], state.gold, state.max_hp, state.act, state.floor, t)
               for d in decks]
        v = builder.values(sts)
        gain += w * float(v[1:].max() - v[0])
    return gain


def is_deadly(weights: dict, act: int, enc: int, is_boss: bool, top: int) -> bool:
    """치명적인 적인가: 보스는 항상, 그 밖에는 이 막의 사망 가중치 상위 top 안."""
    if is_boss:
        return True
    table = weights.get(act, {})
    ranked = sorted(((w, e) for (e, b), w in table.items() if not b and w > 0), reverse=True)[:top]
    return any(e == enc for _, e in ranked)


def deck_relevant(opts: list[ChoiceOption]) -> bool:
    """덱이 바뀌는 화면인가 (기록하는 화면 + 덱에서 카드 고르기)."""
    return screen_group(opts) is not None or any(o.kind.startswith("select.") for o in opts)


def combined_pick(agent, run, state: RunState, opts: list[ChoiceOption], weights: dict, cfg, beta: float,
                  gamma: float, greedy: bool) -> int:
    """단기 모듈 logit + β × 덱 빌딩 log 확률. 휴식 장소(쉬기·강화)는 강화에 γ × 덱 가치 상승을 더한다."""
    from .choice import choice_logits  # noqa: PLC0415

    logits = choice_logits(agent, [state], [opts])[0, : len(opts)].float().cpu()
    targets = deck_targets(run, weights, state.act, cfg.deck_targets_top)
    kinds = [o.kind for o in opts]
    if "rest.smith" in kinds and "rest.heal" in kinds:
        logits[kinds.index("rest.smith")] += gamma * upgrade_gain(agent.deck, run, state, targets)
    else:
        logits = logits + beta * deck_logprob(agent.deck, run, state, opts, targets)
    if greedy:
        return int(logits.argmax())
    return int(torch.distributions.Categorical(logits=logits).sample())
