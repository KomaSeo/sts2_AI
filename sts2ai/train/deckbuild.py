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
    real: Delta | None = None  # 실제 런에서 고른 선택의 효과 (덱이 안 바뀌는 이벤트 화면은 재생에서 이걸 그대로 적용)


def changes_deck(o: ROpt) -> bool:
    """이 선택지가 덱(카드)을 바꾸나: 카드가 들어오거나 빠지거나, 뒤에 덱 카드 고르기(제거·강화·변환·복제·제안)가 이어진다."""
    return bool(o.delta.add or o.delta.remove or o.follow is not None)


DECK_KINDS = ("card.take", "shop.card", "shop.remove", "rest.smith")


def deck_choice_screen(run, opts: list[ChoiceOption], rng: np.random.Generator) -> bool:
    """실제 런에서 덱 빌딩 모듈이 의견을 낼 화면인가: 선택지 가운데 덱이 바뀌는 것이 하나라도 있을 때만.

    카드 보상·상점(카드·카드 제거)·휴식 장소(강화)·덱 카드 고르기는 종류로 안다. 이벤트·고대인 선택은 숨은 난수를
    새로 뽑은 복제로 골라 봐서 덱이 바뀌는 선택지가 있는지 본다. HP·최대 HP·골드·유물만 바뀌는 화면은 단기 모듈만 쓴다.
    """
    if not deck_relevant(opts):
        return False
    kinds = {o.kind for o in opts}
    if any(k.startswith("select.") for k in kinds) or kinds & set(DECK_KINDS):
        return True
    for o in opts:
        if o.kind in ("event.option", "neow.relic", "bundle.pick"):
            try:
                if changes_deck(measure(run, o, rng, exact=False)):
                    return True
            except Exception:  # noqa: BLE001  못 재면 덱이 바뀌는 것으로 보지 않는다
                continue
    return False


@dataclass
class Episode:
    base: dict
    screens: list[RScreen]
    ext_final: Delta
    act: int
    floor: int
    target_raw: int
    is_boss: bool


def measure(run, opt: ChoiceOption, rng: np.random.Generator, max_event_steps: int = 6,
            exact: bool = True) -> ROpt:
    """복제로 이 선택지를 골라 보고 효과를 잰다. 이벤트는 이벤트가 끝날 때까지 규칙 정책으로 이어 간다.

    exact=True: 충실한 복제 (실제로 일어날 결과 그대로. 기록·재생용).
    exact=False: 숨은 난수를 새로 뽑은 복제 (실제 런의 판단용. 미래를 엿보지 않는다).
    """
    c = run.clone_exact() if exact else run.clone(int(rng.integers(2**31)))
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
        self._before = None  # 마지막으로 기록한 화면 직전의 상태 (실제 선택의 효과를 재려고)

    def observe(self, run, opts: list[ChoiceOption] | None) -> None:
        """매 스텝, 행동하기 전에 부른다. 실제 선택의 뒤따르는 화면이 끝나면 그 시점 상태를 기대 상태로 잡는다."""
        if not self._pending:
            return
        if opts and any(o.kind.startswith("select.") for o in opts):
            return  # 아직 실제 선택의 뒤따르는 화면
        self.expect = run.raw_state()
        self._pending = False
        if self.screens and self._before is not None:
            self.screens[-1].real = diff(self._before, self.expect)

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
        self._before = now
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
    opt_after: list | None = None  # 선택지마다 "그걸 골랐다면"의 직후 상태 (선택지 전부 평가용)
    opt_values: list | None = None  # 그 상태들의 D (수집 때 계산)


class _Replay:
    def __init__(self, ep: Episode, hp_frac: float = 1.0, temp: float = 1.0):
        self.hp_frac = hp_frac  # 마지막 전투의 HP 비율 (가치 머리의 입력 HP)
        # 온도: softmax(logit / temp) 에서 뽑는다. 1 보다 크면 높은 확률은 낮추고 낮은 확률은 올려
        # 정책이 잘 안 고르는 선택지도 재생에서 시험된다 (순서는 그대로)
        self.temp = temp
        self.deck = list(ep.base["deck"])
        self.relics = list(ep.base["relics"])
        self.gold = ep.base["gold"]
        self.max_hp = ep.base["max_hp"]
        self.samples: list[DeckSample] = []

    def state(self, run, ep: Episode, target: int, floor: int, deck=None, relics=None, max_hp=None) -> RunState:
        """재생의 지금 상태 (deck·relics·max_hp 를 주면 그 값으로 바꾼 미리보기). 남은 층수 = 대상 적의 층 − 지금 층."""
        deck = self.deck if deck is None else deck
        relics = self.relics if relics is None else relics
        mh = self.max_hp if max_hp is None else max_hp
        return builder_state([run.raw_card(c, u) for c, u in deck], [run.relic_vocab(r) for r in relics],
                             self.gold, mh, ep.act, floor, target, horizon=ep.floor - floor,
                             hp=max(1, int(round(mh * self.hp_frac))))

    def preview(self, run, ep, target, floor, d: Delta) -> RunState:
        """이 효과를 적용했다면의 상태 (실제로는 바꾸지 않는다). 뒤따르는 덱 카드 고르기는 반영하지 않는다."""
        deck = list(self.deck)
        for c in d.remove:
            if c in deck:
                deck.remove(c)
        deck += list(d.add)
        return self.state(run, ep, target, floor, deck=deck, relics=self.relics + list(d.relics),
                          max_hp=max(1, self.max_hp + d.max_hp))

    def apply(self, d: Delta) -> None:
        for c in d.remove:
            if c in self.deck:
                self.deck.remove(c)
        self.deck += list(d.add)
        self.relics += list(d.relics)
        self.gold = max(0, self.gold + d.gold)
        self.max_hp = max(1, self.max_hp + d.max_hp)


def _decide(builder, run, rep: _Replay, ep: Episode, target: int, options: list[ChoiceOption], floor: int,
            greedy: bool = False, opt_after: list | None = None) -> int:
    st = rep.state(run, ep, target, floor)
    with torch.no_grad():
        logits, _ = builder([st], [options])
    logits = logits[0, : len(options)]
    # 실제로 뽑은 분포(온도 적용)의 확률을 PPO 의 "이전 확률"로 적는다. 학습도 같은 온도로 비율을 잰다
    dist = torch.distributions.Categorical(logits=logits / rep.temp)
    i = int(logits.argmax()) if greedy else int(dist.sample())
    rep.samples.append(DeckSample(st, options, i, float(dist.log_prob(torch.tensor(i))), opt_after=opt_after))
    return i


def _value_only(run, rep: _Replay, ep: Episode, target: int, floor: int) -> None:
    rep.samples.append(DeckSample(rep.state(run, ep, target, floor), [], -1))


def _follow(builder, run, rep: _Replay, ep: Episode, target: int, follow, floor: int, rng) -> None:
    kind, arg = follow
    if kind == "offer":
        cards = list(arg)
        if not cards:
            return
        opts = [ChoiceOption(action=i, kind="select.offer", card=run.raw_card(c, u)) for i, (c, u) in enumerate(cards)]
        prev = [rep.state(run, ep, target, floor, deck=rep.deck + [cu]) for cu in cards]
        rep.deck.append(cards[_decide(builder, run, rep, ep, target, opts, floor, opt_after=prev)])
        rep.samples[-1].after = rep.state(run, ep, target, floor)
        return
    if kind == "upgrade":
        idx = [i for i, (c, u) in enumerate(rep.deck) if not u and run.card_upgradable(c)]
    else:
        idx = list(range(len(rep.deck)))
    if not idx:
        return
    opts = [ChoiceOption(action=i, kind=f"select.{kind}", card=run.raw_card(*rep.deck[i]), deck_index=i) for i in idx]
    prev = None
    if kind in ("remove", "upgrade", "duplicate"):
        def changed(j):
            d = list(rep.deck)
            if kind == "remove":
                d.pop(j)
            elif kind == "upgrade":
                d[j] = (d[j][0], True)
            else:
                d.append(d[j])
            return d
        prev = [rep.state(run, ep, target, floor, deck=changed(j)) for j in idx]
    i = idx[_decide(builder, run, rep, ep, target, opts, floor, opt_after=prev)]
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
    prev = [rep.preview(run, ep, target, floor, o.delta) for o in cands]
    i = _decide(builder, run, rep, ep, target, [o.opt for o in cands], floor, opt_after=prev)
    smp = rep.samples[-1]
    o = cands[i]
    rep.apply(o.delta)
    smp.after = rep.state(run, ep, target, floor)
    if o.follow is not None:
        _follow(builder, run, rep, ep, target, o.follow, floor, rng)
    return o


def replay(builder, run, ep: Episode, target: int, rng: np.random.Generator, hp_frac: float = 1.0,
           temp: float = 1.0) -> _Replay:
    rep = _Replay(ep, hp_frac, temp)
    for sc in ep.screens:
        rep.apply(sc.ext)
        if sc.kind == "rest":
            heal = next((o for o in sc.options if o.opt.kind == "rest.heal"), None)
            smith = next((o for o in sc.options if o.opt.kind == "rest.smith"), None)
            _value_only(run, rep, ep, target, sc.floor)
            if smith is not None and (heal is None or rng.random() < 0.5):
                _follow(builder, run, rep, ep, target, ("upgrade", 1), sc.floor, rng)
            continue
        if sc.kind == "event" and not any(changes_deck(o) for o in sc.options):
            # 덱이 안 바뀌는 이벤트(HP·최대 HP·골드·유물만): 덱 빌딩 모듈이 고르지 않고 실제 런의 선택을 그대로 적용
            if sc.real is not None:
                rep.apply(sc.real)
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
    hp_frac = float(rng.uniform(cfg.deck_hp_min, 1.0))
    reps = [replay(builder, run, ep, target, rng, hp_frac, temp=getattr(cfg, "deck_temp", 1.0))
            for _ in range(cfg.deck_group_size)]
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
    match = []  # 상성 신경망 M 의 학습 데이터: 같은 제안 목록에서 다르게 고른 덱들과 그 전투 결과
    from .matchup import MatchSample  # noqa: PLC0415

    mstates = {}
    for g, t in zip(owners, trajs):
        per[g].append(probe_score(t, reps[g].max_hp))
        wins[g].append(t.won)
        rep = reps[g]
        if g not in mstates:
            hp = max(1, int(round(rep.max_hp * hp_frac)))
            mstates[g] = RunState(act=ep.act, floor=ep.floor, hp=hp, max_hp=rep.max_hp, gold=rep.gold, ascension=0,
                                  deck=[run.raw_card(c, u) for c, u in rep.deck],
                                  relics=[(run.relic_vocab(r), 0) for r in rep.relics], potions=[], map=None,
                                  boss=target)
        sc = probe_score(t, rep.max_hp)
        match.append(MatchSample(mstates[g], bool(t.won), (sc - 1.0) if t.won else 0.0, 1.0 if t.won else sc,
                                 synthetic=True))
    scores = [float(np.mean(per[g])) if per[g] else 0.0 for g in range(len(reps))]
    adv = advantages(scores, cfg.advantage_mode)
    samples = [s for rep in reps for s in rep.samples]
    # 모든 상태(선택 전·후, 선택지별 미리보기)의 D 를 한 번에 계산한다
    states, where = [], []
    for k, s in enumerate(samples):
        where.append((len(states), s.after is not None, len(s.opt_after or [])))
        states.append(s.state)
        if s.after is not None:
            states.append(s.after)
        states.extend(s.opt_after or [])
    vals = _values(builder, states)
    v_now, v_after = [], []
    for (i0, has_after, n_opt), s in zip(where, samples):
        v_now.append(vals[i0])
        v_after.append(vals[i0 + 1] if has_after else None)
        if n_opt:
            j = i0 + 1 + int(has_after)
            s.opt_values = [float(x) for x in vals[j: j + n_opt]]
        s.opt_after = None  # 상태 묶음은 무거워서 학습 루프로 보내지 않는다 (값만 보낸다)
        s.after = None
    # 가치 목표 TD(λ): 마지막 단계는 실제 전투 점수, 그 앞은 다음 단계의 예측을 λ 만큼 섞어 이어 받는다
    lam = cfg.deck_lambda
    k = 0
    for g, rep in enumerate(reps):
        n = len(rep.samples)
        idx = list(range(k, k + n))
        G = scores[g]
        for t in reversed(range(n)):
            if t < n - 1:
                G = (1 - lam) * v_now[idx[t + 1]] + lam * G
            rep.samples[t].target = float(G)
            rep.samples[t].advantage = float(adv[g]) if cfg.deck_adv != "td" else 0.0
        k += n
    if cfg.deck_adv in ("td", "mix"):
        # 선택별 이점: 그 선택이 덱 가치 D 를 얼마나 올렸나 (그룹 안 표준편차로 정규화)
        pol = [(s, v_after[i] - v_now[i]) for i, s in enumerate(samples) if s.action >= 0 and v_after[i] is not None]
        if pol:
            sd = float(np.std([d for _, d in pol])) + 1e-6
            for s, d in pol:
                s.advantage += float(d) / sd
    info = {"target": ep.target_raw, "boss": ep.is_boss, "screens": len(ep.screens), "decisions": len(samples),
            "score": float(np.mean(scores)), "win": float(np.mean([w for g in wins for w in wins[g]])),
            "score_spread": float(np.std(scores)), "match_samples": match}
    return samples, info


@torch.no_grad()
def _values(builder, states: list[RunState], chunk: int = 512) -> list[float]:
    out = []
    for i in range(0, len(states), chunk):
        out += builder.values(states[i: i + chunk]).float().cpu().tolist()
    return out


def deck_update(builder, optim, samples: list[DeckSample], epochs: int = 2, clip: float = 0.2,
                ent_coef: float = 0.01, value_coef: float = 0.5, minibatch: int = 128, all_coef: float = 1.0,
                temp: float = 1.0, ent_floor: float = 0.0, ent_floor_coef: float = 1.0) -> dict:
    """선택은 PPO 클리핑(이점 0 인 표본은 빼고) + 선택지 전부 평가, 가치 머리는 TD(λ) 목표 회귀.

    선택지 전부 평가: 선택지마다 "그걸 골랐다면"의 D 가 있으면, 확률을 D 가 높은 선택지 쪽으로 옮긴다
    (손실 = −Σ π(o) · (D(o) − Σ π D) / 척도, D 에는 기울기를 흘리지 않는다). 성장 단계마다 모든 선택지가 신호가 된다.

    temp: 재생에서 뽑을 때 쓴 온도. PPO 비율은 같은 온도의 확률로 잰다 (뽑은 분포 = 이전 확률).
    엔트로피 하한: 선택지 수로 나눈 엔트로피 H/log K 가 ent_floor 아래로 내려가면 ent_floor_coef × (ent_floor − H/log K)
    벌점. 실제 런에서 쓰는 온도 1 의 확률표가 한 선택지로 굳지 않게 한다.
    """
    if not samples:
        return {}
    from ..batching import module_device  # noqa: PLC0415

    dev = module_device(builder)
    stats = {"deck_loss": 0.0, "deck_value_loss": 0.0, "deck_entropy": 0.0, "deck_all_loss": 0.0,
             "deck_entropy_norm": 0.0, "deck_floor_loss": 0.0}
    n = 0
    # 선택지 전부 평가의 척도: 화면 안에서 D 가 선택지끼리 얼마나 벌어지나의 평균 (전체 표본에서 한 번)
    spreads = [float(np.std(s.opt_values)) for s in samples if s.opt_values and len(s.opt_values) > 1]
    q_scale = (float(np.mean(spreads)) if spreads else 1.0) + 1e-6
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
                lp = torch.log_softmax(logits[idx] / temp, -1).gather(1, a.unsqueeze(1)).squeeze(1)
                old = torch.tensor([chunk[i].logp for i in act], device=dev)
                adv = torch.tensor([chunk[i].advantage for i in act], dtype=torch.float, device=dev)
                ratio = (lp - old).exp()
                p_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                ent = -(lp_all.exp() * lp_all.clamp(min=-30)).sum(-1).mean()
            all_loss = torch.zeros((), device=dev)
            qs = [i for i, s in enumerate(chunk) if s.action >= 0 and s.opt_values and len(s.opt_values) > 1]
            if qs and all_coef > 0:
                terms = []
                for i in qs:
                    q = torch.tensor(chunk[i].opt_values, dtype=torch.float, device=dev)
                    p = torch.softmax(logits[i, : len(q)], -1)
                    a_q = (q - (p.detach() * q).sum()) / q_scale  # 이 화면에서 각 선택지가 평균보다 D 를 얼마나 올리나
                    terms.append(-(p * a_q).sum())
                all_loss = torch.stack(terms).mean()
            floor_loss = torch.zeros((), device=dev)
            ent_norm = torch.zeros((), device=dev)
            multi = [i for i, s in enumerate(chunk) if s.action >= 0 and len(s.options) > 1]
            if multi:
                hn = []
                for i in multi:
                    lpi = torch.log_softmax(logits[i, : len(chunk[i].options)], -1)
                    hn.append(-(lpi.exp() * lpi.clamp(min=-30)).sum() / float(np.log(len(chunk[i].options))))
                hn = torch.stack(hn)
                ent_norm = hn.mean()
                if ent_floor > 0:
                    floor_loss = torch.relu(ent_floor - hn).mean()
            loss = p_loss - ent_coef * ent + value_coef * v_loss + all_coef * all_loss + ent_floor_coef * floor_loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(builder.parameters(), 1.0)
            optim.step()
            stats["deck_loss"] += float(p_loss.detach())
            stats["deck_value_loss"] += float(v_loss.detach())
            stats["deck_entropy"] += float(ent.detach())
            stats["deck_all_loss"] += float(all_loss.detach())
            stats["deck_entropy_norm"] += float(ent_norm.detach())
            stats["deck_floor_loss"] += float(floor_loss.detach())
            n += 1
    out = {k: v / max(1, n) for k, v in stats.items()}
    out["deck_samples"] = len(samples)
    return out


# ---- 실제 런에서 합치기 ----

def deck_targets(run, weights: dict, state: RunState, top: int = 2) -> list[tuple[int, float, int]]:
    """대비할 적들 [(encounter 인덱스, 가중치, 남은 층수)]. 가중치 합 = 1.

    - 이번 런 보스: 항상. 남은 층수 = 보스 줄 − 지금 줄.
    - 이 막의 사망 상위 일반·엘리트 적 top 개: 그 적이 나올 수 있는 종류(일반/엘리트)의 노드가 앞으로 도달 가능한
      줄마다 나눠 넣는다. 줄 r 의 몫 ∝ 그 줄의 도달 가능한 노드 중 그 종류의 비율, 남은 층수 = r − 지금 줄.
      이번 막에 그 종류 노드가 더 없으면 뺀다.
    """
    from ..schema import NODE_BOSS, NODE_ELITE, NODE_NORMAL  # noqa: PLC0415

    table = weights.get(state.act, {})
    boss_raw = run.boss_raw()
    m = state.map
    cur = m.rows[m.current] if (m is not None and m.current is not None) else -1
    boss_rows = [m.rows[i] for i, ty in enumerate(m.node_types) if ty == NODE_BOSS] if m is not None else []
    boss_h = (max(boss_rows) - cur) if boss_rows else 0
    boss_w = max(1.0, sum(w for (e, b), w in table.items() if b and e == boss_raw))
    out = [(run.encounter_vocab(boss_raw), boss_w, boss_h)]
    reach = m.reachable() if m is not None else []
    others = sorted(((w, e) for (e, b), w in table.items() if not b and w > 0), reverse=True)[:top]
    for w, e in others:
        ty = NODE_ELITE if getattr(run, "encounter_kind", lambda _e: "Normal")(e) == "Elite" else NODE_NORMAL
        rows: dict[int, list[int]] = {}
        for i, ok in enumerate(reach):
            if ok:
                rows.setdefault(m.rows[i], []).append(m.node_types[i])
        share = {r: sum(t == ty for t in types) / len(types) for r, types in rows.items()}
        share = {r: s for r, s in share.items() if s > 0}
        tot = sum(share.values())
        for r, s in share.items():
            out.append((run.encounter_vocab(e), w * s / tot, r - cur))
    tot = sum(w for _, w, _ in out)
    return [(t, w / tot, h) for t, w, h in out]


COMPARE_HP_FRAC = 0.85  # 실제 런에서 선택지를 비교할 때 가정하는 "그 적과 싸울 때의 HP" 비율


def _builder_states(state: RunState, targets, deck=None):
    deck = state.deck if deck is None else deck
    hp = max(1, int(round(state.max_hp * COMPARE_HP_FRAC)))
    return [builder_state(deck, [r for r, _ in state.relics], state.gold, state.max_hp, state.act, state.floor, t,
                          horizon=h, hp=hp) for t, _, h in targets]


def fight_value_samples(run, start: RunState, enc_raw: int, outcomes: list[tuple[bool, float, float]]) -> list:
    """실제로 일어난 전투들 → 가치 머리의 남은 층수 0 정답. outcomes = [(이겼나, 남은 HP 비율, 준 피해 비율)].

    점수는 재생과 같다: 이기면 1 + 남은 HP 비율, 지면 준 피해 비율.
    """
    st = builder_state(start.deck, [r for r, _ in start.relics], start.gold, start.max_hp, start.act, start.floor,
                       run.encounter_vocab(enc_raw), horizon=0, hp=start.hp)
    return [DeckSample(st, [], -1, target=(1.0 + hp) if won else dmg) for won, hp, dmg in outcomes]


@torch.no_grad()
def deck_logprob(builder, run, state: RunState, opts: list[ChoiceOption], targets) -> torch.Tensor:
    """대비할 적들(과 그 적을 만날 줄들)에 대한 덱 빌딩 모듈 log 확률의 가중 평균 [len(opts)]."""
    sts = _builder_states(state, targets)
    logits, _ = builder(sts, [opts] * len(sts))
    lp = torch.log_softmax(logits[:, : len(opts)], -1)
    w = torch.tensor([w for _, w, _ in targets], dtype=lp.dtype, device=lp.device).unsqueeze(1)
    return (lp * w).sum(0).cpu()


@torch.no_grad()
def upgrade_gain(builder, run, state: RunState, targets) -> float:
    """D(가장 좋은 강화 뒤 덱) − D(지금 덱), 대비할 적들(과 만날 줄들) 가중 평균."""
    raw = run.deck_raw()
    idx = [i for i, (c, u) in enumerate(raw) if not u and run.card_upgradable(c)]
    if not idx or len(raw) != len(state.deck):
        return 0.0
    decks = [state.deck] + [[run.raw_card(c, True) if j == i else state.deck[j] for j, (c, _) in enumerate(raw)]
                            for i in idx]
    sts = [s for d in decks for s in _builder_states(state, targets, deck=d)]
    v = builder.values(sts).view(len(decks), len(targets))  # [덱, 대상]
    w = torch.tensor([w for _, w, _ in targets], dtype=v.dtype, device=v.device)
    per_deck = (v * w).sum(1)
    return float(per_deck[1:].max() - per_deck[0])

@torch.no_grad()
def upgrade_gains(builder, run, state: RunState, targets, opts: list[ChoiceOption]) -> list[float]:
    """휴식 장소 선택지마다 덱 가치 상승: 강화 선택지가 카드 하나를 가리키면 그 카드를 강화한 덱의 D − 지금 덱의 D.
    카드를 가리키지 않는 강화(뒤에 덱 카드 고르기가 이어짐)는 가장 좋은 강화의 상승, 쉬기 등은 0."""
    raw = [tuple(x) for x in run.deck_raw()]
    if len(raw) != len(state.deck):
        return [0.0] * len(opts)
    upg = [i for i, (c, u) in enumerate(raw) if not u and run.card_upgradable(c)]
    decks = [state.deck] + [[run.raw_card(c, True) if j == i else state.deck[j] for j, (c, _) in enumerate(raw)]
                            for i in upg]
    sts = [s for d in decks for s in _builder_states(state, targets, deck=d)]
    v = builder.values(sts).float().cpu().view(len(decks), len(targets))
    w = torch.tensor([w for _, w, _ in targets], dtype=v.dtype)
    per = (v * w).sum(1)
    gain_of = {i: float(per[k + 1] - per[0]) for k, i in enumerate(upg)}
    best = max(gain_of.values()) if gain_of else 0.0
    out = []
    for o in opts:
        if o.kind != "rest.smith":
            out.append(0.0)
            continue
        i = o.deck_index
        if (i is None or i < 0) and o.card is not None:  # 카드만 있으면 덱에서 같은 (id, 강화 안 됨) 카드를 찾는다
            key = (run.card_raw(o.card.id), False)
            i = next((j for j in upg if raw[j] == key), -1)
        out.append(gain_of.get(i, 0.0) if i is not None and i >= 0 else best)
    return out


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


@dataclass
class MixStep:
    """β·γ 신경망 학습용: 실제 런에서 합쳐 고른 화면 하나."""
    x: torch.Tensor  # MixNet 입력
    is_rest: bool
    short: torch.Tensor  # 단기 모듈 logit [K]
    extra: torch.Tensor  # 덱 화면: 덱 빌딩 log 확률 [K] / 휴식: 강화 칸에만 강화 이득 (나머지 0) [K]
    base_log: float  # 그 런의 막별 log β (또는 log γ), 진화 전략 잡음 포함
    action: int
    advantage: float = 0.0  # 런이 끝난 뒤 오른 층 수로 채운다


def mixed_logits(mix_net, steps: list[MixStep]) -> list[torch.Tensor]:
    """MixStep 들의 합친 logit (MixNet 에 기울기가 흐른다)."""
    if not steps:
        return []
    delta = mix_net(torch.stack([s.x for s in steps]).to(next(mix_net.parameters()).device)).cpu()
    out = []
    for s, dl in zip(steps, delta):
        coef = torch.exp(s.base_log + (dl[1] if s.is_rest else dl[0]))
        out.append(s.short + coef * s.extra)
    return out


def combined_pick(agent, run, state: RunState, opts: list[ChoiceOption], weights: dict, cfg, log_beta: float,
                  log_gamma: float, greedy: bool) -> tuple[int, MixStep | None]:
    """단기 모듈 logit + β × 덱 빌딩 log 확률. 휴식 장소(쉬기·강화)는 강화에 γ × 덱 가치 상승을 더한다.

    β, γ = exp(막별 기본값 + β·γ 신경망 보정). 고른 화면을 MixStep 으로 돌려줘 β·γ 신경망을 학습시킨다.
    """
    from .choice import choice_logits  # noqa: PLC0415

    short = choice_logits(agent, [state], [opts])[0, : len(opts)].float().cpu()
    targets = deck_targets(run, weights, state, cfg.deck_targets_top)
    kinds = [o.kind for o in opts]
    is_rest = "rest.smith" in kinds and "rest.heal" in kinds
    if is_rest:
        # 강화 선택지마다 그 카드를 강화했을 때의 덱 가치 상승 (예전에는 첫 강화 선택지에만 최고 상승을 더했다)
        extra = torch.tensor(upgrade_gains(agent.deck, run, state, targets, opts), dtype=torch.float)
    elif getattr(cfg, "deck_pick_mode", "head") == "value":
        # 가치만으로: log_softmax(덱 가치 / τ). 선택 머리의 log 확률과 같은 자리에 들어간다
        d = deck_value_scores(agent.deck, run, state, opts, targets, np.random.default_rng(),
                              getattr(cfg, "deck_value_samples", 4))
        extra = torch.log_softmax(d / getattr(cfg, "deck_value_tau", 0.01), -1).float()
    else:
        extra = deck_logprob(agent.deck, run, state, opts, targets).float()
    mix_net = getattr(agent, "mix_net", None)
    base = log_gamma if is_rest else log_beta
    if mix_net is not None:
        with torch.no_grad():
            x = mix_net.features(agent.run_vec(state), state.act, is_rest)
            step = MixStep(x, is_rest, short, extra, float(base), -1)
            logits = mixed_logits(mix_net, [step])[0]
    else:
        step = None
        logits = short + float(np.exp(base)) * extra
    i = int(logits.argmax()) if greedy else int(torch.distributions.Categorical(logits=logits).sample())
    if step is not None:
        step.action = i
    return i, step


# 가치만으로 고르기 (deck_pick_mode = "value"): 선택지마다 "고른 뒤 덱" 의 덱 가치 기대값으로 점수를 낸다
STOCHASTIC_KINDS = ("event.option", "neow.relic", "bundle.pick", "shop.remove", "shop.relic", "reward.relic")
TRANSFORM_POOL_SAMPLE = 16


@torch.no_grad()
def _deck_values(builder, run, state: RunState, targets, raws: list[list]) -> torch.Tensor:
    """원시 덱 [(카드, 강화)] 여러 개의 덱 가치 (대상 가중 평균) [len(raws)]."""
    if not raws:
        return torch.zeros(0)
    sts = [s for r in raws for s in _builder_states(state, targets, deck=[run.raw_card(c, u) for c, u in r])]
    v = builder.values(sts).float().cpu().view(len(raws), len(targets))
    w = torch.tensor([w for _, w, _ in targets], dtype=v.dtype)
    return (v * w).sum(1)


def _follow_value(builder, run, state, targets, raw: list, follow, rng) -> float:
    """이벤트 등의 뒤에 덱에서 카드를 고르는 화면이 이어질 때: 가장 좋은 한 장을 골랐을 때의 가치."""
    if follow is None:
        return float(_deck_values(builder, run, state, targets, [raw])[0])
    kind, arg = follow
    if kind == "offer":
        cands = [raw + [tuple(x)] for x in arg] or [raw]
    elif kind == "remove":
        cands = [raw[:i] + raw[i + 1:] for i in range(len(raw))] or [raw]
    elif kind == "upgrade":
        cands = [raw[:i] + [(c, True)] + raw[i + 1:] for i, (c, u) in enumerate(raw)
                 if not u and run.card_upgradable(c)] or [raw]
    elif kind == "duplicate":
        cands = [raw + [raw[i]] for i in range(len(raw))] or [raw]
    elif kind == "transform":
        return max(_transform_value(builder, run, state, targets, raw, i, rng) for i in range(len(raw))) if raw             else float(_deck_values(builder, run, state, targets, [raw])[0])
    else:
        cands = [raw]
    return float(_deck_values(builder, run, state, targets, cands).max())


def _transform_value(builder, run, state, targets, raw: list, i: int, rng) -> float:
    """i 번째 카드를 변환: 결과 후보(직업·무색 카드)에서 뽑은 TRANSFORM_POOL_SAMPLE 장의 덱 가치 평균."""
    pool = run.transform_pool()
    if not pool:
        return float(_deck_values(builder, run, state, targets, [raw])[0])
    pick = rng.choice(len(pool), min(TRANSFORM_POOL_SAMPLE, len(pool)), replace=False)
    cands = [raw[:i] + [(int(pool[int(j)]), False)] + raw[i + 1:] for j in pick]
    return float(_deck_values(builder, run, state, targets, cands).mean())


def deck_value_scores(builder, run, state: RunState, opts: list[ChoiceOption], targets, rng,
                      samples: int = 4) -> torch.Tensor:
    """선택지마다 고른 뒤 덱의 덱 가치 기대값 [len(opts)].

    결과가 정해진 선택(카드 집기·덱 카드 제거·강화·복제)은 그 덱. 변환은 결과 후보의 평균.
    결과가 무작위일 수 있는 선택(이벤트 등)은 숨은 난수를 새로 뽑은 복제로 samples 번 골라 본 평균
    (그 뒤에 덱 카드 고르기가 이어지면 가장 좋은 한 장). 그 밖(건너뛰기 등)은 지금 덱.
    """
    raw = [tuple(x) for x in run.deck_raw()]
    base = None
    out = []
    for o in opts:
        k = o.kind
        if k in ("card.take", "shop.card", "select.offer") and o.card is not None:
            out.append(float(_deck_values(builder, run, state, targets, [raw + [(run.card_raw(o.card.id),
                                                                             bool(o.card.upgraded))]])[0]))
        elif k.startswith("select.") and 0 <= o.deck_index < len(raw):
            i = o.deck_index
            if k == "select.remove":
                out.append(float(_deck_values(builder, run, state, targets, [raw[:i] + raw[i + 1:]])[0]))
            elif k == "select.upgrade":
                out.append(float(_deck_values(builder, run, state, targets, [raw[:i] + [(raw[i][0], True)] + raw[i + 1:]])[0]))
            elif k == "select.duplicate":
                out.append(float(_deck_values(builder, run, state, targets, [raw + [raw[i]]])[0]))
            elif k in ("select.transform", "select.transform_to"):
                out.append(_transform_value(builder, run, state, targets, raw, i, rng))
            else:
                base = base if base is not None else float(_deck_values(builder, run, state, targets, [raw])[0])
                out.append(base)
        elif k in STOCHASTIC_KINDS:
            vals = []
            for _ in range(samples):
                try:
                    m = measure(run, o, rng, exact=False)
                except Exception:  # noqa: BLE001  복제로 못 재면 지금 덱으로 본다
                    continue
                r2 = list(raw)
                for x in m.delta.remove:
                    if tuple(x) in r2:
                        r2.remove(tuple(x))
                r2 += [tuple(x) for x in m.delta.add]
                vals.append(_follow_value(builder, run, state, targets, r2, m.follow, rng))
            if not vals:
                base = base if base is not None else float(_deck_values(builder, run, state, targets, [raw])[0])
                vals = [base]
            out.append(float(np.mean(vals)))
        else:
            base = base if base is not None else float(_deck_values(builder, run, state, targets, [raw])[0])
            out.append(base)
    return torch.tensor(out)


def mix_update(mix_net, optim, steps: list[MixStep], clip: float = 1.0) -> dict:
    """정책 경사: 오른 층 수가 평균보다 높았던 런의 선택 확률을 올리는 쪽으로 β·γ 보정을 옮긴다.

    손실 = −mean(advantage × log π_합친(고른 것)). β 에 대한 기울기는 (고른 선택지의 덱 빌딩 log 확률 − 기대값):
    덱 빌딩 모듈 말을 따른 런이 잘됐으면 그런 상황의 β 가 커진다.
    """
    steps = [s for s in steps if s.advantage != 0.0 and s.action >= 0]
    if not steps:
        return {}
    logits = mixed_logits(mix_net, steps)
    lp = torch.stack([torch.log_softmax(lg, -1)[s.action] for lg, s in zip(logits, steps)])
    adv = torch.tensor([s.advantage for s in steps])
    loss = -(adv * lp).mean()
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mix_net.parameters(), clip)
    optim.step()
    with torch.no_grad():
        d = mix_net(torch.stack([s.x for s in steps]).to(next(mix_net.parameters()).device)).cpu()
    rest = torch.tensor([s.is_rest for s in steps])
    return {"mix_loss": float(loss.detach()), "mix_steps": len(steps),
            "mix_dbeta": float(d[~rest, 0].mean()) if (~rest).any() else 0.0,
            "mix_dbeta_sd": float(d[~rest, 0].std()) if int((~rest).sum()) > 1 else 0.0,
            "mix_dgamma": float(d[rest, 1].mean()) if rest.any() else 0.0}
