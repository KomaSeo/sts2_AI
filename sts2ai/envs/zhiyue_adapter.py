"""zhiyue/sts2-rl-agent 시뮬레이터(순수 Python StS2 전투) 어댑터: 전투만 따로 학습할 때 쓴다.

그 저장소의 코드는 이 저장소에 넣지 않는다 (라이선스 파일이 없고, 게임 디컴파일 소스를 포함한다).
사용자가 따로 받아 둔 체크아웃 경로를 받아서 import 만 한다.

    git clone https://github.com/zhiyue/sts2-rl-agent ../sts2-rl-agent
    python scripts/train_combat.py --env zhiyue --zhiyue-path ../sts2-rl-agent

그 시뮬레이터의 특이점
- 카드 메타데이터를 처음 만들 때 그 저장소의 docs/, decompiled/ 를 '현재 폴더 기준' 경로로 읽고
  lru_cache 에 담는다. 그래서 초기화할 때 한 번 그 폴더로 들어가 모든 카드를 만들어 캐시를 채운다.
- 모듈 import 순서에 따라 순환 import 가 난다. gym_env.combat_env 를 먼저 import 한다.

관측 (우리 CombatObs 토큰, 행동 i ↔ 토큰 1+i)
- 평소: GLOBAL, 손패 카드(행동 0..n-1), 턴 종료(행동 n), 포션 칸(행동 n+1+s), 더미·버린 더미·소멸 더미,
  플레이어 파워, 유물, 적(살아 있는 적만) + 의도(한 행동에 여러 개면 전부) + 적 파워.
- 카드 효과로 선택 화면이 뜨면: GLOBAL, 선택지 카드(행동 0..k-1), 확인/건너뛰기(행동 k), 나머지는 같다.
  선택지는 MAX_ACTIONS - 1 개까지만 고를 수 있다 (그보다 많으면 앞에서부터).
- 합법 행동은 그 시뮬레이터의 get_action_mask 를 그대로 옮긴다. 고른 타깃이 그 행동에 합법이 아니면
  합법인 첫 타깃으로 바꾼다 (무작위 타깃·아군 타깃 카드 등).
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from pathlib import Path

import numpy as np

from .. import schema as S
from ..schema import CombatObs, Vocab


@contextlib.contextmanager
def _cwd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class _Sim:
    """그 시뮬레이터의 모듈 묶음과 우리 id 표."""

    def __init__(self, repo: str | Path):
        self.repo = Path(repo).resolve()
        if not (self.repo / "sts2_env").is_dir():
            raise FileNotFoundError(f"{self.repo} 에 sts2_env 가 없습니다 (zhiyue/sts2-rl-agent 체크아웃 경로)")
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        with _cwd(self.repo):
            import sts2_env.gym_env.combat_env  # noqa: F401  (순환 import 회피: 이 모듈을 먼저)
            from sts2_env.cards.factory import create_card
            from sts2_env.core import constants as C
            from sts2_env.core.card_pools import IRONCLAD_CARD_POOL
            from sts2_env.core.combat import CombatState
            from sts2_env.core.enums import CardId, IntentType, PowerId, TargetType
            from sts2_env.core.rng import Rng
            from sts2_env.encounters import act1, act2, act3, act4
            from sts2_env.gym_env.action_space import get_action_mask
            from sts2_env.potions.base import all_potion_models, create_potion
            from sts2_env.relics.base import RelicId

            for cid in CardId:  # 카드 메타데이터 캐시 채우기 (docs/, decompiled/ 를 상대 경로로 읽는다)
                for up in (False, True):
                    with contextlib.suppress(Exception):
                        create_card(cid, upgraded=up)
        self.C, self.CombatState, self.Rng = C, CombatState, Rng
        self.create_card, self.create_potion, self.get_action_mask = create_card, create_potion, get_action_mask
        self.TargetType = TargetType

        self.card_ids = {cid: i + 1 for i, cid in enumerate(CardId)}
        self.cards_by_id = {i: cid for cid, i in self.card_ids.items()}
        self.power_ids = {pid: i + 1 for i, pid in enumerate(PowerId)}
        self.intent_ids = {it: i + 1 for i, it in enumerate(IntentType)}
        self.relic_ids = {rid: i + 1 for i, rid in enumerate(RelicId)}
        models = sorted(all_potion_models(), key=lambda m: m.potion_id)
        self.potion_ids = {m.potion_id: i + 1 for i, m in enumerate(models)}
        self.potions_by_id = {i: pid for pid, i in self.potion_ids.items()}
        # 몬스터 id 는 문자열이다. monster_id="..." 리터럴과, 몬스터 모듈의 *_ID 문자열 상수를 모아
        # 이름순으로 번호를 매긴다 (모르는 것은 0).
        import importlib

        names = set()
        for f in sorted((self.repo / "sts2_env" / "monsters").glob("*.py")):
            names.update(re.findall(r'monster_id\s*=\s*"([A-Z0-9_]+)"', f.read_text()))
            with contextlib.suppress(Exception):
                mod = importlib.import_module(f"sts2_env.monsters.{f.stem}")
                names.update(v for k, v in vars(mod).items() if k.endswith("_ID") and isinstance(v, str))
        self.monster_ids = {n: i + 1 for i, n in enumerate(sorted(names))}
        # 조우: 막마다 약한/일반/엘리트/보스. (조우 id, 막) 은 1:1
        self.encounters: list[tuple[str, int, object]] = []  # (이름, 막, setup)
        for act, mod in enumerate((act1, act2, act3, act4)):
            for kind in ("WEAK_ENCOUNTERS", "NORMAL_ENCOUNTERS", "ELITE_ENCOUNTERS", "BOSS_ENCOUNTERS"):
                for setup in getattr(mod, kind, []):
                    self.encounters.append((getattr(setup, "__name__", str(setup)), act, setup))
        self.ironclad_pool = [self.card_ids[c] for c in IRONCLAD_CARD_POOL]
        self.vocab = Vocab(
            n_cards=len(self.card_ids) + 1,
            n_relics=len(self.relic_ids) + 1,
            n_potions=len(self.potion_ids) + 1,
            n_buffs=len(self.power_ids) + 1,
            n_enemies=len(self.monster_ids) + 1,
            n_intents=len(self.intent_ids) + 1,
            n_encounters=len(self.encounters) + 1,
            n_enchants=16,
            n_acts=4,
        )


class ZhiyueCombatSim:
    """CombatSim 프로토콜 구현."""

    def __init__(self, sim: _Sim, combat, encounter: int):
        self.s, self.c, self.encounter = sim, combat, encounter
        self._actions: list = []  # 우리 행동 i → 그 시뮬레이터 행동 번호 후보들 (observe 때 채움)
        self._mask = None

    # ---- 상태 ----
    @property
    def done(self) -> bool:
        return bool(self.c.is_over)

    @property
    def won(self) -> bool:
        return bool(self.c.is_over and self.c.player_won)

    def hp(self) -> int:
        return max(0, int(self.c.player.current_hp))

    def max_hp(self) -> int:
        return int(self.c.player.max_hp)

    def potions(self) -> list[int]:
        return [self.s.potion_ids.get(p.potion_id, 0) if p is not None else 0 for p in self.c.potions]

    def enemy_damage_frac(self) -> float:
        total = sum(max(0, e.max_hp) for e in self.c.enemies)
        left = sum(max(0, e.current_hp) for e in self.c.enemies if e.is_alive)
        return 1.0 - left / max(1, total)

    def close(self) -> None:
        self.c = None

    # ---- 관측 ----
    def observe(self) -> CombatObs:
        s, c, C = self.s, self.c, self.s.C
        mask = s.get_action_mask(c)
        self._mask = mask
        types, ids, owners, feats = [], [], [], []
        legal = np.zeros(S.MAX_ACTIONS, dtype=bool)
        needs_target = np.zeros(S.MAX_ACTIONS, dtype=bool)
        actions: list[tuple[list[int], list[int]]] = []  # (타깃 없는 후보, 적 j 별 후보[j] 또는 -1)

        def push(t, i=0, owner=0, **ch):
            row = np.zeros(S.N_COMBAT_FEATURES, dtype=np.float32)
            for k, v in ch.items():
                row[getattr(S, k)] = v
            types.append(t)
            ids.append(i)
            owners.append(owner)
            feats.append(row)

        def card_feats(card) -> dict:
            return dict(F_UPGRADED=float(card.upgraded), F_COST=max(0, card.cost) / 5,
                        F_ENCHANT=min(1.0, len(card.enchantments) / 3))

        p = c.player
        push(S.TYPE_GLOBAL, self.encounter, F_HP=p.current_hp / max(1, p.max_hp), F_HP_ABS=p.current_hp / 50,
             F_MAX_HP=p.max_hp / 100, F_BLOCK=p.block / 50, F_ENERGY=c.energy / 10,
             F_DRAW_N=len(c.draw_pile) / 20, F_DISCARD_N=len(c.discard_pile) / 20,
             F_EXHAUST_N=len(c.exhaust_pile) / 20)
        n_enemy = min(len(c.enemies), C.MAX_ENEMIES)

        def targeted(base: int) -> list[int]:
            return [base + j if mask[base + j] else -1 for j in range(n_enemy)]

        pc = c.pending_choice
        if pc is not None:
            k = min(pc.num_options, S.MAX_ACTIONS - 1)
            for i in range(k):
                opt = pc.options[i]
                push(S.TYPE_SELECT, s.card_ids.get(opt.card.card_id, 0), **card_feats(opt.card),
                     F_MAGNITUDE=float(i in pc.selected_indices))
                legal[i] = bool(mask[1 + i])
                actions.append(([1 + i], []))
            push(S.TYPE_DECLINE)
            legal[k] = bool(mask[C.ACTION_END_TURN])
            actions.append(([C.ACTION_END_TURN], []))
        else:
            hand = c.hand[: C.MAX_HAND_SIZE]
            for i, card in enumerate(hand):
                push(S.TYPE_HAND, s.card_ids.get(card.card_id, 0), **card_feats(card))
                plain = [1 + i] if mask[1 + i] else []
                tgt = targeted(1 + C.MAX_HAND_SIZE + i * C.MAX_ENEMIES)
                legal[i] = bool(plain) or any(t >= 0 for t in tgt)
                tt = card.target_type_for(c.primary_player)
                needs_target[i] = not plain and tt == s.TargetType.ANY_ENEMY and any(t >= 0 for t in tgt)
                actions.append((plain, tgt))
            n = len(hand)
            push(S.TYPE_END_TURN)
            legal[n] = bool(mask[C.ACTION_END_TURN])
            actions.append(([C.ACTION_END_TURN], []))
            for slot, pot in enumerate(c.potions[: C.MAX_POTION_SLOTS]):
                a = n + 1 + slot
                if a >= S.MAX_ACTIONS:
                    break
                push(S.TYPE_POTION, s.potion_ids.get(pot.potion_id, 0) if pot is not None else 0,
                     F_MAGNITUDE=float(pot is not None))
                base = C.POTION_ACTION_START + slot * C.POTION_TARGET_OPTIONS
                plain = [base] if mask[base] else []
                tgt = targeted(base + 1)
                legal[a] = bool(plain) or any(t >= 0 for t in tgt)
                needs_target[a] = not plain and any(t >= 0 for t in tgt)
                actions.append((plain, tgt))
        self._actions = actions

        for pile, t in ((c.draw_pile, S.TYPE_DRAW), (c.discard_pile, S.TYPE_DISCARD), (c.exhaust_pile, S.TYPE_EXHAUST)):
            for card in sorted(pile, key=lambda x: (s.card_ids.get(x.card_id, 0), x.upgraded)):  # 순서는 숨긴다
                push(t, s.card_ids.get(card.card_id, 0), **card_feats(card))
        for pid, pw in p.powers.items():
            push(S.TYPE_BUFF, s.power_ids.get(pid, 0), F_MAGNITUDE=pw.amount / 20)
        for r in c.relics:
            push(S.TYPE_RELIC, s.relic_ids.get(getattr(r, "relic_id", None), 0))
        enemy_tokens, enemy_slots = [], []
        for slot, e in enumerate(c.enemies[: C.MAX_ENEMIES]):
            if not e.is_alive:
                continue
            enemy_tokens.append(len(types))
            enemy_slots.append(slot)
            push(S.TYPE_ENEMY, s.monster_ids.get(e.monster_id, 0), owner=slot + 1, F_ENEMY_HP=e.current_hp / 50,
                 F_ENEMY_HP_FRAC=e.current_hp / max(1, e.max_hp), F_ENEMY_BLOCK=e.block / 50)
            ai = c.enemy_ais.get(e.combat_id)
            if ai is not None and ai.current_move is not None:
                for it in ai.current_move.intents:
                    push(S.TYPE_INTENT, s.intent_ids.get(it.intent_type, 0), owner=slot + 1,
                         F_INTENT_MAG=it.damage / 30, F_MAGNITUDE=it.hits / 5)
            for pid, pw in e.powers.items():
                push(S.TYPE_ENEMY_BUFF, s.power_ids.get(pid, 0), owner=slot + 1, F_MAGNITUDE=pw.amount / 20)
        return CombatObs(
            types=np.array(types, dtype=np.int64),
            ids=np.array(ids, dtype=np.int64),
            owners=np.array(owners, dtype=np.int64),
            feats=np.stack(feats),
            legal=legal,
            needs_target=needs_target,
            enemy_tokens=np.array(enemy_tokens, dtype=np.int64),
            enemy_slots=np.array(enemy_slots, dtype=np.int64),
            enemy_alive=np.ones(len(enemy_tokens), dtype=bool),
        )

    # ---- 사람이 읽는 설명 (scripts/show_combat.py) ----
    @staticmethod
    def _card(card) -> str:
        return card.card_id.name.lower().removesuffix("_card") + ("+" if card.upgraded else "")

    def _enemy(self, e) -> str:
        return (e.monster_id or "?").lower()

    def _powers(self, cr) -> str:
        return ", ".join(f"{pid.name.lower()} {pw.amount}" for pid, pw in cr.powers.items())

    def describe_state(self) -> str:
        c, p = self.c, self.c.player
        lines = [f"플레이어 HP {p.current_hp}/{p.max_hp}  방어도 {p.block}  에너지 {c.energy}"
                 + (f"  [{self._powers(p)}]" if p.powers else "")]
        for e in c.enemies:
            if not e.is_alive:
                continue
            ai = c.enemy_ais.get(e.combat_id)
            intents = []
            if ai is not None and ai.current_move is not None:
                for it in ai.current_move.intents:
                    t = it.intent_type.name.lower()
                    intents.append(f"{t} {it.damage}x{it.hits}" if it.damage else t)
            lines.append(f"  적 {self._enemy(e)}: HP {e.current_hp}/{e.max_hp}  방어도 {e.block}  의도 {' + '.join(intents) or '-'}"
                         + (f"  [{self._powers(e)}]" if e.powers else ""))
        lines.append("  손패: " + ", ".join(self._card(x) for x in c.hand))
        pots = [p_.potion_id.lower() for p_ in c.potions if p_ is not None]
        if pots:
            lines.append("  포션: " + ", ".join(pots))
        return "\n".join(lines)

    def describe_action(self, action: int, target_slot: int = -1) -> str:
        """observe() 직후, step() 전에 부른다."""
        c, C = self.c, self.s.C
        a = self._their_action(action, target_slot)
        tgt = ""
        if c.pending_choice is not None:
            if a == C.ACTION_END_TURN:
                return "선택 확인/건너뛰기"
            prompt = (c.pending_choice.prompt or "").strip()
            return f"선택{f'({prompt})' if prompt else ''}: {self._card(c.pending_choice.options[a - 1].card)}"
        if a == C.ACTION_END_TURN:
            return "턴 종료"
        if a >= C.POTION_ACTION_START:
            slot, t = divmod(a - C.POTION_ACTION_START, C.POTION_TARGET_OPTIONS)
            if t:
                tgt = f" → {self._enemy(c.enemies[t - 1])}"
            return f"포션 {c.potions[slot].potion_id.lower()}{tgt}"
        if a <= C.MAX_HAND_SIZE:
            return f"{self._card(c.hand[a - 1])}"
        hand_idx, enemy = divmod(a - 1 - C.MAX_HAND_SIZE, C.MAX_ENEMIES)
        return f"{self._card(c.hand[hand_idx])} → {self._enemy(c.enemies[enemy])}"

    def snapshot(self) -> dict:
        """행동 전후 비교용 수치."""
        p = self.c.player
        return {"hp": p.current_hp, "block": p.block, "max_hp": p.max_hp,
                "enemies": {e.combat_id: (self._enemy(e), max(0, e.current_hp), e.block) for e in self.c.enemies}}

    @staticmethod
    def describe_change(before: dict, after: dict) -> str:
        out = []
        for key, label in (("hp", "HP"), ("block", "방어도"), ("max_hp", "최대 HP")):
            if before[key] != after[key]:
                out.append(f"{label} {before[key]}→{after[key]}")
        for cid, (name, hp, blk) in after["enemies"].items():
            if cid not in before["enemies"]:
                out.append(f"{name} 등장(HP {hp})")
                continue
            _, hp0, blk0 = before["enemies"][cid]
            if hp0 != hp:
                out.append(f"{name} HP {hp0}→{hp}" + (" (처치)" if hp == 0 else ""))
            elif blk0 != blk:
                out.append(f"{name} 방어도 {blk0}→{blk}")
        return ", ".join(out)

    @property
    def turn(self) -> int:
        return int(self.c.turn_count)

    # ---- 행동 ----
    def _their_action(self, action: int, target_slot: int) -> int:
        plain, tgt = self._actions[action]
        if 0 <= target_slot < len(tgt) and tgt[target_slot] >= 0:
            return tgt[target_slot]
        if plain:
            return plain[0]
        ok = [t for t in tgt if t >= 0]
        if not ok:
            raise ValueError(f"행동 {action} 은 지금 합법이 아닙니다")
        return ok[0]

    def step(self, action: int, target_slot: int = -1) -> None:
        if not self._actions:
            self.observe()
        a = self._their_action(action, target_slot)
        c, C = self.c, self.s.C
        self._actions = []
        if c.pending_choice is not None:
            c.resolve_pending_choice(None if a == C.ACTION_END_TURN else a - 1)
        elif a == C.ACTION_END_TURN:
            c.end_player_turn()
        elif a >= C.POTION_ACTION_START:
            off = a - C.POTION_ACTION_START
            slot, t = divmod(off, C.POTION_TARGET_OPTIONS)
            c.use_potion(slot, target_index=None if t == 0 else t - 1)
        elif a <= C.MAX_HAND_SIZE:
            c.play_card(a - 1, None)
        else:
            hand_idx, enemy = divmod(a - 1 - C.MAX_HAND_SIZE, C.MAX_ENEMIES)
            c.play_card(hand_idx, enemy)


class ZhiyueCombatEnv:
    """train_combat.py 가 쓰는 전투 환경 (combat_pool / card_pool / potion_pool / make_combat)."""

    def __init__(self, repo: str | Path, character: str = "Ironclad"):
        if character != "Ironclad":
            raise NotImplementedError("지금은 아이언클래드만 연결했습니다")
        self.s = _Sim(repo)
        self.vocab = self.s.vocab

    def combat_pool(self) -> list[tuple[int, int]]:
        """(조우 id, 막) 전부. 조우 id 는 1부터."""
        return [(i + 1, act) for i, (_, act, _) in enumerate(self.s.encounters)]

    def encounter_name(self, encounter: int) -> str:
        name, act, _ = self.s.encounters[encounter - 1]
        return f"{name.removeprefix('setup_')}@{act + 1}막"

    def card_pool(self) -> list[int]:
        return list(self.s.ironclad_pool)

    def potion_pool(self) -> list[int]:
        return sorted(self.s.potions_by_id)

    def make_combat(self, encounter: int, act: int, deck: list[tuple[int, bool]], hp: int, max_hp: int,
                    potions: list[int], relics: list[int], seed: int) -> ZhiyueCombatSim:
        s = self.s
        if relics:
            raise NotImplementedError("유물은 아직 연결하지 않았습니다")
        cards = [s.create_card(s.cards_by_id[cid], upgraded=up) for cid, up in deck]
        pots = [s.create_potion(s.potions_by_id[pid], slot=i) for i, pid in enumerate(potions)]
        combat = s.CombatState(player_hp=hp, player_max_hp=max_hp, deck=cards, rng_seed=seed,
                               character_id="Ironclad", potions=pots, max_potion_slots=max(3, len(pots)))
        _, _, setup = s.encounters[encounter - 1]
        setup(combat, s.Rng(seed))
        combat.start_combat()
        return ZhiyueCombatSim(s, combat, encounter)
