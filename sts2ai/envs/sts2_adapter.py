"""실제 시뮬레이터 r33hab/sts2 (Slay the Spire 2 emulator, MIT) 어댑터.

주의: 이 파일은 시뮬레이터 소스(src/sts2_gym, rl/combat_tokens.py, 커밋 기준
2026-10-05)를 읽고 작성했지만, 이 저장소를 만든 환경에서는 네이티브 라이브러리를 빌드할 수
없어 실제로 실행해 보지 못했다. 처음 돌릴 때 tests/test_sts2_adapter.py 부터 확인할 것.

검증이 필요한 가정
- 맵 선택지 좌표 (run_state_list 7번) 의 (x, y) 가 map_graph() 의 (col, row) 와 같은 좌표계다.
- 전투 중 run_action_mask 의 앞 32칸이 전투 행동 번호와 같다 (rl/run_train/engine.py 가 그렇게 쓴다).

준비
    git clone https://github.com/r33hab/sts2.git ../sts2
    cd ../sts2 && bash scripts/build.sh && uv sync
"""

from __future__ import annotations

import copy
import ctypes
import sys
from pathlib import Path

import numpy as np

from .. import schema as S
from ..schema import Card, CombatObs, MapGraph, RunState, Vocab
from .base import PHASE_COMBAT, PHASE_DONE, PHASE_MAP, PHASE_OTHER

ACT_ORDINAL = {"overgrowth": 0, "underdocks": 0, "hive": 1, "glory": 2}


def _card_pool(repo: Path, name: str) -> list[int]:
    import re  # noqa: PLC0415

    text = (repo / "src" / "Sts2Emulator" / "Generated" / "CardPools.g.cs").read_text(encoding="utf-8")
    m = re.search(rf"{name} =>\s*\[([^\]]*)\]", text)
    if m is None:
        raise RuntimeError(f"card pool {name} not found in CardPools.g.cs")
    return [int(x) for x in m.group(1).replace("\n", "").split(",") if x.strip()]


class _Lib:
    """시뮬레이터 모듈 묶음 (경로를 받아 한 번만 import)."""

    def __init__(self, repo: str | Path):
        repo = Path(repo).resolve()
        self.repo = repo
        for p in (repo / "src", repo / "rl"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        import combat_tokens as ct  # noqa: PLC0415
        from sts2_gym import names, native  # noqa: PLC0415
        from sts2_gym import run_constants as rc  # noqa: PLC0415
        from sts2_gym.run_env import Sts2RunEnv  # noqa: PLC0415
        # 시뮬레이터 작성자의 선택 화면 읽기 (rl/run_train/engine.py): 화면의 선택지마다 종류와
        # 대상(카드·유물·포션·노드·덱 위치·가격)을 붙여 준다. 선택 정책의 입력으로 쓴다.
        from run_train import engine as run_engine  # noqa: PLC0415

        self.ct, self.names, self.native, self.rc, self.Sts2RunEnv = ct, names, native, rc, Sts2RunEnv
        self.run_engine = run_engine
        # 모의전 덱에 넣을 카드 후보: 직업(아이언클래드) 카드 풀 + 무색 카드 풀 + 저주·상태이상 카드 전부 (멀티 전용 제외).
        # 직업·무색은 시뮬레이터의 생성 표(CardPools.g.cs)를 읽는다. sts2_gym.position_sampler 도 같은 표를 읽지만
        # 시스템 기본 인코딩으로 열어서 Windows(cp949)에서는 실패한다. 저주·상태이상은 카드 종류로 고른다.
        curse_status = {cid for cid, info in names.cards().items() if info.card_type in ("Curse", "Status")}
        # 멀티플레이 전용 카드(Cards.g.cs 의 MultiplayerOnly)는 혼자 하는 런에 나오지 않으므로 뺀다.
        import re  # noqa: PLC0415

        cards_src = (repo / "src" / "Sts2Emulator" / "Generated" / "Cards.g.cs").read_text(encoding="utf-8")
        self.multiplayer_only = {int(m.group(1)) for m in
                                 re.finditer(r"CardDef\(Id: (\d+),[^\n]*MultiplayerOnly: true", cards_src)}
        self.probe_card_pool = sorted((set(_card_pool(repo, "Ironclad")) | set(_card_pool(repo, "Colorless"))
                                       | curse_status) - self.multiplayer_only)
        # 로컬 시뮬레이터에 추가한 함수들. 없는 빌드면 기본 비용만 쓰고 인챈트·자원 토큰을 붙이지 않는다.
        # Sts2_GetPileDetails: 카드마다 (지금 비용, 인챈트, 인챈트 수치). Sts2_GetResources: 별·오스티·오브 등.
        def bind(name, args):
            fn = getattr(native._lib, name, None)  # noqa: SLF001
            if fn is not None:
                fn.restype = ctypes.c_int
                fn.argtypes = args
            return fn

        ip = ctypes.POINTER(ctypes.c_int)
        self._pile_details = bind("Sts2_GetPileDetails", [ctypes.c_int, ctypes.c_int, ip, ctypes.c_int])
        self._pile_costs = bind("Sts2_GetPileCosts", [ctypes.c_int, ctypes.c_int, ip, ctypes.c_int])
        self._resources = bind("Sts2_GetResources", [ctypes.c_int, ip, ctypes.c_int])
        self.cards = names.cards()
        # 날개 달린 부츠: 유물 카운터 = 이미 쓴 이동 횟수, 3 번 쓰면 끝 (RunConstants.WingedBootsTravels)
        self.winged_boots = next((rid for rid, n in names.relics().items() if n.replace(" ", "") == "WingedBoots"), None)
        self.winged_boots_travels = 3
        self.potion_index = {pid: i + 1 for i, pid in enumerate(sorted(names.potions()))}
        self.potion_targets = names.potion_targets()
        self.enchant_index = {eid: i + 1 for i, eid in enumerate(sorted(names.enchantments()))}
        v = ct.VOCAB
        self.vocab = Vocab(
            n_cards=v.n_cards,
            n_relics=v.n_relics,
            n_potions=len(self.potion_index) + 1,
            n_buffs=v.n_buffs,
            n_enemies=v.n_enemies,
            n_intents=v.n_intents,
            n_encounters=v.n_encounters,
            n_enchants=len(self.enchant_index) + 1,
            n_acts=3,
        )

    def pile_details(self, handle: int, pile_id: int) -> list[tuple[int, int, int]] | None:
        """전투 더미 카드들의 (지금 비용, 인챈트, 인챈트 수치) (get_pile 과 같은 순서). 지원하지 않는 빌드면 None.

        인챈트를 내보내지 않는 빌드(Sts2_GetPileCosts 만 있음)면 비용만 채우고 인챈트는 0.
        """
        fn, width = (self._pile_details, 3) if self._pile_details is not None else (self._pile_costs, 1)
        if fn is None:
            return None
        cap = 64
        while True:
            buf = (ctypes.c_int * (cap * width))()
            n = fn(handle, pile_id, buf, cap)
            if n < 0:
                return None
            if n <= cap:
                if width == 1:
                    return [(buf[i], 0, 0) for i in range(n)]
                return [(buf[i * 3], buf[i * 3 + 1], buf[i * 3 + 2]) for i in range(n)]
            cap = n

    def _fill_cards(self, handle: int, tok) -> np.ndarray:
        """카드 토큰의 F_COST 를 지금 비용으로 바꾸고, F_ENCHANT 에 인챈트 수치를 넣는다. 인챈트 인덱스 [T] 를 돌려준다.

        손패 토큰은 손패 순서 그대로, 다른 더미 토큰은 (카드 id, 강화) 로 정렬돼 나오므로 같은 순서로 맞춘다.
        같은 카드끼리는 토큰으로 구분되지 않으니 (비용, 인챈트) 가 낮은 것부터 배정한다.
        """
        native = self.native
        enchants = np.zeros(len(tok.types), dtype=np.int64)
        piles = (("hand", 1, S.TYPE_HAND, False), ("hand", 1, S.TYPE_HAND_LOCKED, True), ("draw", 0, S.TYPE_DRAW, True),
                 ("discard", 2, S.TYPE_DISCARD, True), ("exhaust", 3, S.TYPE_EXHAUST, True))
        feats = None
        for name, pid, ttype, sort in piles:
            idx = np.flatnonzero(tok.types == ttype)
            if not len(idx):
                continue
            details = self.pile_details(handle, pid)
            if details is None:
                return enchants
            pairs = list(zip(native.get_pile(handle, name), details))
            if sort:
                pairs.sort(key=lambda p: (p[0][0], p[0][1], p[1]))
            if len(pairs) != len(idx):
                continue  # 토큰 수가 맞지 않으면(예상 못 한 화면) 건드리지 않는다
            if feats is None:
                feats = tok.feats.copy()
            for ti, (_, (cost, ench, amt)) in zip(idx, pairs):
                feats[ti, S.F_COST] = max(0, cost) / 5.0
                feats[ti, S.F_ENCHANT] = amt / 5.0
                enchants[ti] = self.enchant_index.get(ench, 0) if ench else 0
        if feats is not None:
            tok.feats = feats
        return enchants

    def resource_tokens(self, handle: int) -> list[tuple[int, int, dict]]:
        """(토큰 종류, id, 특징) 목록: 골드·최대 에너지·오브 칸·별·오스티, 오브마다 하나. 지원하지 않는 빌드면 빈 목록."""
        if self._resources is None:
            return []
        buf = (ctypes.c_int * 64)()
        n = self._resources(handle, buf, 64)
        if n < 0:
            return []
        stars, osty, osty_max, orb_cap, max_energy, gold, n_orbs = (buf[i] for i in range(7))
        out = [(S.TYPE_RESOURCE, S.RESOURCE_GOLD, {S.F_MAGNITUDE: gold / 300.0}),
               (S.TYPE_RESOURCE, S.RESOURCE_MAX_ENERGY, {S.F_MAGNITUDE: max_energy / 10.0})]
        if orb_cap:
            out.append((S.TYPE_RESOURCE, S.RESOURCE_ORB_SLOTS, {S.F_MAGNITUDE: orb_cap / 10.0}))
        if stars:
            out.append((S.TYPE_RESOURCE, S.RESOURCE_STARS, {S.F_MAGNITUDE: stars / 10.0}))
        if osty_max:
            out.append((S.TYPE_RESOURCE, S.RESOURCE_OSTY, {S.F_ENEMY_HP: osty / 50.0,
                                                           S.F_ENEMY_HP_FRAC: osty / max(1, osty_max)}))
        for i in range(n_orbs):
            kind, passive, evoke = buf[7 + i * 3], buf[7 + i * 3 + 1], buf[7 + i * 3 + 2]
            out.append((S.TYPE_ORB, min(S.N_ORB_KINDS - 1, kind + 1), {S.F_MAGNITUDE: passive / 10.0,
                                                                        S.F_INTENT_MAG: evoke / 30.0}))
        return out

    def card(self, raw_id: int, upgraded: bool, enchant: int = 0) -> Card:
        info = self.cards.get(raw_id)
        return Card(
            self.ct.VOCAB.card.get(raw_id, 0),
            bool(upgraded),
            self.enchant_index.get(enchant, 0),
            (info.cost if info else 0) / 5.0,
        )

    def combat_obs(self, handle: int, obs, legal: np.ndarray, probe: bool = False) -> CombatObs:
        """combat_tokens.encode 결과에 타깃 정보를 붙인다. handle 은 전투 핸들.

        probe: 턴 종료 예측 특징(지금 턴을 끝내면 적 턴에 HP 가 평균·최악 얼마 줄고 죽을 확률이 얼마인지,
        숨은 난수를 다시 뽑은 복사본 5개로 계산)을 채운다. 끄면 그 칸은 0.
        """
        ct, native = self.ct, self.native
        tok = ct.encode(handle, obs, native.encounter_id(handle), probe=probe)
        # combat_tokens 는 포션 토큰에 "이 칸에 포션이 있다"만 넣고 종류 id 는 0 으로 둔다.
        # 포션 토큰은 칸 순서대로 나오므로, 칸의 포션 id 를 포션 임베딩 인덱스로 채워 넣는다.
        pot = np.flatnonzero(tok.types == S.TYPE_POTION)
        if len(pot):
            ids = tok.ids.copy()
            for slot, ti in enumerate(pot[: ct.MAX_POTIONS]):
                ids[ti] = self.potion_index.get(int(obs[ct.POTION_OFFSET + slot * 2]), 0)
            tok.ids = ids
        enchants = self._fill_cards(handle, tok)
        # 오브·캐릭터 자원 토큰, 카드 고르기 화면 종류 토큰을 맨 뒤에 붙인다 (행동 토큰 1+i 와 적 토큰 위치는 그대로)
        extra = self.resource_tokens(handle)
        sel_kind = int(obs[ct.SELECTION_KIND_OFFSET])
        if sel_kind:
            extra = extra + [(S.TYPE_SELECTION, min(sel_kind, S.N_SELECTION_KINDS - 1), {})]
        if extra:
            feats_x = np.zeros((len(extra), tok.feats.shape[1]), dtype=tok.feats.dtype)
            for r, (_, _, f) in enumerate(extra):
                for ch, v in f.items():
                    feats_x[r, ch] = v
            tok.types = np.concatenate([tok.types, np.array([t for t, _, _ in extra], dtype=tok.types.dtype)])
            tok.ids = np.concatenate([tok.ids, np.array([i for _, i, _ in extra], dtype=tok.ids.dtype)])
            tok.owners = np.concatenate([tok.owners, np.zeros(len(extra), dtype=tok.owners.dtype)])
            tok.feats = np.concatenate([tok.feats, feats_x])
            enchants = np.concatenate([enchants, np.zeros(len(extra), dtype=np.int64)])
        needs = np.zeros(S.MAX_ACTIONS, dtype=bool)
        if not obs[ct.SELECTION_KIND_OFFSET]:
            hand = native.get_pile(handle, "hand")
            for i, (cid, _up) in enumerate(hand[: S.MAX_ACTIONS]):
                info = self.cards.get(cid)
                needs[i] = bool(info and info.targets_an_enemy())
            for s in range(ct.MAX_POTIONS):
                a = len(hand) + 1 + s
                pid = obs[ct.POTION_OFFSET + s * 2]
                if pid and a < S.MAX_ACTIONS:
                    needs[a] = self.potion_targets.get(pid) == "AnyEnemy"
        is_enemy = tok.types == S.TYPE_ENEMY
        enemy_tokens = np.flatnonzero(is_enemy).astype(np.int64)
        return CombatObs(
            types=tok.types,
            ids=tok.ids,
            owners=tok.owners,
            feats=tok.feats,
            legal=legal[: S.MAX_ACTIONS].astype(bool),
            needs_target=needs,
            enemy_tokens=enemy_tokens,
            enemy_slots=(tok.owners[enemy_tokens] - 1).astype(np.int64),
            enemy_alive=tok.feats[enemy_tokens, S.F_ENEMY_HP] > 0,
            enchants=enchants,
        )

    def potions_from_obs(self, obs) -> list[int]:
        ct = self.ct
        return [self.potion_index.get(int(obs[ct.POTION_OFFSET + s * 2]), 0) for s in range(ct.MAX_POTIONS)]


class Sts2CombatFork:
    """Sts2Run_ForkCombat 으로 만든 전투 핸들 (숨은 정보 재샘플)."""

    def __init__(self, lib: _Lib, handle: int, obs, probe: bool = False):
        self.lib, self.h, self.obs, self.probe = lib, handle, obs, probe
        self._rew = (ctypes.c_float * 1)()
        self.done = False
        self.won = False

    def observe(self) -> CombatObs:
        legal = np.ctypeslib.as_array(self.lib.native.valid_actions(self.h, S.MAX_ACTIONS)).astype(bool)
        return self.lib.combat_obs(self.h, self.obs, legal, probe=self.probe)

    def step(self, action: int, target_slot: int = -1) -> None:
        terminal = self.lib.native.step(self.h, action, self.obs, self._rew, target_slot)
        if terminal:
            self.done = True
            self.won = self.lib.native.player_won(self.h)

    def hp(self) -> int:
        return int(self.obs[0])

    def potions(self) -> list[int]:
        return self.lib.potions_from_obs(self.obs)

    def close(self) -> None:
        if self.h is not None:
            self.lib.native.destroy(self.h)
            self.h = None


class Sts2RunCombatFork:
    """런 전체를 복제(숨은 난수 재추첨)해서 그 안에서 전투만 진행하는 분기.

    전투만 떼어 내는 Sts2CombatFork 와 달리, 전투가 끝나면 시뮬레이터가 덱·골드·최대 HP·유물 카운터를 런에
    반영한다(SyncAfterCombat). 그래서 post_state() 가 "전투 직후 실제 런 상태"를 준다 (보상은 아직 받기 전).
    """

    def __init__(self, run: "Sts2Run"):
        self.r = run
        self.done = False
        self.won = False

    def observe(self) -> CombatObs:
        return self.r.combat_obs()

    def step(self, action: int, target_slot: int = -1) -> None:
        self.r.combat_step(action, target_slot)
        if self.r.phase != PHASE_COMBAT:
            self.done = True
            self.won = not self.r.died

    def hp(self) -> int:
        return 0 if self.r.died else int(self.r._info["player_hp"])  # noqa: SLF001

    def potions(self) -> list[int]:
        return [self.r.lib.potion_index.get(p, 0) for p in self.r._info["potions"]]  # noqa: SLF001

    def post_state(self) -> RunState | None:
        """전투가 끝난 뒤의 실제 런 상태. 죽었으면 None."""
        if not self.done or self.r.died:
            return None
        return self.r.run_state()

    def close(self) -> None:
        if self.r is not None:
            self.r.close()
            self.r = None


class Sts2Run:
    """RunSim 프로토콜 구현."""

    def __init__(self, repo: str | Path = "../sts2", ascension: int = 0, combat_probe: bool = False,
                 combat_fork: str = "run"):
        self.lib = _Lib(repo)
        self.vocab = self.lib.vocab
        self.ascension = ascension
        # GRPO 전투 분기 방식. "run" = 런 전체를 복제해 전투 진행 (전투 뒤 덱·골드·최대 HP 변화까지 채점),
        # "combat" = 전투만 복사 (빠르지만 채점은 HP·포션만).
        self.combat_fork = combat_fork
        # 전투 관측에 턴 종료 예측 특징을 채울지 (_Lib.combat_obs 의 probe). 행동마다 작은 시뮬레이션 5번이 들어
        # 수집이 약 16% 느려진다. 기본은 꺼 둔다 (꺼 두면 그 칸은 0).
        self.combat_probe = combat_probe
        self.env = None

    # ---- 진행 ----
    def reset(self, seed: int) -> None:
        if self.env is not None:
            self.env.close()
        self.env = self.lib.Sts2RunEnv(seed=str(seed), max_episode_steps=10**6, ascension=self.ascension)
        self._obs, self._info = self.env.reset()
        self._terminal = False
        self._combat_counter = 0
        self._last_encounter = None
        self._map_key = None
        self._visit = self.lib.run_engine.VisitTracker()
        self._screen_cache = None
        # 상태 재현용 기록 (snapshot/restore). 시뮬레이터는 결정적이라 시드 + 행동 목록으로 같은 상태를 다시 만든다.
        self._seed = seed
        self._origin = None  # 숨은 난수를 다시 뽑기 직전 상태의 토큰 (restore 로 시작한 런)
        self._resample = None  # 그때 쓴 재추첨 시드
        self._actions: list[tuple[int, int]] = []  # 마지막 리셋/재추첨 이후 env 에 보낸 행동
        self._sync()

    def _step(self, action: int, target: int = -1) -> None:
        self._screen_cache = None
        self._actions.append((int(action), int(target)))
        self._obs, _r, terminal, truncated, self._info = self.env.step(action, target)
        self._terminal = self._terminal or terminal or truncated
        self._sync()

    def _sync(self) -> None:
        info, rc = self._info, self.lib.rc
        self.act = ACT_ORDINAL.get(info["act"], 0)
        self.floor = info["floor"]
        ph = info["phase"]
        # info["player_won"] 은 런 클리어가 아니라 "마지막 전투를 이겼나"(LastPlayerWon)다.
        # 런이 끝나면 단계가 COMPLETE 가 되고, 그때 마지막 전투(또는 The Architect)를 이겼으면 클리어.
        finished = self._terminal or ph == rc.PHASE_COMPLETE
        self.won = finished and bool(info["player_won"]) and info["player_hp"] > 0
        self.died = (info["player_hp"] <= 0) or (finished and not self.won)
        if self.won or self.died or ph == rc.PHASE_COMPLETE:
            self.phase = PHASE_DONE
        elif ph == rc.PHASE_COMBAT:
            self.phase = PHASE_COMBAT
            if self._last_encounter != info["encounter_id"] or self._prev_phase != PHASE_COMBAT:
                self._combat_counter += 1
            self._last_encounter = info["encounter_id"]
        elif ph == rc.PHASE_MAP:
            self.phase = PHASE_MAP
        else:
            self.phase = PHASE_OTHER
        self._prev_phase = self.phase
        self._load_map()

    _prev_phase = None

    def _load_map(self) -> None:
        g = self.env.map_graph()
        key = (self.act, tuple(sorted(g["nodes"])))
        if key != self._map_key:
            coords = sorted(g["nodes"], key=lambda cr: (cr[1], cr[0]))  # (row, col) 순
            self._index = {cr: i for i, cr in enumerate(coords)}
            self._map = MapGraph(
                node_types=[g["nodes"][cr] for cr in coords],
                rows=[cr[1] for cr in coords],
                cols=[cr[0] for cr in coords],
                edges=[(self._index[a], self._index[b]) for a, b in g["edges"]
                       if a in self._index and b in self._index],
                current=None,
                n_rows=max(cr[1] for cr in coords) + 1 if coords else 16,
            )
            self._map_key = key
        cur = g["current"]
        self.current_node = self._index.get(tuple(cur)) if cur is not None else None
        self._map.current = self.current_node

    def _choice_slots(self) -> dict[int, int]:
        """맵 화면의 선택지: 노드 인덱스 -> 행동 번호."""
        native, layout = self.lib.native, self.lib.native.RUN_OBS_LAYOUT
        n = layout["map_choices"]
        coords = native.run_state_list(self.env._run_handle, 7, n * 2)  # noqa: SLF001
        types_at = native.OBS_SIZE + layout["map_node_type_offset"]
        out = {}
        for i in range(n):
            if int(self._obs[types_at + i]) == self.lib.rc.NODE_NONE:
                continue
            node = self._index.get((int(coords[2 * i]), int(coords[2 * i + 1])))
            if node is not None:
                out[node] = i
        return out

    # ---- 상태 ----
    def run_state(self) -> RunState:
        info, lib = self._info, self.lib
        m = MapGraph(**{**self._map.__dict__})
        if self.phase == PHASE_MAP:
            m.choices = sorted(self._choice_slots())
        act_summary = lib.native.run_state_list(self.env._run_handle, 14, 4)  # noqa: SLF001
        boss_raw = act_summary[1] if len(act_summary) > 1 else 0
        return RunState(
            act=self.act,
            floor=info["floor"],
            hp=info["player_hp"],
            max_hp=info["player_max_hp"],
            gold=info["gold"],
            ascension=self.ascension,
            deck=[lib.card(c["card_id"], c["upgraded"], c["enchantment"]) for c in info["deck"]],
            relics=[(lib.ct.VOCAB.relic.get(r["relic_id"], 0), r["counter"]) for r in info["relic_slots"]],
            potions=[lib.potion_index.get(p, 0) for p in info["potions"]],
            map=m,
            boss=lib.ct.VOCAB.encounter.get(boss_raw, 0),
            map_jumps=max(0, max((lib.winged_boots_travels - r["counter"] for r in info["relic_slots"]
                                  if r["relic_id"] == lib.winged_boots), default=0)),
        )

    # ---- 전투 ----
    def combat_id(self) -> int:
        return self._combat_counter

    def combat_obs(self) -> CombatObs:
        # 인코딩용으로만 충실한 복사본을 만든다. encode 가 더미 순서를 지우므로 누출은 없다.
        buf = (ctypes.c_int * self.lib.native.OBS_SIZE)()
        h = self.lib.native.run_fork_combat(self.env._run_handle, False, 0, buf)  # noqa: SLF001
        try:
            legal = self.env.action_masks()[: S.MAX_ACTIONS]
            # 턴 종료 예측은 이 복사본을 다시 "숨은 난수를 새로 뽑아" 복제해서 계산하므로 미래를 엿보지 않는다
            return self.lib.combat_obs(h, buf, legal, probe=getattr(self, "combat_probe", False))
        finally:
            self.lib.native.destroy(h)

    def combat_step(self, action: int, target_slot: int = -1) -> None:
        self._step(action, target_slot)

    def fork_combat(self, seed: int):
        if getattr(self, "combat_fork", "run") == "run":
            return Sts2RunCombatFork(self.clone(seed))
        buf = (ctypes.c_int * self.lib.native.OBS_SIZE)()
        h = self.lib.native.run_fork_combat(self.env._run_handle, True, seed, buf)  # noqa: SLF001
        return Sts2CombatFork(self.lib, h, buf, probe=getattr(self, "combat_probe", False))

    # ---- 모의전 (덱을 바꿔 지정한 적과 바로 싸우기) ----
    def encounter_raw(self) -> int:
        """지금(또는 마지막) 전투의 시뮬레이터 encounter id."""
        return int(self._info.get("encounter_id", 0) or 0)

    def boss_raw(self) -> int:
        a = self.lib.native.run_state_list(self.env._run_handle, 14, 4)  # noqa: SLF001
        return int(a[1]) if len(a) > 1 else 0

    def deck_raw(self) -> list[tuple[int, bool]]:
        return [(int(c["card_id"]), bool(c["upgraded"])) for c in self._info["deck"]]

    def probe_card_pool(self) -> list[int]:
        return self.lib.probe_card_pool

    def card_upgradable(self, card_id: int) -> bool:
        """저주·상태이상 카드는 강화할 수 없다."""
        info = self.lib.cards.get(int(card_id))
        return info is not None and info.card_type not in ("Curse", "Status")
    def raw_state(self) -> dict:
        """덱 빌딩 재생용 런 상태: 덱 [(id, 강화)], 유물 [id], 골드, 최대 HP (시뮬레이터 id 그대로)."""
        info = self._info
        return {"deck": [(int(c["card_id"]), bool(c["upgraded"])) for c in info["deck"]],
                "relics": [int(r["relic_id"]) for r in info["relic_slots"]],
                "gold": int(info["gold"]), "max_hp": int(info["player_max_hp"])}

    def raw_card(self, raw_id: int, upgraded: bool) -> Card:
        return self.lib.card(int(raw_id), bool(upgraded), 0)

    def action_token_name(self, ty: int, vid: int) -> str:
        """전투 행동 토큰 → 읽을 이름 (대시보드용). 카드·포션은 시뮬레이터 이름, 턴 종료 등은 한국어."""
        if ty in (S.TYPE_HAND, S.TYPE_SELECT, S.TYPE_HAND_LOCKED):
            info = self.lib.cards.get(self.card_raw(vid))
            n = self.lib.names.spaced(info.name) if info else f"카드{vid}"
            return n if ty == S.TYPE_HAND else f"고르기: {n}"
        if ty == S.TYPE_POTION:
            inv = getattr(self.lib, "_potion_inv", None)
            if inv is None:
                inv = {v: k for k, v in self.lib.potion_index.items()}
                self.lib._potion_inv = inv
            pid = inv.get(int(vid))
            return "포션: " + (self.lib.names.spaced(self.lib.names.potions().get(pid, str(pid))) if pid else "?")
        return {S.TYPE_END_TURN: "턴 종료", S.TYPE_DECLINE: "고르지 않기"}.get(ty, f"행동{ty}")

    def card_raw(self, vocab_id: int) -> int:
        """카드 임베딩 인덱스 → 시뮬레이터 카드 id (없으면 0)."""
        inv = getattr(self.lib, "_card_inv", None)
        if inv is None:
            inv = {v: k for k, v in self.lib.ct.VOCAB.card.items()}
            self.lib._card_inv = inv
        return int(inv.get(int(vocab_id), 0))

    def relic_vocab(self, raw_id: int) -> int:
        return int(self.lib.ct.VOCAB.relic.get(int(raw_id), 0))

    def encounter_vocab(self, raw_id: int) -> int:
        return int(self.lib.ct.VOCAB.encounter.get(int(raw_id), 0))

    def encounter_kind(self, raw_id: int) -> str:
        """적 목록표(EncounterTags.g.cs) 기준 "Weak" | "Normal" | "Elite" | "Boss" (없으면 "Normal")."""
        kinds = getattr(self.lib, "_enc_kind", None)
        if kinds is None:
            from ..train.matchup import load_pools  # noqa: PLC0415

            kinds = {}
            for pools in load_pools(self.lib.repo).values():
                for kind, ids in pools.items():
                    for e in ids:
                        kinds.setdefault(e, kind)
            self.lib._enc_kind = kinds
        return kinds.get(int(raw_id), "Normal")

    def relic_pool(self) -> list[int]:
        """전투만 학습할 때 넣을 유물 후보: 공용 + 아이언클래드 풀 + 어느 직업 풀에도 없는 유물(이벤트·고대인 등).
        다른 직업 풀의 유물은 뺀다 (RelicPools.g.cs)."""
        pool = getattr(self.lib, "_relic_pool", None)
        if pool is None:
            import re  # noqa: PLC0415

            text = (self.lib.repo / "src" / "Sts2Emulator" / "Generated" / "RelicPools.g.cs").read_text(encoding="utf-8")
            pools = {n: {int(x) for x in b.replace("\n", "").split(",") if x.strip()}
                     for n, b in re.findall(r"(\w+) =>\s*\[([^\]]*)\]", text)}
            others = set().union(*(v for k, v in pools.items() if k not in ("Shared", "Ironclad")))
            pool = sorted(r for r in self.lib.names.relics() if r not in others)
            self.lib._relic_pool = pool
        return pool

    def transform_pool(self) -> list[int]:
        """변환 결과 후보: 직업·무색 카드 (저주·상태이상 제외)."""
        return [c for c in self.lib.probe_card_pool if self.card_upgradable(c)]

    def start_probe(self, seed: int, deck: list[tuple[int, bool]], encounter: int, hp: int,
                    combats_before: int, relics: list[int] | None = None,
                    max_hp: int | None = None, potions: list[int] | None = None) -> "Sts2Run | None":
        """런을 복제(숨은 난수는 seed 로 재추첨)한 뒤, 그 복제본에서 덱을 바꿔 encounter 와의 전투를 바로 시작한다.

        유물(relics)·최대 HP(max_hp)·포션(potions) 을 주지 않으면 지금 런 그대로. 골드는 지금 런 그대로.
        Sts2Run_StartCombat 은 카드를 (id, 강화) 로만 받아서 덱 카드의 인챈트와 유물 카운터는 빠진다. 시작하지 못하면 None.
        """
        c = self.clone(seed)
        info = c._info
        ids = [-cid if up else cid for cid, up in deck]
        relics = [int(r["relic_id"]) for r in info["relic_slots"]] if relics is None else [int(r) for r in relics]
        max_hp = int(info["player_max_hp"]) if max_hp is None else int(max_hp)
        pots = [int(p) for p in info["potions"] if p] if potions is None else [int(p) for p in potions]
        if potions is not None and len(pots) > 0:
            # 쓸 수 있는 포션 칸을 준 포션 수만큼 (시뮬레이터 칸 수 한계 PotionSlotCapacity 안에서)
            c.lib.native.run_debug_set_player(c.env._run_handle, c.env._run_obs_buf,  # noqa: SLF001
                                              max_potion_slots=len(pots))
        status = c.lib.native.run_start_combat(
            c.env._run_handle, ids, int(encounter), relics, int(hp), max_hp, pots,  # noqa: SLF001
            int(info["gold"]), int(combats_before), c.env._run_obs_buf)  # noqa: SLF001
        if status < 0:
            c.close()
            return None
        c._obs, c._info = c.env._obs(), c.env._info()  # noqa: SLF001
        c._screen_cache = None
        c._sync()
        if c.phase != PHASE_COMBAT:
            c.close()
            return None
        return c

    # ---- 맵 ----
    def choose_node(self, node: int) -> None:
        slots = self._choice_slots()
        if node not in slots:
            raise RuntimeError(f"node {node} is not a map choice here: {slots}")
        self._visit.took("map.node")
        self._step(slots[node])

    # ---- 상태 저장·복원 (막 시작 상태 풀) ----
    def snapshot(self) -> dict:
        """지금 상태를 다른 프로세스에서도 재현할 수 있는 토큰. 시드 + 행동 목록 (+ 재추첨 이력)."""
        return {"seed": self._seed, "origin": self._origin, "resample": self._resample,
                "actions": list(self._actions), "visit": copy.deepcopy(self._visit),
                "combat_counter": self._combat_counter, "act": self.act, "floor": self.floor}

    def _replay(self, token: dict) -> None:
        if token["origin"] is None:
            self.reset(token["seed"])
        else:
            self._replay(token["origin"])
            self._resample_in_place(token["resample"])
        for a, t in token["actions"]:
            self._step(a, t)
        self._visit = copy.deepcopy(token["visit"])
        self._combat_counter = token["combat_counter"]

    def _resample_in_place(self, seed: int) -> None:
        """이 객체 그대로 숨은 난수(앞으로의 보상·셔플 등)만 새로 뽑는다."""
        origin = self.snapshot()
        new_env = self.env.clone(resample_hidden=True, seed=seed)
        self.env.close()
        self.env = new_env
        self._obs, self._info = new_env._obs(), new_env._info()  # noqa: SLF001
        self._origin, self._resample, self._actions = origin, seed, []
        self._screen_cache = None
        self._sync()

    def restore(self, token: dict, seed: int) -> None:
        """토큰의 상태를 재현한 뒤, seed 로 숨은 난수를 새로 뽑는다 (같은 상태에서 다른 앞날)."""
        self._replay(token)
        if (self.act, self.floor) != (token["act"], token["floor"]):
            raise RuntimeError(f"replay mismatch: got act {self.act} floor {self.floor}, "
                               f"expected act {token['act']} floor {token['floor']}")
        self._resample_in_place(seed)

    # ---- 복사 (숨은 난수 재샘플) ----
    def clone(self, seed: int) -> Sts2Run:
        """run_clone(resample_hidden=True): 변환·무작위 강화 결과, 앞으로의 보상 등을 다시 뽑은 복사본."""
        c = Sts2Run.__new__(Sts2Run)
        c.__dict__.update(self.__dict__)
        c.env = self.env.clone(resample_hidden=True, seed=seed)
        c._obs, c._info = c.env._obs(), c.env._info()  # noqa: SLF001
        c._map = MapGraph(**{**self._map.__dict__})
        c._visit = copy.deepcopy(self._visit)
        c._screen_cache = None
        c._origin, c._resample, c._actions = self.snapshot(), seed, []
        c._sync()
        return c

    def clone_exact(self) -> Sts2Run:
        """숨은 난수까지 그대로인 충실한 복사본. 선택지의 결과를 미리 재 볼 때만 쓴다 (탐색에 쓰면 미래를 엿본다)."""
        c = Sts2Run.__new__(Sts2Run)
        c.__dict__.update(self.__dict__)
        c.env = self.env.clone(resample_hidden=False)
        c._obs, c._info = c.env._obs(), c.env._info()  # noqa: SLF001
        c._map = MapGraph(**{**self._map.__dict__})
        c._visit = copy.deepcopy(self._visit)
        c._screen_cache = None
        c._origin, c._resample, c._actions = None, None, []
        c._sync()
        return c

    # ---- 그 밖의 선택 화면 ----
    def screen_kind(self) -> str:
        names = {v: k for k, v in vars(self.lib.rc).items() if k.startswith("PHASE_") and isinstance(v, int)}
        return names.get(self._info["phase"], str(self._info["phase"]))

    def choice_actions(self) -> list[int]:
        return [int(a) for a in np.flatnonzero(self.env.action_masks())]

    def _screen(self):
        """시뮬레이터 쪽 화면 읽기 결과 (Decision 또는 자동 진행 행동). 스텝마다 한 번만 읽는다."""
        if self._screen_cache is None:
            info = self._info
            self._visit.see(info["phase"], info["event_id"])
            self._screen_cache = self.lib.run_engine._read_screen(  # noqa: SLF001
                self.env, self.ascension, self._visit.event_step, self._visit.rest_option_taken)
        return self._screen_cache

    def choice_options(self) -> list[S.ChoiceOption]:
        """전투·맵 밖 선택 화면의 선택지. 고를 게 없는 화면(자동 진행)이면 빈 목록."""
        if self.phase != PHASE_OTHER:
            return []
        sc = self._screen()
        if sc.decision is None or not sc.decision.options:
            return []
        lib = self.lib
        view = sc.decision.view or lib.run_engine.features.view_from_engine(self.env, self.ascension)
        deck_raw = [(c["card_id"], bool(c["upgraded"]), c["enchantment"]) for c in self._info["deck"]]

        def card(ref):
            return lib.card(ref.card_id, ref.upgraded, ref.enchantment) if ref is not None else None

        def deck_pos(i):
            if i is None or i >= len(view.deck):
                return -1
            ref = view.deck[i]
            key = (ref.card_id, bool(ref.upgraded), ref.enchantment)
            return deck_raw.index(key) if key in deck_raw else -1

        out = []
        for o in sc.decision.options:
            node = -1
            if o.node is not None:
                node = self._index.get((o.node[0], o.node[1]), -1)
            ev, ev_opt = (o.event[0], o.event[1]) if o.event is not None else (0, 0)
            out.append(S.ChoiceOption(
                action=int(o.action), kind=o.kind, card=card(o.card), cards=[card(c) for c in o.cards],
                relic=lib.ct.VOCAB.relic.get(o.relic, 0) if o.relic else 0,
                potion=lib.potion_index.get(o.potion, 0) if o.potion else 0,
                node=node, deck_index=deck_pos(o.deck_index), price=int(o.price), event=int(ev),
                event_option=int(ev_opt),
            ))
        return out

    def step_choice(self, action: int) -> None:
        sc = self._screen() if self.phase == PHASE_OTHER else None
        if sc is not None and sc.decision is not None:
            kind = next((o.kind for o in sc.decision.options if o.action == action), "")
            self._visit.took(kind)
        self._step(action)

    def default_step(self, rng: np.random.Generator) -> None:
        """자동 진행 화면은 시뮬레이터가 정한 행동으로, 나머지는 규칙 정책으로."""
        sc = self._screen() if self.phase == PHASE_OTHER else None
        if sc is not None and sc.decision is None and sc.auto is not None:
            self._step(int(sc.auto))
            return
        self.step_choice(self.rule_action(rng))

    def rule_action(self, rng: np.random.Generator) -> int:
        """비교 기준이 되는 규칙 정책."""
        rc = self.lib.rc
        mask = self.env.action_masks()
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise RuntimeError(f"no legal action in phase {self._info['phase']}")
        ph = self._info["phase"]
        choice = None
        if ph == rc.PHASE_CARD_REWARD:
            cards = [a for a in legal if a < 3]
            if cards and rng.random() < 0.7:
                choice = int(rng.choice(cards))
            elif mask[rc.REWARD_SKIP_ACTION]:
                choice = rc.REWARD_SKIP_ACTION
        elif ph == rc.PHASE_SHOP and mask[rc.SHOP_SKIP_ACTION]:
            choice = rc.SHOP_SKIP_ACTION
        elif ph == rc.PHASE_REST:
            low = self._info["player_hp"] < 0.6 * self._info["player_max_hp"]
            if low and mask[rc.REST_HEAL_ACTION]:
                choice = rc.REST_HEAL_ACTION
            elif mask[rc.REST_UPGRADE_ACTION]:
                choice = rc.REST_UPGRADE_ACTION
        elif ph == rc.PHASE_TREASURE and mask[rc.TREASURE_TAKE_RELIC_ACTION]:
            choice = rc.TREASURE_TAKE_RELIC_ACTION
        if choice is None:
            choice = int(rng.choice(legal))
        return int(choice)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
