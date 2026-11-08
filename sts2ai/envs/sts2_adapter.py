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

import ctypes
import sys
from pathlib import Path

import numpy as np

from .. import schema as S
from ..schema import Card, CombatObs, MapGraph, RunState, Vocab
from .base import PHASE_COMBAT, PHASE_DONE, PHASE_MAP, PHASE_OTHER

ACT_ORDINAL = {"overgrowth": 0, "underdocks": 0, "hive": 1, "glory": 2}


class _Lib:
    """시뮬레이터 모듈 묶음 (경로를 받아 한 번만 import)."""

    def __init__(self, repo: str | Path):
        repo = Path(repo).resolve()
        for p in (repo / "src", repo / "rl"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        import combat_tokens as ct  # noqa: PLC0415
        from sts2_gym import names, native  # noqa: PLC0415
        from sts2_gym import run_constants as rc  # noqa: PLC0415
        from sts2_gym.run_env import Sts2RunEnv  # noqa: PLC0415

        self.ct, self.names, self.native, self.rc, self.Sts2RunEnv = ct, names, native, rc, Sts2RunEnv
        self.cards = names.cards()
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

    def card(self, raw_id: int, upgraded: bool, enchant: int = 0) -> Card:
        info = self.cards.get(raw_id)
        return Card(
            self.ct.VOCAB.card.get(raw_id, 0),
            bool(upgraded),
            self.enchant_index.get(enchant, 0),
            (info.cost if info else 0) / 5.0,
        )

    def combat_obs(self, handle: int, obs, legal: np.ndarray) -> CombatObs:
        """combat_tokens.encode 결과에 타깃 정보를 붙인다. handle 은 전투 핸들."""
        ct, native = self.ct, self.native
        tok = ct.encode(handle, obs, native.encounter_id(handle), probe=False)
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
        )

    def potions_from_obs(self, obs) -> list[int]:
        ct = self.ct
        return [self.potion_index.get(int(obs[ct.POTION_OFFSET + s * 2]), 0) for s in range(ct.MAX_POTIONS)]


class Sts2CombatFork:
    """Sts2Run_ForkCombat 으로 만든 전투 핸들 (숨은 정보 재샘플)."""

    def __init__(self, lib: _Lib, handle: int, obs):
        self.lib, self.h, self.obs = lib, handle, obs
        self._rew = (ctypes.c_float * 1)()
        self.done = False
        self.won = False

    def observe(self) -> CombatObs:
        legal = np.ctypeslib.as_array(self.lib.native.valid_actions(self.h, S.MAX_ACTIONS)).astype(bool)
        return self.lib.combat_obs(self.h, self.obs, legal)

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


class Sts2Run:
    """RunSim 프로토콜 구현."""

    def __init__(self, repo: str | Path = "../sts2", ascension: int = 0):
        self.lib = _Lib(repo)
        self.vocab = self.lib.vocab
        self.ascension = ascension
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
        self._sync()

    def _step(self, action: int, target: int = -1) -> None:
        self._obs, _r, terminal, truncated, self._info = self.env.step(action, target)
        self._terminal = self._terminal or terminal or truncated
        self._sync()

    def _sync(self) -> None:
        info, rc = self._info, self.lib.rc
        self.act = ACT_ORDINAL.get(info["act"], 0)
        self.floor = info["floor"]
        self.won = bool(info["player_won"])
        self.died = (info["player_hp"] <= 0) or (self._terminal and not self.won)
        ph = info["phase"]
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
            return self.lib.combat_obs(h, buf, legal)
        finally:
            self.lib.native.destroy(h)

    def combat_step(self, action: int, target_slot: int = -1) -> None:
        self._step(action, target_slot)

    def fork_combat(self, seed: int) -> Sts2CombatFork:
        buf = (ctypes.c_int * self.lib.native.OBS_SIZE)()
        h = self.lib.native.run_fork_combat(self.env._run_handle, True, seed, buf)  # noqa: SLF001
        return Sts2CombatFork(self.lib, h, buf)

    # ---- 맵 ----
    def choose_node(self, node: int) -> None:
        slots = self._choice_slots()
        if node not in slots:
            raise RuntimeError(f"node {node} is not a map choice here: {slots}")
        self._step(slots[node])

    def spawn(self) -> Sts2Run:
        """네이티브 라이브러리(_Lib)는 같이 쓰고 런 환경만 따로 둔다."""
        c = Sts2Run.__new__(Sts2Run)
        c.lib, c.vocab, c.ascension, c.env = self.lib, self.vocab, self.ascension, None
        return c

    # ---- 복사 (숨은 난수 재샘플) ----
    def clone(self, seed: int) -> Sts2Run:
        """run_clone(resample_hidden=True): 변환·무작위 강화 결과, 앞으로의 보상 등을 다시 뽑은 복사본."""
        c = Sts2Run.__new__(Sts2Run)
        c.__dict__.update(self.__dict__)
        c.env = self.env.clone(resample_hidden=True, seed=seed)
        c._obs, c._info = c.env._obs(), c.env._info()  # noqa: SLF001
        c._map = MapGraph(**{**self._map.__dict__})
        c._sync()
        return c

    # ---- 그 밖의 선택 화면 ----
    def screen_kind(self) -> str:
        names = {v: k for k, v in vars(self.lib.rc).items() if k.startswith("PHASE_") and isinstance(v, int)}
        return names.get(self._info["phase"], str(self._info["phase"]))

    def choice_actions(self) -> list[int]:
        return [int(a) for a in np.flatnonzero(self.env.action_masks())]

    def step_choice(self, action: int) -> None:
        self._step(action)

    def default_step(self, rng: np.random.Generator) -> None:
        self._step(self.rule_action(rng))

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
