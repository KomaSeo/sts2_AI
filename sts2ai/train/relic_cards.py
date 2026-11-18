"""고대의 존재(니오우 포함) 유물을 골랐을 때 덱에 따라오는 카드.

전투만 학습의 시나리오에서 유물을 고른 뒤 덱을 고친다 (relic_scheme="acts"). 시뮬레이터의 유물 획득 효과
(../sts2 RunNonCombatEffects.cs 의 AfterObtained 들)를 옮긴 것이다. 고르는 화면이 뜨는 유물(카드 보상 화면,
덱에서 고르기)은 대부분 넣지 않았고, 덱을 바꾸는 두 개(발톱, 보존된 안개)만 "사람이라면 이렇게 고른다" 로 흉내 낸다.

이 카드들은 무작위 풀에서 빼서, 그 유물을 골랐을 때만 덱에 들어간다 (reserved_cards).
"""

from __future__ import annotations

import re

import numpy as np

# 유물 이름 → 덱에 더하는 카드 이름 (같은 이름 여러 번 = 여러 장). "+" 로 끝나면 강화된 카드.
ADD = {
    "NeowsTorment": ["NeowsFury"],
    "LargeCapsule": ["StrikeIronclad", "DefendIronclad"],
    "CursedPearl": ["Greed"],
    "PaelsHorn": ["Relax", "Relax"],
    "Storybook": ["BrightestFlame"],
    "CallingBell": ["CurseOfTheBell"],
    "DustyTome": ["Corruption+"],  # 아이언클래드는 늘 타락 (박살은 초월 카드라 빠진다), 강화된 채로
    "BloodSoakedRose": ["Enthralled"],
    "TanxsWhistle": ["Whistle"],
    "JewelryBox": ["Apotheosis"],
    "SereTalon": ["Wish", "Wish", "Wish"],
    "DistinguishedCape": ["Apparition", "Apparition", "Apparition"],
    "PreservedFog": ["Folly"],
}
RANDOM_CURSES = {"NeowsBones": 1, "SereTalon": 2}  # 무작위 저주 수 (시뮬레이터 RollGeneratableCurse)
RANDOM_RARE = {"ArcaneScroll": 1}  # 아이언클래드 희귀 카드 무작위 1 장
MAX_HP_DELTA = {"DistinguishedCape": -9}

# 무작위 풀에서 뺄 카드: 위 유물로만 얻는 카드 + 아케익 투스의 박살
RESERVED = {"NeowsFury", "Relax", "BrightestFlame", "CurseOfTheBell", "Corruption", "Enthralled", "Whistle",
            "Apotheosis", "Wish", "Apparition", "Maul", "Break"}


class RelicCards:
    def __init__(self, run):
        lib = run.lib
        self.card_id = {info.name: cid for cid, info in lib.cards.items()}
        self.relic_name = {rid: name.replace(" ", "") for rid, name in lib.names.relics().items()}
        self.lib = lib
        src = (lib.repo / "src" / "Sts2Emulator" / "Core" / "Run" / "RunNonCombatEffects.cs").read_text(encoding="utf-8")
        m = re.search(r"RareIroncladSingleplayerPool\s*=\s*\[([^\]]*)\]", src)
        self.rare_pool = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
        self.rare_pool = [c for c in self.rare_pool if c not in lib.multiplayer_only]
        cards_src = (lib.repo / "src" / "Sts2Emulator" / "Generated" / "Cards.g.cs").read_text(encoding="utf-8")
        no_mod = {int(x) for x in re.findall(r"CardDef\(Id: (\d+),[^\n]*CanBeGeneratedByModifiers: false", cards_src)}
        self.mod_curses = sorted(c for c, info in lib.cards.items()
                                 if info.card_type == "Curse" and c not in no_mod and c not in lib.multiplayer_only)
        self.reserved = {self.card_id[n] for n in RESERVED if n in self.card_id}
        self.bash, self.break_ = self.card_id.get("Bash"), self.card_id.get("Break")
        self.maul = self.card_id.get("Maul")
        self.starters = {self.card_id.get("StrikeIronclad"), self.card_id.get("DefendIronclad")}

    def _is_curse(self, cid: int) -> bool:
        info = self.lib.cards.get(cid)
        return info is not None and info.card_type == "Curse"

    def apply(self, deck: list[tuple[int, bool]], relics: list[int], max_hp: int, hp: int,
              rng: np.random.Generator) -> tuple[list[tuple[int, bool]], int, int, list[str]]:
        """유물마다 덱을 고친다. (새 덱, 최대 HP, HP, 적용한 유물 이름) 을 돌려준다."""
        deck = list(deck)
        applied = []
        for rid in relics:
            name = self.relic_name.get(rid, "")
            touched = False
            for card in ADD.get(name, []):
                up = card.endswith("+")
                cid = self.card_id.get(card.rstrip("+"))
                if cid is not None:
                    deck.append((cid, up))
                    touched = True
            for _ in range(RANDOM_CURSES.get(name, 0)):
                if self.mod_curses:
                    deck.append((int(rng.choice(self.mod_curses)), False))
                    touched = True
            for _ in range(RANDOM_RARE.get(name, 0)):
                if self.rare_pool:
                    deck.append((int(rng.choice(self.rare_pool)), False))
                    touched = True
            if name == "ArchaicTooth" and self.bash is not None:
                # 덱의 첫 강타를 박살로 (시뮬레이터: 덱 순서의 첫 번째, 강화는 이어지지 않음)
                for i, (cid, _up) in enumerate(deck):
                    if cid == self.bash:
                        deck[i] = (self.break_, False)
                        touched = True
                        break
            if name == "Claws" and self.maul is not None:
                # 덱에서 6 장을 골라 할퀴기로 바꾼다. 사람처럼 타격·수비부터, 모자라면 저주 아닌 카드 중 무작위.
                # (게임은 원래 카드가 강화돼 있으면 할퀴기도 강화한다)
                first = [i for i, (c, _) in enumerate(deck) if c in self.starters]
                rest = [i for i, (c, _) in enumerate(deck) if c not in self.starters and not self._is_curse(c)]
                rng.shuffle(rest)
                for i in (first + rest)[:6]:
                    deck[i] = (self.maul, deck[i][1])
                    touched = True
            if name == "PreservedFog":
                # 덱에서 3 장을 지운 뒤 어리석음 (어리석음은 위 ADD 에서 더한다). 사람처럼 저주부터, 다음 타격·수비.
                curses = [i for i, (c, _) in enumerate(deck) if self._is_curse(c) and c != self.card_id.get("Folly")]
                basics = [i for i, (c, _) in enumerate(deck) if c in self.starters]
                for i in sorted((curses + basics)[:3], reverse=True):
                    deck.pop(i)
                    touched = True
            if name in MAX_HP_DELTA:
                max_hp = max(1, max_hp + MAX_HP_DELTA[name])
                hp = min(hp, max_hp)
                touched = True
            if touched:
                applied.append(name)
        return deck, max_hp, hp, applied
