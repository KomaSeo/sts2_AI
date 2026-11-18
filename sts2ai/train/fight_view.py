"""전투만 학습한 폴더의 전투 기록(fights/iter_NNNN.json)을 HTML·Markdown 으로 바꾼다.

학습(train/combat_only.py)이 반복마다 write_all 로 <폴더>/fights.html 을 다시 쓴다.
"""

from __future__ import annotations

import html
import json
import os
import re
from functools import lru_cache
from pathlib import Path


def load(folder: Path) -> list[tuple[int, list[dict]]]:
    out = []
    for p in sorted((folder / "fights").glob("iter_*.json")):
        try:
            out.append((int(p.stem.split("_")[1]), json.loads(p.read_text(encoding="utf-8"))))
        except (ValueError, json.JSONDecodeError):
            continue  # 쓰는 중인 파일
    return out


# ---- 한국어 이름 ----
# 게임 설치 폴더의 SlayTheSpire2.pck 에서 꺼낸 localization/{eng,kor}/*.json 을 쓴다 (저장소에는 넣지 않는다).
# 영문 제목(또는 ID)을 소문자·영숫자만 남겨 맞춘다. 폴더가 없거나 못 찾은 이름은 영문 그대로.
LOC_DIR = Path(os.environ.get("STS2_LOC", r"C:\Users\Ic_Lab\sts2_scratch\loc"))
_LOC_FILES = {"card": ("cards", ".title"), "relic": ("relics", ".title"), "potion": ("potions", ".title"),
              "monster": ("monsters", ".name"), "intent": ("intents", ".title")}
KIND_KO = {"Weak": "약한 적", "Normal": "일반", "Elite": "엘리트", "Boss": "보스"}
# 카드 고르기 화면 종류 (시뮬레이터 CardSelectionKind). 게임 문구가 없어 직접 옮겼다.
SELECT_KO = {
    "DiscardToDrawPileTop": "버린 카드 → 뽑을 더미 맨 위", "ExhaustFromHand": "손패 소멸",
    "ExhaustFromHandThenDraw": "손패 소멸 후 뽑기", "DrawPileToHand": "뽑을 더미 → 손패",
    "HandToDrawPileTop": "손패 → 뽑을 더미 맨 위", "ExhaustFromHandRepeated": "손패 소멸 (여러 장)",
    "GeneratedCardToHand": "새 카드 하나 가져오기", "CurseOfKnowledge": "지식의 저주 고르기",
    "DiscardFromHandRepeated": "손패 버리기 (여러 장)", "MarkHandCardSly": "손패에 교활 표시",
    "QueueHandCardCopies": "손패 카드 복사", "RetainForNextTurn": "다음 턴까지 보존",
    "DiscardToHand": "버린 카드 → 손패", "DiscardAnyThenDraw": "버리고 뽑기", "UpgradeInHand": "손패 강화",
    "TransformDrawPileToSoul": "뽑을 더미 카드를 영혼으로", "ExhaustFromDrawPile": "뽑을 더미 소멸",
    "GrantEtherealInHand": "손패에 휘발성 부여", "TransfigureInHand": "손패 변형",
    "TransformHandToMinionStrike": "손패를 하수인 타격으로", "TransformDrawToMinionDiveBomb": "뽑을 더미를 하수인 급강하로",
    "TransformHandToMinionSacrifice": "손패를 하수인 희생으로", "AutoPlaySkillThrice": "스킬 세 번 자동 사용",
    "CloneColorlessInHand": "무색 카드 복제", "DualWield": "이도류 (복제)", "DiscardToHandFree": "버린 카드 → 손패 (무료)",
    "FreeCardForCombat": "이번 전투 동안 무료로",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


@lru_cache(maxsize=1)
def _tables() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for cat, (f, suffix) in _LOC_FILES.items():
        try:
            eng = json.loads((LOC_DIR / "eng" / f"{f}.json").read_text(encoding="utf-8"))
            kor = json.loads((LOC_DIR / "kor" / f"{f}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        t: dict[str, str] = {}
        for key, v in eng.items():
            if key.endswith(suffix) and key in kor:
                t[_norm(key[: -len(suffix)])] = kor[key]
                t[_norm(v)] = kor[key]
        out[cat] = t
    return out


def ko(cat: str, name: str) -> str:
    """영문 이름 → 한국어. 카드의 + (강화) 는 그대로 붙인다."""
    plus = "+" if name.endswith("+") else ""
    base = name[:-1] if plus else name
    return _tables().get(cat, {}).get(_norm(base), base) + plus


def ko_action(s: str, select: str | None = None) -> str:
    """행동 이름: "카드 → 적0", "포션: 이름 → 적0", "고르기: 카드", "턴 종료", "고르지 않기".
    select(카드 고르기 화면 종류)가 있으면 "고르기(손패 강화): 카드" 처럼 붙인다."""
    act, _, target = s.partition(" → ")
    why = f"({SELECT_KO.get(select, select)})" if select else ""
    if act.startswith("포션: "):
        act = "포션: " + ko("potion", act[len("포션: "):])
    elif act.startswith("고르기: "):
        act = f"고르기{why}: " + ko("card", act[len("고르기: "):])
    elif act == "고르지 않기" and why:
        act = f"고르지 않기{why}"
    elif act not in ("턴 종료", "고르지 않기"):
        act = ko("card", act)
    return act + (f" → {target.replace('적', '적 ')}" if target else "")


def intent_str(x) -> str:
    if not x:
        return "?"
    kind, mag = x
    k = ko("intent", kind)
    return f"{k} {mag}" if mag else k


def enemies_of(s: dict) -> list[dict]:
    """fmt 2: 이름·HP·방어도·의도. 예전 기록(fmt 없음)은 적 HP 만 있고, 배율을 잘못(×100) 되돌려 두 배였다 → 반으로."""
    if "enemies" in s:
        return s["enemies"]
    return [{"name": "?", "hp": round(h / 2)} for h in s.get("enemy_hp", [])]


def enemy_str(e: dict) -> str:
    out = f"{ko('monster', e['name'])} HP {e['hp']}" + (f" 방어 {e['block']}" if e.get("block") else "")
    if "intent" in e:
        out += f" · {intent_str(e['intent'])}" + (f" → 다음 {intent_str(e['next'])}" if e.get("next") else "")
    return out


def hand_str(s: dict) -> str:
    return ", ".join(f"{ko('card', n)}({c})" for n, c, _ in s.get("hand", [])) if "hand" in s else "-"


def hand_html(s: dict) -> str:
    if "hand" not in s:
        return '<span class="na">기록 없음</span>'
    return "".join(f'<span class="card{"" if ok else " off"}" title="{html.escape(n)}">{html.escape(ko("card", n))}'
                   f'<i>{c}</i></span>' for n, c, ok in s["hand"]) or '<span class="na">없음</span>'


def who(rec: dict) -> str:
    return ", ".join(ko("monster", n) for n in rec.get("enemy_names") or []) or f"적 {rec['encounter']}"


def kind_ko(rec: dict) -> str:
    return KIND_KO.get(rec["kind"], rec["kind"])


def names(cat: str, xs) -> str:
    return ", ".join(ko(cat, x) for x in xs) or "없음"


def head(rec: dict) -> str:
    return (f"{'승리' if rec['won'] else '패배'} · {who(rec)} (막 {rec['sim_act']}, {kind_ko(rec)}) · "
            f"HP {rec['hp_start']}/{rec['max_hp']} → {rec['hp_end']} · 점수 {rec['score']}")


def to_md(it: int, recs: list[dict]) -> str:
    lines = [f"# 반복 {it}", ""]
    for r in recs:
        lines += [f"## {head(r)}", "", f"- 덱 {len(r['deck'])}장: " + names("card", r["deck"]),
                  "- 포션: " + names("potion", r["potions"]),
                  "- 유물: " + names("relic", r.get("relics", [])), "",
                  "| 턴 | HP | 방어 | 에너지 | 적 (HP · 이번 턴 행동) | 손패 (비용) | 고른 행동 | 확률 | 확률 높은 행동 3개 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s in r["steps"]:
            top = " / ".join(f"{ko_action(n, s.get('select'))} {p}%" for n, p in s["top"])
            lines.append(f"| {s['turn']} | {s['hp']} | {s.get('block', '')} | {s['energy']} | "
                         f"{'<br>'.join(enemy_str(e) for e in enemies_of(s))} | {hand_str(s)} | "
                         f"{ko_action(s['action'], s.get('select'))} | {s['prob_pct']}% | {top} |")
        lines.append("")
    return "\n".join(lines)


def fight_html(r: dict) -> str:
    e = html.escape
    rows, prev_turn, band = [], None, 0
    for s in r["steps"]:
        if s["turn"] != prev_turn:
            band ^= 1
            prev_turn = s["turn"]
        chosen = s["action"].split(" → ")[0]
        top = "".join(f'<span class="alt{" chosen" if n == chosen else ""}">{e(ko_action(n, s.get("select")))} '
                      f'<b>{p}%</b></span>' for n, p in s["top"])
        p = s["prob_pct"]
        rows.append(
            f'<tr class="b{band}"><td class="n">{s["turn"]}</td><td class="n">{s["hp"]}</td>'
            f'<td class="n">{s.get("block", "")}</td><td class="n">{s["energy"]}</td>'
            f'<td class="en">{"<br>".join(e(enemy_str(x)) for x in enemies_of(s))}</td>'
            f'<td class="hand">{hand_html(s)}</td>'
            f'<td class="act" title="{e(s["action"])}">{e(ko_action(s["action"], s.get("select")))}</td>'
            f'<td class="pr"><div class="bar"><i style="width:{min(100, p)}%"></i></div><span>{p}%</span></td>'
            f'<td class="top">{top}</td></tr>')
    cls = "win" if r["won"] else "loss"
    return f"""<section class="fight {cls}">
<h3><span class="tag">{'승리' if r['won'] else '패배'}</span> {e(who(r))} · 막 {r['sim_act']} · {e(kind_ko(r))}
 <small>HP {r['hp_start']}/{r['max_hp']} → {r['hp_end']} · 점수 {r['score']} · 행동 {len(r['steps'])}번</small></h3>
<details><summary>덱 {len(r['deck'])}장 · 포션 {len(r['potions'])}개 · 유물 {len(r.get('relics', []))}개</summary>
<p>{e(names('card', r['deck']))}</p><p>포션: {e(names('potion', r['potions']))}</p>
<p>유물: {e(names('relic', r.get('relics', [])))}</p></details>
<div class="scroll"><table><thead><tr><th>턴</th><th>HP</th><th>방어</th><th>에너지</th><th>적 (HP · 이번 턴 행동)</th><th>손패 (비용)</th><th>고른 행동</th>
<th>확률</th><th>확률 높은 행동 3개</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"""


CSS = """
/* 레이아웃: 위에 고정된 반복 선택 막대, 아래로 전투 카드(승리·패배) 두 장, 카드 안은 가로 스크롤 표 */
:root{--bg:#f6f7f9;--fg:#1b2430;--mut:#5d6878;--line:#dde2e8;--card:#ffffff;--band:#eef1f5;
--win:#1d7a46;--loss:#b4322b;--acc:#2f6fb3;--accsoft:#d5e3f3;
--f-body:"IBM Plex Sans KR",system-ui,"Malgun Gothic",sans-serif;--f-num:"IBM Plex Mono",ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#12161c;--fg:#e6eaf0;--mut:#98a3b3;--line:#2a313b;--card:#1a2028;--band:#161b22;
--win:#4fbf7f;--loss:#ec6a60;--acc:#6fa6e0;--accsoft:#22344a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#12161c;--fg:#e6eaf0;--mut:#98a3b3;--line:#2a313b;--card:#1a2028;--band:#161b22;
--win:#4fbf7f;--loss:#ec6a60;--acc:#6fa6e0;--accsoft:#22344a;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 var(--f-body)}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:10;background:var(--bg)}
header{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);border-bottom:1px solid var(--line);
padding-block:10px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.hrow{max-width:1240px;margin:0 auto;display:flex;gap:8px 12px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0;text-wrap:balance}.hint{color:var(--mut);font-size:12px;flex-basis:100%}
.nav{display:flex;gap:6px;align-items:center}
select,button{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
button:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
main{max-width:1240px;margin:0 auto;padding-block:16px 32px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:16px}
.iter[hidden]{display:none}.iter{display:grid;gap:16px;min-width:0}
.fight{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--c);border-radius:8px;padding:12px 14px;min-width:0;display:grid;gap:8px}
.win{--c:var(--win)}.loss{--c:var(--loss)}
h3{margin:0;font-size:15px;text-wrap:balance}h3 small{display:block;color:var(--mut);font-weight:400;font-size:12.5px;font-family:var(--f-num)}
.tag{color:var(--c);font-weight:700;margin-right:6px}
details{color:var(--mut);font-size:13px}details p{margin:4px 0;line-height:1.55}summary{cursor:pointer}
.scroll{overflow-x:auto;min-width:0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12px;letter-spacing:.02em;position:sticky;top:0;background:var(--card)}
.n{text-align:right;font-family:var(--f-num);font-variant-numeric:tabular-nums}tr.b1 td{background:var(--band)}
td.en{font-size:12px;line-height:1.4}
td.hand{white-space:normal;min-width:220px;max-width:360px}.na{color:var(--mut);font-size:12px}
.card{display:inline-flex;align-items:baseline;gap:3px;font-size:12px;padding:0 6px;margin:1px 3px 1px 0;border:1px solid var(--line);border-radius:4px;background:var(--bg)}
.card i{font-style:normal;font-family:var(--f-num);color:var(--acc);font-size:11px}.card.off{color:var(--mut);border-style:dashed}
.pr{display:flex;align-items:center;gap:6px}.bar{width:56px;height:6px;background:var(--accsoft);border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--acc)}.pr span{font-family:var(--f-num);font-variant-numeric:tabular-nums;min-width:46px;text-align:right}
.top{color:var(--mut);font-size:12px}.alt{margin-right:10px}.alt b{font-family:var(--f-num);font-weight:500}.alt.chosen{color:var(--fg)}
.empty{color:var(--mut)}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
         '&family=IBM+Plex+Sans+KR:wght@400;600;700&display=swap">')


def body_html(folder: Path, data: list[tuple[int, list[dict]]]) -> str:
    opts = "".join(f'<option value="{it}">반복 {it}</option>' for it, _ in reversed(data))
    secs = "".join(f'<div class="iter" id="it{it}" hidden>' + "".join(fight_html(r) for r in recs) + "</div>"
                   for it, recs in data)
    return f"""<header><div class="hrow"><h1>전투 기록 · {html.escape(folder.name)}</h1>
<div class="nav"><button id="prev" type="button" aria-label="이전 반복">◀</button><select id="sel" aria-label="반복 고르기">{opts}</select>
<button id="next" type="button" aria-label="다음 반복">▶</button></div>
<div class="hint">반복마다 이긴 전투 하나, 진 전투 하나. 확률은 그 전투를 싸운 가중치로 잰 값입니다.
 적 행동의 수치는 한 번 때리는 피해입니다 (여러 번 때리는지는 관측에 없음). 손패에서 점선 카드는 지금 낼 수 없는 카드, 숫자는 지금 비용입니다.</div></div></header>
<main>{secs or '<p class="empty">아직 기록이 없습니다.</p>'}</main>
<script>
const sel=document.getElementById('sel');
function show(v){{document.querySelectorAll('.iter').forEach(d=>{{d.hidden=d.id!=='it'+v;}});sel.value=v;}}
sel.onchange=()=>show(sel.value);
document.getElementById('prev').onclick=()=>{{if(sel.selectedIndex<sel.options.length-1){{sel.selectedIndex++;show(sel.value);}}}};
document.getElementById('next').onclick=()=>{{if(sel.selectedIndex>0){{sel.selectedIndex--;show(sel.value);}}}};
if(sel.options.length)show(sel.options[0].value);
</script>"""


def to_html(folder: Path, data: list[tuple[int, list[dict]]]) -> str:
    """브라우저로 바로 여는 파일."""
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<title>전투 기록</title>{FONTS}<style>{CSS}</style></head><body>{body_html(folder, data)}</body></html>')


def to_artifact(folder: Path, data: list[tuple[int, list[dict]]]) -> str:
    """claude.ai 아티팩트로 올릴 내용 (문서 뼈대는 올릴 때 붙는다)."""
    return (f"<title>전투 학습 기록</title>"
            f'<meta name="claude:viewport-fit-sides" content="cover">{FONTS}<style>{CSS}</style>'
            f'{body_html(folder, data)}')


def write_all(folder: Path, md: bool = False, artifact: str = "") -> int:
    """<폴더>/fights.html 을 다시 쓰고 (md 면 반복마다 .md 도), 반복 수를 돌려준다."""
    folder = Path(folder)
    data = load(folder)
    tmp = folder / "fights.html.tmp"
    tmp.write_text(to_html(folder, data), encoding="utf-8")
    tmp.replace(folder / "fights.html")
    if artifact:
        Path(artifact).write_text(to_artifact(folder, data), encoding="utf-8")
    if md:
        for it, recs in data:
            (folder / "fights" / f"iter_{it:04d}.md").write_text(to_md(it, recs), encoding="utf-8")
    return len(data)
