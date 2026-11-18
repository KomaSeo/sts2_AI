r"""전투만 학습한 폴더의 전투 기록(fights/iter_NNNN.json)을 HTML 한 장으로 만든다.

    .venv\Scripts\python scripts\fights_html.py runs\combat_only_v1          # → runs\combat_only_v1\fights.html
    .venv\Scripts\python scripts\fights_html.py runs\combat_only_v1 --md     # 반복마다 fights\iter_NNNN.md 도

학습 중에도 돌릴 수 있다 (그때까지 쌓인 반복만 보인다).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load(folder: Path) -> list[tuple[int, list[dict]]]:
    out = []
    for p in sorted((folder / "fights").glob("iter_*.json")):
        try:
            out.append((int(p.stem.split("_")[1]), json.loads(p.read_text(encoding="utf-8"))))
        except (ValueError, json.JSONDecodeError):
            continue  # 쓰는 중인 파일
    return out


INTENT_KO = {"Attack": "공격", "Defend": "방어", "Buff": "강화", "Debuff": "약화", "Unknown": "알 수 없음"}


def intent_str(x) -> str:
    if not x:
        return "?"
    kind, mag = x
    k = INTENT_KO.get(kind, kind)
    return f"{k} {mag}" if mag else k


def enemies_of(s: dict) -> list[dict]:
    """fmt 2: 이름·HP·방어도·의도. 예전 기록(fmt 없음)은 적 HP 만 있고, 배율을 잘못(×100) 되돌려 두 배였다 → 반으로."""
    if "enemies" in s:
        return s["enemies"]
    return [{"name": "?", "hp": round(h / 2)} for h in s.get("enemy_hp", [])]


def enemy_str(e: dict) -> str:
    out = f"{e['name']} HP {e['hp']}" + (f" 방어 {e['block']}" if e.get("block") else "")
    if "intent" in e:
        out += f" · {intent_str(e['intent'])}" + (f" → 다음 {intent_str(e['next'])}" if e.get("next") else "")
    return out


def hand_str(s: dict) -> str:
    return ", ".join(f"{n}({c})" for n, c, _ in s.get("hand", [])) if "hand" in s else "-"


def hand_html(s: dict) -> str:
    if "hand" not in s:
        return '<span class="na">기록 없음</span>'
    return "".join(f'<span class="card{"" if ok else " off"}">{html.escape(n)}<i>{c}</i></span>'
                   for n, c, ok in s["hand"]) or '<span class="na">없음</span>'


def who(rec: dict) -> str:
    return ", ".join(rec.get("enemy_names") or []) or f"적 {rec['encounter']}"


def head(rec: dict) -> str:
    return (f"{'승리' if rec['won'] else '패배'} · {who(rec)} (막 {rec['sim_act']}, {rec['kind']}) · "
            f"HP {rec['hp_start']}/{rec['max_hp']} → {rec['hp_end']} · 점수 {rec['score']}")


def to_md(it: int, recs: list[dict]) -> str:
    lines = [f"# 반복 {it}", ""]
    for r in recs:
        lines += [f"## {head(r)}", "", f"- 덱 {len(r['deck'])}장: " + ", ".join(r["deck"]),
                  "- 포션: " + (", ".join(r["potions"]) or "없음"),
                  "- 유물: " + (", ".join(r.get("relics", [])) or "없음"), "",
                  "| 턴 | HP | 방어 | 에너지 | 적 (HP · 이번 턴 행동) | 손패 (비용) | 고른 행동 | 확률 | 확률 높은 행동 3개 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s in r["steps"]:
            top = " / ".join(f"{n} {p}%" for n, p in s["top"])
            lines.append(f"| {s['turn']} | {s['hp']} | {s.get('block', '')} | {s['energy']} | "
                         f"{'<br>'.join(enemy_str(e) for e in enemies_of(s))} | {hand_str(s)} | "
                         f"{s['action']} | {s['prob_pct']}% | {top} |")
        lines.append("")
    return "\n".join(lines)


def fight_html(r: dict) -> str:
    e = html.escape
    rows, prev_turn, band = [], None, 0
    for s in r["steps"]:
        if s["turn"] != prev_turn:
            band ^= 1
            prev_turn = s["turn"]
        top = "".join(f'<span class="alt{" chosen" if n == s["action"].split(" → ")[0] else ""}">{e(n)} '
                      f'<b>{p}%</b></span>' for n, p in s["top"])
        p = s["prob_pct"]
        rows.append(
            f'<tr class="b{band}"><td class="n">{s["turn"]}</td><td class="n">{s["hp"]}</td>'
            f'<td class="n">{s.get("block", "")}</td><td class="n">{s["energy"]}</td>'
            f'<td class="en">{"<br>".join(e(enemy_str(x)) for x in enemies_of(s))}</td>'
            f'<td class="hand">{hand_html(s)}</td>'
            f'<td class="act">{e(s["action"])}</td>'
            f'<td class="pr"><div class="bar"><i style="width:{min(100, p)}%"></i></div><span>{p}%</span></td>'
            f'<td class="top">{top}</td></tr>')
    cls = "win" if r["won"] else "loss"
    return f"""<section class="fight {cls}">
<h3><span class="tag">{'승리' if r['won'] else '패배'}</span> {e(who(r))} · 막 {r['sim_act']} · {e(r['kind'])}
 <small>HP {r['hp_start']}/{r['max_hp']} → {r['hp_end']} · 점수 {r['score']} · 행동 {len(r['steps'])}번</small></h3>
<details><summary>덱 {len(r['deck'])}장 · 포션 {len(r['potions'])}개 · 유물 {len(r.get('relics', []))}개</summary>
<p>{e(', '.join(r['deck']))}</p><p>포션: {e(', '.join(r['potions']) or '없음')}</p>
<p>유물: {e(', '.join(r.get('relics', [])) or '없음')}</p></details>
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
    return (f'<title>{html.escape(folder.name)} 전투 기록</title>'
            f'<meta name="claude:viewport-fit-sides" content="cover">{FONTS}<style>{CSS}</style>'
            f'{body_html(folder, data)}')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--md", action="store_true", help="반복마다 fights/iter_NNNN.md 도 쓴다")
    p.add_argument("--artifact", default="", help="아티팩트로 올릴 HTML 을 이 경로에 쓴다")
    args = p.parse_args()
    folder = Path(args.folder)
    data = load(folder)
    (folder / "fights.html").write_text(to_html(folder, data), encoding="utf-8")
    if args.artifact:
        Path(args.artifact).write_text(to_artifact(folder, data), encoding="utf-8")
    if args.md:
        for it, recs in data:
            (folder / "fights" / f"iter_{it:04d}.md").write_text(to_md(it, recs), encoding="utf-8")
    print(f"{folder / 'fights.html'} ({len(data)}개 반복)")


if __name__ == "__main__":
    main()
