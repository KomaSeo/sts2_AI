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


def head(rec: dict) -> str:
    return (f"{'승리' if rec['won'] else '패배'} · 적 {rec['encounter']} (막 {rec['sim_act']}, {rec['kind']}) · "
            f"HP {rec['hp_start']}/{rec['max_hp']} → {rec['hp_end']} · 점수 {rec['score']}")


def to_md(it: int, recs: list[dict]) -> str:
    lines = [f"# 반복 {it}", ""]
    for r in recs:
        lines += [f"## {head(r)}", "", f"- 덱 {len(r['deck'])}장: " + ", ".join(r["deck"]),
                  "- 포션: " + (", ".join(r["potions"]) or "없음"), "",
                  "| 턴 | HP | 에너지 | 적 HP | 고른 행동 | 확률 | 확률 높은 행동 3개 |", "|---|---|---|---|---|---|---|"]
        for s in r["steps"]:
            top = " / ".join(f"{n} {p}%" for n, p in s["top"])
            lines.append(f"| {s['turn']} | {s['hp']} | {s['energy']} | {', '.join(map(str, s['enemy_hp']))} | "
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
            f'<td class="n">{s["energy"]}</td><td class="n">{e(", ".join(map(str, s["enemy_hp"])))}</td>'
            f'<td class="act">{e(s["action"])}</td>'
            f'<td class="pr"><div class="bar"><i style="width:{min(100, p)}%"></i></div><span>{p}%</span></td>'
            f'<td class="top">{top}</td></tr>')
    cls = "win" if r["won"] else "loss"
    return f"""<section class="fight {cls}">
<h3><span class="tag">{'승리' if r['won'] else '패배'}</span> 적 {r['encounter']} · 막 {r['sim_act']} · {e(r['kind'])}
 <small>HP {r['hp_start']}/{r['max_hp']} → {r['hp_end']} · 점수 {r['score']} · 행동 {len(r['steps'])}번</small></h3>
<details><summary>덱 {len(r['deck'])}장 · 포션 {len(r['potions'])}개</summary>
<p>{e(', '.join(r['deck']))}</p><p>포션: {e(', '.join(r['potions']) or '없음')}</p></details>
<div class="scroll"><table><thead><tr><th>턴</th><th>HP</th><th>에너지</th><th>적 HP</th><th>고른 행동</th>
<th>확률</th><th>확률 높은 행동 3개</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"""


CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--card:#fff;--b1:#f5f5f4;--win:#15803d;--loss:#b91c1c;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#44403c;--card:#292524;--b1:#24201e;--win:#4ade80;--loss:#f87171;--acc:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,"Malgun Gothic",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
h1{font-size:16px;margin:0}select,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
main{padding:16px;max-width:1200px;margin:auto}.iter{display:none}.iter.on{display:block}
.fight{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:16px}
h3{margin:0 0 6px;font-size:15px}h3 small{color:var(--mut);font-weight:normal;margin-left:6px}
.tag{display:inline-block;padding:0 8px;border-radius:10px;color:#fff;font-size:12px;margin-right:6px}
.win .tag{background:var(--win)}.loss .tag{background:var(--loss)}details{color:var(--mut);margin-bottom:8px}details p{margin:4px 0}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:600}.n{text-align:right;font-variant-numeric:tabular-nums}tr.b1{background:var(--b1)}
.pr{display:flex;align-items:center;gap:6px}.bar{width:60px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
.pr span{font-variant-numeric:tabular-nums;min-width:44px;text-align:right}.top{color:var(--mut)}.alt{margin-right:10px}.alt.chosen{color:var(--fg)}
"""


def to_html(folder: Path, data: list[tuple[int, list[dict]]]) -> str:
    opts = "".join(f'<option value="{it}">반복 {it}</option>' for it, _ in reversed(data))
    secs = "".join(f'<div class="iter" id="it{it}"><h2 style="font-size:15px">반복 {it}</h2>'
                   + "".join(fight_html(r) for r in recs) + "</div>" for it, recs in data)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>전투 기록</title><style>{CSS}</style></head><body>
<header><h1>전투 기록 · {html.escape(folder.name)}</h1><button id="prev">◀</button><select id="sel">{opts}</select>
<button id="next">▶</button><span style="color:var(--mut)">반복마다 이긴 전투 하나, 진 전투 하나 · 확률은 그 전투를 싸운 가중치 기준</span></header>
<main>{secs or '<p>아직 기록이 없습니다.</p>'}</main>
<script>
const sel=document.getElementById('sel');
function show(v){{document.querySelectorAll('.iter').forEach(d=>d.classList.toggle('on',d.id==='it'+v));sel.value=v;}}
sel.onchange=()=>show(sel.value);
document.getElementById('prev').onclick=()=>{{if(sel.selectedIndex<sel.options.length-1){{sel.selectedIndex++;show(sel.value);}}}};
document.getElementById('next').onclick=()=>{{if(sel.selectedIndex>0){{sel.selectedIndex--;show(sel.value);}}}};
if(sel.options.length)show(sel.options[0].value);
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--md", action="store_true", help="반복마다 fights/iter_NNNN.md 도 쓴다")
    args = p.parse_args()
    folder = Path(args.folder)
    data = load(folder)
    (folder / "fights.html").write_text(to_html(folder, data), encoding="utf-8")
    if args.md:
        for it, recs in data:
            (folder / "fights" / f"iter_{it:04d}.md").write_text(to_md(it, recs), encoding="utf-8")
    print(f"{folder / 'fights.html'} ({len(data)}개 반복)")


if __name__ == "__main__":
    main()
