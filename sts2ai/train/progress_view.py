"""전투만 학습한 폴더의 학습 추이(history.json)를 progress.html 한 장으로 만든다.

학습(train/combat_only.py)이 반복마다 write_progress 로 다시 쓴다. 손으로는:
    .venv\\Scripts\\python -c "from sts2ai.train.progress_view import write_progress; write_progress('runs/combat_only_v7')"

그래프 세 개 (모두 반복이 x 축, 한 축만 씀)
- 검증 승률: 10 반복마다 고정 검증 전투 156 개를 가장 확률 높은 행동으로. 적 종류별 + 전체.
- 학습 승률: 반복마다 학습 전투 1248 개 (확률대로 뽑음, 잘 지는 적을 더 자주). 최근 5 반복 이동평균.
- 이긴 전투의 남은 HP 비율: 학습(이동평균)과 검증.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

KINDS = (("Weak", "약한 적"), ("Normal", "일반"), ("Elite", "엘리트"), ("Boss", "보스"))


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def _rolling(xs: list, k: int = 5) -> list:
    out = []
    for i in range(len(xs)):
        w = [v for v in xs[max(0, i - k + 1): i + 1] if v is not None]
        out.append(round(sum(w) / len(w), 4) if w else None)
    return out


def collect(history: list[dict]) -> dict:
    """그래프용 데이터. x = 끝낸 반복 수 (학습 전 검증 = 0)."""
    train = [r for r in history if r.get("iter", -1) >= 0 and "win" in r]
    xs = [r["iter"] + 1 for r in train]
    data = {"train_x": xs, "train": {}, "train_raw": {}, "eval": {}, "hp": {}}
    for key, _ in (("", "전체"),) + KINDS:
        col = "win" + (f"_{key}" if key else "")
        raw = [_num(r.get(col)) for r in train]
        data["train_raw"][key or "All"] = raw
        data["train"][key or "All"] = _rolling(raw)
    ev = [r for r in history if "eval_win" in r]
    data["eval_x"] = [r["iter"] + 1 for r in ev]
    for key, _ in (("", "전체"),) + KINDS:
        col = "eval_win" + (f"_{key}" if key else "")
        data["eval"][key or "All"] = [_num(r.get(col)) for r in ev]
    data["hp"]["train"] = _rolling([_num(r.get("hp_left_win")) for r in train])
    data["hp"]["train_raw"] = [_num(r.get("hp_left_win")) for r in train]
    data["hp"]["eval"] = [_num(r.get("eval_hp_left_win")) for r in ev]
    data["sec"] = [_num(r.get("sec")) for r in train]
    return data


PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;600;700&display=swap">
<style>
/* 레이아웃: 맨 위 숫자 세 개(최근 값), 아래로 그래프 세 장(검증 승률, 학습 승률, 남은 HP), 끝에 표. 색: 적 종류 4 가지 = 검증한 순서 고정 팔레트, 전체 = 글자색. */
:root{--bg:#f5f6f8;--surface:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--faint:#8a8984;--grid:#e6e5e0;--line:#d9dee6;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--all:#0b0b0b;
--f-body:"IBM Plex Sans KR",system-ui,"Malgun Gothic",sans-serif;--f-num:"IBM Plex Mono",ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--faint:#8f8e88;--grid:#2e2e2b;--line:#33332f;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--all:#f0efec;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--faint:#8f8e88;--grid:#2e2e2b;--line:#33332f;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--all:#f0efec;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 var(--f-body)}
.wrap{max-width:1080px;margin:0 auto;padding-block:20px 40px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:16px}
h1{font-size:20px;margin:0}
.sub{color:var(--mut);margin:0;font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile .k{color:var(--mut);font-size:12.5px}
.tile .v{font-family:var(--f-num);font-size:28px;font-weight:500;font-variant-numeric:tabular-nums;line-height:1.2}
.tile .d{color:var(--mut);font-size:12.5px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px;display:grid;gap:8px;min-width:0}
.card h2{font-size:15px;margin:0}
.card p{margin:0;color:var(--mut);font-size:12.5px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--mut)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{display:inline-block;width:14px;height:2px;border-radius:1px;background:var(--c)}
.legend i.dot{width:8px;height:8px;border-radius:50%}
.chart{position:relative;min-width:0}
.chart svg{display:block;width:100%;height:auto;overflow:visible}
.chart text{font-family:var(--f-num);font-size:11px;fill:var(--mut)}
.chart .lab{font-family:var(--f-body);font-size:11.5px;fill:var(--fg)}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);white-space:nowrap}
.tip b{font-family:var(--f-num);font-weight:500}
.tip .r{display:flex;gap:8px;justify-content:space-between}
.tip .r span:first-child{display:inline-flex;align-items:center;gap:5px}
.tip .r i{width:8px;height:8px;border-radius:50%;background:var(--c)}
.tip[hidden]{display:none}
details{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 14px}
summary{cursor:pointer;color:var(--mut)}
.scroll{overflow-x:auto;margin-top:8px}
table{border-collapse:collapse;font-size:12.5px;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
th{color:var(--mut);font-weight:600}td{font-family:var(--f-num)}
th:first-child,td:first-child{text-align:left}
</style>
</head>
<body>
<div class="wrap">
  <header style="display:grid;gap:4px"><h1>__TITLE__</h1>
  <p class="sub">학습이 반복마다 이 파일을 다시 씁니다. 새로고침하면 최신입니다. 마우스를 올리면 그 반복의 값이 보입니다.</p></header>
  <div class="tiles" id="tiles"></div>
  <section class="card"><h2>검증 승률</h2>
    <p>고정 검증 전투 156 개(적 78 × 2)를 가장 확률 높은 행동으로 싸운 결과. 10 반복마다. 보스·엘리트는 12~24 판이라 한 판에 4~8%p 씩 움직입니다.</p>
    <div class="legend" id="lg-eval"></div><div class="chart" id="c-eval"></div></section>
  <section class="card"><h2>학습 승률</h2>
    <p>반복마다 학습 전투 1248 개. 확률대로 행동을 뽑고 잘 지는 적과 더 자주 싸워서 검증보다 낮습니다. 선은 최근 5 반복 평균.</p>
    <div class="legend" id="lg-train"></div><div class="chart" id="c-train"></div></section>
  <section class="card"><h2>이긴 전투의 남은 HP</h2>
    <p>이긴 전투에서 끝났을 때 HP ÷ 최대 HP. 승률이 그대로여도 덜 다치고 이기는지 봅니다.</p>
    <div class="legend" id="lg-hp"></div><div class="chart" id="c-hp"></div></section>
  <details><summary>표로 보기</summary><div class="scroll"><table id="tbl"></table></div></details>
</div>
<script>
const D = __DATA__;
const KINDS = [["Weak","약한 적","--s1"],["Normal","일반","--s2"],["Elite","엘리트","--s3"],["Boss","보스","--s4"]];
const pct = v => v == null ? "–" : (v * 100).toFixed(0) + "%";
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }

function legend(id, items) {
  document.getElementById(id).innerHTML = items.map(([name, c, dot]) =>
    `<span style="--c:var(${c})"><i class="${dot ? "dot" : ""}"></i>${name}</span>`).join("");
}

// series: [{name, color, xs, ys, width, dots}]
function lineChart(id, series, opt) {
  const host = document.getElementById(id);
  const W = 760, H = 250, L = 42, R = 86, T = 10, B = 28;
  const xsAll = series.flatMap(s => s.xs);
  if (!xsAll.length) { host.textContent = "아직 값이 없습니다."; return; }
  const x0 = Math.min(0, ...xsAll), x1 = Math.max(10, ...xsAll);
  const X = x => L + (x - x0) / (x1 - x0) * (W - L - R), Y = y => T + (1 - y) * (H - T - B);
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opt.label}, host);
  for (const g of [0, .25, .5, .75, 1]) {
    el("line", {x1: L, x2: W - R, y1: Y(g), y2: Y(g), stroke: "var(--grid)", "stroke-width": 1}, svg);
    el("text", {x: L - 6, y: Y(g) + 4, "text-anchor": "end"}, svg).textContent = (g * 100) + "%";
  }
  const span = x1 - x0, step = [1, 2, 5, 10, 20, 25, 50, 100].find(s => span / s <= 10) || 100;
  for (let t = Math.ceil(x0 / step) * step; t <= x1; t += step)
    el("text", {x: X(t), y: H - 8, "text-anchor": "middle"}, svg).textContent = t;
  const labels = [];
  for (const s of series) {
    let d = "", pen = false;
    s.xs.forEach((x, i) => { const y = s.ys[i]; if (y == null) { pen = false; return; }
      d += (pen ? "L" : "M") + X(x).toFixed(1) + " " + Y(y).toFixed(1); pen = true; });
    if (d) el("path", {d, fill: "none", stroke: `var(${s.color})`, "stroke-width": s.width || 2,
      "stroke-linejoin": "round", "stroke-linecap": "round", opacity: s.faint ? .35 : 1}, svg);
    if (s.dots) s.xs.forEach((x, i) => { if (s.ys[i] == null) return;
      el("circle", {cx: X(x), cy: Y(s.ys[i]), r: 4, fill: `var(${s.color})`, stroke: "var(--surface)", "stroke-width": 2}, svg); });
    if (!s.faint) { let k = s.ys.length - 1; while (k >= 0 && s.ys[k] == null) k--;
      if (k >= 0) labels.push({y: Y(s.ys[k]), x: X(s.xs[k]), name: s.name, v: s.ys[k], color: s.color}); }
  }
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
  for (const lb of labels) {
    el("circle", {cx: W - R + 8, cy: lb.y - 4, r: 3, fill: `var(${lb.color})`}, svg);
    el("text", {x: W - R + 14, y: lb.y, class: "lab"}, svg).textContent = `${lb.name} ${pct(lb.v)}`;
  }
  // 마우스: 가장 가까운 반복에 세로선 + 값 상자
  const cross = el("line", {y1: T, y2: H - B, stroke: "var(--faint)", "stroke-width": 1, visibility: "hidden"}, svg);
  const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; host.appendChild(tip);
  const hit = el("rect", {x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent"}, svg);
  const uniq = [...new Set(xsAll)].sort((a, b) => a - b);
  function move(ev) {
    const r = svg.getBoundingClientRect(), mx = (ev.clientX - r.left) / r.width * W;
    let best = uniq[0]; for (const x of uniq) if (Math.abs(X(x) - mx) < Math.abs(X(best) - mx)) best = x;
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best)); cross.setAttribute("visibility", "visible");
    const rows = series.filter(s => !s.faint).map(s => { const i = s.xs.indexOf(best);
      return i < 0 ? "" : `<div class="r" style="--c:var(${s.color})"><span><i></i>${s.name}</span><b>${pct(s.ys[i])}</b></div>`; }).join("");
    tip.innerHTML = `<div style="color:var(--mut);margin-bottom:2px">반복 ${best}${best === 0 ? " (학습 전)" : ""}</div>` + rows;
    tip.hidden = !rows;
    const px = X(best) / W * r.width, flip = px > r.width * .6;
    tip.style.left = (flip ? px - tip.offsetWidth - 10 : px + 10) + "px"; tip.style.top = "6px";
  }
  hit.addEventListener("mousemove", move);
  hit.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}

const ev = KINDS.map(([k, n, c]) => ({name: n, color: c, xs: D.eval_x, ys: D.eval[k], dots: true}));
ev.push({name: "전체", color: "--all", xs: D.eval_x, ys: D.eval.All, dots: true, width: 2.5});
legend("lg-eval", ev.map(s => [s.name, s.color, false]));
lineChart("c-eval", ev, {label: "검증 승률"});

const tr = KINDS.map(([k, n, c]) => ({name: n, color: c, xs: D.train_x, ys: D.train[k]}));
tr.push({name: "전체", color: "--all", xs: D.train_x, ys: D.train.All, width: 2.5});
legend("lg-train", tr.map(s => [s.name, s.color, false]));
lineChart("c-train", tr, {label: "학습 승률"});

const hp = [{name: "학습", color: "--s1", xs: D.train_x, ys: D.hp.train},
            {name: "검증", color: "--s2", xs: D.eval_x, ys: D.hp.eval, dots: true}];
legend("lg-hp", [["학습 (5 반복 평균)", "--s1", false], ["검증", "--s2", true]]);
lineChart("c-hp", hp, {label: "이긴 전투의 남은 HP"});

// 숫자 세 개
const last = a => { for (let i = a.length - 1; i >= 0; i--) if (a[i] != null) return [a[i], i]; return [null, -1]; };
const [ew, ei] = last(D.eval.All), [tw, ti] = last(D.train.All), [hv] = last(D.hp.train);
const first = a => a.find(v => v != null);
document.getElementById("tiles").innerHTML = [
  ["검증 승률 (최근)", pct(ew), ei >= 0 ? `반복 ${D.eval_x[ei]} · 학습 전 ${pct(first(D.eval.All))}` : ""],
  ["학습 승률 (최근 5 반복)", pct(tw), ti >= 0 ? `반복 ${D.train_x[ti]} · 처음 ${pct(first(D.train.All))}` : ""],
  ["이긴 전투의 남은 HP", pct(hv), `처음 ${pct(first(D.hp.train))}`],
].map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");

// 표
const ex = new Map(D.eval_x.map((x, i) => [x, i]));
let h = "<tr><th>반복</th><th>학습 승률</th>" + KINDS.map(k => `<th>학습 ${k[1]}</th>`).join("") +
  "<th>검증 승률</th>" + KINDS.map(k => `<th>검증 ${k[1]}</th>`).join("") + "<th>남은 HP (학습)</th><th>남은 HP (검증)</th><th>초</th></tr>";
const xs = [...new Set([...D.eval_x, ...D.train_x])].sort((a, b) => b - a);
for (const x of xs) { const i = D.train_x.indexOf(x), j = ex.has(x) ? ex.get(x) : -1;
  h += `<tr><td>${x}${x === 0 ? " (학습 전)" : ""}</td><td>${i < 0 ? "" : pct(D.train_raw.All[i])}</td>` +
    KINDS.map(k => `<td>${i < 0 ? "" : pct(D.train_raw[k[0]][i])}</td>`).join("") +
    `<td>${j < 0 ? "" : pct(D.eval.All[j])}</td>` + KINDS.map(k => `<td>${j < 0 ? "" : pct(D.eval[k[0]][j])}</td>`).join("") +
    `<td>${i < 0 ? "" : pct(D.hp.train_raw[i])}</td><td>${j < 0 ? "" : pct(D.hp.eval[j])}</td><td>${i < 0 || D.sec[i] == null ? "" : D.sec[i].toFixed(0)}</td></tr>`; }
document.getElementById("tbl").innerHTML = h;
</script>
</body>
</html>
"""


def write_progress(folder) -> Path:
    folder = Path(folder)
    history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
    page = PAGE.replace("__TITLE__", f"학습 추이 · {folder.name}").replace(
        "__DATA__", json.dumps(collect(history), ensure_ascii=False))
    tmp = folder / "progress.html.tmp"
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(folder / "progress.html")
    return folder / "progress.html"
