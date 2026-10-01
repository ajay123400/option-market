"""intraday_improve_report.py -- one HTML page comparing the original range
strategy with improvements 1 (rolls), 2 (range-based stop) and 3 (expiry day
only), each checked in-sample (2021-23) and out-of-sample (2024-26).

Needs results/intraday_range_bt/imp_*/days.csv (python intraday_range_bt.py --set improve).
Output: results/intraday_range_bt/improve_report.html
"""
import html
import json
import os
import sys

import pandas as pd

import paths

ROOT = os.path.join(paths.BASE_DIR, "results", "intraday_range_bt")
COLORS = ["#8a8f9c", "#4f5fe0", "#0e9f6e", "#e08a1e", "#c2419b", "#1aa3b8"]


def _dd(net):
    eq = net.cumsum()
    return float((eq - eq.cummax().clip(lower=0)).min()) if len(eq) else 0.0


def _stats(d):
    return {"days": len(d), "net": float(d.net_rs.sum()), "pts": float(d.pnl_pts.sum()),
            "ppd": float(d.pnl_pts.mean()) if len(d) else 0.0,
            "win": float((d.net_rs > 0).mean() * 100) if len(d) else 0.0, "dd": _dd(d.net_rs)}


def load():
    rows, curves = [], {}
    for v in sorted(os.listdir(ROOT)):
        if not v.startswith("imp_") or not os.path.isfile(os.path.join(ROOT, v, "days.csv")):
            continue
        cfg = json.load(open(os.path.join(ROOT, v, "summary.json"), encoding="utf-8"))["config"]
        d = pd.read_csv(os.path.join(ROOT, v, "days.csv")).sort_values("day")
        ins, outs = d[d.day < "2024-01-01"], d[d.day >= "2024-01-01"]
        exits = d.exit.value_counts()
        rows.append({"key": v, "label": cfg["label"], "group": cfg.get("group", ""),
                     "all": _stats(d), "in": _stats(ins), "out": _stats(outs),
                     "charges": float(d.charges.sum()), "rolls": float(d.rolls.mean()),
                     "prem_sl": int(exits.get("STOP LOSS", 0)), "range_sl": int(exits.get("RANGE SL", 0))})
        curves[v] = {"label": cfg["label"], "pts": [[r.day, round(c, 1)] for r, c in zip(d.itertuples(), d.pnl_pts.cumsum())]}
    return rows, curves


def _rs(v):
    return ("-" if v < 0 else "") + "₹" + f"{abs(v):,.0f}"


def _c(v, fmt, extra=""):
    cls = "up" if v > 0 else "down" if v < 0 else ""
    return f'<td class="{cls} {extra}">{fmt(v)}</td>'


def build():
    rows, curves = load()
    order = {"all days": 0, "expiry day": 1}
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    base = next(r for r in rows if r["key"] == "imp_all_orig")

    # the choice a trader could have made in 2021-23: best in-sample points per group
    picks = {g: max(rs, key=lambda r: r["in"]["pts"]) for g, rs in groups.items()}

    tables = []
    for g in sorted(groups, key=lambda x: order.get(x, 9)):
        rs = groups[g]
        if g == "all days":
            rs = sorted(rs, key=lambda r: r["key"] != "imp_all_orig")
        t = ['<div class="scroll"><table><tr><th rowspan="2">Variant</th><th colspan="3">2021–23 (in-sample)</th>'
             '<th colspan="3">2024–26 (out-of-sample)</th><th colspan="6">Whole period</th></tr>'
             '<tr><th>Days</th><th>Points</th><th>Net ₹</th><th>Days</th><th>Points</th><th>Net ₹</th>'
             '<th>Points</th><th>Net ₹</th><th>Win %</th><th>Max DD</th><th>Charges</th><th>SL days (prem / range)</th></tr>']
        for r in rs:
            both = r["in"]["net"] > 0 and r["out"]["net"] > 0
            mark = ("&#9733; " if r is picks[g] else "") + ("<span class='ok'>both periods +</span> " if both else "")
            cls = ' class="base"' if r["key"] == "imp_all_orig" else (' class="pick"' if r is picks[g] else "")
            pts = lambda v: f"{v:,.0f}"
            t.append(f"<tr{cls}><td>{mark}{html.escape(r['label'])}</td>"
                     f"<td>{r['in']['days']}</td>{_c(r['in']['pts'], pts)}{_c(r['in']['net'], _rs)}"
                     f"<td>{r['out']['days']}</td>{_c(r['out']['pts'], pts)}{_c(r['out']['net'], _rs)}"
                     f"{_c(r['all']['pts'], pts)}{_c(r['all']['net'], _rs, 'b')}<td>{r['all']['win']:.1f}%</td>"
                     f"{_c(r['all']['dd'], _rs)}<td>{_rs(r['charges'])}</td><td>{r['prem_sl']} / {r['range_sl']}</td></tr>")
        title = ("All days — one change at a time vs the original" if g == "all days"
                 else "Expiry day only — every combination of rolls × stop")
        tables.append(f'<div class="card"><h2>{title}</h2>{"".join(t)}</table></div>'
                      f'<p class="note">&#9733; = the variant with the most points in 2021–23 in this table — the one a trader '
                      f'would have picked then; its 2024–26 column is the honest test.</p></div>')

    # equity curves in points (fair across lot-size changes): original + each group's in-sample pick + best overall
    best = max(rows, key=lambda r: r["all"]["net"])
    show = []
    for r in [base, picks.get("all days"), picks.get("expiry day"), best]:
        if r and r["key"] not in [s["key"] for s in show]:
            show.append(r)
    series = [{"label": curves[r["key"]]["label"], "pts": curves[r["key"]]["pts"], "color": COLORS[i % len(COLORS)]}
              for i, r in enumerate(show)]

    verdict = []
    for g, r in picks.items():
        verdict.append(f"<li><b>{html.escape(g.title())}:</b> in 2021–23 the best was <b>{html.escape(r['label'])}</b> "
                       f"({r['in']['pts']:,.0f} pts, {_rs(r['in']['net'])}); in 2024–26 it made "
                       f"<b class='{'up' if r['out']['net'] > 0 else 'down'}'>{r['out']['pts']:,.0f} pts, {_rs(r['out']['net'])}</b>.</li>")
    verdict.append(f"<li><b>Original</b> (all days): {base['all']['pts']:,.0f} pts, <b class='down'>{_rs(base['all']['net'])}</b> "
                   f"after {_rs(base['charges'])} charges.</li>")

    page = (TEMPLATE.replace("/*SERIES*/", json.dumps(series))
            .replace("{{VERDICT}}", "".join(verdict)).replace("{{TABLES}}", "".join(tables)))
    out = os.path.join(ROOT, "improve_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Range Strategy Improvements</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--pick:#fff7e0;--base:#eef0f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--pick:#3a321a;--base:#23262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}td.b{font-weight:800}
tr.pick td{background:var(--pick)}tr.base td{background:var(--base);font-style:italic}
.ok{display:inline-block;background:rgba(14,159,110,.14);color:var(--up);border-radius:999px;padding:0 7px;font-size:11px;font-weight:700;font-style:normal}
.note{color:var(--muted);font-size:12.5px;margin:8px 0 0}.scroll{overflow-x:auto}
.rules{display:grid;grid-template-columns:150px 1fr;gap:6px 14px;font-size:13.5px}.rules b{color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;margin-top:6px}.legend i{display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle}
svg text{fill:var(--muted);font-size:11px}ul{margin:0;padding-left:18px}li{margin:4px 0}
</style></head><body><div class="wrap">
<h1>Range strategy — improvements 1 + 2 + 3</h1>
<p class="note" style="margin-bottom:16px">Sep 2021 → Sep 2026 · NIFTY weekly options · 1-min data · 1 lot · 0.5 pt slippage per order · all charges.
Points are per lot (fair across the 50 → 25 → 75 → 65 lot changes); rupees use the lot of that time.</p>
<div class="card"><h2>What was tested</h2><div class="rules">
<b>Original</b><span>09:20 range from the volume leaders (UPPER = CE leader + PE leader's first-candle high, LOWER = PE leader − CE leader's first-candle high); sell the 50-pt strikes nearest UPPER / LOWER; on a leader change the range is recalculated — range up → book PE and sell PE nearest new LOWER, range down → book CE and sell CE nearest new UPPER; stop at 50% of the 09:20 premium; exit 15:15.</span>
<b>1 · Rolls</b><span>No rolls (hold the 09:20 strikes) · at most 1 roll a day · roll only when the range mid moves 50+ points.</span>
<b>2 · Range stop</b><span>Exit both legs when NIFTY <i>closes</i> a 5-min (or 15-min) candle outside the current range (it still updates on leader changes). The 50% premium stop stays as an emergency stop.</span>
<b>3 · Expiry day</b><span>Trade only on the weekly expiry day (Thursday until Aug 2025, Tuesday since).</span>
</div></div>
<div class="card"><h2>Verdict</h2><ul>{{VERDICT}}</ul></div>
<div class="card"><h2>Cumulative points per lot</h2><div id="eq"></div><div class="legend" id="lg"></div></div>
{{TABLES}}
</div><script>
const SERIES = /*SERIES*/;
(function () {
  const W = 1300, H = 340, pl = 60, pr = 12, pt = 12, pb = 26;
  const all = SERIES.flatMap(s => s.pts.map(p => p[0])).sort();
  const t0 = new Date(all[0]).getTime(), t1 = new Date(all[all.length - 1]).getTime();
  const vals = SERIES.flatMap(s => s.pts.map(p => p[1])).concat([0]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = d => pl + (new Date(d).getTime() - t0) / (t1 - t0) * (W - pl - pr);
  const y = v => pt + (hi - v) / (hi - lo || 1) * (H - pt - pb);
  const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4 || 1)));
  const ticks = []; for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  const years = []; for (let yv = new Date(all[0]).getFullYear() + 1; yv <= new Date(all[all.length - 1]).getFullYear(); yv++) years.push(yv);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Cumulative points">`;
  svg += ticks.map(v => `<line x1="${pl}" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)"${v === 0 ? ' stroke-width="2"' : ""}/><text x="${pl - 6}" y="${y(v) + 4}" text-anchor="end">${v.toLocaleString("en-IN")}</text>`).join("");
  svg += years.map(yv => { const xx = x(yv + "-01-01"); return `<line x1="${xx}" x2="${xx}" y1="${pt}" y2="${H - pb}" stroke="var(--border)" stroke-dasharray="3,3"/><text x="${xx + 4}" y="${H - 8}">${yv}</text>`; }).join("");
  const split = x("2024-01-01");
  svg += `<rect x="${split}" y="${pt}" width="${W - pr - split}" height="${H - pt - pb}" fill="rgba(79,95,224,.05)"/><text x="${split + 6}" y="${pt + 12}">out-of-sample →</text>`;
  svg += SERIES.map(s => `<path d="${s.pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("")}" fill="none" stroke="${s.color}" stroke-width="1.8"/>`).join("");
  document.getElementById("eq").innerHTML = svg + "</svg>";
  document.getElementById("lg").innerHTML = SERIES.map(s => `<span><i style="background:${s.color}"></i>${s.label} (${s.pts[s.pts.length - 1][1].toLocaleString("en-IN")} pts)</span>`).join("");
})();
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print(build())
