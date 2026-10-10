"""risk_sizing.py -- which of the plan's four trades, and how much capital per lot?

Uses the saved backtest trades behind the Plan page numbers (1 lot, after
charges, out-of-sample period May 2022 / Sep 2021 -> Sep 2026):

  A  green IV day, 1-4 trading days to expiry, sold at 09:30, held to expiry
     (results/ashish_rules/positions.parquet, set A, green, hold_rs)
  B  green IV at 09:30 with 1 day left on the current expiry, sell NEXT
     week's expiry, exit at the first check where the IV rule reads normal
     (results/next_expiry_exit/trades.parquet, green, ivany_rs)

Per variant: average, win rate, worst trade, 5% worst, max drawdown of the
cumulative P&L, longest losing streak, longest time under water, how many
trades were open at once, P&L by year, and a bootstrap of one year of trades
(95th-percentile drawdown = a forward-looking "bad year").

Capital per lot = exchange margin x the most trades open at once (live Arrow
margin of 10 Oct 2026: straddle ~Rs 2.07 L, 0.15-delta strangle ~Rs 1.85 L)
+ a buffer of 1.5 x the larger of the historical max drawdown and the
bootstrap's 95th-percentile one-year drawdown.

Output: results/risk_sizing/report.html (+ console summary)
"""
import json
import math
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import paths

OUT = os.path.join(paths.BASE_DIR, "results", "risk_sizing")
MARGIN = {"straddle": 207000, "strangle": 185000}   # Rs per lot, Arrow /margin/basket on 2026-10-10
BUFFER = 1.5
BOOT = 10000


def load():
    a = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "ashish_rules", "positions.parquet"))
    a = a[(a.set == "A") & a.green]
    b = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "next_expiry_exit", "trades.parquet"))
    b = b[b.green]
    out = {}
    for name, kind, df, pnl, held, sel in (
            ("A straddle", "straddle", a, "hold_rs", "hold_held", a.strat.str.startswith("Short straddle")),
            ("A strangle", "strangle", a, "hold_rs", "hold_held", a.strat.str.startswith("Short strangle")),
            ("B straddle", "straddle", b, "ivany_rs", "ivany_held", b.strat == "Short straddle ATM"),
            ("B strangle", "strangle", b, "ivany_rs", "ivany_held", b.strat == "Short strangle Δ0.15")):
        x = df[sel][["day", "expiry", pnl, held]].rename(columns={pnl: "pnl", held: "held"}).copy()
        x["day"] = pd.to_datetime(x["day"]).dt.date
        x = x.dropna(subset=["pnl"]).sort_values("day").reset_index(drop=True)
        out[name] = (kind, x)
    return out


def _dd(cum):
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return cum - peak


def stats(kind, x):
    p = x.pnl.to_numpy()
    cum = np.cumsum(p)
    dd = _dd(cum)
    i_dd = int(np.argmin(dd))
    # longest losing streak (consecutive losing trades)
    streak = best = 0
    for v in p:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    # longest time under water (calendar days from a peak to the trade that first beats it)
    days = list(x.day)
    peak_v, peak_d, uw = 0.0, days[0], 0
    for v, d in zip(cum, days):
        if v >= peak_v:
            uw = max(uw, (d - peak_d).days)
            peak_v, peak_d = v, d
    uw = max(uw, (days[-1] - peak_d).days)
    # trades open at once (entry day .. entry + held days)
    ev = []
    for d, h in zip(x.day, x.held.fillna(1)):
        ev += [(d, 1), (d + timedelta(days=max(0.0, float(h))), -1)]
    ev.sort(key=lambda t: (t[0], -t[1]))
    cur = mx = 0
    for _, s in ev:
        cur += s
        mx = max(mx, cur)
    years = (days[-1] - days[0]).days / 365.25
    per_year = len(p) / years if years > 0 else len(p)
    # bootstrap one year of trades
    rng = np.random.default_rng(7)
    n = max(1, int(round(per_year)))
    sims = rng.choice(p, size=(BOOT, n), replace=True)
    boot_dd = np.array([_dd(np.cumsum(r)).min() for r in sims])
    boot_tot = sims.sum(axis=1)
    worst_dd = min(float(dd.min()), float(np.percentile(boot_dd, 5)))
    capital = MARGIN[kind] * mx + BUFFER * abs(worst_dd)
    yearly = x.assign(y=[d.year for d in x.day]).groupby("y").pnl.agg(["count", "sum"])
    return {"n": int(len(p)), "per_year": round(per_year, 1), "avg": round(float(p.mean())), "median": round(float(np.median(p))),
            "win": round(float((p > 0).mean() * 100)), "std": round(float(p.std(ddof=1))), "sharpe_trade": round(float(p.mean() / p.std(ddof=1)), 3),
            "worst": round(float(p.min())), "best": round(float(p.max())), "p5": round(float(np.percentile(p, 5))),
            "max_dd": round(float(dd.min())), "dd_at": str(days[i_dd]), "streak": int(best), "underwater_days": int(uw),
            "max_open": int(mx), "total": round(float(cum[-1])), "per_year_rs": round(float(cum[-1] / years)) if years > 0 else None,
            "avg_over_dd": round(float(p.mean() / abs(dd.min())), 3) if dd.min() < 0 else None,
            "boot_dd95": round(float(np.percentile(boot_dd, 5))), "boot_loss_year": round(float((boot_tot < 0).mean() * 100), 1),
            "boot_p5_year": round(float(np.percentile(boot_tot, 5))), "boot_median_year": round(float(np.median(boot_tot))),
            "margin": MARGIN[kind], "capital": round(capital, -3),
            "return_on_capital": round(float(cum[-1] / years / capital * 100), 1) if years > 0 else None,
            "yearly": [{"y": int(y), "n": int(r["count"]), "rs": round(float(r["sum"]))} for y, r in yearly.iterrows()],
            "first": str(days[0]), "last": str(days[-1]),
            "curve": [{"d": str(d), "cum": round(float(c))} for d, c in zip(days, cum)]}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {k: stats(kind, x) for k, (kind, x) in load().items()}
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps({"v": res, "buffer": BUFFER, "margin": MARGIN}, ensure_ascii=False)))
    return res, path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Risk and Sizing</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--grp:#eef0f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--grp:#23262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
tr.best td{background:rgba(14,159,110,.08)}svg text{font-size:11px;fill:var(--muted)}
</style></head><body><div class="wrap">
<h1>Which plan trade, and how much capital per lot?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="card"><h2>What the numbers say</h2><ul id="verdict"></ul></div>
<div class="card"><h2>Reward vs pain, 1 lot each</h2><div class="scroll"><table id="t1"></table></div>
<p class="note">Max drawdown = deepest fall of the running P&amp;L total from its previous high. Under water = longest calendar stretch before a new high. "Bad year (95%)" = 1 year of trades re-drawn 10,000 times from the history; 1 in 20 such years had a drawdown at least this deep.</p></div>
<div class="card"><h2>Capital per lot</h2><div class="scroll"><table id="t2"></table></div>
<p class="note" id="capNote"></p></div>
<div class="card"><h2>Running P&amp;L (1 lot)</h2><div id="curve"></div></div>
<div class="card"><h2>By calendar year (₹, 1 lot)</h2><div class="scroll"><table id="t3"></table></div></div>
<div class="card"><h2>Method &amp; limits</h2><ul>
<li>Trades from the saved backtests behind the Plan page: A = green IV day 1–4 trading days before expiry, sold at 09:30, held to expiry; B = green at 09:30 with 1 day left, next week's expiry sold, closed at the first check where IV reads normal. 1-min prices, 0.5 pt slippage per order, all charges.</li>
<li>Margin is today's exchange margin for one lot (Arrow, 10 Oct 2026); it changes with NIFTY's level and volatility, and A and B taken on the same day need margin for both.</li>
<li>The past is one path. The bootstrap reshuffles that path; it cannot invent a crash bigger than the worst week already in the sample (2022–26 has no 2020-style crash).</li>
<li>This is a description of the backtest, not advice on how much to trade.</li>
</ul></div>
</div><script>
const D = /*DATA*/null, V = D.v, K = Object.keys(V);
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const L = v => v == null ? "—" : "₹" + (v / 1e5).toFixed(2) + " L";
const cl = v => v > 0 ? "up" : v < 0 ? "down" : "";
document.getElementById("meta").textContent = `Backtest trades, 1 lot each, after charges · A: ${V["A straddle"].first} → ${V["A straddle"].last} · B: ${V["B straddle"].first} → ${V["B straddle"].last}.`;
const bestRatio = K.reduce((a, k) => (V[k].avg_over_dd || 0) > (V[a].avg_over_dd || 0) ? k : a, K[0]);
const bestRoc = K.reduce((a, k) => (V[k].return_on_capital || 0) > (V[a].return_on_capital || 0) ? k : a, K[0]);
document.getElementById("t1").innerHTML = `<tr><th>Trade</th><th>Trades</th><th>/yr</th><th>Avg</th><th>Win</th><th>Worst trade</th><th>5% worst</th><th>Max drawdown</th><th>Avg ÷ |DD|</th><th>Losing streak</th><th>Under water</th><th>Bad year (95%)</th><th>Loss years</th></tr>` +
  K.map(k => { const v = V[k]; return `<tr class="${k === bestRatio ? "best" : ""}"><td><b>${k}</b></td><td>${v.n}</td><td>${v.per_year}</td><td class="${cl(v.avg)}"><b>${rs(v.avg)}</b></td><td>${v.win}%</td><td class="down">${rs(v.worst)}</td><td class="down">${rs(v.p5)}</td><td class="down"><b>${rs(v.max_dd)}</b><div class="note">${v.dd_at}</div></td><td>${v.avg_over_dd}</td><td>${v.streak}</td><td>${v.underwater_days} d</td><td class="down">${rs(v.boot_dd95)}</td><td>${v.boot_loss_year}%</td></tr>`; }).join("");
document.getElementById("t2").innerHTML = `<tr><th>Trade</th><th>Margin / lot</th><th>Max open at once</th><th>Margin needed</th><th>+ buffer (${D.buffer}× worst DD)</th><th>Capital / lot</th><th>Avg P&amp;L / year</th><th>Return on capital / yr</th></tr>` +
  K.map(k => { const v = V[k], need = v.margin * v.max_open; return `<tr class="${k === bestRoc ? "best" : ""}"><td><b>${k}</b></td><td>${L(v.margin)}</td><td>${v.max_open}</td><td>${L(need)}</td><td>${L(v.capital - need)}</td><td><b>${L(v.capital)}</b></td><td class="${cl(v.per_year_rs)}">${rs(v.per_year_rs)}</td><td><b>${v.return_on_capital}%</b></td></tr>`; }).join("");
document.getElementById("capNote").textContent = `Capital per lot = margin × the most trades of that kind open at the same time + ${D.buffer} × the larger of the historical max drawdown and the bad-year (95%) drawdown. Running A and B together needs both.`;
const yrs = [...new Set(K.flatMap(k => V[k].yearly.map(y => y.y)))].sort();
document.getElementById("t3").innerHTML = `<tr><th>Trade</th>${yrs.map(y => `<th>${y}</th>`).join("")}</tr>` + K.map(k => `<tr><td>${k}</td>${yrs.map(y => { const r = V[k].yearly.find(z => z.y === y); return r ? `<td class="${cl(r.rs)}">${rs(r.rs)}<div class="note">${r.n} tr</div></td>` : "<td>—</td>"; }).join("")}</tr>`).join("");
// running P&L
(function () {
  const W = Math.min(1150, document.getElementById("curve").clientWidth || 1100), H = 280, Lm = 70, B = 24, T = 10;
  const all = K.flatMap(k => V[k].curve.map(c => [new Date(c.d).getTime(), c.cum]));
  const x0 = Math.min(...all.map(a => a[0])), x1 = Math.max(...all.map(a => a[0])), y0 = Math.min(0, ...all.map(a => a[1])), y1 = Math.max(...all.map(a => a[1]));
  const X = t => Lm + (t - x0) / (x1 - x0) * (W - Lm - 10), Y = v => T + (y1 - v) / (y1 - y0) * (H - T - B);
  const col = { "A straddle": "#4f5fe0", "A strangle": "#8fa0ff", "B straddle": "#b0413e", "B strangle": "#e8907f" };
  let s = `<svg width="${W}" height="${H + 20}">`;
  for (let k = 0; k <= 4; k++) { const v = y0 + (y1 - y0) * k / 4; s += `<line x1="${Lm}" x2="${W - 10}" y1="${Y(v)}" y2="${Y(v)}" stroke="#ddd" stroke-opacity=".5"/><text x="${Lm - 6}" y="${Y(v) + 4}" text-anchor="end">${rs(v)}</text>`; }
  for (let y = new Date(x0).getFullYear() + 1; y <= new Date(x1).getFullYear(); y++) { const t = new Date(`${y}-01-01`).getTime(); s += `<text x="${X(t)}" y="${H - 4}" text-anchor="middle">${y}</text>`; }
  K.forEach(k => { s += `<polyline fill="none" stroke="${col[k]}" stroke-width="2" points="${V[k].curve.map(c => `${X(new Date(c.d).getTime())},${Y(c.cum)}`).join(" ")}"/>`; });
  K.forEach((k, i) => { s += `<rect x="${Lm + i * 150}" y="${H + 6}" width="12" height="4" fill="${col[k]}"/><text x="${Lm + 16 + i * 150}" y="${H + 12}">${k}</text>`; });
  document.getElementById("curve").innerHTML = s + "</svg>";
})();
const a = V["A straddle"], as_ = V["A strangle"], b = V["B straddle"], bs = V["B strangle"];
document.getElementById("verdict").innerHTML = [
  `<b>Best reward for the pain: ${bestRatio}</b> — it earns ${V[bestRatio].avg_over_dd} of its worst drawdown per trade (A straddle ${a.avg_over_dd}, A strangle ${as_.avg_over_dd}, B straddle ${b.avg_over_dd}, B strangle ${bs.avg_over_dd}).`,
  `<b>Straddles earn about twice as much per trade but hurt far more:</b> A straddle's worst drawdown was ${rs(a.max_dd)} (worst single trade ${rs(a.worst)}) against ${rs(as_.max_dd)} for the A strangle; B straddle ${rs(b.max_dd)} vs B strangle ${rs(bs.max_dd)}.`,
  `<b>Capital per lot</b> (margin for the trades that overlap + ${D.buffer}× the worst drawdown): A straddle ${L(a.capital)}, A strangle ${L(as_.capital)}, B straddle ${L(b.capital)}, B strangle ${L(bs.capital)}. Highest return on that capital: <b>${bestRoc}</b> (${V[bestRoc].return_on_capital}% a year).`,
  `<b>A trades overlap:</b> green days come in runs, so up to ${a.max_open} A positions were open at once — the margin need is that many lots, not one. B was at most ${b.max_open} open.`,
  `<b>Bad years happen:</b> re-drawing a year of trades, ${a.boot_loss_year}% of A-straddle years and ${as_.boot_loss_year}% of A-strangle years ended in a loss; for B ${b.boot_loss_year}% / ${bs.boot_loss_year}%.`,
].map(x => `<li>${x}</li>`).join("");
</script></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    res, path = main()
    for k, v in res.items():
        print(f"{k:11s} n{v['n']:4d} /yr {v['per_year']:5.1f} avg {v['avg']:6d} win {v['win']:3d}% worst {v['worst']:7d} p5 {v['p5']:7d} maxDD {v['max_dd']:8d} ({v['dd_at']}) "
              f"avg/DD {v['avg_over_dd']} streak {v['streak']} uw {v['underwater_days']}d open {v['max_open']} bootDD95 {v['boot_dd95']} lossyr {v['boot_loss_year']}% "
              f"cap {v['capital']:.0f} roc {v['return_on_capital']}% yr {v['per_year_rs']}")
        print("   ", v["yearly"])
    print(path)
