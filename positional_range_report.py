"""positional_range_report.py -- HTML report for positional_range_research.py
(results/positional_range_research/rows.csv -> report.html)."""
import os
import sys

import numpy as np
import pandas as pd

import paths

DIR = os.path.join(paths.BASE_DIR, "results", "positional_range_research")
DTE_LAB = {0: "Expiry day", 1: "1 day before", 2: "2 days before", 3: "3 days before", 4: "4 days before"}


def pct(x):
    return f"{x:.1f}%"


def cls(v):
    return "up" if v > 0 else "down" if v < 0 else ""


def build():
    d = pd.read_csv(os.path.join(DIR, "rows.csv"))
    d["year"] = d.day.str[:4]
    d["touch_any"] = d.touched_up | d.touched_down
    d1 = d[d.day_no == 1].copy()

    # ---- by days-to-expiry ----
    rows = []
    for k, g in d.groupby("dte"):
        if len(g) < 50:
            continue
        c, hw = g.spot_920, g.width / 2
        base = ((g.exp_close >= c - hw) & (g.exp_close <= c + hw)).mean() * 100
        h = g.hold_pts.dropna()
        rows.append(f"<tr><td>{DTE_LAB.get(k, k)}</td><td>{len(g)}</td><td>{g.width.median():.0f}</td>"
                    f"<td><b>{pct((g.side == 'inside').mean() * 100)}</b></td><td>{pct((g.side == 'above').mean() * 100)}</td>"
                    f"<td>{pct((g.side == 'below').mean() * 100)}</td><td>{pct(base)}</td>"
                    f"<td>{pct(g.touch_any.mean() * 100)}</td><td>{pct(((g.side == 'inside') & ~g.touch_any).mean() * 100)}</td>"
                    f"<td>{(g.ce_prem + g.pe_prem).median():.1f}</td><td>{pct((h > 0).mean() * 100)}</td>"
                    f"<td class='{cls(h.mean())}'>{h.mean():.1f}</td><td>{h.median():.1f}</td><td class='down'>{h.min():.0f}</td></tr>")
    t_dte = "".join(rows)

    # ---- Day 1 by year ----
    yr = []
    for y, g in d1.groupby("year"):
        h = g.hold_pts.dropna()
        yr.append(f"<tr><td>{y}</td><td>{len(g)}</td><td>{pct((g.side == 'inside').mean() * 100)}</td>"
                  f"<td>{pct(g.touch_any.mean() * 100)}</td><td>{pct((h > 0).mean() * 100)}</td>"
                  f"<td class='{cls(h.sum())}'><b>{h.sum():,.0f}</b></td><td class='{cls(h.mean())}'>{h.mean():.1f}</td></tr>")
    h1 = d1.hold_pts.dropna()
    yr.append(f"<tr class='tot'><td>All</td><td>{len(d1)}</td><td>{pct((d1.side == 'inside').mean() * 100)}</td>"
              f"<td>{pct(d1.touch_any.mean() * 100)}</td><td>{pct((h1 > 0).mean() * 100)}</td>"
              f"<td class='{cls(h1.sum())}'><b>{h1.sum():,.0f}</b></td><td>{h1.mean():.1f}</td></tr>")

    # ---- Day 1 range width quartiles ----
    wq = []
    d1["wq"] = pd.qcut(d1.width, 4)
    for q, g in d1.groupby("wq", observed=True):
        h = g.hold_pts.dropna()
        wq.append(f"<tr><td>{q.left:.0f} – {q.right:.0f}</td><td>{len(g)}</td><td>{pct((g.side == 'inside').mean() * 100)}</td>"
                  f"<td>{pct((h > 0).mean() * 100)}</td><td class='{cls(h.mean())}'>{h.mean():.1f}</td><td class='{cls(h.sum())}'>{h.sum():,.0f}</td></tr>")

    # ---- Day 1 outside-by + excursion quantiles ----
    out = d1[d1.side != "inside"].outside_by
    qs = [.25, .5, .75, .9]
    exc = (f"<tr><td>Expiry close outside the range, by</td>" + "".join(f"<td>{out.quantile(q):.0f}</td>" for q in qs) + "</tr>"
           f"<tr><td>Max move above UPPER during the week</td>" + "".join(f"<td>{max(0, d1.max_above.quantile(q)):.0f}</td>" for q in qs) + "</tr>"
           f"<tr><td>Max move below LOWER during the week</td>" + "".join(f"<td>{max(0, d1.max_below.quantile(q)):.0f}</td>" for q in qs) + "</tr>")

    # ---- Day 1 hold-to-expiry histogram (svg) ----
    edges = [-800, -400, -300, -200, -100, -50, 0, 50, 100, 150, 200, 300, 500]
    cnt = pd.cut(h1, edges).value_counts().sort_index()
    W, H, pl, pb = 900, 230, 40, 36
    bw = (W - pl) / len(cnt)
    mx = cnt.max()
    bars = []
    for i, (iv, c) in enumerate(cnt.items()):
        bh = (H - pb - 10) * c / mx
        col = "var(--up)" if iv.left >= 0 else "var(--down)"
        x = pl + i * bw
        bars.append(f"<rect x='{x + 3:.0f}' y='{H - pb - bh:.0f}' width='{bw - 6:.0f}' height='{bh:.0f}' fill='{col}' opacity='.8'/>"
                    f"<text x='{x + bw / 2:.0f}' y='{H - pb - bh - 4:.0f}' text-anchor='middle'>{c}</text>"
                    f"<text x='{x + bw / 2:.0f}' y='{H - pb + 14:.0f}' text-anchor='middle'>{iv.left:.0f}…{iv.right:.0f}</text>")
    hist = f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' aria-label='Hold to expiry P&L histogram'>{''.join(bars)}</svg>"

    s = {
        "n_exp": d.expiry.nunique(), "from": d.day.min(), "to": d.day.max(),
        "d1_inside": (d1.side == "inside").mean() * 100, "d1_above": (d1.side == "above").mean() * 100,
        "d1_below": (d1.side == "below").mean() * 100, "d1_touch": d1.touch_any.mean() * 100,
        "d1_width": d1.width.median(), "d1_hold_win": (h1 > 0).mean() * 100, "d1_hold_sum": h1.sum(),
        "d1_hold_avg": h1.mean(), "d1_hold_med": h1.median(), "d1_hold_min": h1.min(),
        "e_inside": (d[d.dte == 0].side == "inside").mean() * 100,
        "d1_base": (((d1.exp_close >= d1.spot_920 - d1.width / 2) & (d1.exp_close <= d1.spot_920 + d1.width / 2)).mean() * 100),
    }
    page = TEMPLATE
    for k, v in {"T_DTE": t_dte, "T_YEAR": "".join(yr), "T_WQ": "".join(wq), "T_EXC": exc, "HIST": hist}.items():
        page = page.replace("{{" + k + "}}", v)
    for k, v in s.items():
        page = page.replace("{{" + k + "}}", f"{v:,.1f}" if isinstance(v, (float, np.floating)) else str(v))
    out_p = os.path.join(DIR, "report.html")
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(page)
    return out_p


TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Range Expiry Research</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px}.kpi .k{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase}
.kpi .v{font-size:21px;font-weight:800;font-variant-numeric:tabular-nums}.kpi .s{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}tr.tot td{font-weight:700;border-top:2px solid var(--border)}
.up{color:var(--up)}.down{color:var(--down)}.scroll{overflow-x:auto}.note{color:var(--muted);font-size:12.5px;margin:8px 0 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
svg text{fill:var(--muted);font-size:11px}ul{margin:0;padding-left:18px}li{margin:5px 0}
</style></head><body><div class="wrap">
<h1>Does NIFTY expire inside the 09:20 volume-leader range?</h1>
<p class="note" style="margin-bottom:16px">{{n_exp}} weekly expiries · {{from}} → {{to}} · range = CE leader + PE leader's first 5-min high / PE leader − CE leader's first 5-min high (100-pt leaders, day's 09:15–09:20 candle) · expiry close = NIFTY's last print on expiry day (≈ settlement).</p>

<div class="kpis">
 <div class="kpi"><div class="k">Day-1 range: expired inside</div><div class="v">{{d1_inside}}%</div><div class="s">above {{d1_above}}% · below {{d1_below}}%</div></div>
 <div class="kpi"><div class="k">Same-width band around spot</div><div class="v">{{d1_base}}%</div><div class="s">a plain band of the same width, centred on NIFTY at 09:20</div></div>
 <div class="kpi"><div class="k">Day-1 range touched during week</div><div class="v">{{d1_touch}}%</div><div class="s">NIFTY traded outside it at least once</div></div>
 <div class="kpi"><div class="k">Expiry-day range: expired inside</div><div class="v">{{e_inside}}%</div><div class="s">range set at 09:20 on expiry day</div></div>
 <div class="kpi"><div class="k">Day-1 range width (median)</div><div class="v">{{d1_width}}</div><div class="s">points</div></div>
 <div class="kpi"><div class="k">Sell Day-1 strikes, hold to expiry</div><div class="v">{{d1_hold_win}}% win</div><div class="s">total {{d1_hold_sum}} pts · avg {{d1_hold_avg}} · median {{d1_hold_med}} · worst {{d1_hold_min}}</div></div>
</div>

<div class="card"><h2>What this says</h2><ul>
<li><b>The Day-1 range holds the expiry close about 4 times in 10</b> ({{d1_inside}}%). Misses are evenly split between above and below, so the range has no directional bias.</li>
<li><b>The leader range is no better than a plain band.</b> A band of the <i>same width</i> centred on NIFTY at 09:20 held the expiry close {{d1_base}}% of the time, as often or more. The volume leaders mainly tell you <i>how wide</i> the market expects the move to be (the range width tracks the straddle premium), not <i>where</i> it will expire.</li>
<li><b>The range is almost always tested:</b> NIFTY trades outside the Day-1 range at some point in {{d1_touch}}% of weeks. A positional version must expect adjustments; a "set and forget" range will be crossed.</li>
<li><b>Selling the Day-1 range strikes and holding to expiry wins about 2 times in 3</b> ({{d1_hold_win}}%, median +{{d1_hold_med}} pts) but the losers are large (worst {{d1_hold_min}} pts) — the classic short-option profile. Over 5 years it is only mildly positive and two years were negative, so the adjustment and stop rules are what decide the result.</li>
<li><b>Wider Day-1 ranges did better</b> (see width table): the widest half expired inside ~50% of the time and was profitable to hold; the narrowest half ~32% and lost. Range width is a useful filter to research further.</li>
</ul></div>

<div class="card"><h2>By the day the range was set</h2><div class="scroll"><table>
<tr><th>Range set on</th><th>Weeks</th><th>Width (med)</th><th>Expired inside</th><th>Above</th><th>Below</th><th>Same-width band on spot</th><th>Touched outside</th><th>Inside &amp; never touched</th><th>Premium (med)</th><th>Hold: win %</th><th>Hold: avg pts</th><th>Hold: median</th><th>Hold: worst</th></tr>
{{T_DTE}}</table></div>
<p class="note">"Hold" = sell the 50-pt strikes nearest UPPER / LOWER at 09:20 of that day, no stop, no adjustment, settle at intrinsic vs the expiry close. Points per lot, 0.5 pt slippage per order, no charges.</p></div>

<div class="grid2">
<div class="card"><h2>Day-1 range by year</h2><div class="scroll"><table><tr><th>Year</th><th>Weeks</th><th>Inside</th><th>Touched</th><th>Hold win %</th><th>Hold pts</th><th>Avg</th></tr>{{T_YEAR}}</table></div></div>
<div class="card"><h2>Day-1 range width (quartiles)</h2><div class="scroll"><table><tr><th>Width</th><th>Weeks</th><th>Inside</th><th>Hold win %</th><th>Avg pts</th><th>Total pts</th></tr>{{T_WQ}}</table></div></div>
</div>
<div class="card"><h2>How far outside (Day-1 range, points)</h2><div class="scroll"><table><tr><th></th><th>25%</th><th>Median</th><th>75%</th><th>90%</th></tr>{{T_EXC}}</table></div></div>
<div class="card"><h2>Sell Day-1 range strikes &amp; hold to expiry — points per week</h2>{{HIST}}</div>
</div></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print(build())
