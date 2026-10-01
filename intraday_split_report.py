"""intraday_split_report.py -- in-sample / out-of-sample check of the leader
straddle variants: pick weekday + SL on 2021-2023 only, then see how the
same choice did on 2024-2026 (data it never saw).

Needs results/intraday_range_bt/leader_sl{30,40,50,60}/days.csv
(python intraday_range_bt.py). Output: results/intraday_range_bt/split_report.html
"""
import html
import os
import sys

import pandas as pd

import paths

ROOT = os.path.join(paths.BASE_DIR, "results", "intraday_range_bt")
SLS = (30, 40, 50, 60)
WDS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
PERIODS = [("2021-23", "In-sample: 2021 – 2023 (choose the rules here)"),
           ("2024-26", "Out-of-sample: 2024 – 2026 (test the chosen rules)")]


def _load():
    frames = []
    for sl in SLS:
        d = pd.read_csv(os.path.join(ROOT, f"leader_sl{sl}", "days.csv"))
        d["sl"] = sl
        frames.append(d)
    d = pd.concat(frames)
    d = d[d.weekday.isin(WDS)]
    d["period"] = d.day.str[:4].map(lambda y: "2021-23" if y <= "2023" else "2024-26")
    d["dte"] = (pd.to_datetime(d.expiry) - pd.to_datetime(d.day)).dt.days
    return d


def _rs(v):
    return ("-" if v < 0 else "") + "₹" + f"{abs(v):,.0f}"


def _cell(v, fmt, best=False):
    cls = "up" if v > 0 else "down" if v < 0 else ""
    return f'<td class="{cls}{" best" if best else ""}">{fmt(v)}</td>'


def _grid(d, period, metric):
    g = d[d.period == period]
    rows = []
    for wd in WDS + ["All days"]:
        x = g if wd == "All days" else g[g.weekday == wd]
        vals = []
        for sl in SLS:
            y = x[x.sl == sl]
            vals.append({"net": y.net_rs.sum(), "pts": y.pnl_pts.sum(), "win": (y.net_rs > 0).mean() * 100 if len(y) else 0,
                         "days": len(y)}[metric])
        rows.append((wd, vals))
    return rows


def build():
    d = _load()
    # the in-sample choice: best weekday x SL by net on 2021-23 only
    ins = d[d.period == "2021-23"].groupby(["weekday", "sl"]).net_rs.sum()
    pick_wd, pick_sl = ins.idxmax()
    body = []
    for p, title in PERIODS:
        n = d[(d.period == p) & (d.sl == 50)]
        body.append(f'<div class="card"><h2>{html.escape(title)}</h2><p class="note">{len(n)} trading days · '
                    f'{n.day.min()} → {n.day.max()}</p><div class="tables">')
        for metric, lab, fmt in (("net", "Net ₹ (1 lot, after charges)", lambda v: _rs(round(v))),
                                 ("pts", "Points per lot", lambda v: f"{v:,.0f}"),
                                 ("win", "Win %", lambda v: f"{v:.0f}%")):
            rows = _grid(d, p, metric)
            best = max((v, r, c) for r, (_, vals) in enumerate(rows[:-1]) for c, v in enumerate(vals))
            t = [f'<table><caption>{lab}</caption><tr><th>Day</th>' + "".join(f"<th>SL {s}%</th>" for s in SLS) + "</tr>"]
            for r, (wd, vals) in enumerate(rows):
                hl = ' class="pick"' if wd == pick_wd else (' class="tot"' if wd == "All days" else "")
                t.append(f"<tr{hl}><td>{wd}</td>" + "".join(
                    _cell(v, fmt, best=(metric != "win" and (r, c) == best[1:])) for c, v in enumerate(vals)) + "</tr>")
            body.append("".join(t) + "</table>")
        body.append("</div></div>")

    # the chosen rule, both periods, split by days-to-expiry
    ch = d[(d.weekday == pick_wd) & (d.sl == pick_sl)]
    dte_lab = {0: "Expiry day", 1: "1 day before expiry", 2: "2 days before expiry"}
    rows = []
    for p, _ in PERIODS:
        g = ch[ch.period == p]
        eq = g.net_rs.cumsum()
        dd = (eq - eq.cummax().clip(lower=0)).min() if len(g) else 0
        for k, x in g.groupby("dte"):
            rows.append(f"<tr><td>{p}</td><td>{dte_lab.get(k, f'{k} days before')}</td><td>{len(x)}</td>"
                        f"<td>{(x.net_rs > 0).mean() * 100:.0f}%</td><td>{x.pnl_pts.sum():,.0f}</td>"
                        f"<td>{x.pnl_pts.mean():.1f}</td>{_cell(round(x.net_rs.sum()), _rs)}</tr>")
        rows.append(f'<tr class="tot"><td>{p}</td><td>All {pick_wd}</td><td>{len(g)}</td><td>{(g.net_rs > 0).mean() * 100:.0f}%</td>'
                    f"<td>{g.pnl_pts.sum():,.0f}</td><td>{g.pnl_pts.mean():.1f}</td>{_cell(round(g.net_rs.sum()), _rs)}</tr>"
                    f'<tr class="tot"><td></td><td>Max drawdown</td><td colspan="4"></td>{_cell(round(dd), _rs)}</tr>')
    chosen = (f'<div class="card"><h2>The rule chosen on 2021-23: {pick_wd} · SL {pick_sl}% — how it did in each period</h2>'
              f'<div class="scroll"><table><tr><th>Period</th><th>Day type</th><th>Days</th><th>Win %</th><th>Points/lot</th>'
              f'<th>Points/day</th><th>Net ₹</th></tr>{"".join(rows)}</table></div></div>')

    page = TEMPLATE.replace("{{BODY}}", chosen + "".join(body)).replace("{{PICK}}", f"{pick_wd} · SL {pick_sl}%")
    out = os.path.join(ROOT, "split_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out, pick_wd, pick_sl


TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Walk-forward Check</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--soft:#eceefc;--pick:#fff7e0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--soft:#252a3a;--pick:#3a321a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 6px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
.tables{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}caption{text-align:left;font-weight:700;padding:4px 0 6px}
th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11.5px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.pick td{background:var(--pick)}tr.tot td{font-weight:700;border-top:2px solid var(--border)}td.best{font-weight:800;outline:2px solid currentColor;outline-offset:-3px;border-radius:4px}
.note{color:var(--muted);font-size:12.5px;margin:0 0 10px}.scroll{overflow-x:auto}
</style></head><body><div class="wrap">
<h1>Walk-forward check — sell leader strikes at 09:20, no re-entry, exit 15:15</h1>
<p class="note">Rules are picked on 2021–2023 only, then checked on 2024–2026, data the choice never saw. Highlighted row = the weekday chosen in-sample
({{PICK}}); boxed cell = best weekday × SL in that table. 1 lot, 0.5 pt slippage per order, all charges.</p>
{{BODY}}
</div></body></html>"""


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print(build())
