"""intraday_expiry_report.py -- expiry-day-only check of the leader straddle:
SL 30/40/50/60% year by year, in-sample (2021-23) vs out-of-sample (2024-26),
and Thursday-expiry era vs Tuesday-expiry era.

Needs results/intraday_range_bt/leader_sl{30,40,50,60}_exp/days.csv
(python intraday_range_bt.py). Output: results/intraday_range_bt/expiry_report.html
"""
import os
import sys

import pandas as pd

import paths
from intraday_split_report import TEMPLATE, _cell, _rs

ROOT = os.path.join(paths.BASE_DIR, "results", "intraday_range_bt")
SLS = (30, 40, 50, 60)
TUE_ERA = "2025-09-01"  # NIFTY weekly expiry moved from Thursday to Tuesday


def _load():
    frames = []
    for sl in SLS:
        d = pd.read_csv(os.path.join(ROOT, f"leader_sl{sl}_exp", "days.csv"))
        d["sl"] = sl
        frames.append(d)
    d = pd.concat(frames)
    d["year"] = d.day.str[:4]
    d["period"] = d.year.map(lambda y: "2021-23" if y <= "2023" else "2024-26")
    d["era"] = d.day.map(lambda x: "Tuesday expiries (Sep 2025 →)" if x >= TUE_ERA else "Thursday expiries (→ Aug 2025)")
    return d


def _dd(x):
    eq = x.sort_values("day").net_rs.cumsum()
    return (eq - eq.cummax().clip(lower=0)).min() if len(eq) else 0


def _table(d, key, order, caption, metric, fmt):
    t = [f"<table><caption>{caption}</caption><tr><th></th>" + "".join(f"<th>SL {s}%</th>" for s in SLS) + "</tr>"]
    for k in order:
        x = d if k == "All" else d[d[key] == k]
        cls = ' class="tot"' if k in ("All", "2021-23", "2024-26") else ""
        cells = []
        for sl in SLS:
            y = x[x.sl == sl]
            v = {"net": y.net_rs.sum(), "pts": y.pnl_pts.sum(), "win": (y.net_rs > 0).mean() * 100 if len(y) else 0,
                 "dd": _dd(y), "ppd": y.pnl_pts.mean() if len(y) else 0}[metric]
            cells.append(_cell(round(v, 1), fmt))
        n = len(x[x.sl == SLS[0]])
        t.append(f"<tr{cls}><td>{k} <span class='note'>({n} d)</span></td>{''.join(cells)}</tr>")
    return "".join(t) + "</table>"


def build():
    d = _load()
    years = sorted(d.year.unique())
    money = lambda v: _rs(round(v))
    pts = lambda v: f"{v:,.0f}"
    by_year = []
    for metric, cap, fmt in (("net", "Net ₹ (1 lot, after charges)", money), ("pts", "Points per lot", pts),
                             ("win", "Win %", lambda v: f"{v:.0f}%")):
        rows = [y for y in years if y <= "2023"] + ["2021-23"] + [y for y in years if y > "2023"] + ["2024-26", "All"]
        # period rows are matched on the "period" column, years on "year"
        t = [f"<table><caption>{cap}</caption><tr><th></th>" + "".join(f"<th>SL {s}%</th>" for s in SLS) + "</tr>"]
        for r in rows:
            x = d if r == "All" else d[d.period == r] if "-" in r else d[d.year == r]
            cls = ' class="tot"' if r in ("All", "2021-23", "2024-26") else ""
            cells = []
            for sl in SLS:
                y = x[x.sl == sl]
                v = {"net": y.net_rs.sum(), "pts": y.pnl_pts.sum(), "win": (y.net_rs > 0).mean() * 100 if len(y) else 0}[metric]
                cells.append(_cell(round(v, 1), fmt))
            t.append(f"<tr{cls}><td>{r} <span class='note'>({len(x[x.sl == SLS[0]])} d)</span></td>{''.join(cells)}</tr>")
        by_year.append("".join(t) + "</table>")
    eras = ["Thursday expiries (→ Aug 2025)", "Tuesday expiries (Sep 2025 →)", "All"]
    era_tables = [_table(d, "era", eras, "Net ₹", "net", money), _table(d, "era", eras, "Points per day", "ppd", lambda v: f"{v:.1f}"),
                  _table(d, "era", eras, "Max drawdown ₹", "dd", money)]
    periods = ["2021-23", "2024-26"]
    per_dd = _table(d, "period", periods, "Max drawdown ₹ by period", "dd", money)

    body = (f'<div class="card"><h2>Year by year · in-sample 2021–23 vs out-of-sample 2024–26</h2>'
            f'<div class="tables">{"".join(by_year)}</div><div class="tables" style="margin-top:14px">{per_dd}</div></div>'
            f'<div class="card"><h2>Thursday-expiry era vs Tuesday-expiry era</h2>'
            f'<p class="note">NIFTY weekly expiry moved from Thursday to Tuesday in Sep 2025 (a few holiday weeks expired on another day).</p>'
            f'<div class="tables">{"".join(era_tables)}</div></div>')
    page = (TEMPLATE.replace("{{BODY}}", body)
            .replace("Walk-forward Check", "Expiry-day Check")
            .replace("Walk-forward check — sell leader strikes at 09:20, no re-entry, exit 15:15",
                     "Expiry day only — sell leader strikes at 09:20, no re-entry, exit 15:15")
            .replace('''Rules are picked on 2021–2023 only, then checked on 2024–2026, data the choice never saw. Highlighted row = the weekday chosen in-sample
({{PICK}}); boxed cell = best weekday × SL in that table. 1 lot, 0.5 pt slippage per order, all charges.''',
                     "Only the expiry day of each weekly expiry is traded. 1 lot (the lot size of that time), 0.5 pt slippage per order, all charges. "
                     "Points per lot compare fairly across years; rupees depend on the lot size (50 → 25 → 75 → 65)."))
    out = os.path.join(ROOT, "expiry_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    print(build())
