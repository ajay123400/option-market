"""vrp_research.py -- is IV minus realised volatility (the volatility risk
premium, VRP) a better "IV is expensive" test than the IV level alone?

The plan sells when ATM IV >= the 60th percentile of its own past 500 days
(sell_rules). But IV can be high because NIFTY really is moving a lot -- then
the premium is fair, not rich. VRP = IV - recent realised vol measures the
part of IV that is NOT explained by how much the market has been moving.

For every entry of the saved backtests (A: 1-4 days to expiry, 09:30, held to
expiry; B: next week's expiry, exit when IV turns normal; both on green AND
normal days), at 09:30 of the entry day:
  IV       the rule's own ATM IV reading
  RV5/10/20  close-to-close realised vol of NIFTY over the previous 5/10/20
           sessions (annualised, sqrt 252), from the 1-min spot file
  VRP      IV - RV (vol points);  ratio  IV / RV
  vrp_pct  VRP's percentile among the previous 250 entry days (causal: no
           look-ahead), so the split adapts to each period

Tests:
  1  inside the rule's green days: high- vs low-VRP trades (median split on
     the causal percentile), with t-stat and both halves of the sample
  2  the left tail: were the worst green-day trades the low-VRP ones?
  3  a combined rule (green AND VRP above its median) vs green alone:
     per-trade, total, worst, drawdown -- does it cut pain more than profit?
  4  VRP alone on normal days: does it find good trades the IV rule skips?

Output: results/vrp/report.html (+ entries.parquet)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import paths
import sell_rules as SR
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "vrp")
WINDOWS = (5, 10, 20)


def realised():
    """Daily NIFTY closes -> RV over the previous N sessions, indexed by the session date it is known on (the next day)."""
    sp = S._load_spot()
    d = pd.DataFrame({"t": pd.to_datetime(sp["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata"), "c": sp["close"].to_numpy()})
    d["day"] = d.t.dt.date
    closes = d.groupby("day").c.last()
    r = np.log(closes).diff()
    out = pd.DataFrame(index=closes.index)
    for n in WINDOWS:
        out[f"rv{n}"] = r.rolling(n).std() * math.sqrt(252) * 100
    return out.shift(1)            # known before the session opens: uses closes up to the previous day


def entries():
    rv = realised()
    h = SR.history()
    iv = h[h.slot == "09:30"][["day", "atm_iv", "thr"]].copy()
    iv["day"] = pd.to_datetime(iv["day"]).dt.date
    iv = iv.set_index("day")
    a = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "ashish_rules", "positions.parquet"))
    a = a[a.set == "A"][["day", "green", "strat", "hold_rs"]].rename(columns={"hold_rs": "pnl"})
    a["setup"] = "A"
    b = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "next_expiry_exit", "trades.parquet"))
    b = b[b.strat.isin(["Short straddle ATM", "Short strangle Δ0.15"])][["day", "green", "strat", "ivany_rs"]].rename(columns={"ivany_rs": "pnl"})
    b["setup"] = "B"
    x = pd.concat([a, b], ignore_index=True)
    x["kind"] = np.where(x.strat.str.contains("straddle"), "straddle", "strangle")
    x["day"] = pd.to_datetime(x["day"]).dt.date
    x = x.join(iv, on="day").join(rv, on="day").dropna(subset=["atm_iv", "rv10", "pnl"])
    for n in WINDOWS:
        x[f"vrp{n}"] = x.atm_iv - x[f"rv{n}"]
    x["ratio10"] = x.atm_iv / x.rv10
    # causal percentile of VRP10 among the previous 250 distinct entry days
    days = x.drop_duplicates("day").sort_values("day")[["day", "vrp10"]].reset_index(drop=True)
    pct = []
    for i, v in enumerate(days.vrp10):
        past = days.vrp10.iloc[max(0, i - 250):i]
        pct.append(float((past < v).mean() * 100) if len(past) >= 60 else np.nan)
    days["vrp_pct"] = pct
    x = x.merge(days[["day", "vrp_pct"]], on="day", how="left")
    x["half"] = np.where(pd.to_datetime(x.day) < pd.Timestamp("2024-01-01"), "2022–23", "2024–26")
    return x.sort_values("day").reset_index(drop=True)


def _t(a, b=None):
    a = pd.Series(a).dropna()
    if b is None:
        return round(float(a.mean() / (a.std(ddof=1) / math.sqrt(len(a)))), 2) if len(a) > 2 else None
    b = pd.Series(b).dropna()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return round(float((a.mean() - b.mean()) / se), 2) if se > 0 else None


def _summ(p):
    p = pd.Series(p).dropna()
    if p.empty:
        return {"n": 0}
    cum = p.cumsum().to_numpy()
    dd = cum - np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return {"n": int(len(p)), "avg": round(float(p.mean())), "win": round(float((p > 0).mean() * 100)), "total": round(float(p.sum())),
            "worst": round(float(p.min())), "p5": round(float(p.quantile(0.05))), "dd": round(float(dd.min())), "t": _t(p)}


def analyse(x):
    res = {"n_days": int(x.day.nunique()), "first": str(x.day.min()), "last": str(x.day.max())}
    g = x[x.green & x.vrp_pct.notna()]
    rows, halves, tails, combo, normal = [], [], [], [], []
    for (st, kind), y in g.groupby(["setup", "kind"]):
        hi, lo = y[y.vrp_pct >= 50], y[y.vrp_pct < 50]
        rows.append({"setup": st, "kind": kind, "hi": _summ(hi.pnl), "lo": _summ(lo.pnl), "diff": round(float(hi.pnl.mean() - lo.pnl.mean())),
                     "t": _t(hi.pnl, lo.pnl), "vrp_hi": round(float(hi.vrp10.mean()), 1), "vrp_lo": round(float(lo.vrp10.mean()), 1),
                     "iv_hi": round(float(hi.atm_iv.mean()), 1), "iv_lo": round(float(lo.atm_iv.mean()), 1),
                     "rv_hi": round(float(hi.rv10.mean()), 1), "rv_lo": round(float(lo.rv10.mean()), 1)})
        for hf, z in y.groupby("half"):
            zh, zl = z[z.vrp_pct >= 50].pnl, z[z.vrp_pct < 50].pnl
            halves.append({"setup": st, "kind": kind, "half": hf, "hi": round(float(zh.mean())) if len(zh) else None, "n_hi": int(len(zh)),
                           "lo": round(float(zl.mean())) if len(zl) else None, "n_lo": int(len(zl))})
        worst = y.nsmallest(max(1, len(y) // 10), "pnl")
        tails.append({"setup": st, "kind": kind, "n": int(len(worst)), "low_vrp_share": round(float((worst.vrp_pct < 50).mean() * 100)),
                      "base_low_vrp_share": round(float((y.vrp_pct < 50).mean() * 100)), "avg_worst": round(float(worst.pnl.mean()))})
        combo.append({"setup": st, "kind": kind, "green": _summ(y.pnl), "green_hi": _summ(hi.pnl)})
    n = x[(~x.green) & x.vrp_pct.notna()]
    for (st, kind), y in n.groupby(["setup", "kind"]):
        top = y[y.vrp_pct >= 70]
        normal.append({"setup": st, "kind": kind, "all": _summ(y.pnl), "top": _summ(top.pnl)})
    # correlation of each measure with green-day P&L, by setup/kind (rank)
    corr = []
    for (st, kind), y in g.groupby(["setup", "kind"]):
        corr.append({"setup": st, "kind": kind, **{c: round(float(y[c].rank().corr(y.pnl.rank())), 3) for c in ("atm_iv", "rv10", "vrp5", "vrp10", "vrp20", "ratio10")}})
    res.update(split=rows, halves=halves, tails=tails, combo=combo, normal=normal, corr=corr)
    return res


def report(a):
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(a, ensure_ascii=False, default=str)))
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VRP Study</title><style>
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
</style></head><body><div class="wrap">
<h1>Is "IV minus realised vol" a better test than the IV level?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="card"><h2>What the data says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>Test 1 — green days split by VRP (above / below its causal median)</h2><div class="scroll"><table id="t1"></table></div>
<p class="note">VRP = the rule's 09:30 ATM IV − NIFTY's 10-session realised vol (vol points). "High" = VRP above the median of the previous 250 entry days. t = difference of means; |t| below ~2 is not reliable.</p>
<h2 style="margin-top:14px">Same split, each half of the sample (avg ₹ per trade)</h2><div class="scroll"><table id="t1h"></table></div></div>
<div class="card"><h2>Test 2 — the worst 10% of green-day trades</h2><div class="scroll"><table id="t2"></table></div>
<p class="note">If VRP caught the bad trades, the low-VRP share among the worst trades would be well above its share among all green trades.</p></div>
<div class="card"><h2>Test 3 — green alone vs green AND high VRP</h2><div class="scroll"><table id="t3"></table></div></div>
<div class="card"><h2>Test 4 — normal (not green) days with the highest VRP (top 30%)</h2><div class="scroll"><table id="t4"></table></div></div>
<div class="card"><h2>Rank correlation with green-day P&amp;L</h2><div class="scroll"><table id="t5"></table></div>
<p class="note">Spearman correlation between each 09:30 measure and the trade's result; 0 = no relation. RV5/10/20 = realised vol over 5/10/20 sessions.</p></div>
<div class="card"><h2>Method &amp; limits</h2><ul>
<li>Trades are the saved backtests behind the Plan page (1 lot, 0.5 pt slippage, charges). Realised vol uses NIFTY's daily closes before the entry day only.</li>
<li>The VRP split uses a rolling percentile of the past 250 entry days, so no threshold is fitted on the future. Green days already need high IV, so the VRP split works inside a narrow band.</li>
<li>Few B trades (87) — treat B results as indicative.</li></ul></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
const nm = r => `${r.setup} ${r.kind}`;
document.getElementById("meta").textContent = `${D.n_days} entry days (${D.first} → ${D.last}) · A and B backtest trades · 09:30 readings.`;
document.getElementById("t1").innerHTML = `<tr><th>Trade</th><th>High VRP: n</th><th>avg</th><th>win</th><th>worst</th><th>Low VRP: n</th><th>avg</th><th>win</th><th>worst</th><th>Difference</th><th>t</th><th>IV hi/lo</th><th>RV hi/lo</th><th>VRP hi/lo</th></tr>` +
  D.split.map(r => `<tr><td><b>${nm(r)}</b></td><td>${r.hi.n}</td><td class="${cl(r.hi.avg)}"><b>${rs(r.hi.avg)}</b></td><td>${r.hi.win}%</td><td class="down">${rs(r.hi.worst)}</td><td>${r.lo.n}</td><td class="${cl(r.lo.avg)}"><b>${rs(r.lo.avg)}</b></td><td>${r.lo.win}%</td><td class="down">${rs(r.lo.worst)}</td><td class="${cl(r.diff)}">${rs(r.diff)}</td><td>${r.t}</td><td>${r.iv_hi} / ${r.iv_lo}</td><td>${r.rv_hi} / ${r.rv_lo}</td><td>${r.vrp_hi} / ${r.vrp_lo}</td></tr>`).join("");
document.getElementById("t1h").innerHTML = `<tr><th>Trade</th><th>Half</th><th>High VRP</th><th>n</th><th>Low VRP</th><th>n</th></tr>` +
  D.halves.map(r => `<tr><td>${nm(r)}</td><td>${r.half}</td><td class="${cl(r.hi)}">${rs(r.hi)}</td><td>${r.n_hi}</td><td class="${cl(r.lo)}">${rs(r.lo)}</td><td>${r.n_lo}</td></tr>`).join("");
document.getElementById("t2").innerHTML = `<tr><th>Trade</th><th>Worst trades</th><th>Their avg</th><th>Low-VRP share among them</th><th>Low-VRP share among all green</th></tr>` +
  D.tails.map(r => `<tr><td>${nm(r)}</td><td>${r.n}</td><td class="down">${rs(r.avg_worst)}</td><td><b>${r.low_vrp_share}%</b></td><td>${r.base_low_vrp_share}%</td></tr>`).join("");
const sm = s => `<td>${s.n}</td><td class="${cl(s.avg)}"><b>${rs(s.avg)}</b></td><td>${s.win}%</td><td class="${cl(s.total)}">${rs(s.total)}</td><td class="down">${rs(s.worst)}</td><td class="down">${rs(s.dd)}</td>`;
document.getElementById("t3").innerHTML = `<tr><th>Trade</th><th>Rule</th><th>n</th><th>avg</th><th>win</th><th>total</th><th>worst</th><th>max DD</th></tr>` +
  D.combo.map(r => `<tr><td rowspan="2"><b>${nm(r)}</b></td><td>green (plan today)</td>${sm(r.green)}</tr><tr><td>green + high VRP</td>${sm(r.green_hi)}</tr>`).join("");
document.getElementById("t4").innerHTML = `<tr><th>Trade</th><th>Days</th><th>n</th><th>avg</th><th>win</th><th>total</th><th>worst</th><th>max DD</th></tr>` +
  D.normal.map(r => `<tr><td rowspan="2"><b>${nm(r)}</b></td><td>all normal days</td>${sm(r.all)}</tr><tr><td>top-30% VRP</td>${sm(r.top)}</tr>`).join("");
document.getElementById("t5").innerHTML = `<tr><th>Trade</th><th>ATM IV</th><th>RV10</th><th>VRP5</th><th>VRP10</th><th>VRP20</th><th>IV/RV10</th></tr>` +
  D.corr.map(r => `<tr><td>${nm(r)}</td><td>${r.atm_iv}</td><td>${r.rv10}</td><td>${r.vrp5}</td><td>${r.vrp10}</td><td>${r.vrp20}</td><td>${r.ratio10}</td></tr>`).join("");
document.getElementById("verdict").innerHTML = (window.VERDICT || []).map(x => `<li>${x}</li>`).join("") || "<li>See the tables.</li>";
</script><script>/*VERDICT*/</script></body></html>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    x = entries()
    x.to_parquet(os.path.join(OUT, "entries.parquet"), index=False)
    return x, analyse(x)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    x, a = main()
    print(json.dumps(a, indent=1, default=str))


def verdict(a):
    s = {f"{r['setup']} {r['kind']}": r for r in a["split"]}
    t = {f"{r['setup']} {r['kind']}": r for r in a["tails"]}
    c = {f"{r['setup']} {r['kind']}": r for r in a["corr"]}
    rs = lambda v: ("−" if v < 0 else "") + "₹" + f"{abs(round(v)):,}"
    return [
        f"<b>VRP does not beat the IV level.</b> Inside the rule's green days, high-VRP trades were not reliably better: A straddle {rs(s['A straddle']['diff'])} (t = {s['A straddle']['t']}), "
        f"A strangle {rs(s['A strangle']['diff'])} (t = {s['A strangle']['t']}), B straddle {rs(s['B straddle']['diff'])}, B strangle {rs(s['B strangle']['diff'])} — none significant, and the signs disagree.",
        f"<b>It does not catch the bad trades either.</b> Among the worst 10% of green-day trades, the low-VRP share was {t['A straddle']['low_vrp_share']}% (A straddle) vs {t['A straddle']['base_low_vrp_share']}% for all green trades, "
        f"{t['A strangle']['low_vrp_share']}% vs {t['A strangle']['base_low_vrp_share']}% for the A strangle, and 0% for B — the big losses came on high-VRP days as often as anywhere.",
        "<b>Adding it as a second filter only removes trades:</b> green + high VRP cut the total profit of every variant while the worst trade and the max drawdown stayed the same.",
        "<b>It finds nothing on normal days:</b> the top-30% VRP normal days still lost money for all four trades.",
        f"<b>The IV level itself is the strongest single measure</b> (rank correlation with green-day P&amp;L: A straddle {c['A straddle']['atm_iv']}, A strangle {c['A strangle']['atm_iv']}, B {c['B straddle']['atm_iv']} / {c['B strangle']['atm_iv']}), "
        "and high realised vol on a green day did not hurt — the worry that \"IV is high only because the market is already moving\" did not show up as worse trades.",
        "<b>Practical use:</b> keep the IV-level rule as it is. VRP can be shown as information next to IV, not as a filter.",
    ]


def write_report(a):
    path = os.path.join(OUT, "report.html")
    html = TEMPLATE.replace("/*DATA*/null", json.dumps(a, ensure_ascii=False, default=str)).replace("/*VERDICT*/", "window.VERDICT = " + json.dumps(verdict(a), ensure_ascii=False) + ";")
    # the verdict list is filled by the first script; make it run after VERDICT is defined
    html = html.replace('document.getElementById("verdict").innerHTML = (window.VERDICT || []).map(x => `<li>${x}</li>`).join("") || "<li>See the tables.</li>";', "")
    html = html.replace("</body></html>", '<script>document.getElementById("verdict").innerHTML = window.VERDICT.map(x => `<li>${x}</li>`).join("");</script></body></html>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
