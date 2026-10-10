"""fwdvol_research.py -- does the FORWARD volatility between this week's and
next week's expiry tell setup B anything the IV rule doesn't?

Setup B sells NEXT week's expiry on a green-IV day when the current expiry
has 1 trading day left, and closes it at the first check where the IV rule
reads normal (results/next_expiry_exit/trades.parquet, ivany_rs). Next
week's option price covers two periods: the last day of this week's expiry
(often event-heavy) and the stretch after it. The forward vol isolates the
second part -- what B is actually short for most of its life:

  sigma_fwd = sqrt( (iv2^2 * T2 - iv1^2 * T1) / (T2 - T1) )

with iv1, T1 the current expiry's ATM IV / time left and iv2, T2 the next
expiry's, both at 09:30 on the entry day (Black-76, put-call-parity forward,
OTM-side IVs, interpolated to the money -- surface.smile()).

Features per B entry day (green AND normal days):
  fwd        forward vol
  fwd_pct    its percentile among the previous 60+ entry days (causal)
  fwd_prem   fwd - iv2      (is the post-expiry stretch priced richer than the
                             next expiry as a whole?)
  fwd_rv     fwd - RV10     (forward vol vs NIFTY's realised 10-session vol)

Tests (1 lot, straddle and 0.15-delta strangle):
  1  green B days: high vs low forward vol (causal median), t-stat, halves
  2  normal B days: does high forward vol find B trades the rule skips?
  3  rank correlations of each measure with B's P&L

Output: results/fwdvol/report.html (+ days.parquet)
"""
import json
import math
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

import market_calendar as mc
import paths
import simulator as S
import surface
import vrp_research

OUT = os.path.join(paths.BASE_DIR, "results", "fwdvol")
ENTRY = "09:30"


def _rows_at(df, day, ts):
    dd = df[(df["day"] == day) & (df.ts <= ts) & (df.ts > ts - 1800)]
    last = dd.sort_values("ts").groupby(["type", "strike"]).close.last()
    rows = {}
    for (typ, k), px in last.items():
        rows.setdefault(int(k), {})["c_px" if typ == "CE" else "p_px"] = float(px)
    return rows


def collect():
    t = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "next_expiry_exit", "trades.parquet"))
    t = t[t.strat.isin(["Short straddle ATM", "Short strangle Δ0.15"])]
    days = t.drop_duplicates("day")[["day", "cur_expiry", "expiry", "green"]].sort_values("day")
    sp = S._load_spot()
    st, sc = sp["ts"].to_numpy(), sp["close"].to_numpy()
    rv = vrp_research.realised()
    out = []
    for r in days.itertuples():
        D = pd.Timestamp(r.day).date()
        ts = int(pd.Timestamp(f"{D} {ENTRY}", tz=S.IST).timestamp())
        e = np.searchsorted(st, ts, side="right") - 1
        if e < 0 or st[e] < ts - 300:
            continue
        spot = float(sc[e])
        ivs = {}
        for key, exp in (("1", str(r.cur_expiry)[:10]), ("2", str(r.expiry)[:10])):
            p = os.path.join(paths.BASE_DIR, "data", "hist1m", "options", f"{exp}.parquet")
            if not os.path.exists(p):
                break
            df = S._load_options(exp)
            rows = _rows_at(df, D, ts)
            if len(rows) < 5:
                break
            T = (surface._close_ts(exp) - ts) / (365 * 86400)
            sm = surface.smile(rows, spot, surface._fwd(rows, spot), T) if T > 0 else None
            if not sm:
                break
            ivs[key] = (sm["atm"], T)
        if len(ivs) < 2:
            continue
        (iv1, T1), (iv2, T2) = ivs["1"], ivs["2"]
        var_f = ((iv2 / 100) ** 2 * T2 - (iv1 / 100) ** 2 * T1) / (T2 - T1)
        fwd = math.sqrt(var_f) * 100 if var_f > 0 else np.nan
        rv10 = rv["rv10"].get(D, np.nan)
        out.append({"day": D, "green": bool(r.green), "iv1": iv1, "iv2": iv2, "T1_days": T1 * 365, "T2_days": T2 * 365,
                    "fwd": fwd, "fwd_prem": fwd - iv2, "slope": iv2 - iv1, "rv10": rv10, "fwd_rv": fwd - rv10})
        print(f"\r{D} {len(out)}", end="", flush=True)
    print()
    d = pd.DataFrame(out).dropna(subset=["fwd"]).sort_values("day").reset_index(drop=True)
    pct = []
    for i, v in enumerate(d.fwd):
        past = d.fwd.iloc[:i]
        pct.append(float((past < v).mean() * 100) if len(past) >= 60 else np.nan)
    d["fwd_pct"] = pct
    pnl = t.pivot_table(index="day", columns="strat", values="ivany_rs", aggfunc="first")
    pnl.columns = ["straddle" if "straddle" in c else "strangle" for c in pnl.columns]
    pnl.index = pd.to_datetime(pnl.index).date
    d = d.join(pnl, on="day")
    d["half"] = np.where(pd.to_datetime(d.day) < pd.Timestamp("2024-01-01"), "2022–23", "2024–26")
    return d


def _summ(p):
    p = pd.Series(p).dropna()
    if p.empty:
        return {"n": 0}
    cum = p.cumsum().to_numpy()
    dd = cum - np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return {"n": int(len(p)), "avg": round(float(p.mean())), "win": round(float((p > 0).mean() * 100)), "total": round(float(p.sum())),
            "worst": round(float(p.min())), "dd": round(float(dd.min())), "t": vrp_research._t(p)}


def analyse(d):
    a = {"n_days": int(len(d)), "n_green": int(d.green.sum()), "first": str(d.day.min()), "last": str(d.day.max()),
         "fwd_med": round(float(d.fwd.median()), 1), "iv1_med": round(float(d.iv1.median()), 1), "iv2_med": round(float(d.iv2.median()), 1)}
    split, halves, normal, corr = [], [], [], []
    for kind in ("straddle", "strangle"):
        for grp, y in (("green", d[d.green & d.fwd_pct.notna()]), ("normal", d[~d.green & d.fwd_pct.notna()])):
            hi, lo = y[y.fwd_pct >= 50], y[y.fwd_pct < 50]
            row = {"kind": kind, "grp": grp, "hi": _summ(hi[kind]), "lo": _summ(lo[kind]), "all": _summ(y[kind]),
                   "diff": round(float(hi[kind].mean() - lo[kind].mean())) if len(hi) and len(lo) else None,
                   "t": vrp_research._t(hi[kind], lo[kind]) if len(hi) > 2 and len(lo) > 2 else None,
                   "fwd_hi": round(float(hi.fwd.mean()), 1) if len(hi) else None, "fwd_lo": round(float(lo.fwd.mean()), 1) if len(lo) else None}
            (split if grp == "green" else normal).append(row)
            for hf, z in y.groupby("half"):
                zh, zl = z[z.fwd_pct >= 50][kind], z[z.fwd_pct < 50][kind]
                halves.append({"kind": kind, "grp": grp, "half": hf, "hi": round(float(zh.mean())) if len(zh) else None, "n_hi": int(len(zh)),
                               "lo": round(float(zl.mean())) if len(zl) else None, "n_lo": int(len(zl))})
        for grp, y in (("green", d[d.green]), ("all B days", d)):
            corr.append({"kind": kind, "grp": grp, **{c: round(float(y[c].rank().corr(y[kind].rank())), 3) for c in ("iv1", "iv2", "fwd", "fwd_prem", "slope", "fwd_rv")}})
    a.update(split=split, normal=normal, halves=halves, corr=corr)
    return a


def verdict(a):
    rs = lambda v: "—" if v is None else ("−" if v < 0 else "") + "₹" + f"{abs(round(v)):,}"
    g = {r["kind"]: r for r in a["split"]}
    n = {r["kind"]: r for r in a["normal"]}
    cg = {r["kind"]: r for r in a["corr"] if r["grp"] == "green"}
    sig = any(r["t"] is not None and abs(r["t"]) >= 2 for r in a["split"] + a["normal"])
    v = [f"<b>Green B days, high vs low forward vol:</b> straddle {rs(g['straddle']['diff'])} (t = {g['straddle']['t']}), strangle {rs(g['strangle']['diff'])} (t = {g['strangle']['t']}).",
         f"<b>Normal B days (the rule skips them):</b> high-forward-vol days averaged {rs(n['straddle']['hi'].get('avg'))} (straddle) / {rs(n['strangle']['hi'].get('avg'))} (strangle) "
         f"vs {rs(n['straddle']['lo'].get('avg'))} / {rs(n['strangle']['lo'].get('avg'))} on low ones (t = {n['straddle']['t']} / {n['strangle']['t']}).",
         f"<b>Rank correlation with green-day B P&amp;L:</b> forward vol {cg['straddle']['fwd']} / {cg['strangle']['fwd']}, next-week IV {cg['straddle']['iv2']} / {cg['strangle']['iv2']}, this-week IV {cg['straddle']['iv1']} / {cg['strangle']['iv1']}."]
    small = min(min(r["hi"].get("n", 0), r["lo"].get("n", 0)) for r in a["split"])
    both_halves = all(any(h["grp"] == "green" and h["half"] == hf and h["n_hi"] >= 10 and h["n_lo"] >= 10 for h in a["halves"]) for hf in ("2022–23", "2024–26"))
    if sig and (small < 15 or not both_halves):
        v.append(f"<b>Verdict: not usable.</b> The green-day split looks significant, but green days almost always have a high forward vol "
                 f"(green IV ⇒ high next-week IV): the low-forward-vol group has only {small} trades, and the split cannot be checked in both halves "
                 "of the sample. Forward vol also tracks next week's ATM IV almost one-for-one (similar correlations), so it adds no new information. "
                 "Not added to the plan.")
    elif not sig:
        v.append("<b>Verdict: no reliable improvement</b> — no split reaches |t| ≥ 2, so forward vol is not added to the plan.")
    else:
        v.append("<b>Verdict: a candidate</b> — significant with enough trades in both halves; needs a walk-forward check before any use.")
    return v


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forward Vol Study</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
</style></head><body><div class="wrap">
<h1>Setup B: does forward volatility help?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="card"><h2>What the data says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>High vs low forward vol (causal median), B trades</h2><div class="scroll"><table id="t1"></table></div>
<p class="note">Forward vol = the implied vol of the stretch between this week's expiry and next week's, from both ATM IVs at 09:30. "High" = above the median of the previous entry days (at least 60). B = sell next week's expiry, exit at the first check where IV reads normal; 1 lot, charges.</p>
<h2 style="margin-top:14px">Each half of the sample (avg ₹ per trade)</h2><div class="scroll"><table id="t2"></table></div></div>
<div class="card"><h2>Rank correlation with B's P&amp;L</h2><div class="scroll"><table id="t3"></table></div>
<p class="note">iv1 = this week's ATM IV (1 day left), iv2 = next week's, fwd = forward vol, fwd − iv2 = forward premium, slope = iv2 − iv1, fwd − RV = forward vol minus realised 10-session vol.</p></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
document.getElementById("meta").textContent = `${D.n_days} B entry days (${D.n_green} green) · ${D.first} → ${D.last} · median forward vol ${D.fwd_med}% (this week ${D.iv1_med}%, next week ${D.iv2_med}%).`;
document.getElementById("verdict").innerHTML = D.verdict.map(x => `<li>${x}</li>`).join("");
const s = x => `<td>${x.n}</td><td class="${cl(x.avg)}"><b>${rs(x.avg)}</b></td><td>${x.win ?? "—"}${x.win != null ? "%" : ""}</td><td class="down">${rs(x.worst)}</td>`;
document.getElementById("t1").innerHTML = `<tr><th>Days</th><th>Trade</th><th>High fwd: n</th><th>avg</th><th>win</th><th>worst</th><th>Low fwd: n</th><th>avg</th><th>win</th><th>worst</th><th>Diff</th><th>t</th><th>fwd hi/lo</th></tr>` +
  D.split.concat(D.normal).map(r => `<tr><td>${r.grp}</td><td><b>${r.kind}</b></td>${s(r.hi)}${s(r.lo)}<td class="${cl(r.diff)}">${rs(r.diff)}</td><td>${r.t ?? "—"}</td><td>${r.fwd_hi ?? "—"} / ${r.fwd_lo ?? "—"}</td></tr>`).join("");
document.getElementById("t2").innerHTML = `<tr><th>Days</th><th>Trade</th><th>Half</th><th>High fwd</th><th>n</th><th>Low fwd</th><th>n</th></tr>` +
  D.halves.map(r => `<tr><td>${r.grp}</td><td>${r.kind}</td><td>${r.half}</td><td class="${cl(r.hi)}">${rs(r.hi)}</td><td>${r.n_hi}</td><td class="${cl(r.lo)}">${rs(r.lo)}</td><td>${r.n_lo}</td></tr>`).join("");
document.getElementById("t3").innerHTML = `<tr><th>Days</th><th>Trade</th><th>iv1</th><th>iv2</th><th>fwd</th><th>fwd − iv2</th><th>slope</th><th>fwd − RV</th></tr>` +
  D.corr.map(r => `<tr><td>${r.grp}</td><td>${r.kind}</td><td>${r.iv1}</td><td>${r.iv2}</td><td>${r.fwd}</td><td>${r.fwd_prem}</td><td>${r.slope}</td><td>${r.fwd_rv}</td></tr>`).join("");
</script></body></html>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "days.parquet")
    if os.path.exists(p) and "--fresh" not in sys.argv:
        d = pd.read_parquet(p)
        d["day"] = pd.to_datetime(d["day"]).dt.date
    else:
        d = collect()
        d.to_parquet(p, index=False)
    a = analyse(d)
    a["verdict"] = verdict(a)
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(a, ensure_ascii=False, default=str)))
    return d, a, path


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    d, a, path = main()
    print(json.dumps({k: v for k, v in a.items() if k != "verdict"}, indent=1, default=str))
    for x in a["verdict"]:
        print("-", x)
    print(path)
