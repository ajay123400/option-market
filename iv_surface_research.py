"""iv_surface_research.py -- does the implied-volatility surface help?

1. Filters: ATM IV term structure (nearest weekly vs the next), put-call skew
   (25-delta put IV - 25-delta call IV) and IV rank, at 09:30 each day -- do
   they separate good from bad days for the Strategy Ideas templates
   (results/template_bt/trades.parquet, 09:30 entries)?
2. Strike selection: a quadratic smile is fitted to each day's out-of-the-money
   options; the residual (IV - fitted IV) says how rich / cheap a strike is
   against its neighbours. Does selling the rich strike beat selling the cheap
   one of the same delta?

Only options that traded IN the 09:30 minute are used (a stale last price
would look falsely rich), and the sale is priced at the NEXT minute's close
(09:31) + slippage, so the signal and the fill are not the same print.

Output: results/iv_surface/report.html (+ options.parquet, days.parquet)
"""
import json
import math
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import greeks as g
import manual_trades
import market_calendar as mc
import paths
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "iv_surface")
LOT = manual_trades.LOT_SIZE
SLIP = 0.5
ENTRY = "09:30"
F_O_CLOSE_CHANGE = date(2026, 8, 3)
R = g.RISK_FREE_RATE


def _dte(day, exp_d):
    n, d = 0, day + timedelta(days=1)
    while d <= exp_d:
        n += mc.is_trading_day(d)
        d += timedelta(days=1)
    return n


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def collect(years=5):
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") == "done")
    cutoff = (date.today() - timedelta(days=int(365.25 * years))).isoformat()
    sp = S._load_spot()
    st, sc = sp["ts"].to_numpy(), sp["close"].to_numpy()
    opts, days = [], []
    for n_e, exp in enumerate(exps):
        if exp < cutoff:
            continue
        df = S._load_options(exp)
        exp_d = pd.Timestamp(exp).date()
        close_t = "15:40" if exp_d >= F_O_CLOSE_CHANGE else "15:30"
        exp_close_ts = int(pd.Timestamp(f"{exp} {close_t}", tz=S.IST).timestamp())
        j = np.searchsorted(st, int(pd.Timestamp(f"{exp} 23:59", tz=S.IST).timestamp()), side="right") - 1
        settle = float(sc[j])
        for D in sorted(df["day"].unique()):
            if D > exp_d:
                continue
            dte = _dte(D, exp_d)
            if dte > 9:
                continue
            t_ent = int(pd.Timestamp(f"{D} {ENTRY}", tz=S.IST).timestamp())
            dd = df[(df["day"] == D) & (df.ts >= t_ent) & (df.ts <= t_ent + 60)]
            now = dd[dd.ts == t_ent].set_index(["type", "strike"]).close
            nxt = dd[dd.ts == t_ent + 60].set_index(["type", "strike"]).close
            if len(now) < 10:
                continue
            spot = float(sc[max(0, np.searchsorted(st, t_ent, side="right") - 1)])
            strikes = sorted({k for (_, k) in now.index})
            near = [(k, float(now[("CE", k)]), float(now[("PE", k)])) for k in strikes
                    if ("CE", k) in now.index and ("PE", k) in now.index]
            near = sorted(near, key=lambda x: abs(x[0] - spot))[:6]
            fwd = g.synthetic_forward(near) if near else None
            T = (exp_close_ts - t_ent) / (365 * 86400)
            if not fwd or T <= 0:
                continue
            rows = []
            for (typ, k), p in now.items():
                if abs(k - fwd) > 2000 or p <= 0.5:
                    continue
                call = typ == "CE"
                iv = g.implied_vol_fwd(float(p), fwd, k, T, R, call)
                if not iv:
                    continue
                d1 = (math.log(fwd / k) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
                delta = math.exp(-R * T) * (_ncdf(d1) if call else _ncdf(d1) - 1)
                rows.append({"type": typ, "strike": int(k), "price": float(p), "iv": iv, "delta": delta,
                             "otm": (k >= fwd) if call else (k < fwd), "next": float(nxt.get((typ, k), np.nan))})
            if not rows:
                continue
            q = pd.DataFrame(rows)
            atm_k = int(round(fwd / 50) * 50)
            a = q[q.strike == atm_k].iv
            if len(a) < 2:
                continue
            atm_iv = float(a.mean())
            o = q[q.otm & (q.delta.abs() >= 0.04) & (q.delta.abs() <= 0.5)].copy()
            if len(o) < 8:
                continue
            # standardised moneyness; quadratic smile through the OTM wing of both sides
            o["x"] = np.log(o.strike / fwd) / (atm_iv * math.sqrt(T))
            c = np.polyfit(o.x, o.iv, 2)
            o["fit"] = np.polyval(c, o.x)
            o["resid"] = o.iv - o.fit

            def at(typ, dlt):
                s = o[o.type == typ]
                if s.empty:
                    return None
                i = (s.delta.abs() - dlt).abs().idxmin()
                return float(s.loc[i, "iv"]) if abs(abs(s.loc[i, "delta"]) - dlt) < 0.07 else None
            p25, c25, p10, c10 = at("PE", 0.25), at("CE", 0.25), at("PE", 0.10), at("CE", 0.10)
            days.append({"expiry": exp, "day": D.isoformat(), "dte": dte, "fwd": fwd, "atm_iv": atm_iv,
                         "skew25": (p25 - c25) if p25 and c25 else None, "skew10": (p10 - c10) if p10 and c10 else None,
                         "curv": float(c[0])})
            intr = np.where(o.type == "CE", np.maximum(0, settle - o.strike), np.maximum(0, o.strike - settle))
            o["pnl_pts"] = (o["next"] - SLIP) - intr          # short, sold at 09:31, held to expiry
            o["expiry"], o["day"], o["dte"] = exp, D.isoformat(), dte
            opts.append(o.drop(columns=["otm"]))
        print(f"\r{exp} {len(days)} days", end="", flush=True)
    print()
    return pd.concat(opts, ignore_index=True), pd.DataFrame(days)


def day_features(days):
    """Per trading day: term = ATM IV of the nearest expiry with >= 1 day left
    / ATM IV of the next expiry; IV rank of that near IV over the past year."""
    rows = []
    for d, x in days.groupby("day"):
        x = x[x.dte >= 1].sort_values("dte")
        if len(x) < 2:
            continue
        rows.append({"day": d, "near_iv": x.atm_iv.iloc[0], "next_iv": x.atm_iv.iloc[1],
                     "term": x.atm_iv.iloc[0] / x.atm_iv.iloc[1]})
    f = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    f["ivr"] = [((f.near_iv.iloc[max(0, i - 250):i] < v).mean() * 100) if i >= 60 else np.nan
                for i, v in enumerate(f.near_iv)]
    return f


KEY = ["Short strangle|Δ0.15", "Short strangle|Δ0.20", "Short straddle|ATM", "Short put|Δ0.20", "Short call|Δ0.20",
       "Iron condor|Δ0.15|w200", "Short strangle|range"]
FEATS = {"term": "Term: this week's ATM IV ÷ next week's", "skew25": "Put skew: 25Δ put IV − 25Δ call IV (this expiry)",
         "ivr": "IV rank (1 year)", "atm_iv": "ATM IV level (this expiry)"}


def filters(days, feats):
    t = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "template_bt", "trades.parquet"))
    t = t[(t.slot == ENTRY) & t.tpl.isin(KEY) & (t.dte >= 1) & (t.dte <= 4)]
    t = t.merge(feats[["day", "term", "ivr"]], on="day", how="left")
    t = t.merge(days[["expiry", "day", "skew25", "atm_iv"]].rename(columns={"atm_iv": "atm_iv_s"}), on=["expiry", "day"], how="left")
    t["atm_iv"] = t["atm_iv_s"] * 100
    t["skew25"] = t["skew25"] * 100
    out = {}
    for f in FEATS:
        x = t.dropna(subset=[f])
        # quintiles over DAYS, not trades, so every bucket spans the same number of days
        edges = np.unique(np.quantile(x.drop_duplicates("day")[f], [0, .2, .4, .6, .8, 1]))
        x = x.assign(q=pd.cut(x[f], edges, include_lowest=True, labels=False))
        res = []
        for tpl in KEY:
            y = x[x.tpl == tpl]
            cells = []
            for qi in range(len(edges) - 1):
                z = y[y.q == qi]
                cells.append({"n": int(len(z)), "avg": round(float(z.hold_rs.mean())) if len(z) else None,
                              "a": round(float(z[z.expiry < "2024"].hold_rs.mean())) if (z.expiry < "2024").any() else None,
                              "b": round(float(z[z.expiry >= "2024"].hold_rs.mean())) if (z.expiry >= "2024").any() else None,
                              "win": round(float((z.hold_rs > 0).mean() * 100), 1) if len(z) else None})
            res.append({"tpl": tpl, "all": round(float(y.hold_rs.mean())), "cells": cells})
        out[f] = {"edges": [round(float(e), 3) for e in edges], "rows": res}
    return out


def strike_selection(opts):
    o = opts[(opts.dte >= 1) & (opts.dte <= 4) & np.isfinite(opts["next"]) & (opts.delta.abs() >= 0.08) & (opts.delta.abs() <= 0.35)].copy()
    o["band"] = (o.delta.abs() // 0.05) * 0.05
    # 1) within the same day / expiry / side / 5-delta band: richest vs cheapest strike
    pairs = []
    for _, x in o.groupby(["expiry", "day", "type", "band"]):
        if len(x) < 2:
            continue
        r, c = x.loc[x.resid.idxmax()], x.loc[x.resid.idxmin()]
        pairs.append({"day": r.day, "type": r.type, "diff": r.pnl_pts - c.pnl_pts, "dres": (r.resid - c.resid) * 100,
                      "ddelta": abs(r.delta) - abs(c.delta), "rich": r.pnl_pts, "cheap": c.pnl_pts})
    p = pd.DataFrame(pairs)

    def tstat(v):
        return float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))) if len(v) > 2 else None
    pair_res = []
    for lab, x in (("All", p), ("Puts", p[p.type == "PE"]), ("Calls", p[p.type == "CE"]),
                   ("2021–23", p[p.day < "2024"]), ("2024–26", p[p.day >= "2024"])):
        pair_res.append({"k": lab, "n": int(len(x)), "rich": round(float(x.rich.mean()), 2), "cheap": round(float(x.cheap.mean()), 2),
                         "diff": round(float(x["diff"].mean()), 2), "rs": round(float(x["diff"].mean() * LOT)),
                         "t": round(tstat(x["diff"]), 2), "dres": round(float(x.dres.mean()), 2), "ddelta": round(float(x.ddelta.mean()), 3)})
    # 2) residual quintiles across all options (same delta range), short P&L per point of premium
    o["rq"] = o.groupby("band").resid.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    quint = [{"q": int(qi), "n": int(len(x)), "resid": round(float(x.resid.mean() * 100), 2), "pnl": round(float(x.pnl_pts.mean()), 2),
              "per_prem": round(float((x.pnl_pts / x["next"]).mean() * 100), 1), "delta": round(float(x.delta.abs().mean()), 3),
              "a": round(float(x[x.day < "2024"].pnl_pts.mean()), 2), "b": round(float(x[x.day >= "2024"].pnl_pts.mean()), 2)}
             for qi, x in o.groupby("rq")]
    # 3) a practical rule: short strangle near 0.20 delta -- nearest-delta strike vs the richest in 0.15-0.25
    rule = []
    for (exp, d), x in o.groupby(["expiry", "day"]):
        legs = {}
        for typ in ("CE", "PE"):
            s = x[(x.type == typ) & (x.delta.abs() >= 0.15) & (x.delta.abs() <= 0.25)]
            if s.empty:
                break
            legs[typ] = {"nearest": s.loc[(s.delta.abs() - 0.20).abs().idxmin()], "rich": s.loc[s.resid.idxmax()],
                         "cheap": s.loc[s.resid.idxmin()]}
        if len(legs) < 2:
            continue
        rule.append({"day": d, **{m: legs["CE"][m].pnl_pts + legs["PE"][m].pnl_pts for m in ("nearest", "rich", "cheap")},
                     **{f"prem_{m}": legs["CE"][m]["next"] + legs["PE"][m]["next"] for m in ("nearest", "rich", "cheap")},
                     "same": bool(legs["CE"]["nearest"].strike == legs["CE"]["rich"].strike and legs["PE"]["nearest"].strike == legs["PE"]["rich"].strike)})
    rl = pd.DataFrame(rule)
    rule_res = []
    for m, lab in (("nearest", "Nearest to Δ0.20 (today's rule)"), ("rich", "Richest IV in Δ0.15–0.25"), ("cheap", "Cheapest IV in Δ0.15–0.25")):
        v = rl[m] * LOT
        rule_res.append({"k": lab, "n": int(len(rl)), "avg": round(float(v.mean())), "total": round(float(v.sum())),
                         "win": round(float((v > 0).mean() * 100), 1), "prem": round(float(rl[f"prem_{m}"].mean() * LOT)),
                         "a": round(float(v[rl.day < "2024"].mean())), "b": round(float(v[rl.day >= "2024"].mean())),
                         "p5": round(float(v.quantile(0.05)))})
    d_rn = (rl["rich"] - rl["nearest"]) * LOT
    rule_diff = {"avg": round(float(d_rn.mean())), "t": round(tstat(d_rn), 2), "same_pct": round(float(rl.same.mean() * 100), 1)}
    return {"pairs": pair_res, "quint": quint, "rule": rule_res, "rule_diff": rule_diff}


OOS_TPL = ["Short strangle|Δ0.15", "Short strangle|Δ0.20", "Short straddle|ATM", "Short strangle|range", "Iron condor|Δ0.15|w200"]


def oos(days, feats):
    """Thresholds set on 2021-23 days (top 40%), then applied to both periods."""
    t = pd.read_parquet(os.path.join(paths.BASE_DIR, "results", "template_bt", "trades.parquet")).drop(columns=["atm_iv"])
    t = t[(t.slot == ENTRY) & (t.dte >= 1) & (t.dte <= 4)]
    t = t.merge(feats[["day", "term"]], on="day").merge(days[["expiry", "day", "atm_iv", "skew25"]], on=["expiry", "day"])
    t["iv"], t["sk"] = t.atm_iv * 100, t.skew25 * 100
    t["half"] = np.where(t.expiry < "2024", "2021–23", "2024–26")
    dA = t[t.half == "2021–23"].drop_duplicates("day")
    thr = {"iv": float(dA.iv.quantile(0.6)), "term": float(dA.term.quantile(0.6)), "sk": float(dA.sk.quantile(0.6))}
    share = {h: {"iv": round(float((x.drop_duplicates("day").iv >= thr["iv"]).mean() * 100)),
                 "term": round(float((x.drop_duplicates("day").term >= thr["term"]).mean() * 100))} for h, x in t.groupby("half")}

    def st(y):
        h = y.hold_rs
        return {"n": int(len(y)), "avg": round(float(h.mean())) if len(y) else None, "total": round(float(h.sum())),
                "win": round(float((h > 0).mean() * 100), 1) if len(y) else None,
                "p5": round(float(h.quantile(0.05))) if len(y) else None, "worst": round(float(h.min())) if len(y) else None}
    rows, grid = [], []
    for tpl in OOS_TPL:
        for half, x in t[t.tpl.isin([tpl])].groupby("half"):
            hi, th = x.iv >= thr["iv"], x.term >= thr["term"]
            rows.append({"tpl": tpl, "half": half, "all": st(x), "iv_hi": st(x[hi]), "iv_lo": st(x[~hi]),
                         "t_hi": st(x[th]), "t_lo": st(x[~th])})
            grid.append({"tpl": tpl, "half": half, "hh": st(x[hi & th]), "hl": st(x[hi & ~th]), "lh": st(x[~hi & th]), "ll": st(x[~hi & ~th])})
    side = []
    for half, x in t.groupby("half"):
        for tpl in ("Short put|Δ0.20", "Short call|Δ0.20"):
            y = x[x.tpl == tpl]
            hs = y.sk >= thr["sk"]
            side.append({"half": half, "tpl": tpl, "steep": st(y[hs]), "flat": st(y[~hs])})
    return {"thr": {k: round(v, 3) for k, v in thr.items()}, "share": share, "rows": rows, "grid": grid, "side": side,
            "corr": round(float(feats.term.corr(feats.near_iv)), 2)}


def report(fl, ss, oo, n_opts, n_days):
    data = {"filters": fl, "feats": FEATS, "ss": ss, "oos": oo, "n_opts": n_opts, "n_days": n_days, "lot": LOT}
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IV Surface Study</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--best:#e9f8f1;--grp:#eef0f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--best:#16302a;--grp:#23262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1360px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 7px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-size:11px;text-transform:uppercase}.up{color:var(--up)}.down{color:var(--down)}
tr.grp td{background:var(--grp);font-weight:700;font-size:12px;color:var(--muted)}
.note{color:var(--muted);font-size:12px}.scroll{overflow-x:auto}ul{margin:0;padding-left:18px}li{margin:6px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px;font-size:12.5px}
button{font:inherit;font-size:12.5px;border:1px solid var(--border);background:var(--card);color:var(--text);border-radius:8px;padding:4px 10px;cursor:pointer}
button.on{background:#4f5fe0;color:#fff;border-color:#4f5fe0}.sm{font-size:11px;color:var(--muted)}
</style></head><body><div class="wrap">
<h1>Does the IV surface help? — filters and strike selection</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>What the data says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>The out-of-sample test — thresholds set on 2021–23, applied to 2024–26</h2>
<p class="note" id="thrNote"></p><div class="scroll"><table id="tOos"></table></div></div>
<div class="card"><h2>Is term structure more than the IV level? (IV high/low × term high/low)</h2>
<div class="scroll"><table id="tGrid"></table></div><p class="note" id="corrNote"></p></div>
<div class="card"><h2>Which side to sell — put skew</h2><div class="scroll"><table id="tSide"></table></div></div>
<div class="card"><h2>Strike selection — selling the rich strike vs the cheap one</h2>
<p class="note">A quadratic smile is fitted each day to the out-of-the-money options; residual = strike IV − fitted IV. Only options that traded in the 09:30 minute; sold at the 09:31 close − 0.5 pt, held to expiry. Days 1–4 before expiry, |Δ| 0.08–0.35.</p>
<h2 style="margin-top:14px">Same day, same side, same 5-delta band: richest vs cheapest strike (points per option)</h2>
<div class="scroll"><table id="tPairs"></table></div>
<h2 style="margin-top:18px">A practical rule: short strangle around Δ0.20 (₹ per lot, both legs)</h2>
<div class="scroll"><table id="tRule"></table></div>
<p class="note" id="ruleNote"></p></div>
<div class="card"><h2>Each IV measure in fifths (all 5 years, entries 1–4 days before expiry at 09:30)</h2>
<div class="bar" id="fBar"></div><div class="scroll"><table id="tQ"></table></div>
<p class="note">Cells: ₹ per trade (2021–23 / 2024–26). Fifths are of days, so each column covers about the same number of days.</p></div>
<div class="card"><h2>Method &amp; limits</h2><ul>
<li>IVs from 1-min closes with Black-76 on the put-call-parity forward; ATM IV = average of the ATM call and put; 25Δ skew = IV of the put nearest 25Δ − IV of the call nearest 25Δ (same expiry).</li>
<li>Term = ATM IV of the nearest expiry with ≥ 1 day left ÷ ATM IV of the one after, at 09:30.</li>
<li>Strategy results are the 5-year template backtest (same strike rules as the live Ideas table), 09:30 entries 1–4 days before expiry, held to expiry, 1 lot, all charges.</li>
<li>High-IV days are fewer and come in clusters (a volatile month gives many of them) — the n are not independent days of evidence.</li>
<li>An absolute IV threshold can drift with the market's regime; re-check it each year.</li>
</ul></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
const O = D.oos, T = O.thr;
document.getElementById("meta").textContent = `${D.n_days.toLocaleString("en-IN")} expiry-days and ${D.n_opts.toLocaleString("en-IN")} option quotes at 09:30 (Sep 2021 → Sep 2026) · 1 lot (${D.lot}) · all charges.`;
const R = (tpl, h) => O.rows.find(r => r.tpl === tpl && r.half === h);
const sB = R("Short straddle|ATM", "2024–26"), gB = R("Short strangle|Δ0.20", "2024–26");
const P0 = D.ss.pairs[0], RD = D.ss.rule_diff;
const kp = (v, l, c = "") => `<div class="kpi"><b class="${c}">${v}</b><span>${l}</span></div>`;
document.getElementById("kpis").innerHTML =
  kp(`${rs(sB.iv_hi.avg)} vs ${rs(sB.iv_lo.avg)}`, `short straddle per trade in 2024–26 on high-IV vs other days (threshold from 2021–23)`) +
  kp(`${D.oos.share["2024–26"].iv}%`, `of 2024–26 days were high-IV (ATM IV ≥ ${T.iv.toFixed(1)}%) — they made ${rs(sB.iv_hi.total)} of the straddle's ${rs(sB.all.total)}`) +
  kp(rs(gB.iv_hi.avg) + " vs " + rs(gB.iv_lo.avg), "short strangle Δ0.20 per trade, 2024–26, high-IV vs other days") +
  kp(`${rs(P0.rs)} (t ${P0.t})`, "extra per lot from selling the richest strike instead of the cheapest of the same delta — noise");
const L = (tpl, h) => R(tpl, h);
document.getElementById("verdict").innerHTML = [
  `<b>The IV level is the filter that works.</b> Selling premium only when this week's ATM IV is high (≥ ${T.iv.toFixed(1)}%, the top 40% of 2021–23 days) held up out of sample: in 2024–26 the short straddle made ${rs(sB.iv_hi.avg)} a trade on those days and ${rs(sB.iv_lo.avg)} on the rest; the strangle Δ0.20 ${rs(gB.iv_hi.avg)} vs ${rs(gB.iv_lo.avg)}. Same direction in 2021–23. On the other days the short-premium templates earned little (strangles) or nothing to a loss (straddle, condor).`,
  `<b>High IV did not bring a worse tail here:</b> the 5% worst straddle trade on those days was ${rs(sB.iv_hi.p5)} vs ${rs(sB.iv_lo.p5)} on other days (the richer premium cushions the moves). The single worst was still ${rs(sB.iv_hi.worst)} — a naked straddle needs sizing for that either way.`,
  `<b>Term structure mostly repeats the IV level</b> (correlation ${O.corr}): when this week's IV is well above next week's, IV is usually high too. Inside high-IV days, 2021–23 favoured a steep term structure but 2024–26 did not — no stable extra edge.`,
  `<b>Put skew helps choose the side.</b> With a steep put skew (25Δ put IV − call IV ≥ ${T.sk.toFixed(2)} vol pts) the short put earned more in both periods; the short call did better on flat-skew days only in 2024–26. Steep skew also coincides with high IV, so it is partly the same signal.`,
  `<b>The surface does not help pick the strike.</b> The fitted smile explains NIFTY's weekly IVs almost fully: the richest and cheapest strike of the same delta differ by only ~${P0.dres} vol points, and selling the rich one earned ${rs(P0.rs)} more per lot (t = ${P0.t}, i.e. zero). A strangle built from the richest strikes in Δ0.15–0.25 beat the nearest-Δ0.20 rule by ${rs(RD.avg)} a trade (t = ${RD.t}) — not a real difference.`,
  `<b>IV rank (vs the past year) was unstable</b> — its top fifth was great in 2024–26 but near zero or negative in 2021–23 — so the plain IV level is the better gauge here.`,
  `<b>Iron condors turn positive only on high-IV days</b> (both periods) — on normal days they lose, as before.`,
].map(x => `<li>${x}</li>`).join("");
document.getElementById("thrNote").innerHTML = `High IV = ATM IV ≥ <b>${T.iv.toFixed(2)}%</b> · high term = this week ÷ next week ≥ <b>${T.term.toFixed(3)}</b> (both the top 40% of 2021–23 days). Share of days passing — 2021–23: IV ${O.share["2021–23"].iv}%, term ${O.share["2021–23"].term}%; 2024–26: IV ${O.share["2024–26"].iv}%, term ${O.share["2024–26"].term}%.`;
const cell = s => `<td class="${cl(s.avg)}"><b>${rs(s.avg)}</b><div class="sm">n ${s.n} · ${rs(s.total)} · 5% ${rs(s.p5)}</div></td>`;
document.getElementById("tOos").innerHTML = `<tr><th>Strategy</th><th>Period</th><th>All days</th><th>IV high</th><th>IV normal/low</th><th>Term high</th><th>Term low</th></tr>` +
  O.rows.map(r => `<tr><td>${r.tpl.replace(/\|/g, " · ")}</td><td>${r.half}</td>${cell(r.all)}${cell(r.iv_hi)}${cell(r.iv_lo)}${cell(r.t_hi)}${cell(r.t_lo)}</tr>`).join("");
document.getElementById("tGrid").innerHTML = `<tr><th>Strategy</th><th>Period</th><th>IV high · term high</th><th>IV high · term low</th><th>IV low · term high</th><th>IV low · term low</th></tr>` +
  O.grid.map(r => `<tr><td>${r.tpl.replace(/\|/g, " · ")}</td><td>${r.half}</td>${cell(r.hh)}${cell(r.hl)}${cell(r.lh)}${cell(r.ll)}</tr>`).join("");
document.getElementById("corrNote").textContent = `Correlation of the term ratio with the near ATM IV across days: ${O.corr}.`;
document.getElementById("tSide").innerHTML = `<tr><th>Period</th><th>Strategy</th><th>Steep put skew</th><th>Flat skew</th></tr>` +
  O.side.map(r => `<tr><td>${r.half}</td><td>${r.tpl.replace(/\|/g, " · ")}</td>${cell(r.steep)}${cell(r.flat)}</tr>`).join("");
document.getElementById("tPairs").innerHTML = `<tr><th>Group</th><th>Pairs</th><th>Rich strike (pts)</th><th>Cheap strike (pts)</th><th>Difference</th><th>₹ per lot</th><th>t-stat</th><th>IV gap (vol pts)</th><th>Δ gap</th></tr>` +
  D.ss.pairs.map(p => `<tr><td>${p.k}</td><td>${p.n}</td><td>${p.rich}</td><td>${p.cheap}</td><td class="${cl(p.diff)}">${p.diff}</td><td class="${cl(p.rs)}">${rs(p.rs)}</td><td>${p.t}</td><td>${p.dres}</td><td>${p.ddelta}</td></tr>`).join("");
document.getElementById("tRule").innerHTML = `<tr><th>Strike choice</th><th>Days</th><th>Win %</th><th>Premium</th><th>₹ / trade</th><th>Total</th><th>2021–23</th><th>2024–26</th><th>5% worst</th></tr>` +
  D.ss.rule.map(r => `<tr><td>${r.k}</td><td>${r.n}</td><td>${r.win}</td><td>${rs(r.prem)}</td><td class="${cl(r.avg)}"><b>${rs(r.avg)}</b></td><td class="${cl(r.total)}">${rs(r.total)}</td><td class="${cl(r.a)}">${rs(r.a)}</td><td class="${cl(r.b)}">${rs(r.b)}</td><td class="down">${rs(r.p5)}</td></tr>`).join("");
document.getElementById("ruleNote").textContent = `Richest − nearest: ${rs(RD.avg)} a trade, t = ${RD.t}. The two rules picked the same strikes on ${RD.same_pct}% of days.`;
const fb = document.getElementById("fBar");
Object.entries(D.feats).forEach(([k, lab], i) => fb.insertAdjacentHTML("beforeend", `<button data-f="${k}" class="${i ? "" : "on"}">${lab}</button>`));
function q(f) {
  const F = D.filters[f], e = F.edges;
  let h = `<tr><th>Strategy</th><th>All</th>${e.slice(0, -1).map((v, i) => `<th>${i === 0 ? "Lowest" : i === e.length - 2 ? "Highest" : "Fifth " + (i + 1)}<div class="sm">${v} – ${e[i + 1]}</div></th>`).join("")}</tr>`;
  h += F.rows.map(r => `<tr><td>${r.tpl.replace(/\|/g, " · ")}</td><td class="${cl(r.all)}">${rs(r.all)}</td>${r.cells.map(c => `<td class="${cl(c.avg)}"><b>${rs(c.avg)}</b><div class="sm">${rs(c.a)} / ${rs(c.b)}</div></td>`).join("")}</tr>`).join("");
  document.getElementById("tQ").innerHTML = h;
}
q(Object.keys(D.feats)[0]);
fb.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; fb.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); q(b.dataset.f); });
</script></body></html>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    po, pd_ = os.path.join(OUT, "options.parquet"), os.path.join(OUT, "days.parquet")
    if os.path.exists(po) and "--fresh" not in sys.argv:
        opts, days = pd.read_parquet(po), pd.read_parquet(pd_)
    else:
        opts, days = collect()
        opts.to_parquet(po, index=False)
        days.to_parquet(pd_, index=False)
    feats = day_features(days)
    return opts, days, feats, filters(days, feats), strike_selection(opts)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    opts, days, feats, fl, ss = main()
    print(len(opts), "options,", len(days), "expiry-days,", len(feats), "days")
    print(feats.describe().round(3).to_string())
    for f, v in fl.items():
        print("\n==", f, v["edges"])
        for r in v["rows"]:
            print(f"  {r['tpl']:26s} all {r['all']:6d} | " + " | ".join(f"{c['avg']} ({c['a']}/{c['b']})" for c in r["cells"]))
    print(json.dumps(ss, indent=1, default=str))
    print(report(fl, ss, oos(days, feats), len(opts), len(days)))
