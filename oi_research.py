"""oi_research.py -- do open-interest walls matter?

Every trading day, at 09:30, for the nearest weekly expiry (0-4 trading days
left), from the 1-min history (data/hist1m, OI included):

  call wall   the 100-pt strike above NIFTY with the most call OI
  put wall    the 100-pt strike below NIFTY with the most put OI
  dOI walls   the same, by OI added since the previous day's close (fresh writing)

Test 1 -- respect: is a wall broken less often, or rejected more often after
being touched, than any other 100-pt strike at the same distance from NIFTY
on the same days? Distance is measured in ATM straddles (the market's own
expected move). Horizons: rest of the day, and to expiry.

Test 5 -- strike placement: selling the wall strike (or the strike just
beyond it) vs selling a non-wall strike of the same delta, same day; and a
"wall strangle" vs a delta-matched strangle. Only options that traded in the
09:30 minute; sold at the 09:31 close - 0.5 pt; held to expiry.

Output: results/oi_walls/report.html (+ levels.parquet, legs.parquet)
"""
import json
import math
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import charges
import greeks as g
import manual_trades
import market_calendar as mc
import paths
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "oi_walls")
LOT = manual_trades.LOT_SIZE
SLIP = 0.5
ENTRY = "09:30"
F_O_CLOSE_CHANGE = date(2026, 8, 3)
R = g.RISK_FREE_RATE
BINS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]


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
    st, sh, sl_, sc = sp["ts"].to_numpy(), sp["high"].to_numpy(), sp["low"].to_numpy(), sp["close"].to_numpy()
    levels, legs = [], []
    for exp in exps:
        if exp < cutoff:
            continue
        df = S._load_options(exp)
        exp_d = pd.Timestamp(exp).date()
        close_t = "15:40" if exp_d >= F_O_CLOSE_CHANGE else "15:30"
        exp_close_ts = int(pd.Timestamp(f"{exp} {close_t}", tz=S.IST).timestamp())
        j_exp = np.searchsorted(st, int(pd.Timestamp(f"{exp} 23:59", tz=S.IST).timestamp()), side="right") - 1
        settle = float(sc[j_exp])
        fdays = sorted(df["day"].unique())
        for i_d, D in enumerate(fdays):
            if D > exp_d or i_d == 0:
                continue
            dte = _dte(D, exp_d)
            if dte > 4:
                continue
            t_ent = int(pd.Timestamp(f"{D} {ENTRY}", tz=S.IST).timestamp())
            dd = df[df["day"] == D]
            upto = dd[dd.ts <= t_ent]
            if upto.empty:
                continue
            last = upto.sort_values("ts").groupby(["type", "strike"]).last()
            prev = df[df["day"] == fdays[i_d - 1]].sort_values("ts").groupby(["type", "strike"]).oi.last()
            now_px = dd[dd.ts == t_ent].set_index(["type", "strike"]).close
            nxt_px = dd[dd.ts == t_ent + 60].set_index(["type", "strike"]).close
            e = np.searchsorted(st, t_ent, side="right") - 1
            spot = float(sc[e])
            d_end = np.searchsorted(st, int(pd.Timestamp(f"{D} 15:31", tz=S.IST).timestamp()))
            day_hi, day_lo, day_cl = float(sh[e + 1:d_end].max()), float(sl_[e + 1:d_end].min()), float(sc[d_end - 1])
            x_end = j_exp + 1
            exp_hi, exp_lo = float(sh[e + 1:x_end].max()), float(sl_[e + 1:x_end].min())
            atm = int(round(spot / 50) * 50)
            if ("CE", atm) not in last.index or ("PE", atm) not in last.index:
                continue
            straddle = float(last.loc[("CE", atm)].close + last.loc[("PE", atm)].close)
            T = (exp_close_ts - t_ent) / (365 * 86400)
            near = [(k, float(now_px[("CE", k)]), float(now_px[("PE", k)])) for k in range(atm - 150, atm + 151, 50)
                    if ("CE", k) in now_px.index and ("PE", k) in now_px.index]
            fwd = g.synthetic_forward(near) if near else None
            # ---- OI table for 100-pt strikes on the OTM side of each type
            rows = []
            for (typ, k), r in last.iterrows():
                k = int(k)
                if k % 100 or abs(k - spot) > 3.2 * straddle:
                    continue
                if (typ == "CE" and k <= spot) or (typ == "PE" and k >= spot):
                    continue
                rows.append({"type": typ, "strike": k, "oi": float(r.oi), "doi": float(r.oi - prev.get((typ, k), np.nan))})
            q = pd.DataFrame(rows)
            if q.empty or q.oi.le(0).all():
                continue
            walls = {}
            for typ in ("CE", "PE"):
                s = q[q.type == typ]
                if s.empty:
                    continue
                walls[(typ, "oi")] = int(s.loc[s.oi.idxmax(), "strike"])
                s2 = s.dropna(subset=["doi"])
                if not s2.empty and s2.doi.max() > 0:
                    walls[(typ, "doi")] = int(s2.loc[s2.doi.idxmax(), "strike"])
            base = {"expiry": exp, "day": D.isoformat(), "dte": dte, "spot": spot, "straddle": straddle}
            for _, r in q.iterrows():
                up = r.type == "CE"
                k = r.strike
                dist = (k - spot) / straddle if up else (spot - k) / straddle
                levels.append({**base, "type": r.type, "strike": k, "dist": dist,
                               "oi_wall": walls.get((r.type, "oi")) == k, "doi_wall": walls.get((r.type, "doi")) == k,
                               "oi_rank": int((q[q.type == r.type].oi > r.oi).sum()) + 1,
                               "touch_d": (day_hi >= k) if up else (day_lo <= k),
                               "break_d": (day_cl > k) if up else (day_cl < k),
                               "touch_x": (exp_hi >= k) if up else (exp_lo <= k),
                               "break_x": (settle > k) if up else (settle < k)})
            # ---- short legs for the placement test (traded in the 09:30 minute, sold at 09:31)
            if not fwd or T <= 0 or dte < 1:
                continue
            for (typ, k), p in now_px.items():
                k = int(k)
                call = typ == "CE"
                if (call and k <= fwd) or (not call and k >= fwd) or abs(k - fwd) > 3.2 * straddle or p <= 0.5:
                    continue
                nx = nxt_px.get((typ, k), np.nan)
                if not np.isfinite(nx):
                    continue
                iv = g.implied_vol_fwd(float(p), fwd, k, T, R, call)
                if not iv:
                    continue
                d1 = (math.log(fwd / k) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
                delta = abs(math.exp(-R * T) * (_ncdf(d1) if call else _ncdf(d1) - 1))
                intr = max(0.0, settle - k) if call else max(0.0, k - settle)
                w = walls.get((typ, "oi"))
                legs.append({**base, "type": typ, "strike": k, "delta": delta, "sold": float(nx) - SLIP,
                             "pnl_pts": float(nx) - SLIP - intr, "wall": w,
                             "vs_wall": None if w is None else ((k - w) if call else (w - k))})
        print(f"\r{exp} levels {len(levels)} legs {len(legs)}", end="", flush=True)
    print()
    return pd.DataFrame(levels), pd.DataFrame(legs)


def respect(lv):
    """Wall vs non-wall 100-pt strikes, matched by distance bin (and day)."""
    lv = lv.copy()
    lv["bin"] = pd.cut(lv.dist, BINS, labels=[f"{a}–{b}" for a, b in zip(BINS, BINS[1:])])
    lv = lv.dropna(subset=["bin"])
    lv["half"] = np.where(lv.day < "2024", "2021–23", "2024–26")
    out = {}
    for kind in ("oi_wall", "doi_wall"):
        res = []
        for side, x in (("Call side (resistance)", lv[lv.type == "CE"]), ("Put side (support)", lv[lv.type == "PE"]), ("Both", lv)):
            for hz, tc, bc in (("Rest of day", "touch_d", "break_d"), ("To expiry", "touch_x", "break_x")):
                rows, wsum, osum, wn = [], 0.0, 0.0, 0
                for b, y in x.groupby("bin", observed=True):
                    w, o = y[y[kind]], y[~y[kind]]
                    if len(w) < 15 or len(o) < 15:
                        continue
                    wt, ot = w[w[tc]], o[o[tc]]
                    rows.append({"bin": str(b), "n_w": int(len(w)), "n_o": int(len(o)),
                                 "touch_w": round(float(w[tc].mean() * 100), 1), "touch_o": round(float(o[tc].mean() * 100), 1),
                                 "brk_w": round(float(w[bc].mean() * 100), 1), "brk_o": round(float(o[bc].mean() * 100), 1),
                                 "hold_w": round(float((~wt[bc]).mean() * 100), 1) if len(wt) >= 10 else None,
                                 "hold_o": round(float((~ot[bc]).mean() * 100), 1) if len(ot) >= 10 else None,
                                 "n_wt": int(len(wt))})
                    # distance-weighted expected break rate for the walls if they behaved like other strikes
                    wsum += w[bc].sum()
                    osum += o[bc].mean() * len(w)
                    wn += len(w)
                halves = {}
                for h, z in x.groupby("half"):
                    a_, b_ = 0.0, 0.0
                    for _, y in z.groupby("bin", observed=True):
                        w, o = y[y[kind]], y[~y[kind]]
                        if len(w) and len(o):
                            a_ += w[bc].sum()
                            b_ += o[bc].mean() * len(w)
                    halves[h] = {"actual": round(float(a_)), "expected": round(float(b_))}
                res.append({"side": side, "hz": hz, "rows": rows, "walls": int(wn), "actual": round(float(wsum)),
                            "expected": round(float(osum)), "halves": halves})
        out[kind] = res
    return out


def placement(lg):
    lg = lg.copy()
    lg["band"] = (lg.delta // 0.04) * 0.04
    lg["half"] = np.where(lg.day < "2024", "2021–23", "2024–26")

    def tstat(v):
        return float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))) if len(v) > 2 else None
    res = []
    for lab, which in (("At the wall", 0), ("One strike (100 pts) beyond the wall", 100), ("One strike inside the wall", -100)):
        diffs = []
        for (_, _, typ), x in lg.groupby(["expiry", "day", "type"]):
            w = x[x.vs_wall == which]
            if w.empty:
                continue
            w = w.iloc[0]
            o = x[(x.vs_wall != which) & (x.band == w.band) & (x.strike % 100 == w.strike % 100)]
            if o.empty:
                o = x[(x.vs_wall != which)]
                o = o.loc[[(o.delta - w.delta).abs().idxmin()]] if not o.empty else o
            if o.empty:
                continue
            m = o.loc[(o.delta - w.delta).abs().idxmin()]
            diffs.append({"day": w.day, "type": typ, "diff": w.pnl_pts - m.pnl_pts, "w": w.pnl_pts, "o": m.pnl_pts,
                          "dd": w.delta - m.delta, "delta": w.delta})
        d = pd.DataFrame(diffs)
        for grp, z in (("All", d), ("Calls", d[d.type == "CE"]), ("Puts", d[d.type == "PE"]),
                       ("2021–23", d[d.day < "2024"]), ("2024–26", d[d.day >= "2024"])):
            res.append({"k": lab, "grp": grp, "n": int(len(z)), "w": round(float(z.w.mean()), 2), "o": round(float(z.o.mean()), 2),
                        "diff": round(float(z["diff"].mean()), 2), "rs": round(float(z["diff"].mean() * LOT)),
                        "t": round(tstat(z["diff"]), 2), "delta": round(float(z.delta.mean()), 3), "dd": round(float(z.dd.mean()), 3)})
    # wall strangle vs delta-matched strangle (both legs), and the plain delta-0.20 strangle
    st = []
    for (exp, day), x in lg.groupby(["expiry", "day"]):
        pick = {}
        for typ in ("CE", "PE"):
            s = x[x.type == typ]
            w = s[s.vs_wall == 0]
            if s.empty or w.empty:
                break
            w = w.iloc[0]
            o = s[s.vs_wall != 0]
            if o.empty:
                break
            m = o.loc[(o.delta - w.delta).abs().idxmin()]
            d20 = s.loc[(s.delta - 0.20).abs().idxmin()]
            pick[typ] = (w, m, d20)
        if len(pick) < 2:
            continue
        rs = {}
        for i, nm in enumerate(("wall", "matched", "d20")):
            legs_ = [pick["CE"][i], pick["PE"][i]]
            pts = sum(l.pnl_pts for l in legs_)
            ch = sum(charges.order_charges("SELL", l.sold, LOT) for l in legs_)
            rs[nm] = pts * LOT - ch
        st.append({"day": day, **rs, "dw": pick["CE"][0].delta + pick["PE"][0].delta})
    s = pd.DataFrame(st)
    strangle = []
    for nm, lab in (("wall", "Sell both OI walls"), ("matched", "Same deltas, not the walls"), ("d20", "Plain Δ0.20 strangle")):
        v = s[nm]
        strangle.append({"k": lab, "n": int(len(v)), "avg": round(float(v.mean())), "total": round(float(v.sum())),
                         "win": round(float((v > 0).mean() * 100), 1), "p5": round(float(v.quantile(0.05))),
                         "a": round(float(v[s.day < "2024"].mean())), "b": round(float(v[s.day >= "2024"].mean()))})
    dwm = s.wall - s.matched
    return {"legs": res, "strangle": strangle, "wall_vs_matched": {"avg": round(float(dwm.mean())), "t": round(tstat(dwm), 2)},
            "wall_delta": round(float(s.dw.mean() / 2), 3)}


def placement_fair(lg):
    """Wall leg vs the P&L a strike of EXACTLY its delta would have made that
    day on that side -- a straight line through the 4 non-wall strikes
    nearest in delta, interpolated only (no extrapolation). Removes the
    delta mismatch of a plain nearest-strike comparison."""
    def tstat(v):
        return float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))) if len(v) > 2 else None
    res, strangle = [], {}
    for which, lab in ((0, "At the wall"), (100, "One strike (100 pts) beyond the wall"), (-100, "One strike inside the wall")):
        out = []
        for (exp, day, typ), x in lg.groupby(["expiry", "day", "type"]):
            w, o = x[x.vs_wall == which], x[x.vs_wall != which]
            if w.empty or len(o) < 4:
                continue
            w = w.iloc[0]
            o = o.iloc[(o.delta - w.delta).abs().argsort()[:4]]
            if o.delta.max() < w.delta or o.delta.min() > w.delta:
                continue
            exp_pnl = float(np.polyval(np.polyfit(o.delta, o.pnl_pts, 1), w.delta))
            out.append({"day": day, "expiry": exp, "type": typ, "d": w.pnl_pts - exp_pnl, "act": w.pnl_pts, "exp": exp_pnl, "delta": w.delta})
        d = pd.DataFrame(out)
        if which == 0:
            both = d.groupby(["expiry", "day"]).filter(lambda z: len(z) == 2).groupby(["expiry", "day"]).d.sum() * LOT
            strangle = {"n": int(len(both)), "avg": round(float(both.mean())), "t": round(tstat(both), 2)}
        for grp, z in (("All", d), ("Calls", d[d.type == "CE"]), ("Puts", d[d.type == "PE"]),
                       ("2021–23", d[d.day < "2024"]), ("2024–26", d[d.day >= "2024"])):
            res.append({"k": lab, "grp": grp, "n": int(len(z)), "act": round(float(z.act.mean()), 2), "exp": round(float(z.exp.mean()), 2),
                        "diff": round(float(z.d.mean()), 2), "rs": round(float(z.d.mean() * LOT)), "t": round(tstat(z.d), 2),
                        "delta": round(float(z.delta.mean()), 3)})
    return {"legs": res, "strangle": strangle}


def report(lv, rp, pf):
    data = {"respect": rp, "place": pf, "n_days": int(lv.day.nunique()), "n_levels": int(len(lv)), "lot": LOT}
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, default=str)))
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OI Walls Study</title><style>
:root{--bg:#f3f4f8;--card:#fff;--border:#e6e8ef;--text:#14161c;--muted:#6f7482;--up:#0e9f6e;--down:#e0434a;--grp:#eef0f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111318;--card:#1a1d24;--border:#2a2e38;--text:#e8e9ee;--muted:#9097a6;--grp:#23262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1300px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}
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
button.on{background:#4f5fe0;color:#fff;border-color:#4f5fe0}
</style></head><body><div class="wrap">
<h1>Do OI walls act as support / resistance?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>What the data says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>Test 1 — walls vs other strikes at the same distance</h2>
<div class="bar"><span class="note">Wall:</span><button data-k="oi_wall" class="on">Biggest OI</button><button data-k="doi_wall">Biggest OI added today (ΔOI)</button></div>
<div class="scroll"><table id="tSum"></table></div>
<p class="note">"Expected" = how many walls would have been broken if they behaved like the other 100-pt strikes at the same distance (in ATM straddles) — fewer actual breaks than expected would mean the wall holds. Broken = NIFTY closes beyond it (rest of day: at 15:30; to expiry: expiry close).</p>
<h2 style="margin-top:16px">By distance from NIFTY (in ATM straddles)</h2><div class="scroll"><table id="tBins"></table></div>
<p class="note">Touch = NIFTY reached the level. Held after touch = of the touches, the share that closed back on the original side.</p></div>
<div class="card"><h2>Test 5 — selling the wall strike vs a strike of the same delta</h2>
<p class="note">Days 1–4 before expiry, 09:30. Only options traded in the 09:30 minute; sold at the 09:31 close − 0.5 pt; held to expiry. "Same-delta P&amp;L" = a straight line through the 4 non-wall strikes nearest in delta that day, read at the wall's exact delta.</p>
<div class="scroll"><table id="tPlace"></table></div><p class="note" id="strNote"></p></div>
<div class="card"><h2>Method &amp; limits</h2><ul>
<li>Walls from 09:30 OI of the nearest weekly expiry, 100-pt strikes on the out-of-the-money side, within 3.2 ATM straddles of NIFTY. ΔOI = OI now − OI at the previous day's close.</li>
<li>OI does not say who is short: a call wall can be writers or buyers. The test only asks whether the level behaves differently.</li>
<li>Levels of the same day are not independent, and a big wall and its neighbours share the same day's move.</li>
</ul></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
document.getElementById("meta").textContent = `${D.n_days.toLocaleString("en-IN")} trading days (Sep 2021 → Sep 2026) · ${D.n_levels.toLocaleString("en-IN")} strike-days · nearest weekly expiry, 09:30 OI · 1-min NIFTY path.`;
const get = (k, side, hz) => D.respect[k].find(r => r.side === side && r.hz === hz);
const pct = (a, e) => e ? Math.round((a / e - 1) * 100) : 0;
const bd = get("oi_wall", "Both", "Rest of day"), bx = get("oi_wall", "Both", "To expiry"), cx = get("oi_wall", "Call side (resistance)", "To expiry"), px = get("oi_wall", "Put side (support)", "To expiry");
const P = D.place.legs.find(r => r.k === "At the wall" && r.grp === "All"), S = D.place.strangle;
const bin1 = bd.rows.find(r => r.bin === "0.5–1.0");
const kp = (v, l) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`;
document.getElementById("kpis").innerHTML =
  kp(`${bx.actual} vs ${bx.expected}`, "walls broken by expiry vs expected for ordinary strikes at the same distance — no difference") +
  kp(`${bd.actual} vs ${bd.expected}`, "walls broken the same day vs expected — walls are broken MORE, not less") +
  kp(`${bin1.touch_w}% vs ${bin1.touch_o}%`, "touched the same day, 0.5–1 straddle away: walls vs other strikes") +
  kp(`${rs(P.rs)} (t ${P.t})`, "extra per lot from selling the wall strike instead of a same-delta strike — zero");
document.getElementById("verdict").innerHTML = [
  `<b>OI walls do not hold NIFTY back.</b> By expiry, ${bx.actual} of ${bx.walls} walls were broken — ordinary strikes at the same distance would have given ${bx.expected}. Same in both periods (2021–23 ${bx.halves["2021–23"].actual} vs ${bx.halves["2021–23"].expected}, 2024–26 ${bx.halves["2024–26"].actual} vs ${bx.halves["2024–26"].expected}).`,
  `<b>Call walls vs put walls to expiry:</b> call walls were broken ${Math.abs(pct(cx.actual, cx.expected))}% ${cx.actual < cx.expected ? "less" : "more"} than expected (${cx.actual} vs ${cx.expected}, similar in both periods), put walls ${Math.abs(pct(px.actual, px.expected))}% ${px.actual < px.expected ? "less" : "more"} (${px.actual} vs ${px.expected}). A mild call-side effect at most, and none on the put side.`,
  `<b>Within the day, walls act more like magnets than barriers.</b> A wall 0.5–1 straddle away was touched ${bin1.touch_w}% of days vs ${bin1.touch_o}% for other strikes at that distance, and broken more often (${bd.actual} vs ${bd.expected} for the day). After a touch, walls held ${bin1.hold_w}% of the time vs ${bin1.hold_o}% for other strikes — no extra rejection. A likely reason: the biggest OI sits at strikes where NIFTY traded recently, and the market often drifts back there after a gap. This is an observation, not a tested trade.`,
  `<b>Fresh OI (ΔOI walls) is no better</b> — even more "magnetic" intraday, and no different by expiry.`,
  `<b>Selling at (or just beyond) the wall does not earn more</b> than selling any strike of the same delta: ${rs(P.rs)} per lot per leg (t = ${P.t}), and a "wall strangle" ${rs(S.avg)} a trade vs its same-delta equivalent (t = ${S.t}). A plain comparison with the nearest strike made walls look ~₹40 better, but that was only because the wall strikes had a higher delta.`,
  `<b>Practical use:</b> OI walls are fine as a picture of where positions sit, but as a support/resistance level or a reason to pick a strike, they add nothing over distance/delta. Keep strikes by delta or by the expected move (straddle).`,
].map(x => `<li>${x}</li>`).join("");
function show(k) {
  const res = D.respect[k];
  document.getElementById("tSum").innerHTML = `<tr><th>Side</th><th>Horizon</th><th>Walls</th><th>Broken</th><th>Expected</th><th>Difference</th><th>2021–23 (act / exp)</th><th>2024–26 (act / exp)</th></tr>` +
    res.map(r => `<tr><td>${r.side}</td><td>${r.hz}</td><td>${r.walls}</td><td><b>${r.actual}</b></td><td>${r.expected}</td><td class="${r.actual < r.expected ? "up" : "down"}">${pct(r.actual, r.expected) > 0 ? "+" : ""}${pct(r.actual, r.expected)}%</td><td>${r.halves["2021–23"].actual} / ${r.halves["2021–23"].expected}</td><td>${r.halves["2024–26"].actual} / ${r.halves["2024–26"].expected}</td></tr>`).join("");
  let h = `<tr><th>Side · horizon</th><th>Distance</th><th>Walls</th><th>Touched: wall</th><th>other</th><th>Broken: wall</th><th>other</th><th>Held after touch: wall</th><th>other</th></tr>`;
  res.filter(r => r.side !== "Both").forEach(r => {
    h += `<tr class="grp"><td colspan="9">${r.side} · ${r.hz}</td></tr>`;
    r.rows.forEach(b => { h += `<tr><td></td><td>${b.bin}</td><td>${b.n_w}</td><td>${b.touch_w}%</td><td>${b.touch_o}%</td><td>${b.brk_w}%</td><td>${b.brk_o}%</td><td>${b.hold_w ?? "—"}${b.hold_w != null ? "%" : ""}</td><td>${b.hold_o ?? "—"}${b.hold_o != null ? "%" : ""}</td></tr>`; });
  });
  document.getElementById("tBins").innerHTML = h;
}
show("oi_wall");
document.querySelectorAll("[data-k]").forEach(b => b.addEventListener("click", () => { document.querySelectorAll("[data-k]").forEach(x => x.classList.toggle("on", x === b)); show(b.dataset.k); }));
let ph = `<tr><th>Strike sold</th><th>Group</th><th>n</th><th>Avg Δ</th><th>Actual P&amp;L (pts)</th><th>Same-delta P&amp;L (pts)</th><th>Difference</th><th>₹ per lot</th><th>t-stat</th></tr>`;
let last = "";
D.place.legs.forEach(r => { if (r.k !== last) { ph += `<tr class="grp"><td colspan="9">${r.k}</td></tr>`; last = r.k; }
  ph += `<tr><td></td><td>${r.grp}</td><td>${r.n}</td><td>${r.delta}</td><td>${r.act}</td><td>${r.exp}</td><td class="${cl(r.diff)}">${r.diff}</td><td class="${cl(r.rs)}">${rs(r.rs)}</td><td>${r.t}</td></tr>`; });
document.getElementById("tPlace").innerHTML = ph;
document.getElementById("strNote").textContent = `Wall strangle (sell the call wall + the put wall) vs the same-delta strangle: ${rs(S.avg)} a trade over ${S.n} days, t = ${S.t}. |t| below ~2 = no real difference.`;
</script></body></html>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    pl, pg = os.path.join(OUT, "levels.parquet"), os.path.join(OUT, "legs.parquet")
    if os.path.exists(pl) and "--fresh" not in sys.argv:
        lv, lg = pd.read_parquet(pl), pd.read_parquet(pg)
    else:
        lv, lg = collect()
        lv.to_parquet(pl, index=False)
        lg.to_parquet(pg, index=False)
    return lv, lg, respect(lv), placement(lg)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    lv, lg, rp, pm = main()
    print(len(lv), "levels", lv.day.nunique(), "days;", len(lg), "legs")
    for kind, res in rp.items():
        print("\n###", kind)
        for r in res:
            print(f"{r['side']:24s} {r['hz']:12s} walls {r['walls']:4d} broke {r['actual']:4d} expected {r['expected']:4d}  halves {r['halves']}")
            for b in r["rows"]:
                print("    ", b)
    pf = placement_fair(lg)
    print(json.dumps(pf["strangle"]), [(r["k"][:12], r["grp"], r["rs"], r["t"]) for r in pf["legs"]])
    print(report(lv, rp, pf))
