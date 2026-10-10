"""gex_research.py -- does dealer gamma exposure (GEX) tell an option seller
anything the IV rule doesn't?

Every trading day at 09:30, for the nearest weekly expiry (0-4 trading days
left), from the 1-min history (data/hist1m, OI included):

  gamma per strike   Black-76 on the put-call-parity forward, IV taken from
                     the OUT-of-the-money option at that strike (its price is
                     the reliable one) and used for both the call and the put
  net GEX            sum over strikes of gamma x (call OI - put OI) x S^2 x 1%
                     = rupees of hedging per 1% NIFTY move, under the usual
                     convention that dealers are long the calls and short the
                     puts customers trade (net > 0 = "positive gamma": dealers
                     sell rallies / buy dips, which should damp moves)
  total gamma        the same with call OI + put OI (how much gamma exists)
  tilt               net / total, in [-1, 1] -- comparable across years even
                     though OI has grown many times since 2021
  flip distance      how far NIFTY sits above (+) or below (-) the level where
                     net GEX would cross zero (sticky-strike IVs), in %

Outcomes after 09:30:
  range ratio        rest-of-day NIFTY high-low / the move ATM IV implied
                     for one day (IV x sqrt(1/252)); < 1 = quieter than priced
  same-day straddle  ATM straddle sold at the 09:31 close - 0.5 pt, bought
                     back at the 15:25 close + 0.5 pt, as % of the premium
  expiry straddle    the same straddle held to expiry (setup A style), in Rs
                     for one lot, after charges

The question for the plan: inside the IV rule's sell days (and on all days),
do positive-GEX days give quieter sessions and better straddle results than
negative-GEX days, consistently in both halves of the sample?

Output: results/gex/report.html (+ days.parquet)
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
import sell_rules as SR
import simulator as S

OUT = os.path.join(paths.BASE_DIR, "results", "gex")
LOT = manual_trades.LOT_SIZE
SLIP = 0.5
ENTRY = "09:30"
F_O_CLOSE_CHANGE = date(2026, 8, 3)
R = g.RISK_FREE_RATE
BAND = 0.08            # strikes within +/-8% of NIFTY
GRID = np.linspace(-0.03, 0.03, 61)


def _dte(day, exp_d):
    n, d = 0, day + timedelta(days=1)
    while d <= exp_d:
        n += mc.is_trading_day(d)
        d += timedelta(days=1)
    return n


def _gamma(F, K, T, iv):
    """Black-76 gamma w.r.t. the forward (vectorised over K / iv)."""
    sq = iv * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * iv * iv * T) / sq
    return np.exp(-R * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (F * sq)


def _snapshot(last, now_px, spot, fwd, T):
    """Per-strike IV (OTM side) + OI -> arrays for the GEX sums."""
    ks, ivs, oic, oip = [], [], [], []
    strikes = sorted({int(k) for (_, k) in last.index})
    for k in strikes:
        if abs(k - spot) > BAND * spot:
            continue
        c = last.loc[("CE", k)] if ("CE", k) in last.index else None
        p = last.loc[("PE", k)] if ("PE", k) in last.index else None
        oc = float(c.oi) if c is not None else 0.0
        op = float(p.oi) if p is not None else 0.0
        if oc <= 0 and op <= 0:
            continue
        iv = None
        for typ, row in ((("CE", c), ("PE", p)) if k >= fwd else (("PE", p), ("CE", c))):
            px = now_px.get((typ, k), np.nan) if row is not None else np.nan
            if not np.isfinite(px):
                px = float(row.close) if row is not None else np.nan
            if np.isfinite(px) and px >= 0.5:
                iv = g.implied_vol_fwd(float(px), fwd, k, T, R, typ == "CE")
                if iv:
                    break
        if not iv:
            continue
        ks.append(k); ivs.append(iv); oic.append(oc); oip.append(op)
    return np.array(ks, float), np.array(ivs, float), np.array(oic, float), np.array(oip, float)


def collect(years=5):
    man = S._manifest()
    exps = sorted(e for e, r in man.items() if isinstance(r, dict) and r.get("status") == "done")
    cutoff = (date.today() - timedelta(days=int(365.25 * years))).isoformat()
    sp = S._load_spot()
    st, sh, sl_, sc = sp["ts"].to_numpy(), sp["high"].to_numpy(), sp["low"].to_numpy(), sp["close"].to_numpy()
    rows = []
    for exp in exps:
        if exp < cutoff:
            continue
        df = S._load_options(exp)
        exp_d = pd.Timestamp(exp).date()
        close_t = "15:40" if exp_d >= F_O_CLOSE_CHANGE else "15:30"
        exp_close_ts = int(pd.Timestamp(f"{exp} {close_t}", tz=S.IST).timestamp())
        j_exp = np.searchsorted(st, int(pd.Timestamp(f"{exp} 23:59", tz=S.IST).timestamp()), side="right") - 1
        settle = float(sc[j_exp])
        for D in sorted(df["day"].unique()):
            if D > exp_d:
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
            now_px = dd[dd.ts == t_ent].set_index(["type", "strike"]).close
            nxt = dd[dd.ts == t_ent + 60].set_index(["type", "strike"]).close
            t_out = int(pd.Timestamp(f"{D} 15:25", tz=S.IST).timestamp())
            out_px = dd[dd.ts == t_out].set_index(["type", "strike"]).close
            e = np.searchsorted(st, t_ent, side="right") - 1
            if e < 0 or st[e] < t_ent - 120:
                continue
            spot = float(sc[e])
            d_end = np.searchsorted(st, int(pd.Timestamp(f"{D} 15:31", tz=S.IST).timestamp()))
            if d_end <= e + 30:
                continue
            day_hi, day_lo, day_cl = float(sh[e + 1:d_end].max()), float(sl_[e + 1:d_end].min()), float(sc[d_end - 1])
            atm = int(round(spot / 50) * 50)
            near = [(k, float(now_px[("CE", k)]), float(now_px[("PE", k)])) for k in range(atm - 150, atm + 151, 50)
                    if ("CE", k) in now_px.index and ("PE", k) in now_px.index]
            fwd = g.synthetic_forward(near) if near else None
            T = (exp_close_ts - t_ent) / (365 * 86400)
            if not fwd or T <= 0:
                continue
            K, IV, OC, OP = _snapshot(last, now_px, spot, fwd, T)
            if len(K) < 8:
                continue
            gam = _gamma(fwd, K, T, IV)
            scale = spot * spot * 0.01
            net = float(np.sum(gam * (OC - OP)) * scale)
            tot = float(np.sum(gam * (OC + OP)) * scale)
            # zero-gamma level (sticky strike): net GEX at hypothetical spots
            nets = np.array([np.sum(_gamma(fwd * (1 + m), K, T, IV) * (OC - OP)) * (spot * (1 + m)) ** 2 * 0.01 for m in GRID])
            sgn = np.sign(nets)
            cross = np.where(np.diff(sgn) != 0)[0]
            if len(cross):
                i = cross[np.argmin(np.abs(GRID[cross]))]
                m0 = GRID[i] - nets[i] * (GRID[i + 1] - GRID[i]) / (nets[i + 1] - nets[i])
                flip = float(spot * (1 + m0))
            else:
                flip = None
            # ATM IV and the straddle
            if ("CE", atm) not in now_px.index or ("PE", atm) not in now_px.index:
                continue
            ce0, pe0 = float(now_px[("CE", atm)]), float(now_px[("PE", atm)])
            iv_c = g.implied_vol_fwd(ce0, fwd, atm, T, R, True)
            iv_p = g.implied_vol_fwd(pe0, fwd, atm, T, R, False)
            atm_iv = np.nanmean([x for x in (iv_c, iv_p) if x]) if (iv_c or iv_p) else np.nan
            ce1, pe1 = nxt.get(("CE", atm), np.nan), nxt.get(("PE", atm), np.nan)
            if not (np.isfinite(ce1) and np.isfinite(pe1)) or not np.isfinite(atm_iv):
                continue
            sold = float(ce1 + pe1) - 2 * SLIP
            day_move = atm_iv * math.sqrt(1 / 252)
            rec = {"expiry": exp, "day": D.isoformat(), "dte": dte, "spot": spot, "fwd": fwd, "atm": atm, "atm_iv": float(atm_iv) * 100,
                   "net_gex_cr": net / 1e7, "tot_gex_cr": tot / 1e7, "tilt": net / tot if tot else np.nan,
                   "flip": flip, "flip_dist": (spot / flip - 1) * 100 if flip else np.nan, "n_strikes": int(len(K)),
                   "range_ratio": (day_hi - day_lo) / spot / day_move, "move_ratio": abs(day_cl - spot) / spot / day_move,
                   "straddle": sold}
            co, po = out_px.get(("CE", atm), np.nan), out_px.get(("PE", atm), np.nan)
            if np.isfinite(co) and np.isfinite(po) and D != exp_d:
                rec["sd_pnl_pct"] = (sold - float(co + po) - 2 * SLIP) / sold * 100
            intr = max(0.0, settle - atm) + max(0.0, atm - settle)
            ch = charges.order_charges("SELL", float(ce1), LOT) + charges.order_charges("SELL", float(pe1), LOT)
            rec["exp_pnl_rs"] = (sold - intr) * LOT - ch
            rows.append(rec)
        print(f"\r{exp} days {len(rows)}", end="", flush=True)
    print()
    d = pd.DataFrame(rows)
    # the IV rule's own 09:30 reading for that day (green = sell day)
    h = SR.history()
    h = h[h.slot == "09:30"][["day", "atm_iv", "thr"]].rename(columns={"atm_iv": "rule_iv", "thr": "rule_thr"})
    h["day"] = h["day"].astype(str)
    d = d.merge(h, on="day", how="left")
    d["green"] = d.rule_iv >= d.rule_thr
    return d


def _t(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std(ddof=1) > 0 else None


def analyse(d):
    d = d.copy()
    d["half"] = np.where(d.day < "2024-01-01", "2021–23", "2024–26")
    d["dteg"] = pd.cut(d.dte, [-1, 0, 2, 4], labels=["expiry day", "1–2 days", "3–4 days"]).astype(str)
    d["q"] = pd.qcut(d.tilt, 5, labels=["Q1 most negative", "Q2", "Q3", "Q4", "Q5 most positive"]).astype(str)
    d["sign"] = np.where(d.net_gex_cr > 0, "positive", "negative")

    def summ(x):
        return {"n": int(len(x)), "range_ratio": round(float(x.range_ratio.mean()), 3), "move_ratio": round(float(x.move_ratio.mean()), 3),
                "sd_pnl_pct": round(float(x.sd_pnl_pct.mean()), 2) if x.sd_pnl_pct.notna().any() else None,
                "sd_win": round(float((x.sd_pnl_pct > 0).mean() * 100)) if x.sd_pnl_pct.notna().any() else None,
                "exp_pnl": round(float(x.exp_pnl_rs.mean())), "exp_win": round(float((x.exp_pnl_rs > 0).mean() * 100)),
                "exp_t": _t(x.exp_pnl_rs), "iv": round(float(x.atm_iv.mean()), 1), "tilt": round(float(x.tilt.mean()), 2)}

    out = {"n_days": int(d.day.nunique()), "first": d.day.min(), "last": d.day.max(),
           "share_positive": round(float((d.net_gex_cr > 0).mean() * 100)),
           "by_q": [{"q": q, **summ(x)} for q, x in sorted(d.groupby("q"))],
           "by_sign": [{"sign": s, **summ(x)} for s, x in d.groupby("sign")],
           "by_sign_half": [{"sign": s, "half": hf, **summ(x)} for (s, hf), x in d.groupby(["sign", "half"])],
           "by_sign_dte": [{"sign": s, "dte": t, **summ(x)} for (s, t), x in d.groupby(["sign", "dteg"])],
           "green_by_sign": [{"sign": s, **summ(x)} for s, x in d[d.green == True].groupby("sign")],
           "green_by_sign_half": [{"sign": s, "half": hf, **summ(x)} for (s, hf), x in d[d.green == True].groupby(["sign", "half"])],
           "flip": [{"bin": b, **summ(x)} for b, x in d.dropna(subset=["flip_dist"]).groupby(
               pd.cut(d.flip_dist.dropna(), [-99, -1, 0, 1, 99], labels=["below flip >1%", "below 0–1%", "above 0–1%", "above >1%"]).astype(str))]}
    # does GEX add anything once IV is known? within IV terciles
    d["ivq"] = pd.qcut(d.atm_iv, 3, labels=["low IV", "mid IV", "high IV"]).astype(str)
    out["iv_x_sign"] = [{"iv": a, "sign": s, **summ(x)} for (a, s), x in d.groupby(["ivq", "sign"])]
    # simple regression: range_ratio ~ tilt + atm_iv + dte (standardised tilt coefficient)
    X = np.column_stack([np.ones(len(d)), (d.tilt - d.tilt.mean()) / d.tilt.std(), d.atm_iv, d.dte])
    y = d.range_ratio.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(y) - X.shape[1])
    out["reg"] = {"tilt_coef": round(float(beta[1]), 4), "tilt_t": round(float(beta[1] / math.sqrt(cov[1, 1])), 2), "n": int(len(y)),
                  "mean_range_ratio": round(float(y.mean()), 3)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pd_ = os.path.join(OUT, "days.parquet")
    if os.path.exists(pd_) and "--fresh" not in sys.argv:
        d = pd.read_parquet(pd_)
    else:
        d = collect()
        d.to_parquet(pd_, index=False)
    return d, analyse(d)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    d, a = main()
    print(json.dumps(a, indent=1, default=str))


def report(d, a):
    g_ = d[d.green == True]
    p, n = g_[g_.net_gex_cr > 0].exp_pnl_rs, g_[g_.net_gex_cr <= 0].exp_pnl_rs
    se = math.sqrt(p.var() / len(p) + n.var() / len(n))
    a = dict(a, green_diff=round(float(p.mean() - n.mean())), green_diff_t=round(float((p.mean() - n.mean()) / se), 2),
             iv_x_sign=[dict(r, ivq=lab) for r, lab in zip(a["iv_x_sign"], ["high IV", "high IV", "low IV", "low IV", "mid IV", "mid IV"])])
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/null", json.dumps(a, ensure_ascii=False, default=str)))
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEX Study</title><style>
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
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
</style></head><body><div class="wrap">
<h1>Does dealer gamma exposure (GEX) help an option seller?</h1>
<p class="note" id="meta" style="margin-bottom:16px"></p>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>What the data says</h2><ul id="verdict"></ul></div>
<div class="card"><h2>IV-rule sell days, split by GEX sign</h2><div class="scroll"><table id="tGreen"></table></div>
<p class="note">Straddle to expiry = ATM straddle sold at the 09:31 close − 0.5 pt, held to the expiry close, 1 lot, after charges (setup A style).</p></div>
<div class="card"><h2>All days, by GEX sign within IV terciles</h2><div class="scroll"><table id="tIv"></table></div>
<p class="note">Range ratio = rest-of-day NIFTY high−low ÷ the one-day move ATM IV implied. Theory says negative-GEX days should be wilder (higher ratio).</p></div>
<div class="card"><h2>All days, by GEX tilt quintile (net ÷ total gamma)</h2><div class="scroll"><table id="tQ"></table></div></div>
<div class="card"><h2>Method &amp; limits</h2><ul>
<li>09:30 snapshot of the nearest weekly expiry (0–4 trading days left), strikes within ±8% of NIFTY. Gamma from Black-76 on the put-call-parity forward, IV from the out-of-the-money option at each strike; OI at 09:30.</li>
<li>Net GEX assumes dealers are long the calls and short the puts that customers trade (the usual convention). Who is really short in NIFTY options is not known from OI; the tilt quintiles and the total-gamma check cover the other readings.</li>
<li>Only the nearest expiry is counted (the next week's contracts are not in the 1-min files on most days). Total gamma mostly tracks days-to-expiry, so it is not used as a separate signal.</li>
</ul></div>
</div><script>
const D = /*DATA*/null;
const rs = v => v == null ? "—" : (v < 0 ? "−" : "") + "₹" + Math.abs(Math.round(v)).toLocaleString("en-IN");
const cl = v => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
document.getElementById("meta").textContent = `${D.n_days.toLocaleString("en-IN")} trading days (${D.first} → ${D.last}) · nearest weekly expiry · 09:30 GEX · positive net GEX on ${D.share_positive}% of days.`;
const gs = s => D.green_by_sign.find(r => r.sign === s), as = s => D.by_sign.find(r => r.sign === s);
const kp = (v, l) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`;
document.getElementById("kpis").innerHTML =
  kp(`${rs(gs("positive").exp_pnl)} vs ${rs(gs("negative").exp_pnl)}`, `IV-rule sell days: straddle to expiry, positive vs negative GEX — difference ${rs(D.green_diff)}, t = ${D.green_diff_t} (not significant)`) +
  kp(`${as("negative").range_ratio} vs ${as("positive").range_ratio}`, "range ratio on negative vs positive GEX days — negative-GEX days were NOT wilder") +
  kp(`t = ${D.reg.tilt_t}`, "GEX tilt → day's range, after controlling for IV and days to expiry (|t| < 2 = nothing)");
document.getElementById("verdict").innerHTML = [
  `<b>GEX adds nothing the IV rule doesn't already give.</b> On the rule's sell days, positive-GEX days made ${rs(gs("positive").exp_pnl)} a straddle vs ${rs(gs("negative").exp_pnl)} on negative-GEX days — a ${rs(D.green_diff)} gap with t = ${D.green_diff_t}, and the two halves disagree (2021–23 the negative days were as good).`,
  `<b>The textbook effect is not there in NIFTY.</b> Negative-GEX days are supposed to move more; here they moved slightly less relative to their IV, in every IV tercile, and the tilt has no significant effect once IV and days-to-expiry are known (t = ${D.reg.tilt_t}).`,
  `<b>Total gamma is just "days to expiry" in disguise:</b> the most-gamma days are expiry days, the least-gamma days are 3–4 days out — already part of the plan's 1–4 day window.`,
  `<b>Practical use:</b> no change to the plan. A GEX number can be shown as information, but it should not decide a trade.`,
].map(x => `<li>${x}</li>`).join("");
const row = (lab, r) => `<tr><td>${lab}</td><td>${r.n}</td><td>${r.iv}%</td><td>${r.range_ratio}</td><td class="${cl(r.exp_pnl)}"><b>${rs(r.exp_pnl)}</b></td><td>${r.exp_win}%</td><td>${r.exp_t ?? "—"}</td><td class="${cl(r.sd_pnl_pct)}">${r.sd_pnl_pct ?? "—"}${r.sd_pnl_pct != null ? "%" : ""}</td></tr>`;
const head = `<tr><th>Group</th><th>Days</th><th>ATM IV</th><th>Range ratio</th><th>Straddle to expiry</th><th>Win</th><th>t</th><th>Same-day straddle</th></tr>`;
document.getElementById("tGreen").innerHTML = head + D.green_by_sign.map(r => row(`${r.sign} GEX · all`, r)).join("") + D.green_by_sign_half.map(r => row(`${r.sign} GEX · ${r.half}`, r)).join("");
document.getElementById("tIv").innerHTML = head + D.iv_x_sign.map(r => row(`${r.ivq} · ${r.sign} GEX`, r)).join("");
document.getElementById("tQ").innerHTML = head + D.by_q.map(r => row(r.q, r)).join("");
</script></body></html>"""
