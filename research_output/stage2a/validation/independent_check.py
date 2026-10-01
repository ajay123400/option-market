"""Independent re-calculation of Stage 2A outputs (needs scipy; NOT part of the package or its tests).

Nothing here imports `optionsengine`: IV is solved with scipy.brentq on a separately written Black-76 pricer,
time to expiry / forward / ATM interpolation / 25-delta metrics are re-derived from the CSV outputs and (for
the forward) from the raw parquet. Usage:
    python research_output/stage2a/validation/independent_check.py research_output/stage2a/full
"""
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

IST = timezone(timedelta(hours=5, minutes=30))
R = 0.065


def expiry_close(exp):  # independent restatement of the F&O close schedule
    return datetime.fromisoformat(exp + ("T15:40:00" if exp >= "2026-08-03" else "T15:30:00")).replace(tzinfo=IST)


def T_years(day, hhmm, exp):
    obs = datetime.fromisoformat(f"{day}T{hhmm}:00").replace(tzinfo=IST) + timedelta(seconds=60)
    return max((expiry_close(exp) - obs).total_seconds(), 0.0) / (365 * 86400)


def b76(F, K, T, sigma, call):
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T)); d2 = d1 - sigma * math.sqrt(T)
    df = math.exp(-R * T)
    return df * (F * norm.cdf(d1) - K * norm.cdf(d2)) if call else df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def iv_brent(price, F, K, T, call):
    f = lambda s: b76(F, K, T, s, call) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-14, rtol=1e-14, maxiter=500)
    except ValueError:
        return None


def fdelta(F, K, T, s, call):
    d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return norm.cdf(d1) if call else norm.cdf(-d1)


def interp_delta(wing, target):
    """wing sorted by strike; points dicts with strike, iv, delta, used. Adjacent pair bracketing `target`."""
    for a, b in zip(wing, wing[1:]):
        if a["used"] and b["used"] and b["strike"] - a["strike"] <= 200:
            lo, hi = sorted((a["delta"], b["delta"]))
            if lo <= target <= hi and a["delta"] != b["delta"]:
                return a["iv"] + (target - a["delta"]) / (b["delta"] - a["delta"]) * (b["iv"] - a["iv"])
    return None


def main(d):
    s = pd.read_csv(f"{d}/smiles.csv"); p = pd.read_csv(f"{d}/points.csv")
    rng = random.Random(11)
    print("=== A. point-level IV / delta / interval vs scipy brentq (separately coded Black-76) ===")
    conv = p[(p.iv.notna()) & (p.forward_status == "ok")]
    samp = pd.concat([conv[conv.used].sample(600, random_state=1), conv[conv.resolution_limited].sample(min(300, int(conv.resolution_limited.sum())), random_state=1)])
    rows = []
    for r in samp.itertuples():
        T = T_years(r.day, r.time, r.expiry); call = r.kind == "call"
        iv = iv_brent(r.price, r.forward, r.strike, T, call)
        lo = iv_brent(max(r.price - 0.025, 1e-12), r.forward, r.strike, T, call) if r.price - 0.025 > 0 else 0.0
        hi = iv_brent(r.price + 0.025, r.forward, r.strike, T, call)
        rows.append((abs(T * 365 - r.T_days), abs(iv - r.iv) * 100 if iv else np.nan, abs(fdelta(r.forward, r.strike, T, r.iv, call) - r.delta_fwd),
                     abs((lo or 0) - r.iv_low) * 100 if r.iv_low is not None and not pd.isna(r.iv_low) else np.nan,
                     abs((hi if hi else np.inf) - r.iv_high) * 100 if hi and not pd.isna(r.iv_high) and np.isfinite(r.iv_high) else np.nan))
    a = pd.DataFrame(rows, columns=["T_days_diff", "iv_diff_volpts", "delta_diff", "iv_low_diff_volpts", "iv_high_diff_volpts"])
    print(f"points checked: {len(samp)} ({int(samp.used.sum())} used, {int(samp.resolution_limited.sum())} resolution-limited)")
    print(a.describe().loc[["count", "50%", "max"]].round(10).to_string())
    print("\n=== B. smile-level ATM / RR25 / BF25 re-derived from points.csv (numpy) ===")
    prim = s[(s.forward_status == "ok") & s.atm_iv.notna()]
    ids = rng.sample(list(prim.smile_id), 400)
    g = p[p.smile_id.isin(ids)].groupby("smile_id")
    out = []
    for sid in ids:
        sm = s[s.smile_id == sid].iloc[0]; pts = g.get_group(sid).sort_values("strike")
        F = sm.forward_used; x = np.log(pts.strike / F); lo = pts[x < 0].iloc[-1]; hi = pts[x >= 0].iloc[0]
        atm = np.interp(0.0, [np.log(lo.strike / F), np.log(hi.strike / F)], [lo.iv, hi.iv])
        rec = dict(atm_diff=abs(atm - sm.atm_iv) * 100, rr_diff=np.nan, bf_diff=np.nan, rr_both=False)
        T = T_years(sm.day, sm.time, sm.expiry)
        if T * 365 > 1:
            pl = [dict(strike=q.strike, iv=q.iv, delta=fdelta(F, q.strike, T, q.iv, q.kind == "call") if q.iv == q.iv else np.nan, used=bool(q.used)) for q in pts.itertuples()]
            cw = [q for q in pl if q["strike"] >= F]; pw = [q for q in pl if q["strike"] < F]
            c25, p25 = interp_delta(cw, 0.25), interp_delta(pw, 0.25)
            if c25 is not None and p25 is not None:
                rec.update(rr_diff=abs((c25 - p25) - sm.rr25) * 100 if sm.rr25 == sm.rr25 else np.nan,
                           bf_diff=abs(0.5 * (c25 + p25) - atm - sm.bf25) * 100 if sm.bf25 == sm.bf25 else np.nan, rr_both=True)
            else:
                rec["rr_missing_mismatch"] = sm.rr25 == sm.rr25      # library produced a value we cannot reproduce
        out.append(rec)
    b = pd.DataFrame(out)
    print(f"smiles checked: {len(b)}; with independently reproducible RR/BF: {int(b.rr_both.sum())}")
    print(b[["atm_diff", "rr_diff", "bf_diff"]].describe().loc[["count", "50%", "max"]].round(10).to_string())
    if "rr_missing_mismatch" in b: print("library RR/BF present but independent bracketing found none:", int(b.rr_missing_mismatch.fillna(False).sum()))
    print("\n=== C. forward re-derived from RAW parquet (no outlier removal) for 40 OK smiles ===")
    diffs = []
    for sid in rng.sample(list(s[s.forward_status == "ok"].smile_id), 40):
        sm = s[s.smile_id == sid].iloc[0]
        df = pd.read_parquet(f"data/hist1m/options/{sm.expiry}.parquet", columns=["symbol", "type", "strike", "ts", "close", "volume"]).sort_values(["symbol", "ts"])
        t0 = int(datetime.fromisoformat(f"{sm.day}T{sm.time}:00").replace(tzinfo=IST).timestamp())
        df = df[df.ts <= t0]
        last = df[df.volume > 0].groupby("symbol").ts.max()
        at = df[df.ts == t0].set_index("symbol")
        age = ((t0 + 60 - last.reindex(at.index)) / 60.0)
        fresh = at[(age <= 5.0)]
        ce = fresh[fresh.type == "CE"].set_index("strike").close.astype(float).round(2); pe = fresh[fresh.type == "PE"].set_index("strike").close.astype(float).round(2)
        ks = sorted(set(ce.index) & set(pe.index), key=lambda k: (abs(k - sm.spot), k))[:6]
        T = T_years(sm.day, sm.time, sm.expiry)
        Fi = [k + math.exp(R * T) * (ce[k] - pe[k]) for k in ks]
        diffs.append((float(np.median(Fi)) - sm.forward_used, max(Fi) - min(Fi)))
    dd = pd.DataFrame(diffs, columns=["median6_minus_library_forward", "raw_range_pts"])
    print(dd.describe().loc[["count", "50%", "max", "min"]].round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2a/full")
