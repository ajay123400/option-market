"""Independent check of the annualization-alignment audit (no optionsengine imports).

Recomputes, from the RAW 1-minute spot file and plain datetime arithmetic:
  * T (ACT/365 seconds from observation = bar START + 60 s to the expiry instant 15:30 IST, 15:40 from 2026-08-03),
  * the realized total variance V of the window (pure Python: hybrid = overnight + 1-minute log returns),
  * window span in calendar days, RV_session, RV_calendar, kappa, implied total variance IV^2*T, ratios and spreads
and compares them with research_output/stage2c/annualization_audit/aligned_observations.csv.
Representative cases are named (normal expiry, Friday/weekend, holiday gap, expiry-day snapshot, 10:00, 13:00, 15:00);
in addition every EXP row is re-derived for T/span and a random sample of fixed-horizon and EXP rows for V.
Usage: python research_output/stage2c/validation/independent_annualization_check.py [stage2c_dir]
"""
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2c"
IST = timezone(timedelta(hours=5, minutes=30))
al = pd.read_csv(f"{OUT}/annualization_audit/aligned_observations.csv")

# ------------------------------------------------------------------ raw bars
df = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet")
ist = df.ts.to_numpy(dtype="int64") + 19800
dn, slot = ist // 86400, (ist % 86400) // 60 - 555
o_, h_, l_, c_ = (df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
ok = np.isfinite(o_) & np.isfinite(h_) & np.isfinite(l_) & np.isfinite(c_) & (o_ > 0) & (h_ >= l_) & (o_ >= l_) & (o_ <= h_) & (c_ >= l_) & (c_ <= h_)
win = (slot >= 0) & (slot < 375) & ok
bars = {}
for d, s, o, c in zip(dn[win], slot[win], o_[win], c_[win]):
    bars.setdefault(int(d), {})[int(s)] = (float(o), float(c))
chain = [d for d in sorted(bars) if (d + 3) % 7 < 5 and len(bars[d]) >= 150]
pos = {d: i for i, d in enumerate(chain)}
complete = {d: len(bars[d]) == 375 for d in chain}
to_dn = lambda iso: (date.fromisoformat(iso) - date(1970, 1, 1)).days
to_date = lambda n: date(1970, 1, 1) + timedelta(days=n)


def sess(d, prev_close, overnight):
    b = bars[d]
    ss = math.log(b[0][1] / b[0][0]) ** 2
    for i in range(1, 375):
        ss += math.log(b[i][1] / b[i - 1][1]) ** 2
    if overnight:
        ss += math.log(b[0][0] / prev_close) ** 2
    return ss


def V_fixed(day, k, measure):
    p = pos[day]
    if p + k >= len(chain):
        return None
    tot = 0.0
    for j in range(p + 1, p + 1 + k):
        d, pd_ = chain[j], chain[j - 1]
        if 374 not in bars[pd_]:
            return None
        if measure == "close_to_close":
            if 374 not in bars[d]:
                return None
            tot += math.log(bars[d][374][1] / bars[pd_][374][1]) ** 2
        else:
            if not complete[d]:
                return None
            tot += sess(d, bars[pd_][374][1], measure == "hybrid")
    return tot, k, to_date(chain[p]), to_date(chain[p + k])


def V_exp(day, hhmm, expiry, measure="hybrid"):
    p, q = pos.get(day), pos.get(expiry)
    if p is None or q is None or q <= p:
        return None
    s0 = int(hhmm[:2]) * 60 + int(hhmm[3:]) - 555
    b = bars[day]
    if any(i not in b for i in range(s0, 375)):
        return None
    tot = sum(math.log(b[i][1] / b[i - 1][1]) ** 2 for i in range(s0 + 1, 375))
    n_part = 374 - s0
    for j in range(p + 1, q + 1):
        d = chain[j]
        if not complete[d] or 374 not in bars[chain[j - 1]]:
            return None
        tot += sess(d, bars[chain[j - 1]][374][1], measure == "hybrid")
    return tot, n_part / 375 + (q - p)


def act365(a, b):
    return (b - a).total_seconds() / (365 * 86400)


def expiry_instant(e):
    d = date.fromisoformat(e)
    hh, mm = (15, 40) if d >= date(2026, 8, 3) else (15, 30)          # exchange close used for the option's T
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=IST)


def check_row(r, label=""):
    """Independent recomputation for one aligned row -> dict of max abs deviations."""
    iv = float(r.iv_pct)
    h, m = r.horizon, r.measure
    d0 = date.fromisoformat(r.day)
    hh, mm = (int(x) for x in r.time.split(":"))
    obs = datetime(d0.year, d0.month, d0.day, hh, mm, tzinfo=IST) + timedelta(seconds=60)
    T = act365(obs, expiry_instant(r.expiry))
    if h == "EXP":
        res = V_exp(to_dn(r.day), r.time, to_dn(r.expiry), m)
        if res is None:
            return None
        V, se = res
        end = datetime(*(date.fromisoformat(r.expiry).timetuple()[:3]), 15, 30, tzinfo=IST)
        span = (end - obs).total_seconds() / 86400
        implied = (iv / 100) ** 2 * T
    else:
        res = V_fixed(to_dn(r.day), int(h[1:]), m)
        if res is None:
            return None
        V, se, ds, de = res
        span = ((datetime(de.year, de.month, de.day, 15, 30, tzinfo=IST)) - datetime(ds.year, ds.month, ds.day, 15, 30, tzinfo=IST)).total_seconds() / 86400
        implied = (iv / 100) ** 2 * span / 365
    rv_s = 100 * math.sqrt(252 * V / se)
    rv_c = 100 * math.sqrt(365 * V / span)
    kap = (span / 365) / (se / 252)
    out = dict(T_years=abs(T - r.T_years) if h == "EXP" else 0.0, V=abs(V - r.rv_total_variance), span=abs(span - r.span_days),
               rv_session=abs(rv_s - r.rv_session_pct), rv_calendar=abs(rv_c - r.rv_calendar_pct), kappa=abs(kap - r.kappa),
               implied_tv=abs(implied - r.implied_total_variance), tv_diff=abs((implied - V) - r.total_variance_diff),
               spread_session=abs((iv - rv_s) - r.spread_session), spread_calendar=abs((iv - rv_c) - r.spread_calendar),
               effect=abs((rv_c - rv_s) - r.annualization_effect))
    if V > 0:
        out["ratio"] = abs(implied / V - r.total_variance_ratio)
    out["_vals"] = (T, V, span, se, rv_s, rv_c, kap, implied)
    return out


def pick(mask, n=1, seed=1):
    pool = al[mask & al.interval_complete]
    return pool.sample(min(n, len(pool)), random_state=seed)


# ------------------------------------------------------------------ named representative cases
ex = (al.horizon == "EXP") & (al.measure == "hybrid")
f1 = (al.horizon == "F1") & (al.measure == "hybrid")
f5 = (al.horizon == "F5") & (al.measure == "hybrid")
wd = pd.to_datetime(al.day).dt.weekday
# holiday gap: window whose elapsed calendar days exceed what weekends explain (F1 over a weekday, or F5 > 7 days)
holiday = (f5 & (al.span_days > 7.5)) | (f1 & (al.span_days > 3.5)) | (f1 & (wd < 4) & (al.span_days > 1.5))
cases = [
    ("normal expiry (EXP, 7-14d, Monday 10:00)", ex & (al.dte_bucket == "7-14d") & (wd == 0) & (al.time == "10:00")),
    ("normal expiry (EXP, 3-7d, 13:00)", ex & (al.dte_bucket == "3-7d") & (al.time == "13:00")),
    ("Friday / weekend (EXP, snapshot Friday 15:00)", ex & (wd == 4) & (al.time == "15:00")),
    ("Friday / weekend (F1 hybrid, snapshot Friday 13:00 -> 3 calendar days)", f1 & (wd == 4) & (al.time == "13:00") & (al.span_days == 3)),
    ("holiday gap (F1 hybrid, weekday snapshot, > 1 calendar day)", f1 & (wd < 4) & (al.span_days > 1.5)),
    ("holiday gap (F5 hybrid, 8+ calendar days)", f5 & (al.span_days >= 8)),
    ("expiry-day snapshot 10:00 (F1 hybrid)", f1 & al.expiry_day & (al.time == "10:00")),
    ("expiry-day snapshot 13:00 (F5 hybrid)", f5 & al.expiry_day & (al.time == "13:00")),
    ("expiry-day snapshot 15:00 (F1 close_to_close)", (al.horizon == "F1") & (al.measure == "close_to_close") & al.expiry_day & (al.time == "15:00")),
    ("10:00 snapshot (EXP)", ex & (al.time == "10:00") & (al.dte_bucket == "1-3d")),
    ("13:00 snapshot (EXP)", ex & (al.time == "13:00") & (al.dte_bucket == "1-3d")),
    ("15:00 snapshot (EXP)", ex & (al.time == "15:00") & (al.dte_bucket == "1-3d")),
    ("post-2026-08-03 expiry (T runs to 15:40, spot to 15:30)", ex & al.expiry_after_close_change & (al.time == "13:00")),
]
tol = 1e-9
bad = 0
print(f"{'case':75s} {'obs_id':32s} {'h':4s} {'T_years':>10s} {'V':>10s} {'span_d':>8s} {'sess_eq':>8s} {'RVsess':>8s} {'RVcal':>8s} {'kappa':>6s} {'IV^2T':>9s}  max|dev|")
for label, mask in cases:
    for _, r in pick(mask, 2, 3).iterrows():
        res = check_row(r)
        if res is None:
            print(f"{label:75s} {r.obs_id}: independent target unavailable (DISAGREEMENT)")
            bad += 1
            continue
        vals = res.pop("_vals")
        dev = max(res.values())
        bad += dev > tol
        T, V, span, se, rs, rc, kap, imp = vals
        print(f"{label:75s} {r.obs_id:32s} {r.horizon:4s} {T:10.6f} {V:10.3e} {span:8.4f} {se:8.4f} {rs:8.3f} {rc:8.3f} {kap:6.3f} {imp:9.2e}  {dev:.1e}")

# ------------------------------------------------------------------ bulk checks
rng = random.Random(11)
sample = al[al.interval_complete].sample(1200, random_state=5)
mx, n = {}, 0
for _, r in sample.iterrows():
    res = check_row(r)
    if res is None:
        bad += 1
        print("independent target unavailable:", r.obs_id, r.horizon, r.measure)
        continue
    res.pop("_vals")
    n += 1
    for k, v in res.items():
        mx[k] = max(mx.get(k, 0.0), v)
print(f"\nrandom sample: {n} rows recomputed independently from raw bars; max abs deviation by quantity")
for k, v in mx.items():
    print(f"  {k:16s} {v:.3e}")
    bad += v > tol

# all EXP rows: T and window span from plain datetime arithmetic (no spot needed)
e = al[ex]
Tv, Sv = [], []
for r in e.itertuples():
    d0 = date.fromisoformat(r.day)
    hh, mm = (int(x) for x in r.time.split(":"))
    obs = datetime(d0.year, d0.month, d0.day, hh, mm, tzinfo=IST) + timedelta(seconds=60)
    Tv.append(act365(obs, expiry_instant(r.expiry)) - r.T_years)
    ed = date.fromisoformat(r.expiry)
    Sv.append((datetime(ed.year, ed.month, ed.day, 15, 30, tzinfo=IST) - obs).total_seconds() / 86400 - r.span_days)
print(f"all {len(e)} EXP/hybrid rows: max |T - T_independent| = {max(map(abs, Tv)):.2e} years; max |span - span_independent| = {max(map(abs, Sv)):.2e} days")
bad += max(map(abs, Tv)) > tol or max(map(abs, Sv)) > tol

# algebraic identities over the whole table (every row)
c = al[al.interval_complete]
ident1 = (c.rv_session_pct - c.rv_calendar_pct * np.sqrt(c.kappa)).abs().max()
ident2 = (c.annualization_effect - (c.spread_session - c.spread_calendar)).abs().max()
ex_c = al[ex]
ident3 = ((ex_c.iv_pct / ex_c.rv_calendar_pct) ** 2 - ex_c.total_variance_ratio).abs()[ex_c.total_variance_ratio_valid & (ex_c.t_minus_span_minutes.abs() < 1e-6)].max()
ident4 = (ex_c.spread_calendar - (ex_c.iv_pct - 100 * np.sqrt(ex_c.rv_total_variance / (ex_c.span_days / 365)))).abs().max()
print(f"identity RV_sess = RV_cal*sqrt(kappa): max abs error {ident1:.2e}; effect = spread_sess - spread_cal: {ident2:.2e}; (IV/RV_cal)^2 = W_imp/V (pre-change expiries): {ident3:.2e}; spread_cal recomputed: {ident4:.2e}")
bad += any(x > 1e-8 for x in (ident1, ident2, ident3, ident4))
print("\nRESULT:", "ALL AGREE (tolerance 1e-9)" if not bad else f"{int(bad)} DISAGREEMENTS")
