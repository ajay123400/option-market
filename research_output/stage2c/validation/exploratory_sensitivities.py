"""EXPLORATORY sensitivity checks (labelled exploratory; none of them changes the primary methodology).

1. Partial-session convention of the expiry-aligned target: primary = UNIFORM trading time (session-equivalents = n/375).
   Alternative = intraday-variance PROFILE measured on DEVELOPMENT sessions only (share of a session's 1-minute variance in
   the remaining minutes). Reports the change of the expiry-aligned RV by snapshot time.
2. Time-basis: implied variance IV^2*T (calendar ACT/365) vs realized total variance, see summary_expiry_total_variance.csv;
   here: the ratio of the calendar-time year fraction to the session-equivalent year fraction (252) for the same window.
Usage: python research_output/stage2c/validation/exploratory_sensitivities.py [stage2c_dir]"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from optionsengine.research import realized_vol as rv, session_calendar as sc, spot_quality as sq

OUT = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2c"
df = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet", columns=["ts", "open", "high", "low", "close"])
sessions = sq.assess(df, sc.SessionCalendar(sc.participant_oi_dates("data/participant_oi")))
prof = np.zeros(375); n = 0
for s in sessions:
    q = s.quality
    if q.session_type == "regular" and q.complete and q.day.isoformat() < "2025-01-01":      # development only
        r = rv.minute_returns(s.open, s.close)
        prof += r ** 2; n += 1
prof /= n
print(f"intraday variance profile from {n} development sessions: share of daily variance in first 30 min {prof[:30].sum() / prof.sum():.3f}, last 30 min {prof[-30:].sum() / prof.sum():.3f}")
long = pd.read_csv(f"{OUT}/iv_rv_observations.csv")
e = long[(long.population == "primary") & (long.horizon == "EXP") & long.target_available].copy()
e["slot0"] = e.time.map({"10:00": 45, "13:00": 225, "15:00": 345})
e["share_remaining_profile"] = e.slot0.map(lambda s0: prof[s0 + 1:].sum() / prof.sum())
e["share_remaining_uniform"] = (374 - e.slot0) / 375
e["equiv_profile"] = e.session_equivalents - e.share_remaining_uniform + e.share_remaining_profile
e["rv_profile"] = 100 * np.sqrt(252 * e.rv_total_variance / e.equiv_profile)
e["d"] = e.rv_profile - e.future_rv_pct
t = e.groupby(["measure", "time"]).agg(n=("d", "size"), share_uniform=("share_remaining_uniform", "first"), share_profile=("share_remaining_profile", "first"),
                                       median_change_volpts=("d", "median"), p5=("d", lambda x: x.quantile(.05)), p95=("d", lambda x: x.quantile(.95)))
print(t.round(4).to_string())
print(f"all expiry-aligned rows: median change {e.d.median():+.3f} vol pts, p95 |change| {e.d.abs().quantile(.95):.3f}; median IV-minus-RV changes from "
      f"{(e.iv_pct - e.future_rv_pct).median():+.3f} to {(e.iv_pct - e.rv_profile).median():+.3f} vol pts")
print("\n=== time-basis: calendar year fraction of T vs session-equivalent year fraction ===")
e["cal_over_sess"] = (e.T_days / 365.0) / (e.session_equivalents / 252.0)
print(e.groupby("time").cal_over_sess.describe()[["count", "mean", "50%", "min", "max"]].round(4).to_string())
print("ratio >1 means the calendar-time year fraction exceeds the session-based one (weekends/holidays inside the window); sqrt of it is the approximate vol-point scaling between the two annualizations")
