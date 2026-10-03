"""Real-data look-ahead probe for Stage 2C.

For random observations, every spot bar AFTER the observation point (snapshot bar start + 60 s) is deliberately corrupted
(smooth multiplicative distortion per minute: valid OHLC, different returns). Then:
  * every FEATURE of the observation (recent 20-session RV known at t, spot at t, snapshot-bar close) must be bit-identical,
  * the TARGETS must change (proving the probe has teeth and that targets, not features, are the only consumers of the future).
Uses the repository's own functions (this is a pipeline probe, not an independent calculation).
Usage: python research_output/stage2c/validation/lookahead_check_stage2c.py [stage2c_dir] [n]"""
import random
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from optionsengine.research import iv_rv, realized_vol as rv, session_calendar as sc, spot_quality as sq

OUT = sys.argv[1] if len(sys.argv) > 1 else "research_output/stage2c"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 25
long = pd.read_csv(f"{OUT}/iv_rv_observations.csv")
prim = long[(long.population == "primary") & (long.horizon == "F5") & (long.measure == "hybrid") & long.target_available].drop_duplicates("obs_id")
sample = prim.sample(N, random_state=11)
df0 = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet", columns=["ts", "open", "high", "low", "close"])
cal = sc.SessionCalendar(sc.participant_oi_dates("data/participant_oi"))


def features_and_targets(df, row):
    sessions = sq.assess(df, cal)
    daily = rv.daily_table(sessions)
    rolling = rv.rolling_table(daily)
    idx = iv_rv.SessionIndex(sessions, daily)
    d = date.fromisoformat(row.day)
    r20 = rolling[(rolling.measure == "hybrid") & (rolling.window_sessions == 20) & (rolling.method == "strict")]
    r20 = {date.fromisoformat(a): b for a, b in zip(r20.session_date, r20.rv_ann_pct) if pd.notna(b)}
    recent = iv_rv.recent_rv_before(r20, idx.chain_dates, d)
    slot0 = iv_rv.slot_of(row.time)
    ref_close = float(idx.by_date[d].close[slot0])
    f5 = iv_rv.forward_sessions_target(idx, d, 5, "hybrid").rv_pct
    exp = iv_rv.expiry_aligned_target(idx, d, row.time, date.fromisoformat(row.expiry), "hybrid").rv_pct
    return recent, ref_close, f5, exp


bad_features = unchanged_targets = 0
for row in sample.itertuples():
    cutoff = int(pd.Timestamp(f"{row.day} {row.time}", tz="Asia/Kolkata").timestamp()) + 60
    df = df0.copy()
    fut = df.ts >= cutoff                                           # bars starting at/after the observation point
    k = 1 + 0.004 * np.sin(np.arange(fut.sum()) / 7.0)
    for c in ("open", "high", "low", "close"):
        df.loc[fut, c] = df.loc[fut, c].to_numpy() * k
    a = features_and_targets(df0, row)
    b = features_and_targets(df, row)
    if not (a[0] == b[0] or (pd.isna(a[0]) and pd.isna(b[0]))) or a[1] != b[1]:
        bad_features += 1
        print("FEATURE CHANGED", row.obs_id, a[:2], b[:2])
    if a[2] == b[2] and (pd.isna(a[3]) or a[3] == b[3]):
        unchanged_targets += 1
print(f"observations probed: {len(sample)}; features changed by corrupting the future: {bad_features} (must be 0); "
      f"targets NOT changed by the corruption: {unchanged_targets} (must be 0)")
sys.exit(1 if bad_features or unchanged_targets else 0)
