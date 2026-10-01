"""Real-data look-ahead test for Stage 2B: truncating the spot file right after a session's last bar (nothing from later
sessions exists) must reproduce that session's daily and rolling rows exactly.
Usage: python research_output/stage2b/validation/lookahead_check_rv.py [n_sessions]   (needs the Stage 2B outputs)"""
import random
import sys

import pandas as pd

sys.path.insert(0, ".")
from optionsengine.research import realized_vol as rv, session_calendar as sc, spot_quality as sq

n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
df = pd.read_parquet("data/hist1m/NIFTY50_1m.parquet", columns=["ts", "open", "high", "low", "close"])
cal = sc.SessionCalendar(sc.participant_oi_dates("data/participant_oi"))
full_daily = rv.daily_table(sq.assess(df, cal))
full_roll = rv.rolling_table(full_daily)
dates = list(full_daily[full_daily.session_type == "regular"].session_date)
random.Random(9).shuffle(dates)
bad = 0
for d in sorted(dates[:n]):
    # last record of that IST date (end of the session) -> keep everything up to and including it
    cutoff = int(pd.Timestamp(f"{d} 15:30:00", tz="Asia/Kolkata").timestamp())
    sub = df[df.ts < cutoff]
    dd = rv.daily_table(sq.assess(sub, cal))
    rr = rv.rolling_table(dd)
    a = full_daily[full_daily.session_date == d].reset_index(drop=True)
    b = dd[dd.session_date == d].reset_index(drop=True)
    ra = full_roll[full_roll.session_date == d].reset_index(drop=True)
    rb = rr[rr.session_date == d].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(ra, rb)
    except AssertionError as e:
        bad += 1
        print("MISMATCH", d, str(e)[:200])
print(f"sessions tested: {min(n, len(dates))} (daily rows + all {len(full_roll.columns)}-column rolling rows per session); mismatches: {bad}")
sys.exit(1 if bad else 0)
