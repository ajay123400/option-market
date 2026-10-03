"""Real-data look-ahead test: rebuilding every snapshot from ONLY the bars with ts <= t0 must reproduce the
full-file result exactly. Usage: python research_output/stage2a/validation/lookahead_check.py [n_expiries]"""
import random
import sys
from datetime import date, datetime

import pandas as pd

sys.path.insert(0, ".")
from optionsengine import expiry_at_close
from optionsengine.research import loaders
from optionsengine.surface import SurfaceConfig, build_smile

ROOT = "data/hist1m"
spot = loaders.load_spot_closes(f"{ROOT}/NIFTY50_1m.parquet")
files = [f for f in loaders.list_expiry_files(ROOT) if f[0] >= "2021-10-01"]
random.Random(5).shuffle(files)
n_snap = n_bad = 0
for expiry, path in sorted(files[: int(sys.argv[1]) if len(sys.argv) > 1 else 8]):
    raw = pd.read_parquet(path, columns=["symbol", "type", "strike", "ts", "close", "volume"])
    full = loaders.prepare_frame(raw)
    exp_dt = expiry_at_close(date.fromisoformat(expiry))
    days = loaders.trading_days(full)
    for day in days[-12:]:
        for hhmm in ("10:00", "13:00", "15:00"):
            t0 = loaders.snapshot_ts(day, hhmm)
            if t0 not in spot:
                continue
            q_full = loaders.quotes_at(full, t0)
            if not q_full:
                continue
            trunc = loaders.prepare_frame(raw[raw["ts"] <= t0])           # the future does not exist
            q_trunc = loaders.quotes_at(trunc, t0)
            obs = datetime.fromtimestamp(t0 + 60, tz=loaders.IST)
            T = max((exp_dt - obs).total_seconds(), 0.0) / (365 * 86400)
            a = build_smile(q_full, spot[t0], T, 0.065, SurfaceConfig())
            b = build_smile(q_trunc, spot[t0], T, 0.065, SurfaceConfig())
            n_snap += 1
            if q_full != q_trunc or a != b:
                n_bad += 1
                print("MISMATCH", expiry, day, hhmm)
print(f"snapshots compared: {n_snap}; mismatches: {n_bad}")
sys.exit(1 if n_bad else 0)
