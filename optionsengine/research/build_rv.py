"""Stage 2B driver: NIFTY 1-minute spot -> data-quality table -> realized-volatility tables.

    python -m optionsengine.research.build_rv --spot data/hist1m/NIFTY50_1m.parquet --out research_output/stage2b

Writes (source data is only read): data_quality.csv, daily_rv.csv, rolling_rv.csv, sensitivity.csv, run_metadata.json.
Independent of the options strike universe (it reads only the spot file and the calendar evidence).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time as _time
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from . import realized_vol as rv
from . import session_calendar as sc
from . import spot_quality as sq

DQ_COLUMNS = ("session_date", "session_type", "expected_trading_date", "calendar_source", "in_chain", "n_records",
              "n_regular_window", "n_outside_window", "n_valid_bars", "n_missing_minutes", "longest_gap_minutes",
              "n_out_of_order", "n_duplicate_exact", "n_duplicate_conflict", "n_invalid_ohlc", "first_bar", "last_bar",
              "complete", "exclusion_reasons")


def quality_table(sessions) -> pd.DataFrame:
    rows = []
    for s in sessions:
        q = s.quality
        rows.append(dict(session_date=q.day.isoformat(), session_type=q.session_type, expected_trading_date=q.expected_trading_date,
                         calendar_source=q.calendar_source, in_chain=q.in_sequence, n_records=q.n_records,
                         n_regular_window=q.n_regular_window, n_outside_window=q.n_outside_window, n_valid_bars=q.n_valid_bars,
                         n_missing_minutes=q.n_missing_minutes if q.session_type == "regular" else None,
                         longest_gap_minutes=q.longest_gap_minutes if q.session_type == "regular" else None,
                         n_out_of_order=q.n_out_of_order, n_duplicate_exact=q.n_duplicate_exact,
                         n_duplicate_conflict=q.n_duplicate_conflict, n_invalid_ohlc=q.n_invalid_ohlc,
                         first_bar=q.first_bar, last_bar=q.last_bar, complete=q.complete,
                         exclusion_reasons="|".join(q.reasons)))
    return pd.DataFrame(rows).reindex(columns=list(DQ_COLUMNS))


def sensitivity_table(sessions, base_cfg: rv.RVConfig) -> pd.DataFrame:
    """How eligibility and level move with the data-quality thresholds (strict rolling is the reference)."""
    daily0 = rv.daily_table(sessions, base_cfg)
    ref = rv.rolling_table(daily0, base_cfg)
    ref_s = ref[ref.method == "strict"].set_index(["session_date", "measure", "window_sessions"])
    rows = []
    for cov, edge, frac in [(c, e, f) for c in (1.0, 0.99, 0.98, 0.95, 0.90, 0.80) for e in (15, 0) for f in (1.0, 0.8)]:
        cfg = rv.RVConfig(base_cfg.annualization_days, cov, frac, base_cfg.windows, edge)
        d = rv.daily_table(sessions, cfg)
        r = rv.rolling_table(d, cfg)
        q = r[r.method == "coverage_qualified"]
        reg = d[d.session_type == "regular"]
        for (measure, N), g in q.groupby(["measure", "window_sessions"]):
            m = g.set_index("session_date")
            both = m.join(ref_s.xs((measure, N), level=("measure", "window_sessions"))[["rv_ann_pct"]], rsuffix="_strict", how="inner")
            both = both[both.rv_ann_pct.notna() & both.rv_ann_pct_strict.notna()]
            rows.append(dict(min_session_coverage=cov, edge_minutes_required=edge, min_window_fraction=frac, measure=measure,
                             window_sessions=N, sessions_valid_intraday=int(np.isfinite(reg["intraday_variance_scaled"]).sum()),
                             windows=len(g), windows_with_value=int(g.rv_ann_pct.notna().sum()),
                             share_with_value=round(float(g.rv_ann_pct.notna().mean()), 4),
                             share_complete=round(float(g.complete.mean()), 4),
                             median_abs_diff_vs_strict_volpts=round(float((both.rv_ann_pct - both.rv_ann_pct_strict).abs().median()), 6) if len(both) else None,
                             max_abs_diff_vs_strict_volpts=round(float((both.rv_ann_pct - both.rv_ann_pct_strict).abs().max()), 6) if len(both) else None))
    return pd.DataFrame(rows)


def build(spot_path: str, out: str, participant_dir: str = "data/participant_oi", annualization_days: float = rv.DEFAULT_ANNUALIZATION_DAYS,
          special_max_bars: int = sq.SPECIAL_MAX_BARS, with_sensitivity: bool = True) -> dict:
    os.makedirs(out, exist_ok=True)
    t0 = _time.time()
    df = pd.read_parquet(spot_path, columns=["ts", "open", "high", "low", "close"])      # raw file order preserved
    cal = sc.SessionCalendar(sc.participant_oi_dates(participant_dir))
    sessions = sq.assess(df, cal, special_max_bars)
    cfg = rv.RVConfig(annualization_days=annualization_days)
    dq = quality_table(sessions)
    daily = rv.daily_table(sessions, cfg)
    rolling = rv.rolling_table(daily, cfg)
    dq.to_csv(os.path.join(out, "data_quality.csv"), index=False)
    daily.to_csv(os.path.join(out, "daily_rv.csv"), index=False)
    rolling.to_csv(os.path.join(out, "rolling_rv.csv"), index=False)
    if with_sensitivity:
        sensitivity_table(sessions, cfg).to_csv(os.path.join(out, "sensitivity.csv"), index=False)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "optionsengine"], stderr=subprocess.DEVNULL, text=True).strip()
        commit = head + ("+dirty" if dirty else "")
    except Exception:
        commit = None
    meta = dict(git_commit=commit, spot=spot_path, spot_bytes=os.path.getsize(spot_path), n_records=int(len(df)),
                n_sessions=len(sessions), annualization_days=annualization_days, special_max_bars=special_max_bars,
                config=asdict(cfg), regular_bars=sq.N_REGULAR, seconds=round(_time.time() - t0, 1),
                started=datetime.fromtimestamp(t0).isoformat(), pandas=pd.__version__)
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spot", default="data/hist1m/NIFTY50_1m.parquet")
    ap.add_argument("--out", default="research_output/stage2b")
    ap.add_argument("--participant-dir", default="data/participant_oi")
    ap.add_argument("--annualization-days", type=float, default=rv.DEFAULT_ANNUALIZATION_DAYS)
    ap.add_argument("--special-max-bars", type=int, default=sq.SPECIAL_MAX_BARS)
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args(argv)
    meta = build(a.spot, a.out, a.participant_dir, a.annualization_days, a.special_max_bars, not a.no_sensitivity)
    print(json.dumps({k: meta[k] for k in ("n_records", "n_sessions", "seconds", "git_commit")}))


if __name__ == "__main__":
    main()
