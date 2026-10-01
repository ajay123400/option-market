"""Realized-volatility maths on quality-assessed sessions (Stage 2B). Data-quality decisions live in
`spot_quality`; this module only turns valid bars into returns, variances and rolling windows.

DEFINITIONS (all log returns; variance = SUM of squared returns, i.e. second moment about zero)
------------------------------------------------------------------------------------------------
Session grid: 375 one-minute bars, bar i starts at 09:15 + i minutes (IST) and is known at the end of that minute.

Intraday 1-minute returns of a session
    r_0 = ln(close_0 / open_0)            (the 09:15 bar's own open -> close move)
    r_i = ln(close_i / close_{i-1})       i = 1..374, only if bars i and i-1 are BOTH valid
  so a complete session has 375 returns and  sum_i r_i = ln(C_t / O_t)  (open of the first bar to the close of
  the last). A missing/invalid bar k removes r_k and r_{k+1}; nothing is interpolated or forward-filled and no
  return ever spans a gap. Overnight moves are never one-minute returns.
  V_intraday(t) = sum_i r_i^2   -- a variance PER SESSION (open to close). Because it is a SUM over the
  session, no "minutes per day" factor exists to get wrong.
  Coverage-qualified variant (explicitly labelled): V * 375 / n_valid_returns, which assumes the lost minutes
  carry average variance. Stress test on real sessions (validation/missing_bar_stress_test.py): scattered or
  mid-session gaps give a variance error of ~1 % (median) at 98 % coverage and ~3 % at 90 %, but a gap at the
  OPEN biases it down by 10-20 % (the first minutes carry far more variance than average). Therefore a session is
  coverage-qualified only if coverage >= cq_min_session_coverage (default 98 %) AND the first and last
  cq_edge_minutes (default 15) bars are all present.

Overnight return      r_on(t) = ln(O_t / C_{t-1})   O_t = open of the 09:15 bar, C_{t-1} = close of the 15:29 bar of
  the PREVIOUS REGULAR SESSION in the calendar chain. Spans weekends/holidays when the sessions are not adjacent
  calendar days (`overnight_calendar_days`). Valid only if both prices exist and no regular session lies between.
Close-to-close        r_cc(t) = ln(C_t / C_{t-1}) = r_on(t) + ln(C_t / O_t); V_cc(t) = r_cc(t)^2.
Hybrid (overnight + intraday)   V_hyb(t) = r_on(t)^2 + V_intraday(t)  -- the high-frequency analogue of V_cc.

ANNUALIZATION: vol_pct = 100 * sqrt( A * mean_over_sessions(V) ), A = trading sessions per year (default 252,
a convention; the data average ~248-250). Intraday-only RV EXCLUDES overnight movement and is not comparable
with close-to-close RV; use hybrid for an overnight-inclusive high-frequency measure.

ROLLING WINDOWS (N = 5, 10, 20): the window is the last N sessions of the REGULAR-SESSION CHAIN ending at session t.
Special sessions (weekend, Muhurat, short) are not in the chain: they neither count nor break it. A missing
regular session IS in the chain (as an invalid observation). Observation time = 15:30 IST of day t, when the
session has completed; later sessions are never used. method = "strict": all N observations valid, else no value.
method = "coverage_qualified": value over the valid observations if at least ceil(window_fraction * N) of the N
are valid; `n_obs` reports how many. Windows are never silently shortened: `n_obs` and `complete` always shown.
For close-to-close, N sessions means N returns from N+1 consecutive valid closes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .spot_quality import N_REGULAR, SessionData

DEFAULT_ANNUALIZATION_DAYS = 252
WINDOWS = (5, 10, 20)
MEASURES = ("close_to_close", "intraday", "hybrid")
METHODS = ("strict", "coverage_qualified")


@dataclass(frozen=True)
class RVConfig:
    annualization_days: float = DEFAULT_ANNUALIZATION_DAYS
    cq_min_session_coverage: float = 0.98     # coverage-qualified: a session counts if >= 98 % of its 375 returns are valid ...
    cq_min_window_fraction: float = 0.80      # coverage-qualified: need >= 80 % of the N sessions valid
    windows: tuple = WINDOWS
    cq_edge_minutes: int = 15                 # ... AND the first/last 15 bars are all present (see below)


def minute_returns(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """375 returns with NaN where not computable (see module docstring)."""
    r = np.full(N_REGULAR, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[0] = np.log(close[0] / open_[0])
        r[1:] = np.log(close[1:] / close[:-1])
    r[~np.isfinite(r)] = np.nan
    return r


def sampled_returns(close: np.ndarray, step: int) -> np.ndarray:
    """Non-overlapping `step`-minute close-to-close returns (bars 0, step, 2*step, ...), both ends must be valid.
    Used only for the sampling-convention checks."""
    idx = np.arange(step - 1, N_REGULAR, step)           # closes at the end of minutes step-1, 2*step-1, ...
    c = close[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(c[1:] / c[:-1])
    return r[np.isfinite(r)]


def _edges_present(s: SessionData, k: int) -> bool:
    """First and last k one-minute bars all valid (their variance is far from average, so gaps there make scaling unsafe)."""
    if k <= 0:
        return True
    return bool(np.isfinite(s.close[:k]).all() and np.isfinite(s.close[N_REGULAR - k:]).all() and np.isfinite(s.open[0]))


def daily_table(sessions: Sequence[SessionData], cfg: RVConfig = RVConfig()) -> pd.DataFrame:
    rows: List[dict] = []
    prev: Optional[SessionData] = None                    # previous member of the regular chain
    for s in sessions:
        q = s.quality
        row = dict(session_date=q.day.isoformat(), session_type=q.session_type, in_chain=q.in_sequence,
                   expected_trading_date=q.expected_trading_date, calendar_source=q.calendar_source,
                   n_records=q.n_records, n_regular_bars=q.n_regular_window, n_valid_bars=q.n_valid_bars,
                   complete=q.complete, quality_flags="|".join(q.reasons))
        if q.session_type not in ("regular",):
            rows.append(row)
            if q.in_sequence:                              # missing_session stays in the chain as an invalid link
                prev = s
            continue
        r = minute_returns(s.open, s.close)
        ok = np.isfinite(r)
        n = int(ok.sum())
        sumsq = float(np.sum(r[ok] ** 2)) if n else float("nan")
        cov = n / N_REGULAR
        o0, c_last = s.open[0], s.close[-1]
        c_last = c_last if np.isfinite(c_last) else float("nan")
        o0 = o0 if np.isfinite(o0) else float("nan")
        row.update(n_valid_returns=n, coverage=cov, intraday_sumsq=sumsq, intraday_sum_r=float(np.sum(r[ok])) if n else float("nan"),
                   open_price=o0, close_price=c_last,
                   intraday_variance=sumsq if q.complete else float("nan"),
                   intraday_variance_scaled=(sumsq * N_REGULAR / n) if (n and cov >= cfg.cq_min_session_coverage
                                                                          and q.n_duplicate_conflict == 0 and q.n_out_of_order == 0
                                                                          and _edges_present(s, cfg.cq_edge_minutes)) else float("nan"))
        row["intraday_vol_ann_pct"] = 100.0 * math.sqrt(cfg.annualization_days * row["intraday_variance"]) if q.complete else float("nan")
        # ---- chain-dependent quantities ---------------------------------------------------------
        on = cc = float("nan"); on_reason = cc_reason = None
        if prev is None:
            on_reason = cc_reason = "no_previous_session"
        elif prev.quality.session_type != "regular":
            on_reason = cc_reason = "previous_regular_session_missing"
        else:
            pc = prev.close[-1]
            row["prev_session_date"] = prev.quality.day.isoformat()
            row["overnight_calendar_days"] = (q.day - prev.quality.day).days
            if not np.isfinite(pc):
                on_reason = cc_reason = "previous_close_bar_missing"
            else:
                if np.isfinite(o0):
                    on = math.log(o0 / pc)
                else:
                    on_reason = "open_bar_missing"
                if np.isfinite(c_last):
                    cc = math.log(c_last / pc)
                else:
                    cc_reason = "close_bar_missing"
        row.update(overnight_return=on, overnight_variance=on * on if np.isfinite(on) else float("nan"),
                   cc_return=cc, cc_variance=cc * cc if np.isfinite(cc) else float("nan"),
                   overnight_reason=on_reason, cc_reason=cc_reason)
        row["cc_vol_ann_pct_single_day"] = 100.0 * math.sqrt(cfg.annualization_days * row["cc_variance"]) if np.isfinite(cc) else float("nan")
        hyb_ok = np.isfinite(on) and q.complete
        row["hybrid_variance"] = on * on + sumsq if hyb_ok else float("nan")
        row["hybrid_vol_ann_pct_single_day"] = 100.0 * math.sqrt(cfg.annualization_days * row["hybrid_variance"]) if hyb_ok else float("nan")
        cq_ok = np.isfinite(row["intraday_variance_scaled"]) and np.isfinite(on)
        row["hybrid_variance_scaled"] = on * on + row["intraday_variance_scaled"] if cq_ok else float("nan")
        rows.append(row)
        prev = s
    return pd.DataFrame(rows)


def rolling_table(daily: pd.DataFrame, cfg: RVConfig = RVConfig()) -> pd.DataFrame:
    """Rolling windows over the regular-session chain. One row per (regular session, measure, window, method)."""
    chain = daily[daily["in_chain"]].reset_index(drop=True)
    var = {"close_to_close": chain["cc_variance"], "intraday": chain["intraday_variance"], "hybrid": chain["hybrid_variance"]}
    var_cq = {"close_to_close": chain["cc_variance"], "intraday": chain["intraday_variance_scaled"], "hybrid": chain["hybrid_variance_scaled"]}
    out: List[dict] = []
    A = cfg.annualization_days
    for i, row in chain.iterrows():
        if row["session_type"] != "regular":
            continue
        for N in cfg.windows:
            for measure in MEASURES:
                for method in METHODS:
                    base = dict(session_date=row["session_date"], asof_ts_ist=f"{row['session_date']}T15:30:00+05:30",
                                measure=measure, window_sessions=N, method=method)
                    if i - N + 1 < 0:
                        out.append({**base, "n_obs": 0, "n_required": N, "complete": False, "rv_ann_pct": float("nan"),
                                    "mean_variance": float("nan"), "sum_variance": float("nan"), "first_session": None,
                                    "note": "insufficient_history"})
                        continue
                    w = (var if method == "strict" else var_cq)[measure].iloc[i - N + 1:i + 1].to_numpy(dtype=float)
                    valid = np.isfinite(w)
                    n_obs = int(valid.sum())
                    need = N if method == "strict" else math.ceil(cfg.cq_min_window_fraction * N)
                    first = chain["session_date"].iloc[i - N + 1]
                    if n_obs >= need and n_obs > 0:
                        mv = float(w[valid].mean())
                        out.append({**base, "n_obs": n_obs, "n_required": N, "complete": n_obs == N,
                                    "rv_ann_pct": 100.0 * math.sqrt(A * mv), "mean_variance": mv,
                                    "sum_variance": float(w[valid].sum()), "first_session": first,
                                    "note": "" if n_obs == N else f"coverage_qualified: {n_obs}/{N} sessions valid"})
                    else:
                        out.append({**base, "n_obs": n_obs, "n_required": N, "complete": False, "rv_ann_pct": float("nan"),
                                    "mean_variance": float("nan"), "sum_variance": float("nan"), "first_session": first,
                                    "note": f"insufficient_valid_sessions {n_obs}/{N}"})
    return pd.DataFrame(out)
