"""T9: selection-on-outcome / rejected-observation audit (Stage 2D, Gate 3a). Measurement only.

Universes: U_all = Stage 2A attempted snapshot x expiry groups with T <= 14 calendar days; U_exp = U_all without expiry-day groups (no expiry-aligned target by
construction); O = observed EXP/hybrid rows. Missing from U_exp: layer L1 (snapshot-quality rejections: forward not OK, no spot bar, no option bars, no strict ATM IV) and
layer L2 (kept groups without an EXP target for data reasons).

INFORMATION RULE: the inclusion model uses only quantities known at the snapshot time t -- calendar (year, snapshot time, weekday), contract metadata (DTE), and spot
features from sessions completed BEFORE the snapshot date plus the session's own open (09:15 <= t): ln RV20 (strict hybrid), previous-session range, |overnight gap|.
Features that depend on the option quotes (counts) or on the snapshot bar are excluded: they encode the rejection mechanism. The outcome variable (ln Y1, the F5
hybrid realized volatility after the snapshot session) enters ONLY the outcome-dependence test, never the covariate model used for weighting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import build_iv_rv as biv, iv_rv, realized_vol as rv, session_calendar as sc, spot_quality as sq
from . import clusters as C
from .contrasts import weighted_median

COVARIATE_NAMES = ["year_2022", "year_2023", "year_2024", "year_2025", "year_2026", "time_1300", "time_1500", "wd_tue", "wd_wed", "wd_thu", "wd_fri",
                   "dte_days", "ln_rv20", "prev_range", "gap_abs"]
OUTCOME_COLUMN = "ln_y1"
WEIGHT_CAP = 10.0


# ------------------------------------------------------------------------------ spot features (known at t) and the outcome series
def load_spot_features(spot: str, participant_dir: str = "data/participant_oi") -> pd.DataFrame:
    """One row per regular session date: ln_rv20 (previous completed sessions), prev_range, gap_abs (09:15 open vs previous close) -- all known at t -- and
    ln_y1 = ln of the F5 hybrid session-basis RV AFTER the snapshot session (the outcome; used only in the outcome-dependence test)."""
    df = pd.read_parquet(spot, columns=["ts", "open", "high", "low", "close"])
    cal = sc.SessionCalendar(sc.participant_oi_dates(participant_dir))
    sessions = sq.assess(df, cal)
    daily = rv.daily_table(sessions)
    rolling = rv.rolling_table(daily)
    idx = iv_rv.SessionIndex(sessions, daily)
    feats = biv.per_date_features(idx, rolling)
    feats["day"] = feats.day.astype(str)
    prev_range, gap = {}, {}
    for j, s in enumerate(idx.chain):
        if j == 0:
            continue
        p = idx.chain[j - 1]
        if np.isfinite(p.high).any() and np.isfinite(p.low).any():
            prev_range[s.quality.day.isoformat()] = math.log(np.nanmax(p.high) / np.nanmin(p.low))
        if np.isfinite(s.open[0]) and np.isfinite(p.close[-1]):
            gap[s.quality.day.isoformat()] = abs(math.log(s.open[0] / p.close[-1]))
    feats["prev_range"] = feats.day.map(prev_range)
    feats["gap_abs"] = feats.day.map(gap)
    feats["ln_rv20"] = np.log(feats.recent_rv20_pct.where(feats.recent_rv20_pct > 0))
    feats["ln_y1"] = np.log(feats.f5_hybrid_rv_pct.where(feats.f5_hybrid_rv_pct > 0))
    return feats[["day", "ln_rv20", "prev_range", "gap_abs", "ln_y1"]]


# ------------------------------------------------------------------------------ universe
REJECT_REASONS = ("no_spot_bar_at_snapshot", "no_option_bars_at_snapshot")


def build_groups(smiles: pd.DataFrame, features: pd.DataFrame, observed_ids: Sequence[str], exp_unavailable: Dict[str, str], dte_max: float = 14.0) -> pd.DataFrame:
    """Attempted groups (T <= dte_max) with layer labels and the known-at-t covariates. `observed_ids` = smile_ids of the observed EXP/hybrid rows;
    `exp_unavailable` maps smile_id -> unavailable reason for kept groups without an EXP target."""
    g = smiles[smiles.T_days <= dte_max].copy()
    g["kept_l1"] = (g.forward_status == "ok") & g.atm_iv.notna()

    def reason(r) -> str:
        if r.kept_l1:
            return "kept"
        if r.group_reason in REJECT_REASONS:
            return r.group_reason
        if r.forward_status != "ok":
            return f"forward_{r.forward_status}"
        return "forward_ok_but_no_strict_atm"

    g["l1_reason"] = [reason(r) for r in g.itertuples()]
    obs = set(observed_ids)
    g["observed"] = g.smile_id.isin(obs)
    g["in_u_exp"] = ~g.expiry_day.astype(bool)
    g["layer"] = np.where(g.observed, "observed", np.where(~g.in_u_exp, "expiry_day_by_construction", np.where(~g.kept_l1, "L1_rejected", "L2_no_exp_target")))
    g["l2_reason"] = g.smile_id.map(exp_unavailable)
    g["year"] = g.day.str[:4].astype(int)
    g["wd"] = pd.to_datetime(g.day).dt.weekday
    g = g.merge(features, on="day", how="left")
    return g.reset_index(drop=True)


def design_matrix(g: pd.DataFrame, with_outcome: bool = False) -> Tuple[np.ndarray, List[str]]:
    """Intercept + known-at-t covariates (+ ln_y1 as the LAST column when with_outcome). No option-quote or snapshot-bar column exists here by construction."""
    cols = {"intercept": np.ones(len(g))}
    for y in (2022, 2023, 2024, 2025, 2026):
        cols[f"year_{y}"] = (g.year == y).to_numpy(float)
    cols["time_1300"] = (g.time == "13:00").to_numpy(float)
    cols["time_1500"] = (g.time == "15:00").to_numpy(float)
    for k, n in ((1, "tue"), (2, "wed"), (3, "thu"), (4, "fri")):
        cols[f"wd_{n}"] = (g.wd == k).to_numpy(float)
    cols["dte_days"] = g.T_days.to_numpy(float)
    cols["ln_rv20"] = g.ln_rv20.to_numpy(float)
    cols["prev_range"] = g.prev_range.to_numpy(float)
    cols["gap_abs"] = g.gap_abs.to_numpy(float)
    if with_outcome:
        cols[OUTCOME_COLUMN] = g[OUTCOME_COLUMN].to_numpy(float)
    names = list(cols)
    X = np.column_stack([cols[n] for n in names])
    return X, names


def standardize(X: np.ndarray) -> np.ndarray:
    Z = X.copy()
    mu, sd = X[:, 1:].mean(axis=0), X[:, 1:].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z[:, 1:] = (X[:, 1:] - mu) / sd
    return Z


def complete_cases(g: pd.DataFrame, need_outcome: bool = True) -> pd.DataFrame:
    cols = ["ln_rv20", "prev_range", "gap_abs"] + ([OUTCOME_COLUMN] if need_outcome else [])
    return g[np.isfinite(g[cols].to_numpy(float)).all(axis=1)].reset_index(drop=True)


# ------------------------------------------------------------------------------ ridge logistic regression
def ridge_logit(X: np.ndarray, y: np.ndarray, lam: float = 1.0, max_iter: int = 60, tol: float = 1e-9) -> np.ndarray:
    """Maximises  loglik - 0.5 * lam * ||beta[1:]||^2  (column 0 is the unpenalised intercept) by Newton-Raphson."""
    n, p = X.shape
    beta = np.zeros(p)
    pen = np.full(p, float(lam))
    pen[0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1 - mu)
        grad = X.T @ (y - mu) - pen * beta
        hess = (X * w[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))


def outcome_coefficient(g: pd.DataFrame, missing: np.ndarray, lam: float = 1.0) -> float:
    """Coefficient of the standardised outcome in the ridge logit of `missing` on covariates + ln_y1."""
    X, names = design_matrix(g, with_outcome=True)
    return float(ridge_logit(standardize(X), missing.astype(float), lam)[-1])


def date_shift_permutation_test(g: pd.DataFrame, missing: np.ndarray, series_dates: Sequence[str], series: np.ndarray, draws: int = 2000, min_shift: int = 20,
                                lam: float = 1.0, seed: int = 20260101) -> Dict[str, float]:
    """Null distribution of the outcome coefficient: the date-level outcome series is circularly shifted (offset >= min_shift sessions), which keeps its autocorrelation
    but breaks its alignment with rejections. Two-sided p = (1 + #{|b*| >= |b|}) / (1 + draws)."""
    series = np.asarray(series, float)
    n = len(series)
    if n < 2 * min_shift + 1:
        raise C.ClusterError("series too short for the requested minimum shift")
    pos = {d: i for i, d in enumerate(series_dates)}
    di = np.array([pos[d] for d in g.day])
    X0, names = design_matrix(g, with_outcome=False)
    X0 = standardize(X0)
    m = missing.astype(float)

    def fit(vals: np.ndarray) -> float:
        z = (vals - vals.mean()) / (vals.std() if vals.std() > 0 else 1.0)
        return float(ridge_logit(np.column_stack([X0, z]), m, lam)[-1])

    obs = fit(series[di])
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for i in range(draws):
        k = int(rng.integers(min_shift, n - min_shift + 1))
        null[i] = fit(series[(di + k) % n])
    return dict(coefficient=obs, p_two_sided=float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-15)) / (1 + draws)), null_sd=float(null.std(ddof=1)), draws=draws, min_shift=min_shift)


# ------------------------------------------------------------------------------ balance
def standardized_difference(x_rej: np.ndarray, x_kept: np.ndarray) -> float:
    v = 0.5 * (np.var(x_rej, ddof=1) + np.var(x_kept, ddof=1)) if len(x_rej) > 1 and len(x_kept) > 1 else math.nan
    return float((np.mean(x_rej) - np.mean(x_kept)) / math.sqrt(v)) if v and v > 0 else math.nan


def balance_table(g: pd.DataFrame, flag: np.ndarray, variables: Sequence[str], block: int = 5, reps: int = 1000, seed: int = 20260101) -> pd.DataFrame:
    """Standardised difference (flagged minus others) per variable with an expiry-block bootstrap interval (clusters = expiries)."""
    d = g[np.isfinite(g[list(variables)].to_numpy(float)).all(axis=1)].copy()
    f = np.asarray(flag)[np.isfinite(g[list(variables)].to_numpy(float)).all(axis=1)]
    cs = C.ClusterSet(d.expiry.to_numpy(), {**{v: d[v].to_numpy(float) for v in variables}, "f": f.astype(float)})

    def ev(rows, seq):
        fl = cs.cols["f"][rows] > 0.5
        if fl.sum() < 2 or (~fl).sum() < 2:
            return np.full(len(variables), np.nan)
        return np.array([standardized_difference(cs.cols[v][rows][fl], cs.cols[v][rows][~fl]) for v in variables])

    seq = np.arange(cs.n_clusters)
    pt = ev(cs.rows_for(seq), seq)
    dr = C.bootstrap(cs, ev, "moving_block", block, reps, seed, k=len(variables))
    lo, hi = C.percentile_ci(dr)
    rows = []
    for i, v in enumerate(variables):
        fl = f > 0.5
        rows.append(dict(variable=v, n_flagged=int(fl.sum()), n_other=int((~fl).sum()), mean_flagged=float(d[v].to_numpy(float)[fl].mean()) if fl.any() else math.nan,
                         mean_other=float(d[v].to_numpy(float)[~fl].mean()), std_difference=pt[i], ci_lo=lo[i], ci_hi=hi[i], block=block, reps=reps))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ Manski bounds and break-down shares
def median_with_missing(values: np.ndarray, k: int, fill: float) -> float:
    v = np.asarray(values, float)
    return float(np.median(np.concatenate([v, np.full(k, fill)])))


def manski_median_bounds(values: np.ndarray, k_missing: int) -> Tuple[float, float]:
    """Worst-case bounds on the median of the full population if `k_missing` unobserved members could take ANY value."""
    return median_with_missing(values, k_missing, -np.inf), median_with_missing(values, k_missing, np.inf)


def breakdown_share(values: np.ndarray, null: float, direction: str = "above") -> Dict[str, float]:
    """Smallest number k (share k/(n+k) of the full population) of missing members, all placed at the extreme, that would move the median to the null.
    direction 'above': observed median > null; the missing members are placed at -inf."""
    v = np.asarray(values, float)
    n = len(v)
    med = float(np.median(v))
    if direction == "above" and med <= null or direction == "below" and med >= null:
        return dict(k_needed=0, share_needed=0.0, observed_median=med)
    fill = -np.inf if direction == "above" else np.inf
    lo, hi = 0, 10 * n + 10
    while lo < hi:
        mid = (lo + hi) // 2
        m = median_with_missing(v, mid, fill)
        if (m <= null if direction == "above" else m >= null):
            hi = mid
        else:
            lo = mid + 1
    return dict(k_needed=lo, share_needed=lo / (n + lo), observed_median=med)


def required_missing_mean_log_ratio(mean_ln_r: float, n: int, k: int) -> float:
    """Mean ln(ratio) the k missing groups would need for the full-population geometric-mean ratio to equal 1."""
    return -n * mean_ln_r / k


# ------------------------------------------------------------------------------ inverse-probability weighting
IPW_NAMES = ["S1_calendar", "S1_session", "R1_geometric_mean", "R2_median_ratio"]


class IPWEvaluator:
    """Weighted estimates over observed groups with weights 1/p from a covariate-only ridge logit of `observed` on known-at-t covariates, refit inside every draw."""

    def __init__(self, cs: C.ClusterSet, lam: float = 1.0, cap: float = WEIGHT_CAP):
        self.cs, self.lam, self.cap = cs, lam, cap
        self.X = cs.cols["X"]
        self.y = cs.cols["observed"]
        self.sc, self.ss, self.lr, self.ratio = cs.cols["sc"], cs.cols["ss"], cs.cols["lr"], cs.cols["ratio"]

    def weights(self, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z = standardize(self.X[rows])
        p = predict(Z, ridge_logit(Z, self.y[rows], self.lam))
        w = np.minimum(1.0 / np.clip(p, 1e-6, 1.0), self.cap)
        return w, p

    def evaluate(self, rows: np.ndarray, seq: np.ndarray) -> np.ndarray:
        w, _ = self.weights(rows)
        o = self.y[rows] > 0.5
        out = np.full(2 * len(IPW_NAMES), np.nan)
        if o.sum() < 5:
            return out
        wo = w[o]
        sc, ss, lr, ra = self.sc[rows][o], self.ss[rows][o], self.lr[rows][o], self.ratio[rows][o]
        out[0], out[1] = weighted_median(sc, wo), weighted_median(ss, wo)
        out[2] = math.exp(float(np.sum(wo * lr) / np.sum(wo)))
        out[3] = weighted_median(ra, wo)
        out[4], out[5] = float(np.median(sc)), float(np.median(ss))
        out[6] = math.exp(float(lr.mean()))
        out[7] = float(np.median(ra))
        return out

    def point(self) -> np.ndarray:
        seq = np.arange(self.cs.n_clusters)
        return self.evaluate(self.cs.rows_for(seq), seq)
