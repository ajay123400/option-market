"""T10: bid/ask and quote-noise sensitivity (Stage 2D, Gate 3a). Measurement only.

THERE IS NO BID/ASK IN THIS DATASET. Every scenario here is an ASSUMPTION or a BOUND about how far last-trade prices could sit from a fair mid price; none of it is observed
spread data, and nothing here estimates a spread. Scenarios perturb only the two prices that bracket the forward (the points interpolated into the ATM IV), re-solve their implied
volatility with an independent Black-76 solver (r = 6.5%, the Stage 2A assumption), re-interpolate linearly in ln(K/F) to 0 (the Stage 2A rule) and recompute the spread and the
total-variance ratio from the STORED realized variance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from math import erf
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

R_RATE = 0.065
TICK = 0.05
SQ2 = math.sqrt(2.0)
SQ2PI = math.sqrt(2.0 * math.pi)


def _cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.array([erf(v / SQ2) for v in np.ravel(x)]).reshape(np.shape(x)))


def _pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / SQ2PI


def b76_price(F, K, T, r, sig, call) -> np.ndarray:
    F, K, T, sig = (np.asarray(a, float) for a in (F, K, T, sig))
    call = np.asarray(call, bool)
    sd = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    disc = np.exp(-r * T)
    return np.where(call, disc * (F * _cdf(d1) - K * _cdf(d2)), disc * (K * _cdf(-d2) - F * _cdf(-d1)))


def b76_vega(F, K, T, r, sig) -> np.ndarray:
    """Rupees per one VOL POINT (1% absolute IV)."""
    F, K, T, sig = (np.asarray(a, float) for a in (F, K, T, sig))
    sd = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return np.exp(-r * T) * F * _pdf(d1) * np.sqrt(T) / 100.0


def implied_vol(price, F, K, T, r, call, init=None, max_newton: int = 25) -> np.ndarray:
    """Vectorised Black-76 implied volatility (decimal). NaN where the price is outside the no-arbitrage bounds or the solver fails. Newton with a bisection fallback."""
    price, F, K, T = (np.asarray(a, float) for a in (price, F, K, T))
    call = np.asarray(call, bool)
    disc = np.exp(-r * T)
    intrinsic = np.where(call, np.maximum(disc * (F - K), 0.0), np.maximum(disc * (K - F), 0.0))
    upper = np.where(call, disc * F, disc * K)
    ok = (price > intrinsic + 1e-12) & (price < upper - 1e-12) & (price > 0) & (T > 0)
    sig = np.full(price.shape, 0.2) if init is None else np.where(np.isfinite(init) & (np.asarray(init) > 0), init, 0.2)
    sig = np.clip(sig, 1e-4, 5.0)
    done = ~ok
    for _ in range(max_newton):
        act = ~done
        if not act.any():
            break
        p = b76_price(F[act], K[act], T[act], r, sig[act], call[act])
        v = b76_vega(F[act], K[act], T[act], r, sig[act]) * 100.0
        diff = p - price[act]
        conv = np.abs(diff) < 1e-12 * np.maximum(price[act], 1.0)
        step = np.where(v > 1e-14, diff / np.where(v > 1e-14, v, 1.0), 0.0)
        new = np.clip(sig[act] - step, 1e-4, 5.0)
        idx = np.flatnonzero(act)
        sig[idx] = np.where(conv, sig[idx], new)
        done[idx[conv]] = True
    left = ~done
    if left.any():                                                          # bisection fallback for the few stragglers
        lo, hi = np.full(left.sum(), 1e-4), np.full(left.sum(), 5.0)
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            low = b76_price(F[left], K[left], T[left], r, mid, call[left]) < price[left]
            lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
        sig[left] = 0.5 * (lo + hi)
    out = np.where(ok, sig, np.nan)
    return out


# ------------------------------------------------------------------------------ bracket table
@dataclass
class Brackets:
    """For each observed row: the two points bracketing the forward (Stage 2A rule: largest strike with ln(K/F) < 0 and smallest with >= 0)."""
    frame: pd.DataFrame


def bracket_table(points: pd.DataFrame, smile_ids: Sequence[str]) -> pd.DataFrame:
    p = points[points.smile_id.isin(set(smile_ids))]
    lo = p[p.log_moneyness < 0].sort_values(["smile_id", "strike"]).groupby("smile_id").tail(1).set_index("smile_id")
    hi = p[p.log_moneyness >= 0].sort_values(["smile_id", "strike"]).groupby("smile_id").head(1).set_index("smile_id")
    cols = ["strike", "kind", "price", "age_min", "log_moneyness", "iv", "used"]
    b = lo[cols + ["forward", "T_days"]].add_suffix("_lo").join(hi[cols].add_suffix("_hi"), how="inner")
    b = b.rename(columns={"forward_lo": "forward", "T_days_lo": "T_days"})
    return b.reset_index()


def interpolate_atm(x_lo, x_hi, iv_lo, iv_hi) -> np.ndarray:
    w = (0.0 - np.asarray(x_lo, float)) / (np.asarray(x_hi, float) - np.asarray(x_lo, float))
    return np.asarray(iv_lo, float) + w * (np.asarray(iv_hi, float) - np.asarray(iv_lo, float))


class RowData:
    """Observed rows (aligned EXP/hybrid) joined to their bracketing points; scenario evaluation is a pure function of perturbed prices."""

    def __init__(self, aligned_exp: pd.DataFrame, brackets: pd.DataFrame, r: float = R_RATE):
        b = brackets.set_index("smile_id")
        a = aligned_exp.set_index("obs_id")
        common = a.index.intersection(b.index)
        self.base = a.loc[common].reset_index().rename(columns={"index": "obs_id"})
        bb = b.loc[common]
        self.r = r
        self.F = bb.forward.to_numpy(float)
        self.T = bb.T_days.to_numpy(float) / 365.0
        self.K_lo, self.K_hi = bb.strike_lo.to_numpy(float), bb.strike_hi.to_numpy(float)
        self.call_lo, self.call_hi = (bb.kind_lo == "call").to_numpy(), (bb.kind_hi == "call").to_numpy()
        self.p_lo, self.p_hi = bb.price_lo.to_numpy(float), bb.price_hi.to_numpy(float)
        self.x_lo, self.x_hi = bb.log_moneyness_lo.to_numpy(float), bb.log_moneyness_hi.to_numpy(float)
        self.iv_lo_stored, self.iv_hi_stored = bb.iv_lo.to_numpy(float), bb.iv_hi.to_numpy(float)
        self.age_lo, self.age_hi = bb.age_min_lo.to_numpy(float), bb.age_min_hi.to_numpy(float)
        self.used = (bb.used_lo & bb.used_hi).to_numpy(bool)
        self.n = len(self.base)
        self.w = (0.0 - self.x_lo) / (self.x_hi - self.x_lo)

    def atm_stored_points(self) -> np.ndarray:
        return interpolate_atm(self.x_lo, self.x_hi, self.iv_lo_stored, self.iv_hi_stored)

    def solve(self, p_lo: np.ndarray, p_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ivl = implied_vol(p_lo, self.F, self.K_lo, self.T, self.r, self.call_lo, init=self.iv_lo_stored)
        ivh = implied_vol(p_hi, self.F, self.K_hi, self.T, self.r, self.call_hi, init=self.iv_hi_stored)
        return ivl, ivh

    def atm_iv(self, p_lo: np.ndarray, p_hi: np.ndarray) -> np.ndarray:
        ivl, ivh = self.solve(p_lo, p_hi)
        return interpolate_atm(self.x_lo, self.x_hi, ivl, ivh)

    def vega_atm(self) -> np.ndarray:
        """Rupees per vol point of the interpolated ATM IV (weights of the two bracketing strikes)."""
        vl = b76_vega(self.F, self.K_lo, self.T, self.r, self.iv_lo_stored)
        vh = b76_vega(self.F, self.K_hi, self.T, self.r, self.iv_hi_stored)
        return (1 - self.w) * vl + self.w * vh

    def price_atm(self) -> np.ndarray:
        return (1 - self.w) * self.p_lo + self.w * self.p_hi

    def frame(self, iv_atm: np.ndarray) -> pd.DataFrame:
        """Baseline rows with the IV-side columns recomputed from `iv_atm` (decimal); rows whose IV could not be solved are dropped."""
        d = self.base.copy()
        ok = np.isfinite(iv_atm)
        d["iv_pct"] = 100.0 * iv_atm
        d["implied_total_variance"] = iv_atm ** 2 * self.T
        d["spread_session"] = d.iv_pct - d.rv_session_pct
        d["spread_calendar"] = d.iv_pct - d.rv_calendar_pct
        return d[ok].reset_index(drop=True)


# ------------------------------------------------------------------------------ scenarios (ASSUMPTIONS)
def floor_tick(p: np.ndarray) -> np.ndarray:
    return np.maximum(p, TICK)


def scenario_prices(rd: RowData, kind: str, param: float, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """kind: 'abs' (+param rupees on both), 'rel' (x(1+param)), 'opposite' (calls +param, puts -param, rupees), 'noise_abs' (N(0, param) rupees), 'noise_rel' (x(1+N(0, param)))."""
    pl, ph = rd.p_lo.copy(), rd.p_hi.copy()
    if kind == "abs":
        pl, ph = pl + param, ph + param
    elif kind == "rel":
        pl, ph = pl * (1 + param), ph * (1 + param)
    elif kind == "opposite":
        pl = pl + np.where(rd.call_lo, param, -param)
        ph = ph + np.where(rd.call_hi, param, -param)
    elif kind == "noise_abs":
        pl, ph = pl + rng.normal(0, param, rd.n), ph + rng.normal(0, param, rd.n)
    elif kind == "noise_rel":
        pl, ph = pl * (1 + rng.normal(0, param, rd.n)), ph * (1 + rng.normal(0, param, rd.n))
    else:
        raise ValueError(f"unknown scenario kind {kind!r}")
    return floor_tick(pl), floor_tick(ph)


def tipping_bias(rd: RowData, baseline_spread: float, baseline_r1: float) -> Dict[str, float]:
    """Systematic IV bias that would explain the whole median spread, and the matching price bias per row; and the multiplicative IV factor for a unit geometric-mean ratio."""
    delta = baseline_spread                                            # vol points: median(spread - delta) = 0 when delta = median(spread)
    eps = delta * rd.vega_atm()
    price = rd.price_atm()
    k = 1.0 / math.sqrt(baseline_r1)
    return dict(delta_vol_points=delta, price_bias_rs_median=float(np.median(eps)), price_bias_rs_p25=float(np.percentile(eps, 25)), price_bias_rs_p75=float(np.percentile(eps, 75)),
                price_bias_pct_of_price_median=float(np.median(eps / price) * 100), vega_rs_per_vol_point_median=float(np.median(rd.vega_atm())),
                atm_bracket_price_rs_median=float(np.median(price)), iv_factor_for_unit_ratio=k, iv_reduction_pct_for_unit_ratio=(1 - k) * 100)


def eiv_corrected_slope(slope: float, var_iv: float, noise_sd_vol_points: float) -> float:
    """Slope of RV on IV corrected for classical measurement noise in IV: b / (1 - sigma^2 / Var(IV)). A scenario, not an estimate."""
    rel = 1.0 - noise_sd_vol_points ** 2 / var_iv
    return float(slope / rel) if rel > 0 else math.nan


# ------------------------------------------------------------------------------ observed call/put proxy
def callput_disagreement(quotes: Sequence, forward: float, T: float, r: float = R_RATE, moneyness_max: float = 0.01, age_max: float = 5.0) -> pd.DataFrame:
    """IV_call - IV_put (vol points) at strikes with BOTH quotes fresh and |ln K/F| <= moneyness_max. Contains quote noise AND forward error AND ITM-side effects."""
    by: Dict[float, Dict[str, tuple]] = {}
    for q in quotes:
        if q.price is None or q.age_minutes is None or q.age_minutes > age_max or q.price <= 0:
            continue
        by.setdefault(float(q.strike), {})["call" if q.kind.value == "call" else "put"] = (q.price, q.age_minutes)
    rows = []
    for K, d in by.items():
        if "call" in d and "put" in d and abs(math.log(K / forward)) <= moneyness_max:
            ivc = implied_vol(np.array([d["call"][0]]), np.array([forward]), np.array([K]), np.array([T]), r, np.array([True]))[0]
            ivp = implied_vol(np.array([d["put"][0]]), np.array([forward]), np.array([K]), np.array([T]), r, np.array([False]))[0]
            if np.isfinite(ivc) and np.isfinite(ivp):
                rows.append(dict(strike=K, moneyness=math.log(K / forward), iv_call=ivc, iv_put=ivp, diff_vol_points=100 * (ivc - ivp), age_call=d["call"][1], age_put=d["put"][1]))
    return pd.DataFrame(rows)
