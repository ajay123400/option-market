"""Pricing primitives: all solvers/forward logic come from the committed research engine (optionsengine); only thin wrappers and the ATM construction live here."""
import math
from typing import Dict, Optional, Tuple

from optionsengine.bsm import OptionType, implied_carry_yield, norm_pdf
from optionsengine.forward import ForwardConfig, ForwardStatus, ParityObservation, estimate_parity_forward
from optionsengine.implied_vol import SolverConfig, implied_volatility

SECONDS_PER_YEAR = 365.0 * 86400.0


def time_to_expiry_years(expiry_ts: float, capture_ts: float, skew_s: float) -> float:
    """ACT/365 from the observation instant to the provider-supplied expiry instant. The observation is the capture time MINUS the estimated local-minus-server clock skew
    (our clock runs ahead of the exchange by about 1-1.5 s)."""
    return (expiry_ts - (capture_ts - (skew_s or 0.0))) / SECONDS_PER_YEAR


def vega_per_vol_point(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 vega in rupees per ONE vol point (1% = 0.01 of sigma): e^{-rT} F phi(d1) sqrt(T) / 100. Identical for the call and the put at a strike."""
    if not (F > 0 and K > 0 and T > 0 and sigma > 0):
        return float("nan")
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return math.exp(-r * T) * F * norm_pdf(d1) * math.sqrt(T) / 100.0


def solve_iv(price: Optional[float], spot: float, K: float, T: float, r: float, q: float, kind: OptionType, cfg: SolverConfig = SolverConfig()) -> Tuple[Optional[float], bool]:
    """(iv, usable): usable = converged AND reliable (not ill-conditioned, not tick-resolution-limited), the same criterion as Stage 2A's `used` points."""
    if price is None or not (price > 0):
        return None, False
    res = implied_volatility(price, spot, K, T, r, q, kind, cfg)
    if not res.converged:
        return None, False
    return res.iv, res.reliable


def parity_forward(pairs, r: float, T: float, spot: float, cfg: ForwardConfig = ForwardConfig()):
    """pairs: [(strike, call_mid, put_mid)] -> ForwardEstimate (same gate as Stage 2A: nearest strikes, MAD outliers, dispersion limit)."""
    return estimate_parity_forward([ParityObservation(k, c, p) for k, c, p in pairs], r, T, spot, cfg)


def atm_iv(points: Dict[float, Tuple[float, Optional[float], bool]], max_bracket_pts: float = 100.0):
    """points: {strike: (log_moneyness x = ln(K/F), iv, usable)}. Linear interpolation in x at x = 0 between the adjacent strikes below/above the forward (never extrapolated), both usable, at most
    `max_bracket_pts` apart: exactly Stage 2A's strict ATM. Returns (iv, strike_low, strike_high) or (None, None, None)."""
    below = [(k, v) for k, v in points.items() if v[0] < 0]
    above = [(k, v) for k, v in points.items() if v[0] >= 0]
    if not below or not above:
        return None, None, None
    klo, lo = max(below, key=lambda kv: kv[0])
    khi, hi = min(above, key=lambda kv: kv[0])
    if khi - klo > max_bracket_pts or not (lo[2] and hi[2] and lo[1] is not None and hi[1] is not None):
        return None, None, None
    w = (0.0 - lo[0]) / (hi[0] - lo[0])
    return lo[1] + w * (hi[1] - lo[1]), klo, khi
