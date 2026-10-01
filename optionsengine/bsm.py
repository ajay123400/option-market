"""Black-Scholes-Merton pricing for European options.

Model: the underlying S follows geometric Brownian motion with constant
volatility sigma, a constant continuously-compounded risk-free rate r and a
constant continuous dividend yield q:

    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)
    call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

Conventions
-----------
* Rates, yields and volatility are decimals per annum (0.065 = 6.5%).
* T is in YEARS. This module does not decide the day-count; see
  `analytics.time_to_expiry_years` (ACT/365 calendar time).
* Prices are per ONE unit of the underlying (not per lot).

Assumptions / limitations (all inherited from BSM; none are corrected here)
---------------------------------------------------------------------------
* European exercise only. NSE index options (NIFTY) are European, so this
  is appropriate for them; it is NOT valid for American-style options.
* Constant volatility, r and q. Real vol has a smile/skew and term
  structure; the "IV" derived from this model is a quote convention, not a
  forecast.
* q is a continuous yield. For an index the discrete dividends are
  approximated by q. When the futures/forward price is known, a
  forward-consistent q can be backed out with `implied_carry_yield`; the
  engine never invents q or r for you.
* No transaction costs, taxes, margin, borrow costs or discrete jumps.

Degenerate inputs
-----------------
* T == 0  -> intrinsic value (the exact limit), documented behaviour.
* sigma == 0 (T > 0) -> discounted forward intrinsic value, the exact limit
  of the BSM price as sigma -> 0.
* T < 0, sigma < 0, S <= 0, K <= 0, or any non-finite input -> InvalidInputError.
"""
from __future__ import annotations

import math
from enum import Enum

from .errors import InvalidInputError

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

    @classmethod
    def coerce(cls, value) -> "OptionType":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("c", "call", "ce"):
                return cls.CALL
            if v in ("p", "put", "pe"):
                return cls.PUT
        raise InvalidInputError(f"option type must be call/put (got {value!r})")


def norm_cdf(x: float) -> float:
    """Standard normal CDF via erfc. Using erfc (not 1 + erf) keeps full
    relative precision in the far left tail, which matters for deep OTM
    option prices that are many orders of magnitude below S."""
    return 0.5 * math.erfc(-x / _SQRT2)


def norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidInputError(f"{name} must be a finite number (got {value!r})")
    return float(value)


def validate_inputs(S, K, T, r, q, sigma=None, *, require_sigma: bool = True) -> None:
    """Raise InvalidInputError for anything outside the model's domain.
    Zero T / sigma are allowed here (handled explicitly by callers)."""
    S = _finite("underlying price S", S)
    K = _finite("strike K", K)
    T = _finite("time to expiry T", T)
    _finite("risk-free rate r", r)
    _finite("dividend yield q", q)
    if S <= 0:
        raise InvalidInputError(f"underlying price S must be > 0 (got {S})")
    if K <= 0:
        raise InvalidInputError(f"strike K must be > 0 (got {K})")
    if T < 0:
        raise InvalidInputError(f"time to expiry T must be >= 0 (got {T}); the option has expired")
    if require_sigma:
        sig = _finite("volatility sigma", sigma)
        if sig < 0:
            raise InvalidInputError(f"volatility sigma must be >= 0 (got {sig})")


def d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    """d1, d2. Requires T > 0 and sigma > 0 (callers handle the limits)."""
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    return d1, d1 - sd


def intrinsic_value(S: float, K: float, option_type) -> float:
    """Undiscounted exercise value max(S-K,0) / max(K-S,0)."""
    ot = OptionType.coerce(option_type)
    return max(S - K, 0.0) if ot is OptionType.CALL else max(K - S, 0.0)


def forward_intrinsic(S: float, K: float, T: float, r: float, q: float, option_type) -> float:
    """Model-free lower bound of a European option:
    max(S e^{-qT} - K e^{-rT}, 0) for a call, max(K e^{-rT} - S e^{-qT}, 0)
    for a put. Equals the BSM price in the sigma -> 0 limit."""
    ot = OptionType.coerce(option_type)
    pv_s, pv_k = S * math.exp(-q * T), K * math.exp(-r * T)
    return max(pv_s - pv_k, 0.0) if ot is OptionType.CALL else max(pv_k - pv_s, 0.0)


def upper_bound(S: float, K: float, T: float, r: float, q: float, option_type) -> float:
    """Model-free upper bound: S e^{-qT} for a call, K e^{-rT} for a put."""
    ot = OptionType.coerce(option_type)
    return S * math.exp(-q * T) if ot is OptionType.CALL else K * math.exp(-r * T)


def bsm_price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type) -> float:
    """Theoretical BSM price of a European option (a MODEL value, not a
    market quote). See module docstring for edge-case behaviour."""
    ot = OptionType.coerce(option_type)
    validate_inputs(S, K, T, r, q, sigma)
    if T == 0:
        return intrinsic_value(S, K, ot)
    if sigma == 0:
        return forward_intrinsic(S, K, T, r, q, ot)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    pv_s, pv_k = S * math.exp(-q * T), K * math.exp(-r * T)
    if ot is OptionType.CALL:
        return pv_s * norm_cdf(d1) - pv_k * norm_cdf(d2)
    return pv_k * norm_cdf(-d2) - pv_s * norm_cdf(-d1)


def call_price(S, K, T, r, q, sigma) -> float:
    return bsm_price(S, K, T, r, q, sigma, OptionType.CALL)


def put_price(S, K, T, r, q, sigma) -> float:
    return bsm_price(S, K, T, r, q, sigma, OptionType.PUT)


def implied_carry_yield(S: float, F: float, T: float, r: float) -> float:
    """Continuous yield q* that makes the BSM forward equal an observed
    forward/futures price F:  F = S e^{(r-q)T}  =>  q* = r - ln(F/S)/T.

    Index options price off the forward, not the cash index (NIFTY futures
    carry a basis), so for NIFTY this q* -- not a guessed dividend yield --
    is the consistent input. Requires T > 0."""
    validate_inputs(S, F, T, r, 0.0, require_sigma=False)
    if T <= 0:
        raise InvalidInputError("implied_carry_yield requires T > 0")
    return r - math.log(F / S) / T
