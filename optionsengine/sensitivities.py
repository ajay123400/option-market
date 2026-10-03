"""Black-Scholes-Merton Greeks (closed form, European options, continuous
dividend yield q). All values are per ONE unit of the option/underlying --
multiply by lot size and number of lots for a position.

UNITS (the `Greeks` dataclass fields)
-------------------------------------
delta : dV/dS. Dimensionless; price change per +1.00 move in the underlying
        (1 index point for NIFTY). Call in [0, e^{-qT}], put in [-e^{-qT}, 0].
gamma : d2V/dS2 = d(delta)/dS. Delta change per +1.00 move in the underlying.
vega  : dV/dsigma / 100. Price change per +1 PERCENTAGE POINT of volatility
        (e.g. 20% -> 21%). The raw per-1.00 (=100 points) figure is
        `Greeks.raw().vega`.
theta : -dV/dT / 365. Price change per +1 CALENDAR DAY of passing time
        (time decay; normally negative for long options). Raw per-year
        figure is `Greeks.raw().theta`.
rho   : dV/dr / 100. Price change per +1 PERCENTAGE POINT of the risk-free
        rate (e.g. 6.5% -> 7.5%). Raw per-1.00 figure is `Greeks.raw().rho`.

`Greeks.raw()` returns the same Greeks in "pure calculus" units (vega per
1.00 of sigma, theta per year, rho per 1.00 of r) -- these are what the
finite-difference tests differentiate against.

Degenerate inputs: Greeks are undefined/singular at T == 0 or sigma == 0
(delta is a step function, gamma a delta spike). Rather than return a
misleading finite number, `greeks()` raises DegenerateInputError there and
InvalidInputError for out-of-domain inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .bsm import OptionType, d1_d2, norm_cdf, norm_pdf, validate_inputs
from .errors import DegenerateInputError

DAYS_PER_YEAR = 365.0  # theta is quoted per CALENDAR day


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float   # per 1 percentage point of volatility
    theta: float  # per calendar day
    rho: float    # per 1 percentage point of the risk-free rate

    def raw(self) -> "Greeks":
        """Same Greeks in calculus units: vega per 1.00 sigma, theta per
        year, rho per 1.00 r (delta/gamma unchanged)."""
        return Greeks(self.delta, self.gamma, self.vega * 100.0,
                      self.theta * DAYS_PER_YEAR, self.rho * 100.0)


def greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type) -> Greeks:
    """All five BSM Greeks. Not rounded -- rounding is a presentation
    concern for the caller."""
    ot = OptionType.coerce(option_type)
    validate_inputs(S, K, T, r, q, sigma)
    if T == 0:
        raise DegenerateInputError("Greeks are undefined at T == 0 (expiry)")
    if sigma == 0:
        raise DegenerateInputError("Greeks are undefined at sigma == 0")
    sqrtT = math.sqrt(T)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    pdf1 = norm_pdf(d1)
    eq, er = math.exp(-q * T), math.exp(-r * T)

    gamma = eq * pdf1 / (S * sigma * sqrtT)
    vega_raw = S * eq * pdf1 * sqrtT
    decay = -S * eq * pdf1 * sigma / (2.0 * sqrtT)
    if ot is OptionType.CALL:
        delta = eq * norm_cdf(d1)
        theta_year = decay - r * K * er * norm_cdf(d2) + q * S * eq * norm_cdf(d1)
        rho_raw = K * T * er * norm_cdf(d2)
    else:
        delta = -eq * norm_cdf(-d1)
        theta_year = decay + r * K * er * norm_cdf(-d2) - q * S * eq * norm_cdf(-d1)
        rho_raw = -K * T * er * norm_cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega_raw / 100.0,
                  theta=theta_year / DAYS_PER_YEAR, rho=rho_raw / 100.0)


def raw_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """dV/dsigma per 1.00 of sigma (same for call and put). Used by the IV
    solver; requires T > 0 and sigma > 0."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)
