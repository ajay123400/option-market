"""Implied-volatility solver: find sigma such that BSM(sigma) == market price.

What the result IS: the Black-Scholes-Merton *market-implied* volatility of
the supplied price -- a quoting convention that depends on the model, on r,
q and S you passed, and on the price you chose (mid / last). It is NOT a
model-implied theoretical value; for that use `bsm.bsm_price`. Every
`IVResult` is tagged `kind="market_implied"` to keep the two apart.

Method
------
1. Validate inputs (InvalidInputError for out-of-domain values).
2. Check the price against the model-free no-arbitrage bounds
       forward_intrinsic <= price <= upper_bound
   (call: S e^{-qT} - K e^{-rT} ... S e^{-qT}; put: K e^{-rT} - S e^{-qT} ... K e^{-rT}).
   Prices outside the bounds, or sitting on a bound (where the price carries
   no information about sigma), get an explicit failure status -- never a
   fabricated IV.
3. Bracket sigma in [vol_min, vol_max] (defaults 1e-4 .. 5.0, i.e. 0.01% ..
   500%). BSM price is strictly increasing in sigma, so if the target is
   outside [price(vol_min), price(vol_max)] the status says so.
4. Safeguarded Newton-Raphson: take the Newton step if it stays inside the
   current bracket, otherwise bisect. The bracket shrinks every iteration,
   so convergence is guaranteed within max_iter bisections' worth of
   reduction, and Newton gives quadratic speed in the well-conditioned case.

Convergence criterion (explicit): |BSM(sigma*) - price| <= tol, with
tol = min(price_tol, price_rel_tol * price). price_tol is in PRICE UNITS
(default 1e-8) and price_rel_tol is relative (default 1e-10); the relative
term matters for cheap deep-OTM options, where a fixed 1e-8 would be a large
fraction of the price. If this is not met within max_iter the status is
NOT_CONVERGED and `iv` is None. A tolerance tighter than float arithmetic can
deliver therefore fails loudly instead of being silently relaxed.

Conditioning: when vega is tiny (deep ITM/OTM, very short expiry) a price
error of tol maps to a large vol error. `IVDiagnostics.vol_uncertainty`
= tol/vega reports it and `ill_conditioned` is set when it exceeds
`max_vol_uncertainty`; in that case the IV is returned (it is the
mathematically correct root) but flagged, and callers should not use it for
smile fitting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .bsm import (OptionType, bsm_price, forward_intrinsic, upper_bound,
                  validate_inputs)
from .errors import InvalidInputError
from .sensitivities import raw_vega


class IVStatus(str, Enum):
    CONVERGED = "converged"
    INVALID_INPUT = "invalid_input"
    EXPIRED = "expired"                    # T == 0: IV undefined
    PRICE_NOT_POSITIVE = "price_not_positive"
    BELOW_LOWER_BOUND = "below_lower_bound"  # price < forward intrinsic: arbitrage / bad quote
    ABOVE_UPPER_BOUND = "above_upper_bound"  # price > S e^{-qT} (call) / K e^{-rT} (put)
    AT_BOUND = "at_bound"                  # price on a bound: sigma not identifiable
    OUTSIDE_VOL_RANGE = "outside_vol_range"  # root not within [vol_min, vol_max]
    NOT_CONVERGED = "not_converged"        # max_iter exhausted


@dataclass(frozen=True)
class SolverConfig:
    vol_min: float = 1e-4
    vol_max: float = 5.0
    price_tol: float = 1e-8       # absolute cap, in price units
    price_rel_tol: float = 1e-10  # relative to the market price; effective tol = min(price_tol, price_rel_tol*price)
    max_iter: int = 100
    bound_rel_tol: float = 1e-12  # relative slack when comparing to no-arb bounds
    max_vol_uncertainty: float = 1e-6  # tol/vega above this (0.0001 vol points) => ILL_CONDITIONED

    def __post_init__(self):
        if not (0 < self.vol_min < self.vol_max):
            raise InvalidInputError("need 0 < vol_min < vol_max")
        if self.price_tol <= 0 or self.price_rel_tol <= 0 or self.max_iter < 1:
            raise InvalidInputError("price_tol and price_rel_tol must be > 0 and max_iter >= 1")


@dataclass(frozen=True)
class IVDiagnostics:
    iterations: int
    price_error: Optional[float]      # BSM(iv) - market price at the last iterate
    vega_raw: Optional[float]         # dV/dsigma per 1.00 at the last iterate
    vol_uncertainty: Optional[float]  # tol / vega: sigma error implied by the price tolerance
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    bracket: Optional[tuple]          # final (lo, hi) sigma bracket
    ill_conditioned: bool
    message: str


@dataclass(frozen=True)
class IVResult:
    status: IVStatus
    iv: Optional[float]               # decimal (0.20 = 20%); None unless status is CONVERGED
    diagnostics: IVDiagnostics
    kind: str = "market_implied"      # never "theoretical": see module docstring

    @property
    def converged(self) -> bool:
        return self.status is IVStatus.CONVERGED


def _fail(status: IVStatus, msg: str, *, lb=None, ub=None, iterations=0, bracket=None,
          price_error=None, vega=None) -> IVResult:
    return IVResult(status, None, IVDiagnostics(iterations, price_error, vega, None, lb, ub,
                                                bracket, False, msg))


def implied_volatility(price: float, S: float, K: float, T: float, r: float, q: float,
                       option_type, config: SolverConfig = SolverConfig()) -> IVResult:
    """Solve BSM implied volatility. Never raises for bad *market* prices --
    it returns a non-CONVERGED IVResult with a reason; it never returns an
    `iv` unless the convergence criterion was met."""
    try:
        ot = OptionType.coerce(option_type)
        validate_inputs(S, K, T, r, q, require_sigma=False)
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price):
            raise InvalidInputError(f"price must be a finite number (got {price!r})")
    except InvalidInputError as e:
        return _fail(IVStatus.INVALID_INPUT, str(e))

    if T == 0:
        return _fail(IVStatus.EXPIRED, "T == 0: implied volatility is undefined at expiry")
    if price <= 0:
        return _fail(IVStatus.PRICE_NOT_POSITIVE, f"price {price} <= 0 carries no volatility information")

    lb = forward_intrinsic(S, K, T, r, q, ot)
    ub = upper_bound(S, K, T, r, q, ot)
    slack = config.bound_rel_tol * max(S, K)
    if price < lb - slack:
        return _fail(IVStatus.BELOW_LOWER_BOUND,
                     f"price {price} < forward-intrinsic lower bound {lb:.6f} (arbitrage / stale or bad quote)",
                     lb=lb, ub=ub)
    if price > ub + slack:
        return _fail(IVStatus.ABOVE_UPPER_BOUND,
                     f"price {price} > upper bound {ub:.6f} (bad quote)", lb=lb, ub=ub)
    if price <= lb + slack or price >= ub - slack:
        return _fail(IVStatus.AT_BOUND,
                     f"price {price} sits on a no-arbitrage bound [{lb:.6f}, {ub:.6f}]; sigma is not identifiable",
                     lb=lb, ub=ub)

    lo, hi = config.vol_min, config.vol_max
    p_lo = bsm_price(S, K, T, r, q, lo, ot)
    p_hi = bsm_price(S, K, T, r, q, hi, ot)
    if price < p_lo - config.price_tol:
        return _fail(IVStatus.OUTSIDE_VOL_RANGE,
                     f"price {price} implies sigma below vol_min={lo} (model price there is {p_lo:.8f})",
                     lb=lb, ub=ub, bracket=(lo, hi))
    if price > p_hi + config.price_tol:
        return _fail(IVStatus.OUTSIDE_VOL_RANGE,
                     f"price {price} implies sigma above vol_max={hi} (model price there is {p_hi:.8f})",
                     lb=lb, ub=ub, bracket=(lo, hi))

    tol = min(config.price_tol, config.price_rel_tol * price)
    # Start from the ATM-forward approximation clipped into the bracket.
    fwd = S * math.exp((r - q) * T)
    sigma = min(max(math.sqrt(2.0 * math.pi / T) * price / fwd, lo), hi) if fwd > 0 else 0.2
    err = vega = None
    for it in range(1, config.max_iter + 1):
        err = bsm_price(S, K, T, r, q, sigma, ot) - price
        vega = raw_vega(S, K, T, r, q, sigma)
        if abs(err) <= tol:
            unc = tol / vega if vega > 0 else math.inf
            ill = unc > config.max_vol_uncertainty
            msg = "converged" + ("; ILL-CONDITIONED (vega too small for a reliable vol)" if ill else "")
            return IVResult(IVStatus.CONVERGED, sigma,
                            IVDiagnostics(it, err, vega, unc, lb, ub, (lo, hi), ill, msg))
        if err > 0:
            hi = sigma
        else:
            lo = sigma
        step = sigma - err / vega if vega > 1e-300 else math.nan
        sigma = step if (math.isfinite(step) and lo < step < hi) else 0.5 * (lo + hi)
    return IVResult(IVStatus.NOT_CONVERGED, None,
                    IVDiagnostics(config.max_iter, err, vega, None, lb, ub, (lo, hi), False,
                                  f"no convergence to price tolerance {tol:.3g} in {config.max_iter} iterations; "
                                  f"last sigma estimate {sigma:.10f} is NOT a valid IV"))
