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

Price resolution (tick size) -- a SEPARATE uncertainty from convergence:
exchange prices are quantised (NSE options tick = 0.05), so an observed price
p only says the true price lies in about [p - res/2, p + res/2], whatever the
solver tolerance. `price_resolution` (default 0.05) turns that into an IV
interval by re-solving at p -/+ res/2:
    diagnostics.iv_interval            (iv_low, iv_high); 0.0 / inf when the
                                       perturbed price reaches a no-arbitrage
                                       bound (sigma -> 0 or unbounded above)
    diagnostics.resolution_uncertainty max(iv - iv_low, iv_high - iv)
    diagnostics.resolution_limited     True when that exceeds
                                       `max_resolution_uncertainty` (default
                                       0.01 = 1 vol point)
Behaviour that matters: the IV is still CONVERGED and still returned -- the
contract is NOT rejected; `resolution_limited` is only a flag for callers
(use `IVResult.reliable` to require converged AND well-conditioned AND
resolution-resolved). Why near expiry: vega scales like sqrt(T) (and falls
off further for away-from-the-money strikes), so the same half-tick of price
maps to a vol band ~ (res/2)/vega that grows without bound as T -> 0. Worked
example (S = 24,500, 15 minutes to expiry, true vol 14 %, r = 6.5 %): the ATM
call quotes 7.35 and its IV is resolved to about +/-0.05 vol points, but the
call 200 points ITM quotes 200.05 (0.05 of time value) and the same half-tick
band spans IV from 0 % to 52 %; the 245-point OTM call rounds to 0.00 and has
no IV at all (PRICE_NOT_POSITIVE). Pass `price_resolution=None` (or 0) to disable. If you feed a
bid/ask MID, the mid sits on a 0.025 grid but the real uncertainty is the
spread -- do not use 0.025 as a substitute for quote-quality screening.
"""
from __future__ import annotations

import dataclasses
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
    price_resolution: Optional[float] = 0.05  # tick size of the observed price; None/0 disables resolution diagnostics
    max_resolution_uncertainty: float = 0.01  # IV band half-width (decimal vol) above this => resolution_limited

    def __post_init__(self):
        if not (0 < self.vol_min < self.vol_max):
            raise InvalidInputError("need 0 < vol_min < vol_max")
        if self.price_tol <= 0 or self.price_rel_tol <= 0 or self.max_iter < 1:
            raise InvalidInputError("price_tol and price_rel_tol must be > 0 and max_iter >= 1")
        pr = self.price_resolution
        if pr is not None and (isinstance(pr, bool) or not isinstance(pr, (int, float)) or not math.isfinite(pr) or pr < 0):
            raise InvalidInputError("price_resolution must be None or a finite number >= 0")
        if not (self.max_resolution_uncertainty > 0):
            raise InvalidInputError("max_resolution_uncertainty must be > 0")


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
    # --- price-resolution (tick) uncertainty: independent of convergence ---
    price_resolution: Optional[float] = None
    iv_interval: Optional[tuple] = None          # (iv_low, iv_high) for price -/+ resolution/2
    resolution_uncertainty: Optional[float] = None
    resolution_limited: bool = False


@dataclass(frozen=True)
class IVResult:
    status: IVStatus
    iv: Optional[float]               # decimal (0.20 = 20%); None unless status is CONVERGED
    diagnostics: IVDiagnostics
    kind: str = "market_implied"      # never "theoretical": see module docstring

    @property
    def converged(self) -> bool:
        return self.status is IVStatus.CONVERGED

    @property
    def resolution_limited(self) -> bool:
        return self.diagnostics.resolution_limited

    @property
    def reliable(self) -> bool:
        """Converged AND not ill-conditioned AND not limited by price
        resolution. Stricter than `converged`; use it for surface fitting."""
        d = self.diagnostics
        return self.converged and not d.ill_conditioned and not d.resolution_limited


def _fail(status: IVStatus, msg: str, *, lb=None, ub=None, iterations=0, bracket=None,
          price_error=None, vega=None) -> IVResult:
    return IVResult(status, None, IVDiagnostics(iterations, price_error, vega, None, lb, ub,
                                                bracket, False, msg))


def implied_volatility(price: float, S: float, K: float, T: float, r: float, q: float,
                       option_type, config: SolverConfig = SolverConfig()) -> IVResult:
    """Solve BSM implied volatility. Never raises for bad *market* prices --
    it returns a non-CONVERGED IVResult with a reason; it never returns an
    `iv` unless the convergence criterion was met. When it converges it also
    reports the price-resolution (tick) uncertainty -- see module docstring;
    that flag never changes the status."""
    res = _solve(price, S, K, T, r, q, option_type, config)
    h = config.price_resolution
    if not res.converged or not h:
        return res
    lo = _perturbed_iv(price - 0.5 * h, S, K, T, r, q, option_type, config, side="low")
    hi = _perturbed_iv(price + 0.5 * h, S, K, T, r, q, option_type, config, side="high")
    d = res.diagnostics
    if lo is None or hi is None:      # perturbed solve failed to converge: cannot bound -> be conservative
        unc, limited = math.inf, True
    else:
        unc = max(res.iv - lo, hi - res.iv)
        limited = unc > config.max_resolution_uncertainty
    msg = d.message + ("; RESOLUTION-LIMITED (half-tick price band moves IV by more than "
                       f"{config.max_resolution_uncertainty:g})" if limited else "")
    return IVResult(res.status, res.iv, dataclasses.replace(
        d, message=msg, price_resolution=h, iv_interval=(lo, hi), resolution_uncertainty=unc,
        resolution_limited=limited), res.kind)


def _perturbed_iv(price, S, K, T, r, q, option_type, config, side):
    """IV at a half-tick-perturbed price. A price at/below the lower bound
    maps to sigma -> 0 (returns 0.0) and one at/above the upper bound or
    beyond vol_max to +inf; a non-converged solve returns None."""
    res = _solve(price, S, K, T, r, q, option_type, config)
    if res.converged:
        return res.iv
    if res.status is IVStatus.NOT_CONVERGED or res.status is IVStatus.INVALID_INPUT:
        return None
    if side == "low":
        return 0.0   # price_not_positive / below_lower_bound / at_bound / outside_vol_range(below)
    return math.inf  # above_upper_bound / at_bound / outside_vol_range(above)


def _solve(price: float, S: float, K: float, T: float, r: float, q: float,
           option_type, config: SolverConfig) -> IVResult:
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
