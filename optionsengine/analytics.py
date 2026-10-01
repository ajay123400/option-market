"""Calculated analytics for ONE contract, kept apart from raw data.

`analyze_contract(raw_contract, assumptions)` -> `OptionAnalytics`, a new
immutable object that references the raw contract's identity but never
modifies it. Pipeline:

    raw OptionContract --assess_quote--> price + quality flags
                       --time_to_expiry_years--> T
                       --implied_volatility--> market IV (+ diagnostics)
                       --greeks at that IV--> Greeks
                       --bsm_price at that IV--> model price (round-trip check)

If the quote is blocked by policy, IV and Greeks are NOT computed
(status UNRELIABLE_QUOTE) -- the engine will not manufacture a number from a
price it has been told not to trust.

Time to expiry uses ACT/365 calendar time from the snapshot timestamp to the
expiry datetime (seconds / (365 * 86400)). This is a convention, not truth:
trading-time (business-day) conventions would give different vols/theta.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from .bsm import bsm_price
from .implied_vol import IVResult, IVStatus, SolverConfig, implied_volatility
from .quality import PriceBasis, QualityPolicy, QuoteAssessment, assess_quote
from .schema import MarketAssumptions, OptionContract
from .sensitivities import Greeks, greeks

IST = timezone(timedelta(hours=5, minutes=30))
SECONDS_PER_YEAR = 365.0 * 86400.0


def time_to_expiry_years(timestamp: datetime, expiry: datetime) -> float:
    """ACT/365 year fraction from `timestamp` to `expiry` (both aware).
    Returns 0.0 if expiry <= timestamp (never negative)."""
    seconds = (expiry - timestamp).total_seconds()
    return max(seconds, 0.0) / SECONDS_PER_YEAR


def expiry_at_close(expiry_date: date, close: time = time(15, 30), tz: timezone = IST) -> datetime:
    """Convenience for providers that send only an expiry DATE. The 15:30 IST
    default is the NSE index-option expiry-day close; it is an ASSUMPTION --
    pass the result with `expiry_time_assumed=True` on the contract so the
    quality layer flags it."""
    return datetime.combine(expiry_date, close, tzinfo=tz)


class AnalyticsStatus:
    OK = "ok"
    UNRELIABLE_QUOTE = "unreliable_quote"   # blocked by quality policy; IV not attempted
    IV_FAILED = "iv_failed"                 # price accepted but solver status != CONVERGED
    EXPIRED = "expired"


@dataclass(frozen=True)
class OptionAnalytics:
    """Derived values. Every field here is CALCULATED; the raw inputs remain
    in the `OptionContract` it was computed from."""
    contract_symbol: Optional[str]
    underlying_symbol: str
    timestamp: datetime
    status: str
    quote: QuoteAssessment
    time_to_expiry: float                # years, ACT/365
    price_used: Optional[float]
    price_basis: Optional[PriceBasis]
    iv_result: Optional[IVResult]        # market-implied vol (None if not attempted)
    greeks: Optional[Greeks]             # at the market IV; None unless IV converged
    model_price_at_iv: Optional[float]   # BSM price re-computed at the solved IV (round-trip check, ~ price_used)
    model: str = "BSM (European, continuous q)"

    @property
    def iv(self) -> Optional[float]:
        return self.iv_result.iv if self.iv_result and self.iv_result.converged else None


def analyze_contract(contract: OptionContract, assumptions: MarketAssumptions,
                     policy: QualityPolicy = QualityPolicy(),
                     solver: SolverConfig = SolverConfig()) -> OptionAnalytics:
    quote = assess_quote(contract, policy)
    T = time_to_expiry_years(contract.timestamp, contract.expiry)
    base = dict(contract_symbol=contract.contract_symbol, underlying_symbol=contract.underlying_symbol,
                timestamp=contract.timestamp, quote=quote, time_to_expiry=T,
                price_used=quote.price, price_basis=quote.basis)

    if T == 0:
        return OptionAnalytics(status=AnalyticsStatus.EXPIRED, iv_result=None, greeks=None,
                               model_price_at_iv=None, **base)
    if not quote.usable:
        return OptionAnalytics(status=AnalyticsStatus.UNRELIABLE_QUOTE, iv_result=None, greeks=None,
                               model_price_at_iv=None, **base)

    r, q = assumptions.risk_free_rate, assumptions.dividend_yield
    iv = implied_volatility(quote.price, contract.underlying_price, contract.strike, T, r, q,
                            contract.option_type, solver)
    if not iv.converged:
        return OptionAnalytics(status=AnalyticsStatus.IV_FAILED, iv_result=iv, greeks=None,
                               model_price_at_iv=None, **base)
    g = greeks(contract.underlying_price, contract.strike, T, r, q, iv.iv, contract.option_type)
    mp = bsm_price(contract.underlying_price, contract.strike, T, r, q, iv.iv, contract.option_type)
    return OptionAnalytics(status=AnalyticsStatus.OK, iv_result=iv, greeks=g, model_price_at_iv=mp, **base)
