"""Normalized RAW market-data schema. Nothing in this module computes
analytics: calculated values (IV, Greeks, quality verdicts) live in
`analytics.OptionAnalytics`, so raw data is never mutated or mixed with
derived numbers and can be stored/replayed as-is.

Rules
-----
* Missing data is `None`, never 0 or a guess. A bid of 0.0 means "the
  exchange showed a zero bid"; a bid of None means "we don't know".
* The constructor rejects only STRUCTURALLY impossible records (naive
  datetimes, strike <= 0, spot <= 0, unknown option type). Dubious MARKET
  values (crossed book, zero bid, stale quote, negative price) are accepted
  as raw facts and judged later by `quality.assess_quote`.
* All datetimes must be timezone-aware (NSE data is IST, UTC+05:30).
* Prices are per unit of underlying; `lot_size` converts to a position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .bsm import OptionType
from .errors import InvalidInputError


def _aware(name: str, dt: Optional[datetime], required: bool) -> None:
    if dt is None:
        if required:
            raise InvalidInputError(f"{name} is required")
        return
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise InvalidInputError(f"{name} must be a timezone-aware datetime (got {dt!r})")


@dataclass(frozen=True)
class OptionContract:
    """One option contract observed at one moment, as the provider sent it."""
    underlying_symbol: str              # e.g. "NIFTY"
    underlying_price: float             # spot (or the reference level you intend to price off)
    timestamp: datetime                 # when this snapshot was observed (aware)
    expiry: datetime                    # expiry date-time (aware); NSE index options expire 15:30 IST
    strike: float
    option_type: OptionType
    data_source: str                    # provider/feed name, e.g. "fyers:options-chain-v3"
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None        # last traded price
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    lot_size: Optional[int] = None
    quote_timestamp: Optional[datetime] = None       # exchange time of the bid/ask update
    last_trade_timestamp: Optional[datetime] = None  # exchange time of the last trade
    contract_symbol: Optional[str] = None            # e.g. "NSE:NIFTY2690824500CE"
    expiry_time_assumed: bool = False   # True if `expiry` time-of-day was filled by the caller (e.g. 15:30 IST) rather than sent by the provider
    source_flags: tuple = ()            # provider-supplied quality flags, verbatim

    def __post_init__(self):
        object.__setattr__(self, "option_type", OptionType.coerce(self.option_type))
        object.__setattr__(self, "source_flags", tuple(self.source_flags))
        if not self.underlying_symbol or not self.data_source:
            raise InvalidInputError("underlying_symbol and data_source are required")
        for name, v in (("underlying_price", self.underlying_price), ("strike", self.strike)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise InvalidInputError(f"{name} must be a finite number > 0 (got {v!r})")
        _aware("timestamp", self.timestamp, True)
        _aware("expiry", self.expiry, True)
        _aware("quote_timestamp", self.quote_timestamp, False)
        _aware("last_trade_timestamp", self.last_trade_timestamp, False)
        if self.lot_size is not None and self.lot_size <= 0:
            raise InvalidInputError(f"lot_size must be > 0 (got {self.lot_size})")


@dataclass(frozen=True)
class MarketAssumptions:
    """Rate/dividend inputs. Both are REQUIRED with a stated provenance --
    the engine has no default r or q, because a silent default (the old
    app hardcodes r = 6.5%) quietly changes every IV and Greek.
    For an index, `dividend_yield` may be the forward-consistent carry yield
    from `bsm.implied_carry_yield`; say so in `dividend_source`."""
    risk_free_rate: float
    dividend_yield: float
    rate_source: str                    # e.g. "RBI 91-day T-bill cut-off 2026-09-30"
    dividend_source: str                # e.g. "implied from NIFTY futures, cost-of-carry"
    as_of: Optional[datetime] = None

    def __post_init__(self):
        for name in ("risk_free_rate", "dividend_yield"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise InvalidInputError(f"{name} must be a finite number (got {v!r})")
        if not self.rate_source or not self.dividend_source:
            raise InvalidInputError("rate_source and dividend_source are required (no silent defaults)")
        _aware("as_of", self.as_of, False)
