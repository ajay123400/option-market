"""Quote-quality assessment for a RAW OptionContract. Pure function of the
raw record + a policy; it never alters the contract.

Flags (QuoteFlag)
-----------------
Blocking by default (IV is NOT computed from the price when present):
  MISSING_PRICE      no usable price at all
  NON_FINITE_PRICE   a bid/ask/last is NaN/inf
  NEGATIVE_PRICE     a bid/ask/last is < 0
  CROSSED_MARKET     bid > ask
  ZERO_BID           bid is 0 (or missing while an ask exists): the mid is meaningless
  NO_ASK             no ask (one-sided book)
  STALE_QUOTE        bid/ask older than policy.max_quote_age
  STALE_LAST         last trade older than policy.max_last_age (only matters when
                     the last price is the basis)
  WIDE_SPREAD        (ask-bid)/mid > policy.max_rel_spread
Warning only:
  LOCKED_MARKET      bid == ask > 0
  LAST_OUTSIDE_MARKET last trade outside [bid, ask]
  NO_VOLUME          volume is 0
  ZERO_OPEN_INTEREST open interest is 0
  NO_QUOTE_TIMESTAMP no timestamp to judge staleness (staleness NOT verified)
  EXPIRY_TIME_ASSUMED expiry time-of-day was filled in by the caller
  SOURCE_FLAGGED     provider attached its own flags

Which flags block is `QualityPolicy.blocking`, so it is configurable and
explicit. Thresholds are policy parameters with conservative defaults, not
market truth -- tune them with real NIFTY data.

Price selection: mid when the book is two-sided and uncrossed; otherwise
LAST only if `allow_last_fallback` and the last is fresh; otherwise none.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Optional

from .schema import OptionContract


class QuoteFlag(str, Enum):
    MISSING_PRICE = "missing_price"
    NON_FINITE_PRICE = "non_finite_price"
    NEGATIVE_PRICE = "negative_price"
    CROSSED_MARKET = "crossed_market"
    ZERO_BID = "zero_bid"
    NO_ASK = "no_ask"
    STALE_QUOTE = "stale_quote"
    STALE_LAST = "stale_last"
    WIDE_SPREAD = "wide_spread"
    LOCKED_MARKET = "locked_market"
    LAST_OUTSIDE_MARKET = "last_outside_market"
    NO_VOLUME = "no_volume"
    ZERO_OPEN_INTEREST = "zero_open_interest"
    NO_QUOTE_TIMESTAMP = "no_quote_timestamp"
    EXPIRY_TIME_ASSUMED = "expiry_time_assumed"
    SOURCE_FLAGGED = "source_flagged"


_DEFAULT_BLOCKING = frozenset({
    QuoteFlag.MISSING_PRICE, QuoteFlag.NON_FINITE_PRICE, QuoteFlag.NEGATIVE_PRICE,
    QuoteFlag.CROSSED_MARKET, QuoteFlag.ZERO_BID, QuoteFlag.NO_ASK,
    QuoteFlag.STALE_QUOTE, QuoteFlag.STALE_LAST, QuoteFlag.WIDE_SPREAD,
})


@dataclass(frozen=True)
class QualityPolicy:
    max_quote_age: timedelta = timedelta(seconds=60)
    max_last_age: timedelta = timedelta(minutes=15)
    max_rel_spread: float = 0.5          # (ask-bid)/mid
    allow_last_fallback: bool = False    # use LTP when the book is unusable?
    blocking: frozenset = field(default=_DEFAULT_BLOCKING)


class PriceBasis(str, Enum):
    MID = "mid"
    LAST = "last"


@dataclass(frozen=True)
class QuoteAssessment:
    flags: tuple                         # tuple[QuoteFlag, ...]
    blocking_flags: tuple
    price: Optional[float]               # selected price, or None
    basis: Optional[PriceBasis]

    @property
    def usable(self) -> bool:
        return self.price is not None and not self.blocking_flags


def _num(v) -> Optional[float]:
    return None if v is None else float(v)


def assess_quote(c: OptionContract, policy: QualityPolicy = QualityPolicy()) -> QuoteAssessment:
    flags: list[QuoteFlag] = []
    bid, ask, last = _num(c.bid), _num(c.ask), _num(c.last)

    for v in (bid, ask, last):
        if v is not None and not math.isfinite(v):
            flags.append(QuoteFlag.NON_FINITE_PRICE)
            break
    bid_ok = bid is not None and math.isfinite(bid)
    ask_ok = ask is not None and math.isfinite(ask)
    last_ok = last is not None and math.isfinite(last)
    if any(v is not None and math.isfinite(v) and v < 0 for v in (bid, ask, last)):
        flags.append(QuoteFlag.NEGATIVE_PRICE)

    two_sided = False
    if ask_ok and ask > 0 and (not bid_ok or bid <= 0):
        flags.append(QuoteFlag.ZERO_BID)
    if bid_ok and bid > 0 and (not ask_ok or ask <= 0):
        flags.append(QuoteFlag.NO_ASK)
    if bid_ok and ask_ok and bid > 0 and ask > 0:
        if bid > ask:
            flags.append(QuoteFlag.CROSSED_MARKET)
        else:
            two_sided = True
            if bid == ask:
                flags.append(QuoteFlag.LOCKED_MARKET)
            mid = 0.5 * (bid + ask)
            if (ask - bid) / mid > policy.max_rel_spread:
                flags.append(QuoteFlag.WIDE_SPREAD)
    if bid_ok and bid == 0 and QuoteFlag.ZERO_BID not in flags:
        flags.append(QuoteFlag.ZERO_BID)
    if last_ok and two_sided and not (bid <= last <= ask):
        flags.append(QuoteFlag.LAST_OUTSIDE_MARKET)

    if c.quote_timestamp is None:
        if bid_ok or ask_ok:
            flags.append(QuoteFlag.NO_QUOTE_TIMESTAMP)
    elif c.timestamp - c.quote_timestamp > policy.max_quote_age:
        flags.append(QuoteFlag.STALE_QUOTE)
    last_stale = (c.last_trade_timestamp is not None
                  and c.timestamp - c.last_trade_timestamp > policy.max_last_age)

    if c.volume is not None and c.volume == 0:
        flags.append(QuoteFlag.NO_VOLUME)
    if c.open_interest is not None and c.open_interest == 0:
        flags.append(QuoteFlag.ZERO_OPEN_INTEREST)
    if c.expiry_time_assumed:
        flags.append(QuoteFlag.EXPIRY_TIME_ASSUMED)
    if c.source_flags:
        flags.append(QuoteFlag.SOURCE_FLAGGED)

    # ---- price selection --------------------------------------------------
    price: Optional[float] = None
    basis: Optional[PriceBasis] = None
    if two_sided:
        price, basis = 0.5 * (bid + ask), PriceBasis.MID
    elif policy.allow_last_fallback and last_ok and last > 0:
        price, basis = last, PriceBasis.LAST
        if last_stale:
            flags.append(QuoteFlag.STALE_LAST)
    if price is None and not any(f in flags for f in (QuoteFlag.NON_FINITE_PRICE, QuoteFlag.NEGATIVE_PRICE)):
        flags.append(QuoteFlag.MISSING_PRICE)

    # Flags that describe the book only block when the book (mid) is the basis;
    # with a LAST basis the book-side flags are informational.
    book_flags = {QuoteFlag.ZERO_BID, QuoteFlag.NO_ASK, QuoteFlag.CROSSED_MARKET,
                  QuoteFlag.WIDE_SPREAD, QuoteFlag.STALE_QUOTE}
    blocking = tuple(f for f in dict.fromkeys(flags)
                     if f in policy.blocking and not (basis is PriceBasis.LAST and f in book_flags))
    return QuoteAssessment(tuple(dict.fromkeys(flags)), blocking, price, basis)
