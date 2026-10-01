import dataclasses
import math
from datetime import datetime, timedelta, timezone, date

import pytest

from optionsengine import (AnalyticsStatus, InvalidInputError, IVStatus, MarketAssumptions, OptionContract,
                           OptionType, PriceBasis, QualityPolicy, QuoteFlag, analyze_contract, assess_quote,
                           bsm_price, expiry_at_close, implied_carry_yield, time_to_expiry_years)

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 10, 1, 11, 0, tzinfo=IST)
EXPIRY = datetime(2026, 10, 8, 15, 30, tzinfo=IST)
ASSUME = MarketAssumptions(0.065, 0.012, rate_source="test fixture", dividend_source="test fixture")


def make(**kw):
    base = dict(underlying_symbol="NIFTY", underlying_price=24500.0, timestamp=NOW, expiry=EXPIRY,
                strike=24500.0, option_type="CE", data_source="unit-test", bid=100.0, ask=102.0, last=101.0,
                volume=1000, open_interest=5000, lot_size=75, quote_timestamp=NOW - timedelta(seconds=2),
                last_trade_timestamp=NOW - timedelta(seconds=30), contract_symbol="NSE:NIFTY2610824500CE")
    base.update(kw)
    return OptionContract(**base)


# ---- schema -----------------------------------------------------------------
def test_contract_is_immutable_and_normalizes_type():
    c = make()
    assert c.option_type is OptionType.CALL
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.bid = 1.0


@pytest.mark.parametrize("kw", [dict(strike=0), dict(strike=-1), dict(underlying_price=float("nan")),
                                dict(timestamp=datetime(2026, 10, 1, 11, 0)), dict(expiry=datetime(2026, 10, 8)),
                                dict(option_type="straddle"), dict(data_source=""), dict(lot_size=0),
                                dict(quote_timestamp=datetime(2026, 10, 1))])
def test_structurally_invalid_contract_rejected(kw):
    with pytest.raises(InvalidInputError):
        make(**kw)


def test_dubious_market_values_are_kept_raw_not_rejected():
    c = make(bid=105.0, ask=100.0, last=-3.0)   # crossed + negative: raw fact, flagged later
    assert (c.bid, c.ask, c.last) == (105.0, 100.0, -3.0)


def test_assumptions_require_provenance_and_finite_values():
    with pytest.raises(InvalidInputError):
        MarketAssumptions(0.065, 0.0, rate_source="", dividend_source="x")
    with pytest.raises(InvalidInputError):
        MarketAssumptions(float("nan"), 0.0, rate_source="x", dividend_source="x")
    with pytest.raises(TypeError):
        MarketAssumptions(0.065, 0.0)  # no silent defaults


# ---- quote quality ------------------------------------------------------------
def test_clean_quote_uses_mid():
    a = assess_quote(make())
    assert a.usable and a.price == 101.0 and a.basis is PriceBasis.MID and not a.blocking_flags


def test_zero_bid_blocks():
    a = assess_quote(make(bid=0.0, ask=0.5))
    assert QuoteFlag.ZERO_BID in a.flags and not a.usable and a.price is None


def test_missing_bid_with_ask_is_zero_bid_class():
    a = assess_quote(make(bid=None, ask=0.5))
    assert QuoteFlag.ZERO_BID in a.blocking_flags and not a.usable


def test_crossed_market_blocks():
    a = assess_quote(make(bid=105.0, ask=100.0))
    assert QuoteFlag.CROSSED_MARKET in a.blocking_flags and not a.usable


def test_locked_market_is_warning_only():
    a = assess_quote(make(bid=101.0, ask=101.0))
    assert QuoteFlag.LOCKED_MARKET in a.flags and a.usable


def test_one_sided_no_ask_blocks():
    a = assess_quote(make(ask=None))
    assert QuoteFlag.NO_ASK in a.blocking_flags and not a.usable


def test_stale_quote_blocks():
    a = assess_quote(make(quote_timestamp=NOW - timedelta(minutes=10)))
    assert QuoteFlag.STALE_QUOTE in a.blocking_flags and not a.usable


def test_missing_quote_timestamp_is_flagged_but_not_assumed_stale():
    a = assess_quote(make(quote_timestamp=None))
    assert QuoteFlag.NO_QUOTE_TIMESTAMP in a.flags and QuoteFlag.STALE_QUOTE not in a.flags and a.usable


def test_wide_spread_blocks():
    a = assess_quote(make(bid=1.0, ask=10.0))
    assert QuoteFlag.WIDE_SPREAD in a.blocking_flags and not a.usable


def test_negative_and_nan_prices_blocked():
    assert not assess_quote(make(bid=-1.0, ask=2.0)).usable
    a = assess_quote(make(bid=float("nan"), ask=2.0, last=None))
    assert QuoteFlag.NON_FINITE_PRICE in a.blocking_flags and not a.usable


def test_no_price_at_all():
    a = assess_quote(make(bid=None, ask=None, last=None))
    assert QuoteFlag.MISSING_PRICE in a.blocking_flags and a.price is None


def test_last_not_used_by_default_even_if_available():
    a = assess_quote(make(bid=0.0, ask=0.5, last=0.35))
    assert a.price is None and not a.usable


def test_last_fallback_when_allowed_and_fresh():
    pol = QualityPolicy(allow_last_fallback=True)
    a = assess_quote(make(bid=0.0, ask=0.5, last=0.35), pol)
    assert a.usable and a.basis is PriceBasis.LAST and a.price == 0.35 and QuoteFlag.ZERO_BID in a.flags


def test_last_fallback_blocked_when_stale():
    pol = QualityPolicy(allow_last_fallback=True)
    a = assess_quote(make(bid=0.0, ask=0.5, last=0.35, last_trade_timestamp=NOW - timedelta(hours=3)), pol)
    assert QuoteFlag.STALE_LAST in a.blocking_flags and not a.usable


def test_informational_flags():
    a = assess_quote(make(volume=0, open_interest=0, last=500.0, expiry_time_assumed=True, source_flags=("delayed",)))
    for f in (QuoteFlag.NO_VOLUME, QuoteFlag.ZERO_OPEN_INTEREST, QuoteFlag.LAST_OUTSIDE_MARKET,
              QuoteFlag.EXPIRY_TIME_ASSUMED, QuoteFlag.SOURCE_FLAGGED):
        assert f in a.flags
    assert a.usable  # all warning-level


def test_policy_blocking_set_is_configurable():
    pol = QualityPolicy(blocking=frozenset())
    assert assess_quote(make(bid=105.0, ask=100.0), pol).price is None  # crossed: no mid even if not blocking
    assert assess_quote(make(bid=0.0, ask=0.5, last=0.3), QualityPolicy(allow_last_fallback=True, blocking=frozenset())).usable


# ---- time to expiry ---------------------------------------------------------
def test_time_to_expiry_act365():
    assert time_to_expiry_years(NOW, EXPIRY) == pytest.approx((7 * 86400 + 4.5 * 3600) / (365 * 86400))
    assert time_to_expiry_years(EXPIRY, EXPIRY) == 0.0
    assert time_to_expiry_years(EXPIRY + timedelta(hours=1), EXPIRY) == 0.0


def test_expiry_at_close_helper_is_aware_ist():
    e = expiry_at_close(date(2026, 10, 8))
    assert e == EXPIRY and e.utcoffset() == timedelta(hours=5, minutes=30)


# ---- analytics pipeline -------------------------------------------------------
def theoretical_contract(sigma=0.14, strike=24500.0, ot="CE", spread=0.02):
    T = time_to_expiry_years(NOW, EXPIRY)
    mid = bsm_price(24500.0, strike, T, ASSUME.risk_free_rate, ASSUME.dividend_yield, sigma, ot.replace("CE", "call").replace("PE", "put"))
    return make(strike=strike, option_type=ot, bid=mid * (1 - spread / 2), ask=mid * (1 + spread / 2), last=mid)


def test_full_pipeline_recovers_vol_and_keeps_raw_untouched():
    raw = theoretical_contract(0.14)
    before = dataclasses.asdict(raw)
    out = analyze_contract(raw, ASSUME)
    assert dataclasses.asdict(raw) == before                      # raw not mutated
    assert out.status == AnalyticsStatus.OK and out.iv == pytest.approx(0.14, abs=1e-8)
    assert out.iv_result.kind == "market_implied"
    assert out.price_basis is PriceBasis.MID and out.model_price_at_iv == pytest.approx(out.price_used, abs=1e-6)
    assert out.greeks is not None and 0 < out.greeks.delta < 1 and out.greeks.theta < 0
    assert not hasattr(raw, "iv") and not hasattr(raw, "greeks")  # raw and calculated are separate types


def test_unreliable_quote_yields_no_iv_or_greeks():
    out = analyze_contract(make(bid=0.0, ask=0.5), ASSUME)
    assert out.status == AnalyticsStatus.UNRELIABLE_QUOTE and out.iv is None and out.iv_result is None and out.greeks is None


def test_solver_failure_surfaces_status_not_a_number():
    # Mid below forward-intrinsic for a deep ITM call: arbitrage-violating quote.
    out = analyze_contract(make(strike=20000.0, bid=100.0, ask=102.0, last=None), ASSUME)
    assert out.status == AnalyticsStatus.IV_FAILED and out.iv is None and out.greeks is None
    assert out.iv_result.status is IVStatus.BELOW_LOWER_BOUND


def test_expired_contract():
    out = analyze_contract(make(timestamp=EXPIRY + timedelta(minutes=1), quote_timestamp=EXPIRY + timedelta(minutes=1)), ASSUME)
    assert out.status == AnalyticsStatus.EXPIRED and out.iv is None and out.greeks is None


def test_put_pipeline():
    out = analyze_contract(theoretical_contract(0.2, strike=24000.0, ot="PE"), ASSUME)
    assert out.status == AnalyticsStatus.OK and out.iv == pytest.approx(0.2, abs=1e-6) and out.greeks.delta < 0


def test_pipeline_with_forward_consistent_carry_yield():
    S, F = 24500.0, 24562.0
    T = time_to_expiry_years(NOW, EXPIRY)
    q = implied_carry_yield(S, F, T, 0.065)
    a = MarketAssumptions(0.065, q, rate_source="fixture", dividend_source="implied from futures 24562")
    out = analyze_contract(theoretical_contract(0.14), a)
    assert out.status == AnalyticsStatus.OK and math.isfinite(out.iv)
