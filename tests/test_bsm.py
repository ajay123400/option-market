import math
import itertools
import pytest

from optionsengine import (InvalidInputError, bsm_price, call_price, put_price,
                           forward_intrinsic, implied_carry_yield, intrinsic_value, upper_bound)
from tests.helpers import quad_price

GRID = list(itertools.product(
    [80.0, 100.0, 120.0],          # S
    [90.0, 100.0, 125.0],          # K
    [0.02, 0.25, 1.0, 3.0],        # T
    [-0.01, 0.0, 0.05, 0.10],      # r (incl. negative)
    [0.0, 0.02, 0.07],             # q
    [0.08, 0.2, 0.6],              # sigma
))


@pytest.mark.parametrize("S,K,T,r,q,sigma", GRID[::7])
def test_put_call_parity(S, K, T, r, q, sigma):
    c, p = call_price(S, K, T, r, q, sigma), put_price(S, K, T, r, q, sigma)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10 * S)


# Published benchmarks -------------------------------------------------------
def test_hull_100_100_benchmark():
    # Widely published: S=K=100, T=1, r=5%, q=0, sigma=20%  ->  C=10.4506, P=5.5735
    assert call_price(100, 100, 1, 0.05, 0, 0.2) == pytest.approx(10.4506, abs=5e-5)
    assert put_price(100, 100, 1, 0.05, 0, 0.2) == pytest.approx(5.5735, abs=5e-5)


def test_hull_42_40_benchmark():
    # Hull, Options Futures & Other Derivatives, Ex. "S=42,K=40,r=10%,sigma=20%,T=0.5": c=4.76, p=0.81
    assert call_price(42, 40, 0.5, 0.10, 0, 0.2) == pytest.approx(4.76, abs=5e-3)
    assert put_price(42, 40, 0.5, 0.10, 0, 0.2) == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("S,K,T,r,q,sigma", GRID[3::11])
@pytest.mark.parametrize("is_call", [True, False])
def test_matches_independent_quadrature(S, K, T, r, q, sigma, is_call):
    ref = quad_price(S, K, T, r, q, sigma, is_call)
    assert bsm_price(S, K, T, r, q, sigma, "call" if is_call else "put") == pytest.approx(ref, rel=1e-9, abs=1e-10)


def test_dividend_yield_lowers_calls_raises_puts():
    args = (100, 100, 1, 0.05)
    assert call_price(*args, 0.03, 0.2) < call_price(*args, 0.0, 0.2)
    assert put_price(*args, 0.03, 0.2) > put_price(*args, 0.0, 0.2)


def test_q_equal_r_matches_zero_drift_symmetry():
    # With r == q the forward equals spot and call == put at K == S.
    assert call_price(100, 100, 0.5, 0.04, 0.04, 0.25) == pytest.approx(put_price(100, 100, 0.5, 0.04, 0.04, 0.25), rel=1e-12)


def test_black76_cross_check_with_existing_module():
    # Independent formula path: the repo's existing Black-76 (greeks.b76_price)
    # must agree with BSM when q is the carry-implied yield of the same forward.
    import greeks as legacy  # repo-root module, untouched by this work
    S, F, K, T, r, sigma = 24500.0, 24560.0, 24600.0, 7 / 365, 0.065, 0.14
    q = implied_carry_yield(S, F, T, r)
    for is_call in (True, False):
        mine = bsm_price(S, K, T, r, q, sigma, "call" if is_call else "put")
        assert mine == pytest.approx(legacy.b76_price(F, K, T, r, sigma, is_call), rel=1e-10)


def test_implied_carry_yield_roundtrip():
    S, T, r, q = 24500.0, 30 / 365, 0.065, 0.012
    F = S * math.exp((r - q) * T)
    assert implied_carry_yield(S, F, T, r) == pytest.approx(q, abs=1e-12)


# Deep ITM / OTM, expiry, limits ---------------------------------------------
def test_deep_itm_call_is_forward_intrinsic():
    S, K, T, r, q = 200.0, 50.0, 0.5, 0.05, 0.01
    assert call_price(S, K, T, r, q, 0.2) == pytest.approx(forward_intrinsic(S, K, T, r, q, "call"), rel=1e-12)


def test_deep_otm_is_tiny_nonnegative_and_finite():
    p = call_price(100, 1000, 0.1, 0.05, 0.0, 0.2)
    assert 0.0 <= p < 1e-12 and math.isfinite(p)
    p = put_price(100, 1.0, 0.1, 0.05, 0.0, 0.2)
    assert 0.0 <= p < 1e-12 and math.isfinite(p)


def test_price_within_no_arbitrage_bounds_everywhere():
    for S, K, T, r, q, sigma in GRID[::5]:
        for ot in ("call", "put"):
            p = bsm_price(S, K, T, r, q, sigma, ot)
            assert forward_intrinsic(S, K, T, r, q, ot) - 1e-12 <= p <= upper_bound(S, K, T, r, q, ot) + 1e-12


def test_expiry_returns_intrinsic():
    assert call_price(105, 100, 0.0, 0.05, 0.01, 0.3) == 5.0
    assert put_price(105, 100, 0.0, 0.05, 0.01, 0.3) == 0.0
    assert put_price(95, 100, 0.0, 0.05, 0.01, 0.3) == 5.0
    assert intrinsic_value(100, 100, "call") == 0.0


def test_zero_vol_is_discounted_forward_intrinsic():
    S, K, T, r, q = 100.0, 95.0, 1.0, 0.05, 0.0
    assert call_price(S, K, T, r, q, 0.0) == pytest.approx(S - K * math.exp(-r * T), rel=1e-14)
    assert put_price(S, K, T, r, q, 0.0) == 0.0


def test_vol_to_zero_converges_to_zero_vol_limit():
    assert call_price(100, 95, 1, 0.05, 0, 1e-6) == pytest.approx(call_price(100, 95, 1, 0.05, 0, 0.0), abs=1e-9)


@pytest.mark.parametrize("T", [1e-9, 1e-7, 1e-6, 1e-4])
def test_near_expiry_atm_matches_small_T_asymptotics(T):
    # ATM forward option: price ~ 0.3989 * S * sigma * sqrt(T) (r=q=0)
    S, sigma = 100.0, 0.2
    assert call_price(S, S, T, 0.0, 0.0, sigma) == pytest.approx(S * sigma * math.sqrt(T) / math.sqrt(2 * math.pi), rel=1e-4)


def test_near_expiry_otm_is_finite_and_zero():
    assert call_price(100, 101, 1e-9, 0.05, 0.0, 0.2) == 0.0


@pytest.mark.parametrize("bad", [
    dict(S=0), dict(S=-1), dict(K=0), dict(K=-5), dict(T=-0.1), dict(sigma=-0.2),
    dict(S=float("nan")), dict(K=float("inf")), dict(T=float("nan")), dict(r=float("nan")),
    dict(q=float("inf")), dict(sigma=float("nan")), dict(S=None), dict(S="100"), dict(sigma=True),
])
def test_invalid_inputs_raise(bad):
    kw = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2)
    kw.update(bad)
    with pytest.raises(InvalidInputError):
        call_price(**kw)


def test_invalid_option_type_raises():
    with pytest.raises(InvalidInputError):
        bsm_price(100, 100, 1, 0.05, 0, 0.2, "straddle")


def test_numerical_stability_extremes():
    for S, K, T, sigma in [(1e6, 1e-3 + 1, 5.0, 3.0), (1e-3, 1e6, 5.0, 0.01), (100, 100, 50.0, 5.0),
                           (100, 100, 1e-12, 1e-6), (24500, 24500, 1 / 365 / 24 / 60, 0.9)]:
        for ot in ("call", "put"):
            p = bsm_price(S, K, T, 0.07, 0.0, sigma, ot)
            assert math.isfinite(p) and p >= 0.0


def test_price_increasing_in_sigma():
    prices = [call_price(100, 110, 0.5, 0.05, 0.01, s) for s in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)]
    assert prices == sorted(prices) and len(set(prices)) == len(prices)
