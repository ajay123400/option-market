"""Regression tests for price-resolution (tick) IV uncertainty: flagged separately from convergence,
never rejecting an otherwise valid contract."""
import math

import pytest

from optionsengine import (InvalidInputError, IVStatus, SolverConfig, bsm_price, implied_volatility)

S, R, Q = 24500.0, 0.065, 0.0
MIN15 = 15 / 525600.0            # 15 minutes in years (ACT/365)
TICK = 0.05


def quote(K, T, sigma, ot="call", tick=TICK):
    """Theoretical price rounded onto the exchange tick grid, as a real quote would be."""
    return round(round(bsm_price(S, K, T, R, Q, sigma, ot) / tick) * tick, 2)


def iv(price, K, T, ot="call", cfg=SolverConfig()):
    return implied_volatility(price, S, K, T, R, Q, ot, cfg)


def test_default_resolution_is_the_nse_tick():
    assert SolverConfig().price_resolution == 0.05


def test_atm_liquid_option_is_resolved_and_reliable():
    T = 7 / 365
    res = iv(quote(24500, T, 0.14), 24500, T)
    d = res.diagnostics
    assert res.converged and not d.resolution_limited and res.reliable
    assert d.price_resolution == 0.05 and d.iv_interval[0] < res.iv < d.iv_interval[1]
    assert d.resolution_uncertainty < 1e-3          # well below 0.1 vol point


def test_true_vol_lies_inside_the_reported_interval_across_the_grid():
    for sigma in (0.08, 0.14, 0.3, 0.6):
        for K in (24000, 24400, 24500, 24600, 25000):
            for T in (15 / 525600, 1 / 365, 7 / 365, 30 / 365):
                for ot in ("call", "put"):
                    p = quote(K, T, sigma, ot)
                    res = iv(p, K, T, ot)
                    if not res.converged:
                        continue
                    lo, hi = res.diagnostics.iv_interval
                    assert lo - 1e-9 <= sigma <= hi + 1e-9, (sigma, K, T, ot, p, lo, hi)


def test_near_expiry_itm_option_is_flagged_but_still_returned():
    """200-pt ITM call 15 minutes before expiry: 0.05 of time value -> IV is 'valid' but unresolved."""
    T = MIN15
    p = quote(24300, T, 0.14)
    assert p == pytest.approx(200.05, abs=1e-9)
    res = iv(p, 24300, T)
    d = res.diagnostics
    assert res.status is IVStatus.CONVERGED and res.iv is not None          # NOT rejected
    assert not d.ill_conditioned                                            # solver itself is fine ...
    assert d.resolution_limited and not res.reliable                        # ... resolution is not
    assert d.iv_interval[0] == 0.0 and d.iv_interval[1] > 0.4               # band spans 0 % -> >40 %
    assert d.resolution_uncertainty > 0.3 and "RESOLUTION-LIMITED" in d.message


def test_uncertainty_grows_as_expiry_approaches():
    unc = []
    for T in (30 / 365, 7 / 365, 1 / 365, 60 / 525600, MIN15):
        res = iv(quote(24300, T, 0.14), 24300, T)
        assert res.converged
        unc.append(res.diagnostics.resolution_uncertainty)
    assert unc == sorted(unc) and unc[-1] > 100 * unc[0]


def test_flag_is_independent_of_convergence_and_of_solver_tolerance():
    T = MIN15
    p = quote(24300, T, 0.14)
    tight = iv(p, 24300, T, cfg=SolverConfig(price_tol=1e-12, price_rel_tol=1e-14))
    assert tight.converged and tight.diagnostics.resolution_limited          # tighter tolerance does not help
    off = iv(p, 24300, T, cfg=SolverConfig(price_resolution=None))
    assert off.converged and off.iv == pytest.approx(tight.iv, abs=1e-6)     # same IV
    assert off.diagnostics.iv_interval is None and not off.diagnostics.resolution_limited


def test_threshold_is_configurable():
    T = 1 / 365
    p = quote(24300, T, 0.14)
    strict = iv(p, 24300, T, cfg=SolverConfig(max_resolution_uncertainty=1e-6))
    loose = iv(p, 24300, T, cfg=SolverConfig(max_resolution_uncertainty=1.0))
    assert strict.diagnostics.resolution_limited and not loose.diagnostics.resolution_limited
    assert strict.iv == loose.iv


def test_otm_option_that_rounds_to_zero_has_no_iv_not_a_fake_one():
    T = MIN15
    p = quote(24745, T, 0.14)
    assert p == 0.0
    res = iv(p, 24745, T)
    assert res.status is IVStatus.PRICE_NOT_POSITIVE and res.iv is None


def test_cheap_otm_option_one_day_out_is_resolved_negative_control():
    # price of a single tick, 1 day to expiry, ~1.6 % OTM: vega is large enough that +/-half-tick is only ~0.6 vol point
    res = iv(0.05, 24900, 1 / 365)
    d = res.diagnostics
    assert res.converged and not d.resolution_limited and res.reliable
    assert 0.003 < d.resolution_uncertainty < 0.01      # visible but below the 1-vol-point default threshold


def test_near_expiry_itm_call_50_points_is_still_limited():
    T = MIN15
    p = quote(24450, T, 0.14)
    assert p == pytest.approx(50.05, abs=1e-9)
    res = iv(p, 24450, T)
    assert res.converged and res.diagnostics.resolution_limited and res.diagnostics.iv_interval[0] == 0.0


def test_interval_equals_solved_iv_at_perturbed_prices():
    T = 7 / 365
    p = quote(24800, T, 0.2)
    res = iv(p, 24800, T)
    lo, hi = res.diagnostics.iv_interval
    assert lo == pytest.approx(iv(p - 0.025, 24800, T).iv, abs=1e-12)
    assert hi == pytest.approx(iv(p + 0.025, 24800, T).iv, abs=1e-12)


def test_put_side_is_flagged_too():
    # Spot nudged by 2 paise so that tick-rounding the 200-pt ITM put does not dip below discounted intrinsic
    # (rounding a price down onto the tick grid can itself create BELOW_LOWER_BOUND -- a real quantisation effect).
    S2, K, T = 24500.02, 24700, MIN15
    p = round(round(bsm_price(S2, K, T, R, Q, 0.14, "put") / TICK) * TICK, 2)
    res = implied_volatility(p, S2, K, T, R, Q, "put")
    assert res.converged and res.diagnostics.resolution_limited and not res.reliable


def test_rounding_to_tick_can_push_an_itm_put_below_intrinsic():
    T, K = MIN15, 24700
    p = quote(K, T, 0.14, "put")            # 199.95 < discounted intrinsic 199.954
    assert iv(p, K, T, "put").status is IVStatus.BELOW_LOWER_BOUND


def test_unconverged_results_carry_no_resolution_claims():
    res = iv(0.0, 24500, 7 / 365)
    d = res.diagnostics
    assert not res.converged and d.iv_interval is None and d.resolution_uncertainty is None and not d.resolution_limited


@pytest.mark.parametrize("bad", [-0.05, float("nan"), float("inf"), "0.05", True])
def test_invalid_price_resolution_rejected(bad):
    with pytest.raises(InvalidInputError):
        SolverConfig(price_resolution=bad)


@pytest.mark.parametrize("res_", [None, 0, 0.0])
def test_resolution_can_be_disabled(res_):
    r = iv(quote(24500, 7 / 365, 0.14), 24500, 7 / 365, cfg=SolverConfig(price_resolution=res_))
    assert r.converged and r.diagnostics.iv_interval is None and r.reliable


def test_resolution_flag_does_not_change_phase1_iv_values():
    # regression: adding diagnostics must not move the IV itself
    p = bsm_price(24500, 24600, 7 / 365, R, Q, 0.17, "call")
    a = iv(p, 24600, 7 / 365)
    b = iv(p, 24600, 7 / 365, cfg=SolverConfig(price_resolution=None))
    assert a.iv == b.iv and a.diagnostics.iterations == b.diagnostics.iterations
