"""Stage 2A core: smile construction from one snapshot (synthetic, hand-verifiable inputs)."""
import math

import pytest

from optionsengine import (ForwardConfig, ForwardStatus, OptionType, SolverConfig, bsm_price, implied_carry_yield)
from optionsengine.surface import (ATM_BRACKET_TOO_WIDE, ATM_NO_BRACKET, ATM_POINT_UNUSABLE, FORWARD_LOW_CONFIDENCE,
                                   GROUP_EXPIRED, GROUP_FORWARD_UNAVAILABLE, ILL_CONDITIONED, NEVER_TRADED,
                                   RESOLUTION_LIMITED, RRBF_NOT_BRACKETED, RRBF_SHORT_EXPIRY, STALE, ZERO_PRICE,
                                   OptionQuote, SurfaceConfig, build_smile, forward_delta_abs)

S = F = 24500.0
R = 0.065
T14 = 14 / 365
STRIKES = list(range(22800, 26201, 100))


def chain(sigma_fn, T=T14, F_=F, strikes=STRIKES, age=1.0, tick=None, skip=()):
    """Both sides of every strike priced by BSM at sigma_fn(ln K/F). tick=None keeps exact prices."""
    q = implied_carry_yield(S, F_, T, R)
    out = []
    for k in strikes:
        sig = sigma_fn(math.log(k / F_))
        for kind in ("call", "put"):
            if (k, kind) in skip:
                continue
            p = bsm_price(S, k, T, R, q, sig, kind)
            if tick:
                p = round(round(p / tick) * tick, 2)
            out.append(OptionQuote(float(k), kind, p, age))
    return out


def smile(quotes, T=T14, **cfg):
    return build_smile(quotes, S, T, R, SurfaceConfig(**cfg) if cfg else SurfaceConfig())


def by_strike(res):
    return {(p.strike, p.kind.value): p for p in res.points}


# ---------------------------------------------------------------- flat smile (hand-verifiable)
def test_flat_smile_recovers_sigma_atm_and_zero_rr_bf():
    res = smile(chain(lambda x: 0.18))
    assert res.primary and res.forward.status is ForwardStatus.OK
    assert res.forward_used == pytest.approx(F, abs=1e-6)
    assert res.atm.iv == pytest.approx(0.18, abs=1e-8) and res.atm.n_unreliable_inputs == 0
    assert res.rr_bf is not None
    assert res.rr_bf.rr25 == pytest.approx(0.0, abs=1e-8) and res.rr_bf.bf25 == pytest.approx(0.0, abs=1e-8)
    assert res.rr_bf.iv25_call == pytest.approx(0.18, abs=1e-8)


def test_otm_selection_rule_and_itm_count():
    res = smile(chain(lambda x: 0.18))
    pts = by_strike(res)
    assert all((p.kind is OptionType.CALL) == (p.strike >= F) for p in res.points)
    assert len(res.points) == len(STRIKES) and res.n_itm_side_excluded == len(STRIKES)
    assert pts[(24500.0, "call")].log_moneyness == 0.0                   # K == F counts as a call
    assert pts[(24400.0, "put")].log_moneyness == pytest.approx(math.log(24400 / F))


def test_log_moneyness_is_ln_k_over_forward_not_spot():
    f2 = 24560.0
    res = smile(chain(lambda x: 0.18, F_=f2))
    assert res.forward_used == pytest.approx(f2, abs=1e-6)
    p = by_strike(res)[(24800.0, "call")]
    assert p.log_moneyness == pytest.approx(math.log(24800 / f2), abs=1e-9)


# ---------------------------------------------------------------- linear-in-x skew: ATM interpolation is exact
def test_atm_interpolation_exact_for_linear_smile_with_forward_between_strikes():
    f2 = 24530.0                                           # F sits between 24500 and 24600
    sig = lambda x: 0.17 - 1.5 * x                         # linear in ln(K/F)
    res = smile(chain(sig, F_=f2))
    assert res.atm.strike_low == 24500.0 and res.atm.strike_high == 24600.0
    assert res.atm.iv == pytest.approx(0.17, abs=1e-7)    # sigma(0) exactly, by linear interpolation in x


def test_negative_skew_gives_negative_rr_and_hand_computed_butterfly():
    sig = lambda x: 0.16 - 0.5 * x + 3.0 * x * x
    res = smile(chain(sig))
    rr = res.rr_bf
    assert rr is not None and rr.rr25 < 0 and rr.bf25 > 0
    # hand check of the algorithm: re-derive from the two bracketing pairs
    pts = by_strike(res)
    for (ka, kb), kind, iv25 in ((rr.call_bracket, "call", rr.iv25_call), (rr.put_bracket, "put", rr.iv25_put)):
        a, b = pts[(ka, kind)], pts[(kb, kind)]
        assert min(a.delta_fwd, b.delta_fwd) <= 0.25 <= max(a.delta_fwd, b.delta_fwd)
        w = (0.25 - a.delta_fwd) / (b.delta_fwd - a.delta_fwd)
        assert iv25 == pytest.approx(a.iv + w * (b.iv - a.iv), abs=1e-12)
    assert rr.rr25 == pytest.approx(rr.iv25_call - rr.iv25_put) and rr.bf25 == pytest.approx(
        0.5 * (rr.iv25_call + rr.iv25_put) - res.atm.iv)
    # true 25-delta vols of the generating smile are close (interpolation error only)
    assert rr.rr25 == pytest.approx(-0.5 * (rr_x(res, True) - rr_x(res, False)), abs=3e-3)


def rr_x(res, call):
    """ln(K/F) of the exact 25-delta strike for the quadratic smile, found by bisection."""
    sig = lambda x: 0.16 - 0.5 * x + 3.0 * x * x
    lo, hi = (0.0, 0.2) if call else (-0.2, 0.0)
    for _ in range(80):
        m = 0.5 * (lo + hi)
        d = forward_delta_abs(F, F * math.exp(m), T14, sig(m), OptionType.CALL if call else OptionType.PUT)
        if (d > 0.25) == call:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def test_forward_delta_matches_engine_spot_delta():
    from optionsengine import greeks
    q = implied_carry_yield(S, 24560.0, T14, R)
    g = greeks(S, 24700, T14, R, q, 0.2, "call")
    assert forward_delta_abs(24560.0, 24700, T14, 0.2, OptionType.CALL) == pytest.approx(g.delta * math.exp(q * T14), rel=1e-12)
    gp = greeks(S, 24300, T14, R, q, 0.2, "put")
    assert forward_delta_abs(24560.0, 24300, T14, 0.2, OptionType.PUT) == pytest.approx(-gp.delta * math.exp(q * T14), rel=1e-12)


# ---------------------------------------------------------------- freshness / exclusions
def test_stale_prices_are_excluded_not_used_and_counted():
    q = [OptionQuote(x.strike, x.kind, x.price, 30.0 if x.strike == 24700 else 1.0) for x in chain(lambda x: 0.18)]
    res = smile(q)
    p = by_strike(res)[(24700.0, "call")]
    assert p.exclusion == STALE and not p.used and p.iv is None
    assert res.exclusion_counts[STALE] == 1


def test_never_traded_and_zero_price_reasons():
    qs = chain(lambda x: 0.18)
    qs = [OptionQuote(x.strike, x.kind, None, None) if (x.strike, x.kind.value) == (24800.0, "call") else
          OptionQuote(x.strike, x.kind, 0.0, 1.0) if (x.strike, x.kind.value) == (24900.0, "call") else x for x in qs]
    res = smile(qs)
    pts = by_strike(res)
    assert pts[(24800.0, "call")].exclusion == NEVER_TRADED and pts[(24900.0, "call")].exclusion == ZERO_PRICE


def test_every_point_has_exactly_one_reason_or_is_used():
    res = smile(chain(lambda x: 0.18, tick=0.05))
    assert all((p.used and p.exclusion is None) or (not p.used and p.exclusion) for p in res.points)
    assert sum(res.exclusion_counts.values()) + sum(p.used for p in res.points) == len(res.points)


def test_resolution_limited_points_are_labelled_and_not_used():
    # expiry-eve ITM-ish wing priced on the tick grid is resolution-limited at low T; ATM stays usable
    T = 20 / 525600                                        # 20 minutes to expiry
    res = smile(chain(lambda x: 0.14, T=T, strikes=list(range(24300, 24701, 50)), tick=0.05), T=T)
    lim = [p for p in res.points if p.resolution_limited]
    assert lim and all(p.exclusion == RESOLUTION_LIMITED and not p.used and p.iv is not None for p in lim)
    assert all(p.reliable == (not p.resolution_limited and not p.ill_conditioned) for p in res.points if p.iv is not None)


def test_atm_requires_used_bracketing_points_and_loose_atm_is_labelled():
    T = 20 / 525600
    res = smile(chain(lambda x: 0.14, T=T, strikes=[24450, 24500, 24550], tick=0.05), T=T)
    # whichever of the two ATM neighbours is resolution-limited, strict ATM must refuse; loose ATM must say how many inputs are unreliable
    if res.atm is None:
        assert res.atm_reason == ATM_POINT_UNUSABLE
        assert res.atm_loose is None or res.atm_loose.n_unreliable_inputs >= 1
    else:
        assert res.atm.n_unreliable_inputs == 0


# ---------------------------------------------------------------- forward quality separation
def test_low_confidence_forward_gives_labelled_points_and_no_metrics():
    noisy = []
    for x in chain(lambda x: 0.18):
        bump = {24400: 25, 24450: -15, 24500: 5, 24550: -5, 24600: 15, 24650: -25}.get(int(x.strike), 0)
        noisy.append(OptionQuote(x.strike, x.kind, x.price + (bump * 0.5 if x.kind is OptionType.CALL else -bump * 0.5), x.age_minutes))
    res = smile(noisy)
    assert res.forward.status is ForwardStatus.LOW_CONFIDENCE and not res.primary
    assert res.forward.forward is None and res.forward_used == res.forward.candidate_forward
    assert res.points and all(p.exclusion == FORWARD_LOW_CONFIDENCE and not p.used for p in res.points)
    assert res.atm is None and res.atm_loose is None and res.rr_bf is None
    assert res.atm_reason == FORWARD_LOW_CONFIDENCE and res.rr_bf_reason == "forward_not_ok"   # the REAL reason is reported
    assert any(p.iv is not None for p in res.points)       # IVs still visible for inspection


def test_stale_quotes_never_enter_the_forward_estimate():
    # three of the six strikes nearest the money carry STALE call prices that are 20 pts off; they must not be
    # allowed to contaminate the parity forward (if they were, dispersion would exceed the gate)
    stale_strikes = {24400.0, 24500.0, 24600.0}
    qs = []
    for x in chain(lambda x: 0.18):
        if x.strike in stale_strikes and x.kind is OptionType.CALL:
            qs.append(OptionQuote(x.strike, x.kind, x.price + 20.0, 45.0))
        else:
            qs.append(x)
    res = smile(qs)
    assert res.forward.status is ForwardStatus.OK and res.forward_used == pytest.approx(F, abs=1e-6)
    assert all(by_strike(res)[(k, "call" if k >= F else "put")].exclusion in (None, STALE) for k in stale_strikes)
    assert by_strike(res)[(24500.0, "call")].exclusion == STALE


def test_forward_unavailable_yields_no_smile_and_a_reason():
    res = smile([x for x in chain(lambda x: 0.18) if x.kind is OptionType.CALL])      # no puts at all
    assert res.group_reason == GROUP_FORWARD_UNAVAILABLE and res.points == () and res.atm is None


def test_expired_group():
    res = smile(chain(lambda x: 0.18), T=0.0)
    assert res.group_reason == GROUP_EXPIRED and res.points == ()


def test_threshold_override_is_respected_and_default_unchanged():
    assert SurfaceConfig().forward == ForwardConfig()          # approved default untouched
    grid = list(range(24300, 24701, 50))
    nearest6 = sorted(grid, key=lambda k: (abs(k - F), k))[:6]
    noise = {k: 20.0 * (i / 5 - 0.5) for i, k in enumerate(sorted(nearest6))}      # evenly spread, 20-pt range
    mod = []
    for x in chain(lambda x: 0.18, strikes=grid):
        b = noise.get(int(x.strike), 0.0) * 0.5
        mod.append(OptionQuote(x.strike, x.kind, x.price + (b if x.kind is OptionType.CALL else -b), x.age_minutes))
    default = smile(mod)                                       # 20 pts > 12.25-pt default limit
    lenient = build_smile(mod, S, T14, R, SurfaceConfig(forward=ForwardConfig(max_dispersion_bps=12.0)))
    assert default.forward.status is ForwardStatus.LOW_CONFIDENCE and lenient.forward.status is ForwardStatus.OK


# ---------------------------------------------------------------- no extrapolation / no bridging
def test_no_extrapolation_when_forward_outside_listed_strikes():
    res = smile(chain(lambda x: 0.18, strikes=[24600, 24700, 24800, 24900]))   # all strikes above F
    assert res.atm is None and res.atm_reason == ATM_NO_BRACKET


def test_atm_not_bridged_across_wide_gap():
    res = smile(chain(lambda x: 0.18, strikes=[23800, 24000, 25000, 25200, 25400]))
    assert res.atm is None and res.atm_reason == ATM_BRACKET_TOO_WIDE


def test_missing_wing_means_missing_rr_bf_not_a_made_up_one():
    res = smile(chain(lambda x: 0.18, strikes=list(range(24300, 24701, 50))))  # +-200 pts: 25 delta (~300 pts) not bracketed
    assert res.atm is not None and res.rr_bf is None and res.rr_bf_reason == RRBF_NOT_BRACKETED


def test_unreliable_neighbour_is_not_bridged():
    qs = chain(lambda x: 0.18)
    # make the strike just inside the 25-delta bracket stale on the call side -> that pair cannot be used
    tgt = by_strike(smile(qs)).get
    base = smile(qs).rr_bf
    ka, kb = base.call_bracket
    qs2 = [OptionQuote(x.strike, x.kind, x.price, 99.0 if (x.strike, x.kind.value) == (ka, "call") else x.age_minutes) for x in qs]
    res = smile(qs2)
    assert res.rr_bf is None or res.rr_bf.call_bracket != (ka, kb)
    assert res.rr_bf is None or res.rr_bf.call_bracket[0] != ka


def test_rr_bf_not_produced_within_one_day():
    T = 0.8 / 365
    res = smile(chain(lambda x: 0.18, T=T, strikes=list(range(24200, 24801, 50))), T=T)
    assert res.rr_bf is None and res.rr_bf_reason == RRBF_SHORT_EXPIRY
    exact = smile(chain(lambda x: 0.18, T=1.0 / 365), T=1.0 / 365)       # exactly 1 day: still excluded (> 1 day required)
    assert exact.rr_bf is None and exact.rr_bf_reason == RRBF_SHORT_EXPIRY


def test_rr_bf_min_days_is_configurable():
    T = 0.8 / 365
    res = smile(chain(lambda x: 0.18, T=T, strikes=list(range(24000, 25001, 50))), T=T, rr_bf_min_days=0.5)
    assert res.rr_bf_reason != RRBF_SHORT_EXPIRY


def test_look_ahead_free_pure_function():
    qs = chain(lambda x: 0.18)
    a, b = smile(qs), smile(list(reversed(qs)))
    assert a.atm == b.atm and a.rr_bf == b.rr_bf and a.forward_used == b.forward_used


def test_invalid_config_rejected():
    with pytest.raises(Exception):
        SurfaceConfig(max_age_minutes=0)
    with pytest.raises(Exception):
        SurfaceConfig(target_delta=0.6)
