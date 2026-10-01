"""Greeks vs central finite differences of the (independently tested) price,
plus structural identities and the documented units."""
import math
import pytest

from optionsengine import DegenerateInputError, InvalidInputError, bsm_price, greeks

CASES = [
    (100, 100, 1.0, 0.05, 0.00, 0.20),
    (100, 110, 0.5, 0.05, 0.02, 0.30),
    (100, 90, 0.25, 0.03, 0.01, 0.15),
    (24500, 24600, 7 / 365, 0.065, 0.012, 0.14),
    (100, 100, 2.0, -0.01, 0.03, 0.40),
    (100, 140, 1.0, 0.07, 0.0, 0.25),   # OTM
    (100, 60, 1.0, 0.07, 0.0, 0.25),    # ITM
]
OT = ("call", "put")


def P(S, K, T, r, q, s, ot):
    return bsm_price(S, K, T, r, q, s, ot)


@pytest.mark.parametrize("S,K,T,r,q,s", CASES)
@pytest.mark.parametrize("ot", OT)
def test_against_finite_differences(S, K, T, r, q, s, ot):
    g = greeks(S, K, T, r, q, s, ot).raw()  # calculus units
    h = 1e-4 * S
    fd_delta = (P(S + h, K, T, r, q, s, ot) - P(S - h, K, T, r, q, s, ot)) / (2 * h)
    fd_gamma = (P(S + h, K, T, r, q, s, ot) - 2 * P(S, K, T, r, q, s, ot) + P(S - h, K, T, r, q, s, ot)) / h ** 2
    hs = 1e-5
    fd_vega = (P(S, K, T, r, q, s + hs, ot) - P(S, K, T, r, q, s - hs, ot)) / (2 * hs)
    hr = 1e-6
    fd_rho = (P(S, K, T, r + hr, q, s, ot) - P(S, K, T, r - hr, q, s, ot)) / (2 * hr)
    hT = min(1e-6, T / 100)
    fd_theta = -(P(S, K, T + hT, r, q, s, ot) - P(S, K, T - hT, r, q, s, ot)) / (2 * hT)  # -dV/dT
    assert g.delta == pytest.approx(fd_delta, rel=1e-6, abs=1e-9)
    assert g.gamma == pytest.approx(fd_gamma, rel=1e-4, abs=1e-9)  # 2nd difference: looser by construction
    assert g.vega == pytest.approx(fd_vega, rel=1e-6, abs=1e-8)
    assert g.rho == pytest.approx(fd_rho, rel=1e-5, abs=1e-6)
    assert g.theta == pytest.approx(fd_theta, rel=1e-5, abs=1e-5)


def test_units_vega_rho_theta_scaling():
    g = greeks(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
    raw = g.raw()
    assert raw.vega == pytest.approx(g.vega * 100) and raw.rho == pytest.approx(g.rho * 100)
    assert raw.theta == pytest.approx(g.theta * 365)
    # Vega is per ONE vol point: price(21%) - price(20%) ~ vega
    up = P(100, 100, 1.0, 0.05, 0.0, 0.21, "call") - P(100, 100, 1.0, 0.05, 0.0, 0.20, "call")
    assert up == pytest.approx(g.vega, rel=2e-2)
    # Rho is per ONE percentage point of rate
    up = P(100, 100, 1.0, 0.06, 0.0, 0.2, "call") - P(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
    assert up == pytest.approx(g.rho, rel=2e-2)
    # Theta is per calendar DAY
    dec = P(100, 100, 1.0 - 1 / 365, 0.05, 0.0, 0.2, "call") - P(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
    assert dec == pytest.approx(g.theta, rel=1e-2)


def test_hull_100_100_greeks():
    g = greeks(100, 100, 1, 0.05, 0, 0.2, "call")
    assert g.delta == pytest.approx(0.6368, abs=5e-5)           # N(d1), d1=0.35
    assert g.gamma == pytest.approx(0.018762, abs=5e-7)
    assert g.raw().vega == pytest.approx(37.524, abs=5e-4)
    assert g.raw().theta == pytest.approx(-6.4140, abs=5e-4)    # per year
    assert g.raw().rho == pytest.approx(53.2325, abs=5e-4)
    p = greeks(100, 100, 1, 0.05, 0, 0.2, "put")
    assert p.delta == pytest.approx(-0.3632, abs=5e-5)
    assert p.raw().rho == pytest.approx(-41.8905, abs=5e-4)


@pytest.mark.parametrize("S,K,T,r,q,s", CASES)
def test_call_put_relationships(S, K, T, r, q, s):
    c, p = greeks(S, K, T, r, q, s, "call"), greeks(S, K, T, r, q, s, "put")
    assert c.delta - p.delta == pytest.approx(math.exp(-q * T), abs=1e-12)
    assert c.gamma == pytest.approx(p.gamma, rel=1e-12)
    assert c.vega == pytest.approx(p.vega, rel=1e-12)
    assert c.rho - p.rho == pytest.approx(K * T * math.exp(-r * T) / 100, rel=1e-10)
    # theta parity: d/dT of C - P = S e^{-qT} - K e^{-rT}
    lhs = (c.theta - p.theta) * 365
    rhs = q * S * math.exp(-q * T) - r * K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-9)
    assert 0 <= c.delta <= math.exp(-q * T) and -math.exp(-q * T) <= p.delta <= 0
    assert c.gamma > 0 and c.vega > 0


def test_deep_itm_otm_greeks():
    itm = greeks(100, 20, 0.5, 0.05, 0.0, 0.2, "call")
    assert itm.delta == pytest.approx(1.0, abs=1e-9) and itm.gamma < 1e-12 and itm.vega < 1e-9
    otm = greeks(100, 400, 0.5, 0.05, 0.0, 0.2, "call")
    assert otm.delta == pytest.approx(0.0, abs=1e-12) and otm.gamma >= 0 and math.isfinite(otm.theta)
    pitm = greeks(100, 400, 0.5, 0.05, 0.0, 0.2, "put")
    assert pitm.delta == pytest.approx(-math.exp(0), abs=1e-9)


def test_near_expiry_atm_gamma_large_but_finite():
    g = greeks(100, 100, 1e-6, 0.0, 0.0, 0.2, "call")
    assert math.isfinite(g.gamma) and g.gamma == pytest.approx(1 / (100 * 0.2 * math.sqrt(1e-6) * math.sqrt(2 * math.pi)), rel=1e-6)
    assert g.delta == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("kw", [dict(T=0.0), dict(sigma=0.0)])
def test_degenerate_raises(kw):
    a = dict(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="call")
    a.update(kw)
    with pytest.raises(DegenerateInputError):
        greeks(**a)


@pytest.mark.parametrize("kw", [dict(T=-1.0), dict(sigma=-0.1), dict(S=0), dict(K=-1), dict(S=float("nan"))])
def test_invalid_raises(kw):
    a = dict(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")
    a.update(kw)
    with pytest.raises(InvalidInputError):
        greeks(**a)
