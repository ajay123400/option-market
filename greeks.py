"""greeks.py -- Black-Scholes implied volatility + Greeks (Delta, Gamma,
Theta, Vega) computed from the option chain's own LTP, since Fyers'
options-chain-v3 doesn't expose IV/Greeks directly (confirmed earlier --
only LTP, OI, volume, bid/ask). These are OUR OWN computed approximation,
not exchange-verified figures: they inherit every Black-Scholes assumption
(European exercise, constant volatility or a flat risk-free rate) and will
drift from reality for illiquid strikes with stale/wide-spread LTPs, or in
the last few minutes before expiry where T -> 0 makes the model singular.

Risk-free rate is a hardcoded approximation (India's short-term T-bill
ballpark), not live data -- another simplification worth knowing about.
"""
import math
from statistics import NormalDist

RISK_FREE_RATE = 0.065  # ~6.5%, approximate -- not fetched live
_N = NormalDist().cdf
_SQRT_2PI = math.sqrt(2 * math.pi)


def _pdf(x):
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def bs_price(S, K, T, r, sigma, is_call):
    """Black-Scholes theoretical price for a European option."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * _N(d1) - K * math.exp(-r * T) * _N(d2)
    return K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)


def implied_vol(price, S, K, T, r, is_call, tol=1e-4, max_iter=100):
    """Bisection search for the sigma that reproduces `price`. Bisection
    (not Newton-Raphson) is used deliberately -- BS price is monotonically
    increasing in sigma (vega >= 0 always), so as long as `price` falls
    within [price at sigma=lo, price at sigma=hi] the search is guaranteed
    to converge; Newton-Raphson can diverge or blow up near-zero-vega deep
    ITM/OTM strikes instead. Returns None if T/price/inputs are invalid or
    `price` sits outside what's achievable in the [1bp, 500%] vol range
    (e.g. a stale/crossed quote) -- callers must treat None as "can't compute
    Greeks for this contract right now", not as zero IV.
    """
    if price is None or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    lo, hi = 1e-4, 5.0
    price_lo = bs_price(S, K, T, r, lo, is_call)
    price_hi = bs_price(S, K, T, r, hi, is_call)
    if price < price_lo or price > price_hi:
        return None
    mid = (lo + hi) / 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p_mid = bs_price(S, K, T, r, mid, is_call)
        if abs(p_mid - price) < tol:
            return mid
        if p_mid < price:
            lo = mid
        else:
            hi = mid
    return mid


def compute_greeks(S, K, T, r, sigma, is_call):
    """Delta, Gamma, Theta (per calendar day), Vega (per 1% IV move) --
    standard Black-Scholes closed forms. Returns None if T/sigma/inputs
    are degenerate (e.g. right at/after expiry)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = _pdf(d1)
    gamma = pdf_d1 / (S * sigma * sqrtT)
    vega = S * pdf_d1 * sqrtT / 100  # per 1 percentage-point move in IV
    if is_call:
        delta = _N(d1)
        theta_year = -(S * pdf_d1 * sigma) / (2 * sqrtT) - r * K * math.exp(-r * T) * _N(d2)
    else:
        delta = _N(d1) - 1
        theta_year = -(S * pdf_d1 * sigma) / (2 * sqrtT) + r * K * math.exp(-r * T) * _N(-d2)
    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta_year / 365, 4),  # per calendar day
        "vega": round(vega, 4),
    }


def synthetic_forward(quotes):
    """quotes: iterable of (strike, call_price, put_price). Returns the
    median put-call-parity-implied forward (F ~= C - P + K) across those
    strikes, or None if none have both a call and a put price.

    Index options price off the futures/forward, not the raw cash index --
    for NIFTY the spot-futures basis commonly runs 40-80+ points. Feeding
    the raw index ticker in as S systematically skews call vs put implied
    vol at the very same strike (confirmed live: a consistent ~60pt gap
    between the parity-implied forward and the raw index spot, present at
    every strike -- not scattered noise, so not just illiquid-strike
    staleness, and not a pricing-formula bug either; see this module's
    __main__ self-test for that). Using the median of several near-ATM
    strikes' parity estimates instead of a single strike is robust to any
    one strike having a stale/crossed quote."""
    estimates = sorted(c - p + k for k, c, p in quotes if c and p)
    if not estimates:
        return None
    n = len(estimates)
    mid = n // 2
    return estimates[mid] if n % 2 else (estimates[mid - 1] + estimates[mid]) / 2


def option_greeks(price, S, K, T, r, is_call):
    """IV + Greeks in one call; None if IV can't be solved for this
    contract (e.g. LTP missing/stale, or expiry has effectively passed).
    SPOT-based Black-Scholes -- S must be the raw underlying. Callers that
    have a FORWARD (every caller in this app) must use option_greeks_fwd()."""
    iv = implied_vol(price, S, K, T, r, is_call)
    if iv is None:
        return None
    g = compute_greeks(S, K, T, r, iv, is_call)
    if g is None:
        return None
    g["iv"] = round(iv * 100, 2)  # as a percentage, the usual convention
    return g


# ---- Black-76 (forward-based) --------------------------------------------
# Every caller in this app prices off the put-call-parity FORWARD, not the
# cash index. Feeding a forward into the spot formula above with r > 0
# grows it a second time by e^{rT} (carry counted twice) -- verified to
# split CE vs PE IV at the SAME strike by ~2 vol points ATM at 7 DTE and
# ~5 points at 28 DTE. Black-76 uses F directly and r only for discounting,
# so CE and PE at one strike solve to the same IV.

def b76_price(F, K, T, r, sigma, is_call):
    """Black-76 price of a European option on forward F."""
    df = math.exp(-r * T) if T > 0 else 1.0
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return df * max(0.0, (F - K) if is_call else (K - F))
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return df * (F * _N(d1) - K * _N(d2))
    return df * (K * _N(-d2) - F * _N(-d1))


def implied_vol_fwd(price, F, K, T, r, is_call, tol=1e-4, max_iter=100):
    """implied_vol() for Black-76 -- same bisection and same None contract."""
    if price is None or price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return None
    lo, hi = 1e-4, 5.0
    if price < b76_price(F, K, T, r, lo, is_call) or price > b76_price(F, K, T, r, hi, is_call):
        return None
    mid = (lo + hi) / 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p_mid = b76_price(F, K, T, r, mid, is_call)
        if abs(p_mid - price) < tol:
            return mid
        if p_mid < price:
            lo = mid
        else:
            hi = mid
    return mid


def compute_greeks_fwd(F, K, T, r, sigma, is_call):
    """Black-76 Greeks, same units as compute_greeks(): delta (w.r.t. the
    underlying, per 1 point), gamma, theta per calendar day, vega per 1
    vol point -- all per ONE unit of the option (multiply by lot size and
    lots for a position)."""
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return None
    sqrtT = math.sqrt(T)
    df = math.exp(-r * T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    pdf_d1 = _pdf(d1)
    delta = df * _N(d1) if is_call else -df * _N(-d1)
    gamma = df * pdf_d1 / (F * sigma * sqrtT)
    vega = df * F * pdf_d1 * sqrtT / 100
    price = b76_price(F, K, T, r, sigma, is_call)
    theta_year = -df * F * pdf_d1 * sigma / (2 * sqrtT) + r * price
    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta_year / 365, 4),
        "vega": round(vega, 4),
    }


def option_greeks_fwd(price, F, K, T, r, is_call):
    """option_greeks() for a forward F (Black-76)."""
    iv = implied_vol_fwd(price, F, K, T, r, is_call)
    if iv is None:
        return None
    g = compute_greeks_fwd(F, K, T, r, iv, is_call)
    if g is None:
        return None
    g["iv"] = round(iv * 100, 2)
    return g


if __name__ == "__main__":
    # Cross-check against the standard Hull textbook reference case:
    # S=100, K=100, T=1yr, r=5%, sigma=20% -> call ~10.45, put ~5.57,
    # delta_call ~0.6368, and put-call parity (C - P = S - K*e^-rT) must hold.
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    call_price = bs_price(S, K, T, r, sigma, True)
    put_price = bs_price(S, K, T, r, sigma, False)
    print(f"call={call_price:.4f} (expect ~10.4506)  put={put_price:.4f} (expect ~5.5735)")
    parity_lhs = call_price - put_price
    parity_rhs = S - K * math.exp(-r * T)
    print(f"put-call parity: C-P={parity_lhs:.4f} vs S-K*e^-rT={parity_rhs:.4f}")
    assert abs(parity_lhs - parity_rhs) < 1e-6, "put-call parity violated -- pricing formula is wrong"
    assert abs(call_price - 10.4506) < 1e-3, "call price doesn't match known reference"
    assert abs(put_price - 5.5735) < 1e-3, "put price doesn't match known reference"

    call_greeks = compute_greeks(S, K, T, r, sigma, True)
    put_greeks = compute_greeks(S, K, T, r, sigma, False)
    print("call greeks:", call_greeks, "(expect delta ~0.6368)")
    print("put greeks:", put_greeks, "(expect delta ~-0.3632)")
    assert abs(call_greeks["delta"] - 0.6368) < 1e-3
    assert abs(put_greeks["delta"] - (-0.3632)) < 1e-3
    # Gamma and Vega are identical for call and put at the same strike.
    assert abs(call_greeks["gamma"] - put_greeks["gamma"]) < 1e-9
    assert abs(call_greeks["vega"] - put_greeks["vega"]) < 1e-9

    # Round-trip: feed the known price back through implied_vol and recover sigma.
    recovered_iv = implied_vol(call_price, S, K, T, r, True)
    print(f"recovered IV: {recovered_iv:.4f} (expect ~0.2000)")
    assert abs(recovered_iv - sigma) < 1e-3, "IV solver did not round-trip correctly"

    # Black-76: a CE and PE at the same strike, priced off the same forward,
    # must solve to the SAME IV (the regression the spot formula failed).
    F, r76 = 25060.0, 0.065
    for days in (1, 7, 28):
        T76 = days / 365
        for K76 in (24800.0, 25050.0, 25300.0):
            c = b76_price(F, K76, T76, r76, 0.13, True)
            p = b76_price(F, K76, T76, r76, 0.13, False)
            fwd = synthetic_forward([(K76, c, p)])
            iv_c = implied_vol_fwd(c, fwd, K76, T76, r76, True)
            iv_p = implied_vol_fwd(p, fwd, K76, T76, r76, False)
            assert abs(iv_c - 0.13) < 2e-3 and abs(iv_p - 0.13) < 2e-3, (days, K76, iv_c, iv_p)
    print("Black-76 CE/PE IV parity check passed.")

    print("All cross-checks passed.")
