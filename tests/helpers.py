"""Independent reference calculations for the tests. Nothing here imports
optionsengine, so these are a genuine cross-check of the closed forms:
the price is obtained by numerically integrating the discounted risk-neutral
payoff over the lognormal terminal distribution (Simpson's rule), which uses
only exp/log/sqrt -- no normal CDF, no closed-form BSM."""
import math


def quad_price(S, K, T, r, q, sigma, is_call, n=40000):
    """E[e^{-rT} payoff] with ln S_T = ln S + (r-q-sigma^2/2)T + sigma sqrt(T) z, z~N(0,1).
    The payoff is zero on one side of z* (the strike), so integrate only over
    the smooth region [z*, 12] (call) / [-12, z*] (put) -- no kink inside."""
    m = math.log(S) + (r - q - 0.5 * sigma ** 2) * T
    s = sigma * math.sqrt(T)
    zstar = (math.log(K) - m) / s
    a, b = (zstar, 12.0) if is_call else (-12.0, zstar)
    a, b = max(a, -12.0), min(b, 12.0)
    if b <= a:
        return 0.0
    h = (b - a) / n

    def f(z):
        st = math.exp(m + s * z)
        payoff = (st - K) if is_call else (K - st)
        return payoff * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    tot = f(a) + f(b)
    for i in range(1, n):
        tot += f(a + i * h) * (4 if i % 2 else 2)
    return math.exp(-r * T) * tot * h / 3.0
