"""Per-expiry implied-volatility smile from ONE snapshot of option quotes (Stage 2A).

Pure calculation, standard library only: no file/network I/O, no pandas.
Data loading lives in `optionsengine.research.loaders`.

Pipeline for one (snapshot, expiry)
-----------------------------------
1. FRESHNESS.  A quote is usable only if its last trade is at most
   `max_age_minutes` old. Prices copied forward through zero-volume bars are
   not observations and are excluded from the primary analysis.
2. FORWARD.  Fresh call+put pairs go through `forward.estimate_parity_forward`
   (Phase 1 gate). Status OK -> primary; LOW_CONFIDENCE -> a smile is still
   computed from `candidate_forward` but every point is labelled
   `forward_low_confidence`, is not `used`, and no metrics are produced;
   UNAVAILABLE -> no smile (reason recorded).
3. OTM ONLY.  For each listed strike use the out-of-the-money option:
   call if K >= F, put if K < F (the other side is counted as `itm_side`).
4. IV.  `implied_volatility` with q = carry-implied yield of F (S cancels),
   r = the caller's explicit assumption, tick-rounded prices. A point is
   `reliable` only if converged AND not ill-conditioned AND not
   resolution-limited (Phase 1 `IVResult.reliable`). Unreliable points are
   kept, labelled, and excluded from every metric.
5. METRICS from `used` points only (reliable, forward OK, fresh):
   * log-moneyness  x = ln(K / F).
   * ATM IV: linear interpolation in x at x = 0 between the two strikes that
     bracket F and are ADJACENT in the listed strike grid, both used, and no
     more than `max_atm_bracket_pts` apart. Never extrapolated.
     `atm_loose` is the same construction allowing converged-but-unreliable
     points; it exists so thin expiry-day data is not silently dropped, and
     carries the number of unreliable inputs -- do not treat it as precise.
   * 25-delta RR/BF (only if T > `rr_bf_min_days`, forward OK, ATM available):
     forward delta |Delta| = N(d1) (call) / N(-d1) (put), d1 =
     [ln(F/K) + sigma^2 T/2] / (sigma sqrt(T)), using each point's own IV.
     IV at |Delta| = 0.25 is found by linear interpolation in delta between
     two ADJACENT used strikes that bracket 0.25 (<= `max_delta_bracket_pts`
     apart) on each wing; if either wing cannot be bracketed the metrics are
     missing. RR25 = IV25(call) - IV25(put);
     BF25 = (IV25(call) + IV25(put))/2 - ATM IV.
   Missing pieces are never fabricated; the reason is recorded.

Every excluded point has exactly one `exclusion` reason (None if used).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .bsm import OptionType, implied_carry_yield, norm_cdf
from .errors import InvalidInputError
from .forward import (ForwardConfig, ForwardEstimate, ForwardStatus, ParityObservation,
                      estimate_parity_forward)
from .implied_vol import IVStatus, SolverConfig, implied_volatility

# ---- exclusion reasons (point level) ---------------------------------------
STALE = "stale"                              # last trade older than max_age_minutes
NEVER_TRADED = "never_traded"                # no trade in the loaded history
ZERO_PRICE = "zero_price"                    # price <= 0 after tick rounding
FORWARD_LOW_CONFIDENCE = "forward_low_confidence"
RESOLUTION_LIMITED = "resolution_limited"
ILL_CONDITIONED = "ill_conditioned"
IV_FAILED_PREFIX = "iv_failed:"              # + IVStatus value

# ---- group level reasons ---------------------------------------------------
GROUP_FORWARD_UNAVAILABLE = "forward_unavailable"
GROUP_EXPIRED = "expired"
ATM_NO_BRACKET = "no_adjacent_strikes_bracketing_forward"
ATM_BRACKET_TOO_WIDE = "bracket_wider_than_limit"
ATM_POINT_UNUSABLE = "bracketing_point_not_used"
RRBF_SHORT_EXPIRY = "expiry_within_min_days"
RRBF_NO_ATM = "no_atm_iv"
RRBF_NOT_BRACKETED = "delta_not_bracketed"
RRBF_FORWARD_NOT_OK = "forward_not_ok"


@dataclass(frozen=True)
class OptionQuote:
    """One listed contract at the snapshot. `price` is the last traded price
    on the tick grid; `age_minutes` is an UPPER bound on minutes since that
    trade (None = never traded). Both None-safe: missing stays missing."""
    strike: float
    kind: OptionType
    price: Optional[float]
    age_minutes: Optional[float]

    def __post_init__(self):
        object.__setattr__(self, "kind", OptionType.coerce(self.kind))


@dataclass(frozen=True)
class SurfaceConfig:
    max_age_minutes: float = 5.0
    rr_bf_min_days: float = 1.0
    target_delta: float = 0.25
    max_atm_bracket_pts: float = 100.0
    max_delta_bracket_pts: float = 200.0
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def __post_init__(self):
        if not (self.max_age_minutes > 0 and 0 < self.target_delta < 0.5
                and self.max_atm_bracket_pts > 0 and self.max_delta_bracket_pts > 0 and self.rr_bf_min_days >= 0):
            raise InvalidInputError("invalid SurfaceConfig")


@dataclass(frozen=True)
class SmilePoint:
    strike: float
    kind: OptionType
    price: Optional[float]
    age_minutes: Optional[float]
    log_moneyness: float                 # ln(K / F)
    iv: Optional[float]                  # decimal; None unless the solver converged
    iv_status: Optional[str]             # IVStatus value (None if IV not attempted)
    reliable: bool
    resolution_limited: bool
    ill_conditioned: bool
    iv_low: Optional[float]              # IV at price - tick/2
    iv_high: Optional[float]             # IV at price + tick/2
    delta_fwd: Optional[float]           # |forward delta| at its own IV (None if no IV)
    used: bool                           # entered the metrics
    exclusion: Optional[str]             # None iff used


@dataclass(frozen=True)
class AtmResult:
    iv: float
    strike_low: float
    strike_high: float
    iv_low: float
    iv_high: float
    n_unreliable_inputs: int             # 0 for the strict ATM


@dataclass(frozen=True)
class RiskReversalButterfly:
    iv25_call: float
    iv25_put: float
    rr25: float                          # iv25_call - iv25_put
    bf25: float                          # (iv25_call + iv25_put)/2 - ATM IV
    call_bracket: tuple                  # (strike_a, strike_b)
    put_bracket: tuple


@dataclass(frozen=True)
class SmileResult:
    forward: ForwardEstimate
    forward_used: Optional[float]        # forward OK -> forward; LOW_CONFIDENCE -> candidate; else None
    spot: float
    T_years: float
    T_days: float
    r: float
    q: Optional[float]
    points: tuple
    n_quotes: int
    n_itm_side_excluded: int
    n_strikes_without_otm_quote: int
    atm: Optional[AtmResult]
    atm_loose: Optional[AtmResult]
    atm_reason: Optional[str]            # why `atm` is None
    rr_bf: Optional[RiskReversalButterfly]
    rr_bf_reason: Optional[str]
    group_reason: Optional[str] = None   # set when no smile could be built at all

    @property
    def primary(self) -> bool:
        """True only for OK forwards: LOW_CONFIDENCE groups are never primary."""
        return self.forward.status is ForwardStatus.OK

    @property
    def exclusion_counts(self) -> dict:
        out: dict = {}
        for p in self.points:
            if p.exclusion:
                out[p.exclusion] = out.get(p.exclusion, 0) + 1
        return out


def forward_delta_abs(F: float, K: float, T: float, sigma: float, kind: OptionType) -> float:
    """|forward delta|: N(d1) for a call, N(-d1) for a put (undiscounted, w.r.t. the forward)."""
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if kind is OptionType.CALL else norm_cdf(-d1)


def _empty(fe, spot, T, r, n_quotes, reason) -> SmileResult:
    return SmileResult(fe, None, spot, T, T * 365.0, r, None, (), n_quotes, 0, 0, None, None, reason,
                       None, reason, group_reason=reason)


def build_smile(quotes: Sequence[OptionQuote], spot: float, T: float, r: float,
                config: SurfaceConfig = SurfaceConfig()) -> SmileResult:
    """Build the OTM smile for one expiry at one snapshot. `r` is the caller's
    explicit risk-free assumption (the engine has no default)."""
    quotes = list(quotes)
    if T <= 0:
        fe = ForwardEstimate(ForwardStatus.UNAVAILABLE, None, None, 0, 0, 0, 0, None, None, (), "expired")
        return _empty(fe, spot, 0.0, r, len(quotes), GROUP_EXPIRED)

    def fresh(qt: OptionQuote) -> bool:
        return (qt.price is not None and qt.price > 0 and qt.age_minutes is not None
                and qt.age_minutes <= config.max_age_minutes)

    calls = {qt.strike: qt for qt in quotes if qt.kind is OptionType.CALL and fresh(qt)}
    puts = {qt.strike: qt for qt in quotes if qt.kind is OptionType.PUT and fresh(qt)}
    obs = [ParityObservation(k, calls[k].price, puts[k].price) for k in sorted(set(calls) & set(puts))]
    fe = estimate_parity_forward(obs, r, T, spot, config.forward)
    F = fe.forward if fe.status is ForwardStatus.OK else fe.candidate_forward
    if fe.status is ForwardStatus.UNAVAILABLE or F is None:
        return _empty(fe, spot, T, r, len(quotes), GROUP_FORWARD_UNAVAILABLE)

    q = implied_carry_yield(spot, F, T, r)
    by_key = {(qt.strike, qt.kind): qt for qt in quotes}
    strikes = sorted({qt.strike for qt in quotes})
    n_itm = n_missing = 0
    points: list = []
    for k in strikes:
        kind = OptionType.CALL if k >= F else OptionType.PUT
        other = OptionType.PUT if kind is OptionType.CALL else OptionType.CALL
        if (k, other) in by_key:
            n_itm += 1
        qt = by_key.get((k, kind))
        if qt is None:
            n_missing += 1
            continue
        points.append(_make_point(qt, F, T, spot, r, q, fe, config))

    pts = tuple(points)
    if fe.status is not ForwardStatus.OK:      # LOW_CONFIDENCE: points are kept for inspection, no metrics
        return SmileResult(fe, F, spot, T, T * 365.0, r, q, pts, len(quotes), n_itm, n_missing,
                           None, None, FORWARD_LOW_CONFIDENCE, None, RRBF_FORWARD_NOT_OK)
    atm, atm_reason = _atm(pts, config, loose=False)
    atm_loose, _ = _atm(pts, config, loose=True)
    rr_bf, rr_reason = _rr_bf(pts, F, T, atm, fe, config)
    return SmileResult(fe, F, spot, T, T * 365.0, r, q, pts, len(quotes), n_itm, n_missing,
                       atm, atm_loose, atm_reason, rr_bf, rr_reason)


def _make_point(qt: OptionQuote, F: float, T: float, spot: float, r: float, q: float,
                fe: ForwardEstimate, config: SurfaceConfig) -> SmilePoint:
    x = math.log(qt.strike / F)

    def pt(**kw) -> SmilePoint:
        base = dict(strike=qt.strike, kind=qt.kind, price=qt.price, age_minutes=qt.age_minutes, log_moneyness=x,
                    iv=None, iv_status=None, reliable=False, resolution_limited=False, ill_conditioned=False,
                    iv_low=None, iv_high=None, delta_fwd=None, used=False, exclusion=None)
        base.update(kw)
        return SmilePoint(**base)

    if qt.price is None or qt.age_minutes is None:
        return pt(exclusion=NEVER_TRADED)
    if qt.age_minutes > config.max_age_minutes:
        return pt(exclusion=STALE)
    if qt.price <= 0:
        return pt(exclusion=ZERO_PRICE)

    res = implied_volatility(qt.price, spot, qt.strike, T, r, q, qt.kind, config.solver)
    d = res.diagnostics
    if not res.converged:
        return pt(iv_status=res.status.value, exclusion=IV_FAILED_PREFIX + res.status.value)
    lo, hi = d.iv_interval if d.iv_interval else (None, None)
    delta = forward_delta_abs(F, qt.strike, T, res.iv, qt.kind)
    base = dict(iv=res.iv, iv_status=IVStatus.CONVERGED.value, reliable=res.reliable,
                resolution_limited=d.resolution_limited, ill_conditioned=d.ill_conditioned,
                iv_low=lo, iv_high=hi, delta_fwd=delta)
    if fe.status is not ForwardStatus.OK:
        return pt(exclusion=FORWARD_LOW_CONFIDENCE, **base)
    if d.resolution_limited:
        return pt(exclusion=RESOLUTION_LIMITED, **base)
    if d.ill_conditioned:
        return pt(exclusion=ILL_CONDITIONED, **base)
    return pt(used=True, **base)


def _atm(points: Sequence[SmilePoint], config: SurfaceConfig, loose: bool):
    """Interpolate IV at x = 0 between the two listed strikes bracketing F."""
    below = [p for p in points if p.log_moneyness < 0]
    above = [p for p in points if p.log_moneyness >= 0]
    if not below or not above:
        return None, ATM_NO_BRACKET
    lo, hi = max(below, key=lambda p: p.strike), min(above, key=lambda p: p.strike)
    if hi.strike - lo.strike > config.max_atm_bracket_pts:
        return None, ATM_BRACKET_TOO_WIDE

    def ok(p: SmilePoint) -> bool:
        if loose:
            return p.iv is not None and p.exclusion != FORWARD_LOW_CONFIDENCE
        return p.used

    if not (ok(lo) and ok(hi)):
        return None, ATM_POINT_UNUSABLE
    w = (0.0 - lo.log_moneyness) / (hi.log_moneyness - lo.log_moneyness)
    iv = lo.iv + w * (hi.iv - lo.iv)
    n_unrel = sum(1 for p in (lo, hi) if not p.used)
    return AtmResult(iv, lo.strike, hi.strike, lo.iv, hi.iv, n_unrel), None


def _bracket_025(wing: Sequence[SmilePoint], target: float, max_gap: float):
    """Adjacent used strikes whose |delta| brackets `target`; linear in delta."""
    for a, b in zip(wing, wing[1:]):                     # consecutive in the listed grid
        if not (a.used and b.used) or b.strike - a.strike > max_gap:
            continue
        lo_d, hi_d = sorted((a.delta_fwd, b.delta_fwd))
        if lo_d <= target <= hi_d and a.delta_fwd != b.delta_fwd:
            w = (target - a.delta_fwd) / (b.delta_fwd - a.delta_fwd)
            return a.iv + w * (b.iv - a.iv), (a.strike, b.strike)
    return None


def _rr_bf(points, F, T, atm, fe, config):
    if fe.status is not ForwardStatus.OK:
        return None, RRBF_FORWARD_NOT_OK
    if T * 365.0 <= config.rr_bf_min_days:
        return None, RRBF_SHORT_EXPIRY
    if atm is None:
        return None, RRBF_NO_ATM
    ordered = sorted(points, key=lambda p: p.strike)
    call_wing = [p for p in ordered if p.log_moneyness >= 0]
    put_wing = [p for p in ordered if p.log_moneyness < 0]
    c = _bracket_025(call_wing, config.target_delta, config.max_delta_bracket_pts)
    p_ = _bracket_025(put_wing, config.target_delta, config.max_delta_bracket_pts)
    if c is None or p_ is None:
        return None, RRBF_NOT_BRACKETED
    (iv_c, kc), (iv_p, kp) = c, p_
    return RiskReversalButterfly(iv_c, iv_p, iv_c - iv_p, 0.5 * (iv_c + iv_p) - atm.iv, kc, kp), None
