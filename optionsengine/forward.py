"""Put-call-parity forward estimator with an explicit quality gate.

Parity for European options:  C - P = e^{-rT} (F - K)   =>   F_i = K_i + e^{rT} (C_i - P_i).
Each strike with a usable call AND put gives one forward observation F_i.
Index options price off this forward (NIFTY futures carry a basis and no
weekly futures exist), so a wrong F silently skews every IV and delta.
The estimator therefore returns an explicit status and refuses to hand out a
forward unless the observations agree.

Estimator (all steps deterministic)
-----------------------------------
1. Validate observations: strike > 0 and both prices finite and > 0;
   anything else is dropped and counted (`n_dropped_invalid`). The CALLER is
   responsible for passing only fresh, tradeable quotes (see quality.py /
   last-trade age); this module cannot see staleness.
2. Select the `n_nearest` (default 6) strikes closest to `reference_price`
   (ties -> lower strike). Near-the-money parity is least affected by wide
   spreads and stale ITM prints, and mixing in far strikes just adds noise.
3. If fewer than `min_pairs` (default 3) remain -> UNAVAILABLE.
4. Outlier handling (only when >= 4 candidates; with 3 a median/MAD cannot
   identify an outlier, so none is removed and the dispersion test below
   decides): m = median(F_i), s = 1.4826 * MAD(F_i). A candidate is an
   outlier if |F_i - m| > max(outlier_mad_k * s, outlier_floor_bps * 1e-4 *
   reference_price). The bps floor stops the rule from rejecting tight
   clusters over a few paise (MAD can be ~0).
5. If inliers < `min_pairs` or the outlier fraction > `max_outlier_fraction`
   -> LOW_CONFIDENCE.
6. dispersion = max(F_i) - min(F_i) over inliers. If dispersion >
   `max_dispersion_bps` * 1e-4 * reference_price -> LOW_CONFIDENCE.
7. Otherwise OK with forward = median(inlier F_i).

Statuses -- `forward` is populated ONLY for OK:
  OK              forward usable.
  LOW_CONFIDENCE  enough data, but observations inconsistent (too dispersed
                  or too many outliers). `forward` is None; the median is
                  exposed as `candidate_forward` for diagnostics only.
  UNAVAILABLE     too little valid data (or invalid inputs). `forward` None.

Thresholds are POLICY, set from this project's NIFTY history (see README
"Parity-forward gate"); they are not market laws. Calibration evidence:
on 3,917 EOD (day, expiry) groups the inlier dispersion has median 2.2 bps /
p90 10.7 bps, and the forward's error against the exchange's monthly futures
rises smoothly with it (median |F - fut| 1.1 bps below 1 bp of dispersion,
2.2 bps at 4-6 bps, 4.0 bps above 10 bps) -- there is no natural cliff, so the
limit is a risk choice: 5 bps (~12 index points at 24,500) keeps the implied
forward error (about half the dispersion) near 0.2-0.3 vol points for an ATM
option one week out. The same absolute forward error costs MORE IV near
expiry (IV error ~ delta * dF / vega, vega ~ sqrt(T)): ~1.2 vol points at 1
day, several at 1 hour -- tighten `max_dispersion_bps` for short-dated work.
Tune them for other instruments/data vendors. `r` enters F_i only through
e^{rT}; its effect on F is tiny for short T (+/-1 pp of r moves F by ~0.03
points at 1 week), but it is still an assumption.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from .bsm import validate_inputs
from .errors import InvalidInputError


class ForwardStatus(str, Enum):
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ParityObservation:
    strike: float
    call_price: float
    put_price: float


@dataclass(frozen=True)
class ForwardConfig:
    n_nearest: int = 6
    min_pairs: int = 3
    outlier_mad_k: float = 3.5
    outlier_floor_bps: float = 1.0
    max_outlier_fraction: float = 0.5
    max_dispersion_bps: float = 5.0

    def __post_init__(self):
        if self.min_pairs < 2 or self.n_nearest < self.min_pairs:
            raise InvalidInputError("need min_pairs >= 2 and n_nearest >= min_pairs")
        if not (self.outlier_mad_k > 0 and self.outlier_floor_bps >= 0 and self.max_dispersion_bps > 0
                and 0 <= self.max_outlier_fraction <= 1):
            raise InvalidInputError("invalid forward-gate thresholds")


@dataclass(frozen=True)
class ForwardEstimate:
    status: ForwardStatus
    forward: Optional[float]               # only for OK
    candidate_forward: Optional[float]     # median of inliers (or of candidates) -- diagnostics, NOT for use unless OK
    n_input: int
    n_dropped_invalid: int
    n_candidates: int
    n_inliers: int
    dispersion: Optional[float]            # max-min of inlier forwards, in index points
    dispersion_limit: Optional[float]      # the threshold applied, in index points
    rejected_strikes: tuple = field(default=())
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ForwardStatus.OK


def _unavailable(n_in, n_bad, n_cand, reason):
    return ForwardEstimate(ForwardStatus.UNAVAILABLE, None, None, n_in, n_bad, n_cand, 0, None, None, (), reason)


def estimate_parity_forward(observations: Iterable[ParityObservation], r: float, T: float,
                            reference_price: float,
                            config: ForwardConfig = ForwardConfig()) -> ForwardEstimate:
    """See module docstring. `reference_price` is the spot/ATM level used to
    pick nearest strikes and to scale the bps thresholds. Raises
    InvalidInputError only for bad r/T/reference (programmer error); bad
    observations are dropped and counted, never raised."""
    validate_inputs(reference_price, reference_price, T, r, 0.0, require_sigma=False)
    if T <= 0:
        raise InvalidInputError("parity forward requires T > 0")
    obs = list(observations)
    good = []
    for o in obs:
        vals = (o.strike, o.call_price, o.put_price)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals) \
                and o.strike > 0 and o.call_price > 0 and o.put_price > 0:
            good.append(o)
    n_bad = len(obs) - len(good)
    if len(good) < config.min_pairs:
        return _unavailable(len(obs), n_bad, len(good),
                            f"only {len(good)} valid call+put pairs (< min_pairs={config.min_pairs})")

    good.sort(key=lambda o: (abs(o.strike - reference_price), o.strike))
    cand = good[:config.n_nearest]
    growth = math.exp(r * T)
    fs = [(o.strike, o.strike + growth * (o.call_price - o.put_price)) for o in cand]
    values = [f for _, f in fs]
    if min(values) <= 0:
        return _unavailable(len(obs), n_bad, len(cand), "non-positive parity forward")

    scale = 1e-4 * reference_price
    limit = config.max_dispersion_bps * scale
    inliers, rejected = fs, ()
    if len(fs) >= 4:
        m = statistics.median(values)
        s = 1.4826 * statistics.median(abs(v - m) for v in values)
        tol = max(config.outlier_mad_k * s, config.outlier_floor_bps * scale)
        inliers = [(k, f) for k, f in fs if abs(f - m) <= tol]
        rejected = tuple(k for k, f in fs if abs(f - m) > tol)
    cand_f = statistics.median(f for _, f in inliers) if inliers else statistics.median(values)

    n_rej = len(fs) - len(inliers)
    if len(inliers) < config.min_pairs or n_rej / len(fs) > config.max_outlier_fraction:
        return ForwardEstimate(ForwardStatus.LOW_CONFIDENCE, None, cand_f, len(obs), n_bad, len(cand),
                               len(inliers), None, limit, rejected,
                               f"{n_rej} of {len(fs)} candidate forwards rejected as outliers; "
                               f"{len(inliers)} inliers (< min_pairs={config.min_pairs} or fraction > {config.max_outlier_fraction})")
    inl = [f for _, f in inliers]
    disp = max(inl) - min(inl)
    if disp > limit:
        return ForwardEstimate(ForwardStatus.LOW_CONFIDENCE, None, cand_f, len(obs), n_bad, len(cand),
                               len(inliers), disp, limit, rejected,
                               f"inlier dispersion {disp:.2f} pts > limit {limit:.2f} pts "
                               f"({config.max_dispersion_bps:g} bps of {reference_price:g})")
    return ForwardEstimate(ForwardStatus.OK, statistics.median(inl), cand_f, len(obs), n_bad, len(cand),
                           len(inliers), disp, limit, rejected, "ok")
