"""Analysis configuration. Every threshold that decides which rows enter a statistic is here, fixed in advance, and written into run_metadata.json."""
from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class AnalysisConfig:
    rate: float = 0.065                              # risk-free rate assumption, identical to Stage 2A
    max_quote_age_s: float = 300.0                   # a quote (or last trade) older than this, after removing the clock skew, is not used (Stage 2A used 5 minutes for last trades)
    ltp_age_buckets_s: Tuple[float, ...] = (10.0, 60.0, 300.0)
    dte_edges_days: Tuple[float, ...] = (1.0, 3.0, 7.0, 14.0)           # (0,1], (1,3], (3,7], (7,14], > 14 (outside the research universe)
    moneyness_edges: Tuple[int, ...] = (0, 2, 6, 12)                     # |offset| in strikes: ATM, 1-2, 3-6, 7-12
    time_edges_min: Tuple[Tuple[str, int], ...] = (("open", 10 * 60 + 30), ("midday", 13 * 60), ("afternoon", 14 * 60 + 30), ("close", 24 * 60))
    primary_times: Tuple[str, ...] = ("10:01", "13:01", "15:01")        # the historical observation instants (bar start + 60 s)
    cross_fractions: Tuple[float, ...] = (0.0, 0.5, 1.0)                 # share of the half-spread crossed in the executable-IV scenarios
    atm_max_bracket_pts: float = 100.0                                   # as SurfaceConfig.max_atm_bracket_pts

    def as_dict(self):
        return asdict(self)


def dte_bucket(t_days: float, cfg: AnalysisConfig = AnalysisConfig()) -> str:
    """Same buckets as the research (build_iv_rv.DTE_BUCKETS): (0,1] '<=1d', (1,3], (3,7], (7,14]; beyond 14 days ('>14d') is outside the primary research universe."""
    e = cfg.dte_edges_days
    if t_days <= e[0]:
        return f"<={e[0]:g}d"
    for lo, hi in zip(e, e[1:]):
        if t_days <= hi:
            return f"{lo:g}-{hi:g}d"
    return f">{e[-1]:g}d"


def moneyness_bucket(offset: int, cfg: AnalysisConfig = AnalysisConfig()) -> str:
    a = abs(int(offset))
    e = cfg.moneyness_edges
    if a == e[0]:
        return "ATM"
    for lo, hi in zip(e, e[1:]):
        if a <= hi:
            return f"{lo + 1}-{hi}"
    return f">{e[-1]}"


def time_bucket(hhmm: str, cfg: AnalysisConfig = AnalysisConfig()) -> str:
    h, m = hhmm.split(":")
    mins = int(h) * 60 + int(m)
    for name, edge in cfg.time_edges_min:
        if mins < edge:
            return name
    return cfg.time_edges_min[-1][0]


def ltp_age_bucket(age_s: float, cfg: AnalysisConfig = AnalysisConfig()) -> str:
    e = cfg.ltp_age_buckets_s
    if age_s <= e[0]:
        return f"<={e[0]:g}s"
    for lo, hi in zip(e, e[1:]):
        if age_s <= hi:
            return f"{lo:g}-{hi:g}s"
    return f">{e[-1]:g}s"
