"""Descriptive per-row quality flags. FLAGS ONLY: nothing is filtered, altered or deleted at capture time; thresholds are fixed in Config and recorded in the database."""
from .config import WS_SOURCES, Config

FLAG_NAMES = ["NO_BID", "NO_ASK", "BAD_PRICE", "CROSSED", "LOCKED", "WIDE_SPREAD", "STALE_QUOTE", "FEED_IN_FUTURE", "FEED_REGRESSED", "NO_FEED", "AGE_UNKNOWN", "NO_OI", "NO_VOLUME_TODAY", "ZERO_LTP"]


def row_flags(r: dict, skew_s: float, cfg: Config = Config(), prev_feed_ts=None, kind: str = "option") -> str:
    """r needs: bid, ask, ltp, volume, oi, quote_feed_ts, capture_ts, data_source. skew_s = local clock minus server clock (estimated per cycle; 0 if unknown)."""
    f = []
    bid, ask, ltp = r.get("bid"), r.get("ask"), r.get("ltp")
    if kind != "index":                       # the index has no bid/ask by nature
        if bid is None or bid <= 0:
            f.append("NO_BID")
        if ask is None or ask <= 0:
            f.append("NO_ASK")
    if any(x is not None and x < 0 for x in (bid, ask, ltp)):
        f.append("BAD_PRICE")
    if ltp is not None and ltp == 0:
        f.append("ZERO_LTP")
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        if bid > ask:
            f.append("CROSSED")
        elif bid == ask:
            f.append("LOCKED")
        else:
            mid = (bid + ask) / 2
            if ask - bid > max(cfg.wide_spread_abs, cfg.wide_spread_pct * mid):
                f.append("WIDE_SPREAD")
    feed, cap = r.get("quote_feed_ts"), r.get("capture_ts")
    if r.get("data_source") not in WS_SOURCES or feed is None:
        f.append("NO_FEED" if feed is None and r.get("data_source") in WS_SOURCES else "AGE_UNKNOWN")
    elif cap is not None:
        age = cap - feed - (skew_s or 0.0)
        if age > cfg.stale_quote_s:
            f.append("STALE_QUOTE")
        if feed - cap > 2.0:
            f.append("FEED_IN_FUTURE")
        if prev_feed_ts is not None and feed < prev_feed_ts:
            f.append("FEED_REGRESSED")
    if kind == "option" and r.get("oi") is None:
        f.append("NO_OI")
    if kind != "index" and not r.get("volume"):
        f.append("NO_VOLUME_TODAY")
    return ",".join(sorted(set(f)))
