"""Contract universe: which expiries, which strikes, which symbols. Pure functions (no network, no clock)."""
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from .config import Config, IST

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def grid_instants(day: date, cfg: Config = Config()) -> List[datetime]:
    """Cycle instants of one trading day, IST-aware: first_cycle, +5 min ... last_cycle (09:16 ... 15:36 = 77 instants; includes 10:01, 13:01, 15:01)."""
    t = datetime.combine(day, cfg.first_cycle, tzinfo=IST)
    end = datetime.combine(day, cfg.last_cycle, tzinfo=IST)
    out = []
    while t <= end:
        out.append(t)
        t += timedelta(minutes=cfg.grid_minutes)
    return out


def _parse_date(s: str) -> date:
    d, m, y = s.split("-")
    return date(int(y), int(m), int(d))


def select_expiries(expiry_data: List[dict], today: date, cfg: Config = Config()) -> List[dict]:
    """The nearest up-to-`max_expiries` listed expiries whose calendar distance from `today` is 0..max_expiry_days. `epoch` is the provider-supplied expiry instant (not assumed)."""
    out = []
    for e in expiry_data or []:
        d = _parse_date(e["date"])
        days = (d - today).days
        if days < 0 or days > cfg.max_expiry_days:
            continue
        out.append(dict(index=len(out), date=d.isoformat(), epoch=int(e["expiry"]), flag=e.get("expiry_flag"), dte_days=days, timestamp_param=str(e["expiry"])))
        if len(out) == cfg.max_expiries:
            break
    return out


def future_symbol(expiry_data: List[dict], today: date) -> Optional[str]:
    """Nearest-month NIFTY future, derived from the first monthly ('M') expiry on/after today: 27-10-2026 -> NSE:NIFTY26OCTFUT."""
    for e in expiry_data or []:
        d = _parse_date(e["date"])
        if e.get("expiry_flag") == "M" and d >= today:
            return f"NSE:NIFTY{d.year % 100:02d}{_MONTHS[d.month - 1]}FUT"
    return None


def atm_strike(spot: float, step: int = 50) -> int:
    return int(math.floor(spot / step + 0.5) * step)          # round half up (deterministic at exact mid-points)


def parse_chain(body: dict) -> Optional[dict]:
    """options-chain-v3 response -> {spot, vix, fp, expiries, rows}. rows: one dict per option row (symbol, strike, type, bid, ask, ltp, volume, oi, oich, prev_oi)."""
    data = (body or {}).get("data") or {}
    chain = data.get("optionsChain") or []
    if not chain:
        return None
    spot = next((r.get("ltp") for r in chain if r.get("strike_price", -1) == -1), None)
    fp = next((r.get("fp") for r in chain if r.get("strike_price", -1) == -1), None)
    rows = [dict(symbol=r["symbol"], strike=float(r["strike_price"]), type=r.get("option_type"), bid=r.get("bid"), ask=r.get("ask"), ltp=r.get("ltp"), volume=r.get("volume"), oi=r.get("oi"), oich=r.get("oich"),
                 prev_oi=r.get("prev_oi")) for r in chain if r.get("strike_price", -1) != -1 and r.get("symbol")]
    return dict(spot=spot, fp=fp, vix=(data.get("indiavixData") or {}).get("ltp"), expiries=data.get("expiryData") or [], rows=rows)


def in_window(rows: List[dict], atm: int, n_strikes: int, step: int = 50) -> List[dict]:
    lim = n_strikes * step
    return [r for r in rows if abs(r["strike"] - atm) <= lim]


def subscription_symbols(parsed_by_expiry: Dict[int, dict], atm: int, cfg: Config = Config()) -> List[str]:
    """Websocket subscription set: every option row within ATM +/- subscribe_strikes of every selected expiry."""
    syms = []
    for _, p in sorted(parsed_by_expiry.items()):
        syms += [r["symbol"] for r in in_window(p["rows"], atm, cfg.subscribe_strikes, cfg.strike_step)]
    return syms
