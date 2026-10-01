"""simulator.py -- historical backtest sandbox: candle-by-candle replay of the
downloaded 1-minute history in data/hist1m/ (see history_downloader.py):

  options/<expiry>.parquet   every ATM +/-10 NIFTY option of that expiry,
                             1-min OHLC + volume + OI (Fyers expired-F&O API)
  NIFTY50_1m.parquet         NIFTY 50 spot, 1-min OHLC
  manifest.json              which expiries are on disk + each one's week

Any timeframe that is a multiple of 1 minute is built by resampling the 1-min
bars from 09:15 each day. Positions, payoff and Greeks use the same math as
the live Strategy Builder (manual_trades.py / greeks.py), fed historical inputs.
"""
import json
import os
from datetime import date, timedelta, time as dtime

import numpy as np
import pandas as pd

import greeks as greeks_mod
import manual_trades
import paths

BASE = paths.BASE_DIR
HIST_DIR = os.path.join(BASE, "data", "hist1m")
MANIFEST = os.path.join(HIST_DIR, "manifest.json")
SPOT_FILE = os.path.join(HIST_DIR, "NIFTY50_1m.parquet")
IST = "Asia/Kolkata"
AVAILABLE_TIMEFRAMES = [1, 3, 5, 15, 30, 60]  # minutes (all resampled from 1-min bars)
DEFAULT_TIMEFRAME = 5
MARKET_CLOSE = dtime(15, 40)  # latest candle start kept (NSE F&O close moved 15:30 -> 15:40 on 2026-08-03)
F_O_CLOSE_CHANGE = date(2026, 8, 3)
LOT_SIZE = manual_trades.LOT_SIZE
LOT_CANDIDATES = (75, 65, 50, 25)  # NIFTY lot sizes used over the years (largest first)


def _manifest():
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def list_expiries():
    """Every expiry with downloaded 1-min data, newest first."""
    out = []
    for exp, rec in _manifest().items():
        if not isinstance(rec, dict) or rec.get("status") not in ("done", "live"):
            continue
        if not os.path.exists(os.path.join(HIST_DIR, "options", f"{exp}.parquet")):
            continue
        d = date.fromisoformat(exp)
        out.append({"expiry": exp, "label": d.strftime("%d %b %Y (%a)"), "year": d.year,
                    "week": rec.get("week")})
    return sorted(out, key=lambda e: e["expiry"], reverse=True)


def _latest_expiry():
    e = list_expiries()
    if not e:
        raise ValueError("No downloaded history in data/hist1m -- run history_downloader.py first.")
    return e[0]["expiry"]


def _week_of(expiry):
    rec = _manifest().get(expiry) or {}
    if rec.get("status") not in ("done", "live"):  # live = the current week, updated each evening
        raise ValueError(f"Expiry {expiry!r} has no downloaded data.")
    w = rec.get("week") or [expiry, expiry]
    return date.fromisoformat(w[0]), date.fromisoformat(w[1])


_opt_cache = {}  # expiry -> DataFrame (keeps the last 3 expiries used)


def _load_options(expiry):
    if expiry in _opt_cache:
        return _opt_cache[expiry]
    df = pd.read_parquet(os.path.join(HIST_DIR, "options", f"{expiry}.parquet"))
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(IST)
    df["day"] = df["date"].dt.date
    if len(_opt_cache) >= 3:
        _opt_cache.pop(next(iter(_opt_cache)))
    _opt_cache[expiry] = df
    return df


_spot_df = None


def _load_spot():
    global _spot_df
    if _spot_df is None and os.path.exists(SPOT_FILE):
        s = pd.read_parquet(SPOT_FILE)
        s["date"] = pd.to_datetime(s["ts"], unit="s", utc=True).dt.tz_convert(IST)
        _spot_df = s.set_index("date").sort_index()
    return _spot_df


_lot_cache = {}


def lot_size_for(expiry):
    """The NIFTY lot size this expiry traded at, read off the data itself:
    exchange volume is always a whole number of lots, so the largest
    candidate lot that divides (~all) candle volumes is the lot size."""
    if expiry not in _lot_cache:
        v = _load_options(expiry)["volume"].to_numpy()
        v = v[v > 0]
        lot = LOT_SIZE
        for cand in LOT_CANDIDATES:
            if len(v) and (v % cand == 0).mean() >= 0.98:
                lot = cand
                break
        _lot_cache[expiry] = lot
    return _lot_cache[expiry]


def list_available_dates(expiry=None):
    """Every day this expiry's contracts traded (from listing to expiry).
    The bulk download covered ATM +/-10 for the expiry week; an earlier
    start date is topped up on demand (history_downloader.ensure_strikes)."""
    expiry = expiry or _latest_expiry()
    days = _load_options(expiry)["day"].unique()
    return sorted(d.isoformat() for d in days)


def week_start(expiry):
    return _week_of(expiry)[0].isoformat()


def forget(expiry):
    """Drop the cached frame after the file on disk changed."""
    _opt_cache.pop(expiry, None)
    _lot_cache.pop(expiry, None)


def list_missing_weekdays(expiry=None):
    """NSE trading days inside the expiry's week with no candles at all."""
    import market_calendar as mc
    expiry = expiry or _latest_expiry()
    w0, w1 = _week_of(expiry)
    w1 = min(w1, mc.now_ist().date())  # a live week's future days aren't missing
    have = {d for d in list_available_dates(expiry) if w0.isoformat() <= d}
    missing, d = [], w0
    while d <= w1:
        if mc.is_trading_day(d) and d.isoformat() not in have:
            missing.append(d.isoformat())
        d += timedelta(days=1)
    return missing


def _bucket_labels(idx, timeframe_min):
    """Bucket-start label for each timestamp: `timeframe_min` buckets anchored at 09:15 each day."""
    open_ = idx.normalize() + pd.Timedelta(hours=9, minutes=15)
    k = np.asarray((idx - open_).total_seconds() // (60 * timeframe_min), dtype="int64")
    return open_ + pd.to_timedelta(k * timeframe_min, unit="min")


def _bucket(df, timeframe_min):
    """Resample 1-min option bars to `timeframe_min` candles (label = bucket start)."""
    if timeframe_min == 1:
        return df
    df = df.assign(date=_bucket_labels(pd.DatetimeIndex(df["date"]), timeframe_min))
    return (df.sort_values("ts")
              .groupby(["symbol", "date"], sort=False)
              .agg(type=("type", "first"), strike=("strike", "first"), day=("day", "first"),
                   open=("open", "first"), high=("high", "max"), low=("low", "min"),
                   close=("close", "last"), volume=("volume", "sum"), oi=("oi", "last"))
              .reset_index())


class SimulationSession:
    def __init__(self, date_str, timeframe_min=1, expiry=None, end_date=None):
        """Replay that starts at 09:15 of `date_str` and runs on through every
        later trading day up to `end_date` (default: the expiry) -- positions
        carry overnight, time-to-expiry (theta) decays candle by candle, and a
        replay that reaches the end of expiry day settles open legs at
        intrinsic. The base candle is `timeframe_min` (1 min); step(minutes)
        moves the clock by any number of minutes on top of that."""
        if timeframe_min not in AVAILABLE_TIMEFRAMES:
            raise ValueError(f"timeframe_min must be one of {AVAILABLE_TIMEFRAMES}")
        self.date_str = date_str
        self.timeframe_min = timeframe_min
        self.expiry = expiry or _latest_expiry()
        self.lot_size = lot_size_for(self.expiry)

        start_day = pd.Timestamp(date_str).date()
        exp_day = pd.Timestamp(self.expiry).date()
        end_day = min(pd.Timestamp(end_date).date(), exp_day) if end_date else exp_day
        if end_day < start_day:
            raise ValueError("End date is before the start date.")
        self.positional = end_day > start_day

        raw = _load_options(self.expiry)
        raw = raw[raw["date"].dt.time <= MARKET_CLOSE]
        # "OI Chg" (Fyers' oich) = today's OI minus the PREVIOUS trading day's
        # closing OI, per contract and per day of the replay.
        closes = raw.sort_values("ts").groupby(["symbol", "day"])["oi"].last().reset_index()
        closes["prev_oi"] = closes.groupby("symbol")["oi"].shift(1)
        self.prev_day_oi = {(r.symbol, r.day): float(r.prev_oi) for r in closes.itertuples()
                            if pd.notna(r.prev_oi) and r.prev_oi > 0 and start_day <= r.day <= end_day}

        win = raw[(raw["day"] >= start_day) & (raw["day"] <= end_day)]
        if win.empty:
            raise ValueError(f"No candles for {date_str}" + (f" -> {end_date}" if end_date else "") + ".")
        bars = _bucket(win, timeframe_min).sort_values("date")
        # Volume shown in the chain = the day's cumulative traded volume up to
        # this candle (what a live option chain shows), so it reads the same
        # whatever step size is used. `traded` keeps the candle's own volume.
        bars["dayvol"] = bars.groupby(["symbol", "day"])["volume"].cumsum()
        self.strike_of = dict(zip(bars["symbol"], bars["strike"].astype(int)))
        self.side_of = dict(zip(bars["symbol"], bars["type"]))
        self.frames = {sym: g.set_index("date")[["open", "high", "low", "close", "volume", "dayvol", "oi"]].sort_index()
                       for sym, g in bars.groupby("symbol")}

        self.all_ts = [pd.Timestamp(t) for t in sorted(bars["date"].unique())]
        self.days = sorted({ts.date() for ts in self.all_ts})
        self.index = -1  # -1 = not started; step() moves to 0, 1, 2, ...
        self.finished = False

        # Real NIFTY 50 spot for the same buckets (close of each bucket).
        self.spot_at = {}
        spot = _load_spot()
        if spot is not None:
            s = spot.loc[str(start_day):f"{end_day} 23:59"]
            if not s.empty:
                self.spot_at = s["close"].groupby(_bucket_labels(s.index, timeframe_min)).last().to_dict()

        # Any gap wider than one candle inside the SAME day -- overnight is not a gap.
        self.gaps = []
        step = pd.Timedelta(minutes=timeframe_min)
        for i in range(1, len(self.all_ts)):
            a, b = self.all_ts[i - 1], self.all_ts[i]
            if a.date() == b.date() and b > a + step:
                self.gaps.append({"from": a.isoformat(), "to": b.isoformat()})

        # Manual, user-picked legs -- last_ce/last_pe hold whatever
        # _candles_at() most recently produced, so a leg can be priced (and
        # the payoff/Greeks recomputed) at "wherever the replay currently is".
        self.legs = []
        self.closed_legs = []
        self._leg_seq = 0
        self.last_ce = {}
        self.last_pe = {}
        close_t = "15:40:00" if exp_day >= F_O_CLOSE_CHANGE else "15:30:00"
        self.expiry_close = pd.Timestamp(f"{self.expiry} {close_t}").tz_localize(IST)

    @property
    def total_candles(self):
        return len(self.all_ts)

    def spot_now(self):
        """Real NIFTY 50 spot at the current candle's close (None if not on disk)."""
        if not (0 <= self.index < len(self.all_ts)):
            return None
        v = self._spot_asof(self.all_ts[self.index])
        return round(v, 2) if v is not None else None

    def _spot_asof(self, ts):
        """NIFTY at `ts`, or its last print earlier that same day (the index
        closes at 15:30 while options trade on to 15:40)."""
        v = self.spot_at.get(ts)
        if v is not None and pd.notna(v):
            return float(v)
        same_day = [k for k in self.spot_at if k.date() == ts.date() and k <= ts]
        return float(self.spot_at[max(same_day)]) if same_day else None

    def _candles_at(self, ts):
        """A strike with no candle at `ts` (not yet listed, or a gap in the
        downloaded data) -- blanking the display
        for it is misleading (a real option-chain always shows the last
        traded price/OI, it doesn't go empty). So a strike missing at `ts`
        falls back to its most recent PRIOR candle's close/oi (volume=0,
        since nothing new traded) rather than being dropped from the
        snapshot."""
        ce_candles, pe_candles = {}, {}
        for sym, d in self.frames.items():
            strike = self.strike_of[sym]
            if ts in d.index:
                row = d.loc[ts]
                rec = {"open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]),
                       "close": float(row["close"]),
                       "volume": float(row["dayvol"]) if pd.notna(row["dayvol"]) else 0.0,
                       "traded": float(row["volume"]) if pd.notna(row["volume"]) else 0.0,
                       "oi": float(row["oi"]) if pd.notna(row["oi"]) else 0.0}
            else:
                prior = d.index[d.index <= ts]
                if len(prior) == 0:
                    continue
                last_ts = prior.max()
                last_row = d.loc[last_ts]
                last_close = float(last_row["close"])
                same_day = last_ts.date() == ts.date()
                rec = {"open": last_close, "high": last_close, "low": last_close, "close": last_close,
                       "volume": float(last_row["dayvol"]) if same_day and pd.notna(last_row["dayvol"]) else 0.0,
                       "traded": 0.0,
                       "oi": float(last_row["oi"]) if pd.notna(last_row["oi"]) else 0.0}
            prev_oi = self.prev_day_oi.get((sym, ts.date()))
            if prev_oi:
                rec["oich"] = rec["oi"] - prev_oi
                rec["oichp"] = round(rec["oich"] / prev_oi * 100, 2)
            else:
                rec["oich"] = None
                rec["oichp"] = None
            if self.side_of[sym] == "CE":
                ce_candles[strike] = rec
            else:
                pe_candles[strike] = rec
        return ce_candles, pe_candles

    def step(self, minutes=1):
        """Moves the clock forward `minutes` (never past the day's last
        candle -- from there the next step opens the next day). Returns a
        snapshot dict, or None if the session is already finished."""
        if self.finished:
            return None
        target = self.index + 1
        if self.index >= 0 and minutes > 1:
            cur = self.all_ts[self.index]
            goal = cur + pd.Timedelta(minutes=minutes)
            j = self.index + 1
            while j < self.total_candles and self.all_ts[j].date() == cur.date() and self.all_ts[j] < goal:
                j += 1
            if j < self.total_candles and self.all_ts[j].date() != cur.date() and j - 1 > self.index:
                j -= 1  # stop at the day's close; the next step opens the next day
            target = j
        return self._goto(target)

    def eod(self):
        """Jump to the current day's last candle (open legs are kept)."""
        if self.finished:
            return None
        day = self.all_ts[max(self.index, 0)].date()
        last = max(i for i, ts in enumerate(self.all_ts) if ts.date() == day)
        if last == self.index:
            return self._snapshot_now()
        return self._goto(last)

    def _snapshot_now(self):
        ts = self.all_ts[self.index]
        return self._snapshot(ts, ce_candles=self.last_ce, pe_candles=self.last_pe)

    def _goto(self, i):
        self.index = i
        if self.index >= self.total_candles:
            self.finished = True
            self._settle_at_expiry()
            return self._snapshot(None, is_eod=True)

        ts = self.all_ts[self.index]
        ce_candles, pe_candles = self._candles_at(ts)
        self.last_ce, self.last_pe = ce_candles, pe_candles
        return self._snapshot(ts, ce_candles=ce_candles, pe_candles=pe_candles)

    def next_day(self):
        """Positional replay: jump to the NEXT trading day's first candle
        (positions carried overnight). Returns that candle's snapshot, or the
        end-of-replay snapshot if there's no later day."""
        if self.finished:
            return None
        cur = self.all_ts[self.index].date() if 0 <= self.index < len(self.all_ts) else None
        nxt = next((i for i, ts in enumerate(self.all_ts) if i > self.index and (cur is None or ts.date() > cur)), None)
        return self._goto(nxt if nxt is not None else len(self.all_ts))

    def _settle_at_expiry(self):
        """A replay that runs to the end of EXPIRY day settles every open leg
        at intrinsic value against the NIFTY close."""
        if not self.all_ts or self.all_ts[-1].date() != pd.Timestamp(self.expiry).date() or not self.legs:
            return
        # Real NIFTY 50 close of expiry day (NSE settles on the index), else the forward.
        last = self.all_ts[-1]
        settle = self._spot_asof(last) or self.synthetic_forward_now()
        if not settle:
            return
        for leg in list(self.legs):
            k = leg["strike"]
            intrinsic = max(0.0, settle - k) if leg["type"] == "CE" else max(0.0, k - settle)
            rec = self.close_leg(leg["id"], intrinsic)
            rec["exit_reason"] = f"EXPIRY (settled vs {settle:.1f})"

    def _snapshot(self, ts, ce_candles=None, pe_candles=None, is_eod=False):
        chain = []
        if ce_candles is not None or pe_candles is not None:
            strikes = sorted(set((ce_candles or {}).keys()) | set((pe_candles or {}).keys()))
            for s in strikes:
                chain.append({
                    "strike": s,
                    "ce": (ce_candles or {}).get(s),
                    "pe": (pe_candles or {}).get(s),
                })
        day = ts.date() if ts is not None else None
        prev_ts = getattr(self, "_prev_shown", None)
        if ts is not None:
            self._prev_shown = ts
        return {
            "time": ts.isoformat() if ts is not None else None,
            "day": day.isoformat() if day else None,
            "day_no": (self.days.index(day) + 1) if day in self.days else None,
            "days_total": len(self.days),
            "new_day": bool(ts is not None and (prev_ts is None or prev_ts.date() != day)),
            "positional": self.positional,
            "index": self.index,
            "total": self.total_candles,
            "is_eod": is_eod,
            "finished": self.finished,
            "chain": chain,
            # Put-call-parity forward, computed regardless of whether any
            # legs are open -- the UI's ITM shading needs a spot-like
            # anchor even before the user has added a single position.
            "anchor": self.synthetic_forward_now() if chain else None,
            "spot": self.spot_now() if ts is not None else None,
            "lot_size": self.lot_size,
        }

    # ---- manual legs: add positions straight off the historical chain and
    # get back the same payoff-chart/stats/Greeks bundle the live Strategy
    # Builder shows, computed from wherever the replay currently sits. ----

    def add_leg(self, strike, opt_type, side, qty, entry_price=None):
        candles = self.last_ce if opt_type == "CE" else self.last_pe
        rec = candles.get(strike)
        if entry_price is None:
            if rec is None:
                raise ValueError(f"No historical data for {opt_type} {strike} at this point in the replay.")
            entry_price = rec["close"]
        self._leg_seq += 1
        leg = {
            "id": f"sim{self._leg_seq}",
            "strike": strike, "type": opt_type, "side": side, "qty": int(qty),
            "lot_size": self.lot_size,
            "entry_price": round(float(entry_price), 2),
            "entry_time": self.all_ts[self.index].isoformat() if 0 <= self.index < len(self.all_ts) else None,
        }
        self.legs.append(leg)
        return leg

    def remove_leg(self, leg_id):
        """Undo for a mis-click: only allowed on the SAME candle the leg was
        added. Later on it would erase a position's outcome with no record
        (a losing leg could simply vanish) -- use close_leg() instead, which
        books the P&L."""
        leg = next((l for l in self.legs if l["id"] == leg_id), None)
        if leg is None:
            return False
        now_ts = self.all_ts[self.index].isoformat() if 0 <= self.index < len(self.all_ts) else None
        if leg.get("entry_time") != now_ts:
            raise ValueError("This leg was entered on an earlier candle -- close it (Edit > Close) so its P&L is "
                             "recorded. Remove only undoes a leg added on the current candle.")
        self.legs = [l for l in self.legs if l["id"] != leg_id]
        return True

    def reset_legs(self):
        self.legs = []
        self.closed_legs = []

    def close_leg(self, leg_id, exit_price=None):
        """Exits an open leg at exit_price (defaulting to the current
        candle's close, like add_leg's entry-price default), records it
        in closed_legs with realized P&L, and removes it from the open
        legs list -- the Simulator's equivalent of manual_trades.close_leg()
        for live trading, so a backtest can actually show what a real
        exit here-and-now would have realized instead of only ever
        deleting a position with no record of the outcome."""
        leg = next((l for l in self.legs if l["id"] == leg_id), None)
        if leg is None:
            raise ValueError(f"Leg {leg_id!r} not found.")
        if exit_price is None:
            candles = self.last_ce if leg["type"] == "CE" else self.last_pe
            rec = candles.get(leg["strike"])
            if rec is None:
                raise ValueError(f"No historical data for {leg['type']} {leg['strike']} at this point in the replay.")
            exit_price = rec["close"]
        record = {**leg, "exit_price": round(float(exit_price), 2),
                  "exit_time": self.all_ts[self.index].isoformat() if 0 <= self.index < len(self.all_ts) else None,
                  "pnl": self._leg_pnl(leg, float(exit_price))}
        self.legs = [l for l in self.legs if l["id"] != leg_id]
        self.closed_legs.append(record)
        return record

    def edit_leg(self, leg_id, strike, opt_type, side, qty, entry_price):
        for leg in self.legs:
            if leg["id"] == leg_id:
                leg["strike"] = strike
                leg["type"] = opt_type
                leg["side"] = side
                leg["qty"] = int(qty)
                leg["entry_price"] = round(float(entry_price), 2)
                return leg
        raise ValueError(f"Leg {leg_id!r} not found.")

    def _leg_pnl(self, leg, price):
        if price is None:
            return None
        diff = (leg["entry_price"] - price) if leg["side"] == "SELL" else (price - leg["entry_price"])
        return round(diff * manual_trades.lot_of(leg) * leg["qty"], 2)

    def legs_with_live(self):
        out = []
        for leg in self.legs:
            candles = self.last_ce if leg["type"] == "CE" else self.last_pe
            rec = candles.get(leg["strike"])
            ltp = rec["close"] if rec else None
            row = dict(leg)
            row["ltp"] = ltp
            row["live_pnl"] = self._leg_pnl(leg, ltp)
            out.append(row)
        return out

    def synthetic_forward_now(self):
        """Put-call-parity-implied forward from whatever strikes the
        replay currently has quotes for -- the forward (not raw spot)
        doubles as both the payoff chart's x-axis anchor and the "S" fed
        into Black-Scholes, exactly like the live Greeks/curve code does
        (see greeks.synthetic_forward's docstring)."""
        # Only strikes where BOTH sides actually traded in this candle
        # (volume > 0 -- carried-forward stale closes pair a CE and PE
        # from different times and skew parity), then the 6 nearest the
        # rough centre -- same "near-ATM median" rule as the live chain.
        both = [k for k in set(self.last_ce) & set(self.last_pe)
                if self.last_ce[k].get("traded") and self.last_pe[k].get("traded")]
        if len(both) < 2:
            both = list(set(self.last_ce) & set(self.last_pe))
        if not both:
            return None
        rough = greeks_mod.synthetic_forward([(k, self.last_ce[k]["close"], self.last_pe[k]["close"]) for k in both])
        if rough is None:
            return None
        near = sorted(both, key=lambda k: abs(k - rough))[:6]
        return greeks_mod.synthetic_forward([(k, self.last_ce[k]["close"], self.last_pe[k]["close"]) for k in near])

    def _time_to_expiry(self):
        if not (0 <= self.index < len(self.all_ts)):
            return None
        ts = self.all_ts[self.index]
        T = (self.expiry_close - ts).total_seconds() / (365 * 86400)
        return T if T > 0 else None

    def payoff_curve(self, num_points=41, price_step=50):
        """At-expiry intrinsic-value line, anchored on the current
        synthetic forward (see payoff_curve() in manual_trades.py for the
        live equivalent -- same math, just no live spot to anchor on)."""
        if not self.legs:
            return []
        anchor = self.synthetic_forward_now()
        if not anchor:
            return []
        half = num_points // 2
        prices = [anchor + (i - half) * price_step for i in range(num_points)]
        # Same strike-corner points as manual_trades.payoff_curve().
        lo, hi = prices[0], prices[-1]
        prices = sorted(set(prices) | {float(l["strike"]) for l in self.legs if lo < l["strike"] < hi})
        return [{"price": round(S, 2), "pnl": round(manual_trades._expiry_pnl(self.legs, S), 2)} for S in prices]

    def now_curve(self, prices):
        """The historical analogue of manual_trades.compute_todays_curve():
        at the CURRENTLY SIMULATED candle, solve each leg's IV from its own
        historical LTP at this exact point in the replay (real T-to-expiry
        from this simulated moment, not wall-clock now), then reprice every
        leg via Black-Scholes across `prices`. Returns None if any leg's IV
        can't be solved right here -- same all-or-nothing rule as the live
        version, so the line never silently mixes a real T+0 leg with an
        intrinsic-only one."""
        if not self.legs or not prices:
            return None
        T = self._time_to_expiry()
        if T is None:
            return None
        forward = self.synthetic_forward_now()
        if not forward:
            return None
        leg_calcs = []
        for leg in self.legs:
            candles = self.last_ce if leg["type"] == "CE" else self.last_pe
            rec = candles.get(leg["strike"])
            if rec is None:
                return None
            is_call = leg["type"] == "CE"
            iv = greeks_mod.implied_vol_fwd(rec["close"], forward, leg["strike"], T, greeks_mod.RISK_FREE_RATE, is_call)
            if iv is None:
                return None
            leg_calcs.append((leg, iv, is_call))

        curve = []
        for S in prices:
            total = 0.0
            for leg, iv, is_call in leg_calcs:
                price_now = greeks_mod.b76_price(S, leg["strike"], T, greeks_mod.RISK_FREE_RATE, iv, is_call)
                if leg["side"] == "SELL":
                    total += (leg["entry_price"] - price_now) * manual_trades.lot_of(leg) * leg["qty"]
                else:
                    total += (price_now - leg["entry_price"]) * manual_trades.lot_of(leg) * leg["qty"]
            curve.append({"price": round(S, 2), "pnl": round(total, 2)})
        return curve

    def greeks_table(self):
        if not self.legs:
            return []
        T = self._time_to_expiry()
        forward = self.synthetic_forward_now() if T is not None else None
        out = []
        for leg in self.legs:
            candles = self.last_ce if leg["type"] == "CE" else self.last_pe
            rec = candles.get(leg["strike"])
            g = None
            if rec is not None and forward and T is not None:
                g = greeks_mod.option_greeks_fwd(rec["close"], forward, leg["strike"], T,
                                              greeks_mod.RISK_FREE_RATE, leg["type"] == "CE")
            out.append({"id": leg["id"], "strike": leg["strike"], "type": leg["type"], "side": leg["side"],
                        **(g or {"iv": None, "delta": None, "gamma": None, "theta": None, "vega": None})})
        return out

    def leg_bundle(self):
        """Everything the frontend needs to (re)draw the Positions panel,
        Closed Positions list, Payoff chart and Greeks table -- returned
        as one bundle after every step/add/remove/close so the UI never
        has to guess when to re-fetch."""
        legs = self.legs_with_live()
        curve = self.payoff_curve()
        now_curve = self.now_curve([c["price"] for c in curve]) if curve else None
        stats = manual_trades.summary(self.legs, curve)
        anchor = self.synthetic_forward_now()
        live_pnl = round(sum(l["live_pnl"] for l in legs if l["live_pnl"] is not None), 2) if legs else 0.0
        realized_pnl = round(sum(l["pnl"] for l in self.closed_legs if l["pnl"] is not None), 2) if self.closed_legs else 0.0
        return {
            "legs": legs, "closed_legs": self.closed_legs, "curve": curve, "now_curve": now_curve,
            "stats": stats, "greeks": self.greeks_table(),
            "live_pnl": live_pnl, "realized_pnl": realized_pnl, "total_pnl": round(live_pnl + realized_pnl, 2),
            "forward": round(anchor, 2) if anchor else None,
            "spot": self.spot_now(), "lot_size": self.lot_size,
        }


if __name__ == "__main__":
    dates = list_available_dates()
    print(f"{len(dates)} dates available: {dates[:3]} ... {dates[-3:]}")
    sess = SimulationSession(dates[0])
    print(f"session for {dates[0]}: {sess.total_candles} candles")
    snap = None
    while True:
        snap = sess.step()
        if snap is None:
            break
        if snap["finished"]:
            print("FINAL at index", snap["index"])
            break
