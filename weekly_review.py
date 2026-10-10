"""weekly_review.py -- one Telegram message at the end of each trading week.

Sent by the evening job (daily_history.update, after the IV history is
updated) on the last trading day of the ISO week, so it arrives even when the
app is closed. Nothing here decides anything; it only reports:

  forward test   each plan variant's paper results vs the backtest band
  this week      trades opened / closed, the week's paper P&L
  checks         live IV-rule readings vs the evening history (skew_check)
  data           recorder cycles per day, the evening job's last run
  app            memory of the running OptionMarket.exe, if it is running

Usage:  python weekly_review.py          (prints; add --send to send now)
"""
import json
import os
import sqlite3
import sys
from datetime import date, timedelta

import market_calendar as mc
import paths


def _last_trading_day_of_week(d):
    nxt = d + timedelta(days=1)
    while not mc.is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt.isocalendar()[1] != d.isocalendar()[1]


def _rs(v):
    return ("−" if v < 0 else "") + "₹" + f"{abs(v):,.0f}"


def build(today=None):
    import plan
    today = today or mc.now_ist().date()
    monday = today - timedelta(days=today.weekday())
    lines = [f"🗓 <b>Weekly review · week of {monday:%d %b}</b>"]
    # forward test
    book = plan._load_book()["trades"]
    closed = [t for t in book.values() if t["status"] not in ("open", "void")]
    fw = plan._forward_test(closed)
    lines.append("<b>Forward test</b> (paper, closed trades):")
    for k, f in fw.items():
        if not f["n"]:
            lines.append(f"  {k}: no trades yet (test avg {_rs(f['test_avg'])})")
            continue
        st = "too early" if f["n"] < 10 else ("on track" if f.get("on_track") else ("above test" if f.get("above") else "BELOW TEST — review"))
        lines.append(f"  {k}: {f['n']} tr · {_rs(f['total'])} · avg {_rs(f['avg'])} vs test {_rs(f['test_avg'])} · {st}")
    # this week
    wk = lambda s: s and monday.isoformat() <= s[:10] <= today.isoformat()
    opened = [t for t in book.values() if wk(t["day"]) and t.get("status") != "void"]
    shut = [t for t in closed if wk(t.get("closed") or "") or (t["status"] == "expired" and wk(t["expiry"]))]
    lines.append(f"<b>This week</b>: opened {len(opened)}, closed {len(shut)}" + (f" → {_rs(sum(t['pnl'] for t in shut))}" if shut else ""))
    for t in shut:
        lines.append(f"  {t['setup']} {t['strategy']} {_rs(t['pnl'])}" + (f" (worst {_rs(t['mae'])})" if t.get("mae") is not None else ""))
    op = [t for t in book.values() if t["status"] == "open"]
    if op:
        lines.append("  open: " + "; ".join(f"{t['setup']} {t['strategy']} {_rs(t['pnl'])}" for t in op))
    # live vs history readings
    try:
        import skew_check
        d = skew_check.compare()
        if len(d):
            d = d[(d.day >= monday.isoformat()) & (d.day <= today.isoformat())]
        if len(d):
            parts = []
            if d.iv_gap.notna().any():
                parts.append(f"ATM IV gap mean |{d.iv_gap.abs().mean():.2f}| pts")
            if "skew_gap" in d and d.skew_gap.notna().any():
                parts.append(f"skew gap mean |{d.skew_gap.abs().mean():.2f}| pts")
            lines.append(f"<b>Checks</b>: {len(d)} live readings vs history — " + (", ".join(parts) if parts else "history not in yet"))
        else:
            lines.append("<b>Checks</b>: no live readings logged this week")
    except Exception as e:
        lines.append(f"<b>Checks</b>: comparison failed ({type(e).__name__})")
    # data health
    rec_dir = os.environ.get("NIFTY_MICRO_DIR") or r"E:\nifty_microstructure"
    rec = []
    d0 = monday
    while d0 <= today:
        if mc.is_trading_day(d0):
            db = os.path.join(rec_dir, f"micro_{d0:%Y%m%d}.sqlite")
            if os.path.exists(db):
                try:
                    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                    ok = con.execute("SELECT COUNT(*) FROM cycles WHERE status='ok'").fetchone()[0]
                    con.close()
                    rec.append(f"{d0:%a} {ok}")
                except Exception:
                    rec.append(f"{d0:%a} ?")
            else:
                rec.append(f"{d0:%a} none")
        d0 += timedelta(days=1)
    lines.append("<b>Recorder</b> ok cycles (of 77): " + " · ".join(rec))
    try:
        st = json.load(open(os.path.join(paths.BASE_DIR, "data", "hist1m", "daily_status.json")))
        lines.append(f"<b>Evening job</b>: last run {st.get('last_run', '?')[:16]} · {st.get('seconds')} s")
    except (OSError, ValueError):
        lines.append("<b>Evening job</b>: no status file")
    try:
        import psutil
        mem = [p.info["memory_info"].rss for p in psutil.process_iter(["name", "memory_info"]) if (p.info["name"] or "").lower() == "optionmarket.exe"]
        if mem:
            lines.append(f"<b>App</b> memory: {max(mem) / 1e6:,.0f} MB")
    except Exception:
        pass
    return "\n".join(lines)


def maybe_send(today=None):
    """Called by the evening job: send once on the week's last trading day."""
    import plan
    today = today or mc.now_ist().date()
    if not mc.is_trading_day(today) or not _last_trading_day_of_week(today):
        return False
    y, w, _ = today.isocalendar()
    return plan._send_once(f"weekly|{y}-W{w:02d}", build(today))


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    text = build()
    print(text)
    if "--send" in sys.argv:
        import telegram_notify as tg
        print("sent:", tg.send(text))
