"""launcher.py -- the single entry point for the double-clickable .exe.
Logs in to Fyers once, starts the background services (historical
recorder, risk monitor, Range Strategy) and the Flask dashboard, then opens
the dashboard in the default browser. Closing this window (or Ctrl+C) stops
everything. (The old intraday algo paper-trader was removed from the app at
the user's request on 2026-09-24.)
"""
import sys
import threading
import time
import traceback
import webbrowser

import fyers_auth

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DASHBOARD_URL = "http://localhost:5057"


def run_dashboard():
    import app as flask_app_module
    try:
        flask_app_module.app.run(host="127.0.0.1", port=5057, debug=False, use_reloader=False, threaded=True)
    except Exception:
        print("\n[dashboard] crashed:")
        traceback.print_exc()


def run_historical_recorder():
    import historical_recorder
    try:
        historical_recorder.start_background_recorder()
    except Exception:
        print("\n[historical_recorder] crashed:")
        traceback.print_exc()


def run_record_only():
    """`OptionMarket.exe --record` -- headless: no dashboard, no browser,
    no paper-trading, just today's live chain -> data/*.csv, then exit
    once the market closes. Meant to be launched by a Windows Scheduled
    Task at market open, independently of whether the GUI is ever opened
    that day -- see historical_recorder.run_headless_session()'s
    docstring for why the GUI's own recording can't cover a day the app
    wasn't running for."""
    import historical_recorder
    print("Logging in to Fyers (automated TOTP flow)...")
    try:
        fyers_auth.login()
        print("Fyers login OK.")
    except Exception as e:
        print(f"\nFyers login FAILED: {e}")
        sys.exit(1)
    historical_recorder.run_headless_session()


def run_daily_history():
    """`OptionMarket.exe --daily-history` -- the evening job (scheduled task
    "OptionMarket Daily History", 19:00 Mon-Fri): pulls the day's 1-min
    NIFTY + option history from Fyers into data/hist1m and fills any gaps.
    Replaces the old intraday recorder -- see daily_history.py."""
    import daily_history
    try:
        print(daily_history.update())
    except Exception:
        traceback.print_exc()
        sys.exit(1)


def main():
    if "--daily-history" in sys.argv:
        run_daily_history()
        return
    if "--record" in sys.argv:
        run_record_only()
        return

    print("=" * 60)
    print(" Option Market -- NIFTY Options Strategy Suite")
    print("=" * 60)
    print("\nLogging in to Fyers (automated TOTP flow)...")
    try:
        fyers_auth.login()
        print("Fyers login OK.")
    except Exception as e:
        print(f"\nFyers login FAILED: {e}")
        print("Check FYERS_APP_ID / FYERS_SECRET_KEY / FYERS_CLIENT_ID / "
              "FYERS_PIN / FYERS_TOTP_KEY in .env, then restart.")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # (the old intraday option-chain recorder is retired: data/hist1m is now
    # filled each evening by `OptionMarket.exe --daily-history`)
    import manual_monitor
    manual_monitor.start_background_monitor()  # SL/Target/square-off/expiry settlement, browser or not
    import range_strategy
    range_strategy.start_background()  # the user's weekly range method, paper-traded + Telegram
    import plan
    plan.start_background()            # "Aaj ka Plan": records + follows the plan's trades
    threading.Thread(target=run_dashboard, daemon=True).start()

    time.sleep(2)
    print(f"\nOpening dashboard at {DASHBOARD_URL} ...")
    try:
        webbrowser.open(DASHBOARD_URL)
    except Exception:
        pass

    print(f"\nDashboard:      {DASHBOARD_URL}")
    print("Range Strategy: your weekly range method, paper-traded -- watch Telegram for updates")
    print("Risk monitor:   enforcing SL/Target/15:25 square-off on Strategy Builder legs")
    print("\nKeep this window open. Close it (or Ctrl+C) to stop everything.\n")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
