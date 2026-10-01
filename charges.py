"""charges.py -- estimated transaction costs for NSE index options, so P&L
can be shown NET instead of gross. Every P&L figure in the app used to
ignore costs entirely; for multi-leg / rolled strategies at 1 lot the
charges are often the same size as the edge.

Rates below are the published schedule for NSE equity options as
configured here -- VERIFY THEM against your broker's current charge sheet
and edit the constants if anything has changed (STT and exchange rates
are revised from time to time).
"""
BROKERAGE_PER_ORDER = 20.0        # Fyers options: flat Rs 20 per executed order
STT_SELL_PCT = 0.001              # 0.1% of premium, sell side
STT_EXERCISE_PCT = 0.00125        # 0.125% of intrinsic value, long ITM options exercised at expiry
EXCHANGE_TXN_PCT = 0.0003503      # NSE options: 0.03503% of premium turnover
SEBI_FEE_PCT = 0.000001           # Rs 10 per crore
STAMP_DUTY_BUY_PCT = 0.00003      # 0.003% of premium, buy side
GST_PCT = 0.18                    # on brokerage + exchange txn + SEBI fee


def order_charges(side, price, units):
    """Charges for ONE executed order. side: 'BUY'/'SELL'; price: premium
    per unit; units: contracts (lots x lot size)."""
    turnover = max(0.0, price) * units
    if turnover <= 0:
        return 0.0
    brokerage = BROKERAGE_PER_ORDER
    txn = turnover * EXCHANGE_TXN_PCT
    sebi = turnover * SEBI_FEE_PCT
    stt = turnover * STT_SELL_PCT if side == "SELL" else 0.0
    stamp = turnover * STAMP_DUTY_BUY_PCT if side == "BUY" else 0.0
    gst = (brokerage + txn + sebi) * GST_PCT
    return brokerage + txn + sebi + stt + stamp + gst


def round_trip(side, entry_price, exit_price, units, exit_reason=None):
    """Entry + exit charges for one leg. side is the ENTRY side. A leg
    settled at expiry has no exit order -- only exercise STT on a long
    that finished in the money."""
    total = order_charges(side, entry_price, units)
    if exit_price is None:
        return round(total, 2)
    if exit_reason == "EXPIRED":
        if side == "BUY" and exit_price > 0:
            total += exit_price * units * STT_EXERCISE_PCT
    else:
        total += order_charges("SELL" if side == "BUY" else "BUY", exit_price, units)
    return round(total, 2)


if __name__ == "__main__":
    # 1 lot (65) short at 100, bought back at 60 -- sanity check.
    print("short 100 -> 60, 1 lot:", round_trip("SELL", 100, 60, 65))
    print("long 50 -> 80, 1 lot:", round_trip("BUY", 50, 80, 65))
    print("long 50 expired ITM @ 120:", round_trip("BUY", 50, 120, 65, "EXPIRED"))
