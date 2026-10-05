"""Phase A Step 1: forward bid/ask + microstructure recorder for NIFTY options (read-only data collection).

No orders, no trading actions, no strategy maths, no authentication of its own: the Fyers access token is read from the app's daily cache and the recorder NEVER logs in.
See collector/README.md. Nothing in the existing app imports this package.
"""
SCHEMA_VERSION = 1
