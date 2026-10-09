"""broker.py -- the one place the app logs in to its market-data broker.
Arrow (iRage) when .env has DATA_SOURCE=arrow (the app is moving off Fyers
entirely: from 9 Oct 2026 Fyers' Standard plan allows 5,000 data calls a
day), Fyers otherwise."""
import fyers_option_chain as chain_mod


def name():
    return "Arrow" if chain_mod.DATA_SOURCE == "arrow" else "Fyers"


def login():
    if chain_mod.DATA_SOURCE == "arrow":
        import arrow_auth
        return arrow_auth.get_token()
    import fyers_auth
    return fyers_auth.login()


def env_hint():
    if chain_mod.DATA_SOURCE == "arrow":
        return "Check ARROW_APP_ID / ARROW_APP_SECRET / ARROW_USER_ID / ARROW_PASSWORD / ARROW_TOTP_KEY in .env, then restart."
    return "Check FYERS_APP_ID / FYERS_SECRET_KEY / FYERS_CLIENT_ID / FYERS_PIN / FYERS_TOTP_KEY in .env, then restart."
