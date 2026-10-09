"""arrow_auth.py -- fully automated Arrow (iRage) login, the same idea as
fyers_auth.py: no daily browser click. Endpoints and payloads follow the
broker's own Go SDK (github.com/arrow-trade/go-arrow, arrow/auth.go):

  1. /auth/app/login          -- user id + password -> requestId
  2. /auth/validate-2fa       -- 6-digit TOTP generated locally from
                                 ARROW_TOTP_KEY -> redirectUrl carrying the
                                 request-token
  3. /auth/app/authenticate-token -- sha256("appID:appSecret:request-token")
                                 -> access token (valid ~24 h)

A failed step raises immediately and is NOT retried in a loop: repeated bad
password/TOTP attempts can lock the account. The token is cached in
.arrow_session.json (git-ignored) for the calendar day and never printed.
"""
import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyotp
import requests
from dotenv import dotenv_values

import paths

_ENV = dotenv_values(Path(paths.BASE_DIR) / ".env")
APP_ID = (_ENV.get("ARROW_APP_ID") or "").strip()
_APP_SECRET = (_ENV.get("ARROW_APP_SECRET") or "").strip()
_USER_ID = (_ENV.get("ARROW_USER_ID") or "").strip()
_PASSWORD = _ENV.get("ARROW_PASSWORD") or ""
_TOTP_KEY = (_ENV.get("ARROW_TOTP_KEY") or "").replace(" ", "").upper()

LOGIN_URL = "https://edge.arrow.trade/auth/app/login"
TOTP_URL = "https://edge.arrow.trade/auth/validate-2fa"
TOKEN_URL = "https://edge.arrow.trade/auth/app/authenticate-token"
SESSION_CACHE = Path(paths.BASE_DIR) / ".arrow_session.json"


class ArrowAuthError(RuntimeError):
    pass


def configured():
    return all((APP_ID, _APP_SECRET, _USER_ID, _PASSWORD, _TOTP_KEY))


def _post(url, payload):
    resp = requests.post(url, json=payload, timeout=20)
    try:
        body = resp.json()
    except ValueError:
        raise ArrowAuthError(f"{url} returned non-JSON (HTTP {resp.status_code})")
    if body.get("status") != "success":
        # message only -- the body never carries our secrets, but keep it short
        raise ArrowAuthError(f"{url.rsplit('/', 1)[-1]} failed: {body.get('message') or body.get('status')}")
    return body.get("data") or {}


def _login_steps():
    if not configured():
        raise ArrowAuthError("ARROW_* credentials missing in .env")
    d = _post(LOGIN_URL, {"userID": _USER_ID, "password": _PASSWORD, "captchaValue": "", "captchaID": None,
                          "appID": APP_ID, "isAppLogin": True})
    req_id = d.get("requestId")
    if not req_id:
        raise ArrowAuthError("login: no requestId in response")
    d = _post(TOTP_URL, {"code": pyotp.TOTP(_TOTP_KEY).now(), "requestId": req_id, "userID": _USER_ID})
    rt = (parse_qs(urlparse(d.get("redirectUrl") or "").query).get("request-token") or [None])[0]
    if not rt:
        raise ArrowAuthError("validate-2fa: no request-token in redirectUrl")
    cs = hashlib.sha256(f"{APP_ID}:{_APP_SECRET}:{rt}".encode()).hexdigest()
    d = _post(TOKEN_URL, {"checkSum": cs, "checksum": cs, "token": rt, "appId": APP_ID})
    tok = d.get("token")
    if not tok:
        raise ArrowAuthError("authenticate-token: no token in response")
    return tok


_mem = None          # (date, token)
_lock = threading.Lock()
_last_attempt = 0.0
_RETRY_SEC = 600


def _load_cached():
    global _mem
    today = time.strftime("%Y-%m-%d")
    if _mem and _mem[0] == today:
        return _mem[1]
    try:
        data = json.loads(SESSION_CACHE.read_text())
    except (OSError, ValueError):
        return None
    if data.get("date") != today or not data.get("token"):
        return None
    _mem = (today, data["token"])
    return data["token"]


def _save(tok):
    global _mem
    today = time.strftime("%Y-%m-%d")
    SESSION_CACHE.write_text(json.dumps({"date": today, "token": tok, "saved": time.time()}))
    _mem = (today, tok)


def invalidate():
    """Called when Arrow answers 'invalid session' -- the next call logs in again."""
    global _mem
    _mem = None
    try:
        SESSION_CACHE.unlink()
    except OSError:
        pass


def get_token():
    """Today's access token, logging in when there is none. A failed login is
    retried at most every 10 minutes (never in a tight loop)."""
    global _last_attempt
    tok = _load_cached()
    if tok:
        return tok
    with _lock:
        tok = _load_cached()
        if tok:
            return tok
        if time.time() - _last_attempt < _RETRY_SEC:
            raise ArrowAuthError(f"Arrow login failed recently; next attempt within {_RETRY_SEC // 60} min")
        _last_attempt = time.time()
        tok = _login_steps()
        _save(tok)
        return tok


def headers():
    return {"appId": APP_ID, "token": get_token()}


if __name__ == "__main__":
    t = get_token()
    print(f"Arrow login OK. Token cached in {SESSION_CACHE.name} ({len(t)} chars).")
