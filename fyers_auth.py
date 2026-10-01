"""fyers_auth.py -- fully automated Fyers login (no daily manual browser
click), adapted from the StockDashboard project's proven implementation.
Fyers access tokens are valid for the trading day only, so this needs to
run once per day (e.g. lazily, the first time get_access_token() is called
and no valid cached token exists).

Flow (Fyers' own web login does this in the browser; this replicates it via
direct API calls -- same endpoints, no Selenium):
  1. send_login_otp   -- identifies the account by FY-ID
  2. verify_otp       -- the "OTP" here is the 6-digit TOTP code, generated
                         locally from FYERS_TOTP_KEY (no SMS/app wait)
  3. verify_pin       -- trading PIN, returns a short-lived session token
  4. token (auth code)-- exchanges the session token for an auth_code via
                         the same redirect_uri/app_id the manual flow uses
  5. validate-authcode-- standard OAuth code exchange for the real
                         access_token + refresh_token

Each step is checked for a real error before proceeding to the next -- a
wrong PIN or TOTP must NOT be silently retried (repeated bad attempts can
lock the account), so any non-"ok" response raises immediately instead of
looping.
"""
import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pyotp
import requests
from dotenv import load_dotenv

import paths

load_dotenv(Path(paths.BASE_DIR) / ".env")

APP_ID = os.environ["FYERS_APP_ID"]
SECRET_KEY = os.environ["FYERS_SECRET_KEY"]
REDIRECT_URI = os.environ["FYERS_REDIRECT_URI"]
CLIENT_ID = os.environ["FYERS_CLIENT_ID"]
PIN = os.environ["FYERS_PIN"]
TOTP_KEY = os.environ["FYERS_TOTP_KEY"]

BASE_T1 = "https://api-t1.fyers.in"
BASE_T2 = "https://api-t2.fyers.in"
SESSION_CACHE = Path(paths.BASE_DIR) / ".fyers_session.json"


class FyersAuthError(RuntimeError):
    pass


def _post(url, payload, headers=None):
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    try:
        body = resp.json()
    except ValueError:
        raise FyersAuthError(f"{url} returned non-JSON (HTTP {resp.status_code}): {resp.text[:200]}")
    return resp, body


def _field(body, name):
    if name in body:
        return body[name]
    return body.get("data", {}).get(name)


def _step1_send_login_otp():
    url = f"{BASE_T2}/vagator/v2/send_login_otp"
    resp, body = _post(url, {"fy_id": CLIENT_ID, "app_id": "2"})
    request_key = _field(body, "request_key")
    if body.get("s") != "ok" or not request_key:
        raise FyersAuthError(f"send_login_otp failed: {body}")
    return request_key


def _step2_verify_totp(request_key):
    totp_code = pyotp.TOTP(TOTP_KEY).now()
    url = f"{BASE_T2}/vagator/v2/verify_otp"
    resp, body = _post(url, {"request_key": request_key, "otp": totp_code})
    new_request_key = _field(body, "request_key")
    if body.get("s") != "ok" or not new_request_key:
        raise FyersAuthError(f"verify_otp (TOTP) failed: {body}")
    return new_request_key


def _step3_verify_pin(request_key):
    url = f"{BASE_T2}/vagator/v2/verify_pin"
    resp, body = _post(url, {"request_key": request_key, "identity_type": "pin", "identifier": PIN})
    access_token = _field(body, "access_token")
    if body.get("s") != "ok" or not access_token:
        raise FyersAuthError(f"verify_pin failed: {body}")
    return access_token  # short-lived SESSION token, not the final API token


def _step4_get_auth_code(session_token):
    url = f"{BASE_T1}/api/v3/token"
    payload = {
        "fyers_id": CLIENT_ID,
        "app_id": APP_ID.split("-")[0],
        "redirect_uri": REDIRECT_URI,
        "appType": APP_ID.split("-")[1] if "-" in APP_ID else "100",
        "code_challenge": "",
        "state": "optionmarket",
        "scope": "",
        "nonce": "",
        "response_type": "code",
        "create_cookie": True,
    }
    resp = requests.post(url, json=payload, headers={"Authorization": f"Bearer {session_token}"}, timeout=15)
    body = resp.json()
    redirect_url = _field(body, "Url")
    if body.get("s") != "ok" or not redirect_url:
        raise FyersAuthError(f"auth-code step failed: {body}")
    from urllib.parse import urlparse, parse_qs
    qs = parse_qs(urlparse(redirect_url).query)
    if "auth_code" not in qs:
        raise FyersAuthError(f"no auth_code in redirect URL: {redirect_url}")
    return qs["auth_code"][0]


def exchange_auth_code(auth_code):
    app_id_hash = hashlib.sha256(f"{APP_ID}:{SECRET_KEY}".encode()).hexdigest()
    resp = requests.post(
        f"{BASE_T1}/api/v3/validate-authcode",
        json={"grant_type": "authorization_code", "appIdHash": app_id_hash, "code": auth_code},
        timeout=15,
    )
    body = resp.json()
    if resp.status_code != 200 or "access_token" not in body:
        raise FyersAuthError(f"validate-authcode failed: {resp.status_code} {body}")
    return body["access_token"], body.get("refresh_token")


def _jwt_exp(token):
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("exp")
    except Exception:
        return None


_token_mem_cache = None  # (cached_at_monotonic, token)
_TOKEN_MEM_TTL_SEC = 30  # every Fyers call in the whole app goes through
# get_auth_header() -> get_access_token() -> this function -- with the
# Flask server now threaded and multiple pages polling every few seconds,
# that was a disk read + JSON parse + JWT decode on every single request
# for a file that only actually changes once a day (at login).


def _load_cached_token():
    global _token_mem_cache
    now = time.monotonic()
    if _token_mem_cache is not None and (now - _token_mem_cache[0]) < _TOKEN_MEM_TTL_SEC:
        return _token_mem_cache[1]

    if not SESSION_CACHE.exists():
        return None
    try:
        data = json.loads(SESSION_CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    token = data.get("access_token")
    if not token:
        return None
    exp = _jwt_exp(token)
    if exp is not None and time.time() >= exp:
        return None
    if data.get("date") != time.strftime("%Y-%m-%d"):
        return None
    _token_mem_cache = (now, token)
    return token


def _save_cached_token(access_token, refresh_token=None):
    global _token_mem_cache
    SESSION_CACHE.write_text(json.dumps({
        "date": time.strftime("%Y-%m-%d"),
        "access_token": access_token,
        "refresh_token": refresh_token,
    }))
    _token_mem_cache = None  # force the next _load_cached_token() to pick up the fresh token immediately


def login(force=False):
    if not force:
        cached = _load_cached_token()
        if cached:
            return cached

    request_key = _step1_send_login_otp()
    request_key = _step2_verify_totp(request_key)
    session_token = _step3_verify_pin(request_key)
    auth_code = _step4_get_auth_code(session_token)
    access_token, refresh_token = exchange_auth_code(auth_code)
    _save_cached_token(access_token, refresh_token)
    return access_token


_relogin_lock = threading.Lock()
_last_relogin_attempt = 0.0
_RELOGIN_RETRY_SEC = 600  # a failed login is NOT retried in a tight loop --
# repeated bad TOTP/PIN attempts can lock the Fyers account.


def get_access_token():
    """Today's token, logging in again automatically when the cached one is
    from a previous day. Before this, login() only ran at app startup, so an
    app left running overnight failed every Fyers call the next morning
    (chain, risk monitor, paper trading) until it was restarted by hand."""
    global _last_relogin_attempt
    cached = _load_cached_token()
    if cached:
        return cached
    with _relogin_lock:
        cached = _load_cached_token()  # another thread may have just logged in
        if cached:
            return cached
        if time.time() - _last_relogin_attempt < _RELOGIN_RETRY_SEC:
            raise FyersAuthError("Fyers token expired and the last automatic re-login failed; "
                                 f"retrying within {_RELOGIN_RETRY_SEC // 60} min.")
        _last_relogin_attempt = time.time()
        print("[fyers_auth] Cached token is stale -- logging in again (daily refresh)...")
        return login(force=True)


def get_auth_header():
    return f"{APP_ID}:{get_access_token()}"


if __name__ == "__main__":
    print("Logging in to Fyers (automated TOTP flow)...")
    token = login()
    print(f"Success. Token cached in {SESSION_CACHE.name} ({len(token)} chars).")
    resp = requests.get(f"{BASE_T1}/api/v3/profile", headers={"Authorization": f"{APP_ID}:{token}"}, timeout=10).json()
    if resp.get("s") == "ok":
        d = resp["data"]
        print(f"Profile check OK -- {d.get('name', '?')} ({d.get('fy_id', '?')})")
    else:
        print(f"Profile check response: {resp}")
