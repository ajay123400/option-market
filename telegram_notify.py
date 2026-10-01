"""telegram_notify.py -- Telegram Bot API delivery for paper-trade events.
Plain HTTPS Bot API (sendMessage) via `requests`, adapted from
StockDashboard's telegram_notify.py (same pattern, same env var names, so
the same bot/chat can be reused across both projects, or a fresh bot can be
pointed here independently).

Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env (create a bot via
@BotFather on Telegram, send it /start, then read the chat id from
https://api.telegram.org/bot<token>/getUpdates).
"""
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

import paths

load_dotenv(Path(paths.BASE_DIR) / ".env")

logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def is_configured():
    return bool(BOT_TOKEN and CHAT_ID)


def send(text):
    """Returns True if Telegram accepted the message, False otherwise --
    never raises, so a missing/unconfigured bot or a transient network error
    can't crash the paper-trade loop."""
    if not is_configured():
        logger.warning(
            "Telegram not configured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
            "missing in .env) -- message not sent: %s", text,
        )
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.error("Telegram sendMessage failed: %s %s", resp.status_code, resp.text)
        return resp.ok
    except requests.RequestException:
        logger.exception("Telegram sendMessage request failed")
        return False


if __name__ == "__main__":
    if not is_configured():
        print("Not configured -- add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env first.")
    else:
        ok = send("Test message from Option Market paper-trade bot.")
        print("Sent OK" if ok else "Send failed -- check logs above.")
