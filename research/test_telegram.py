#!/usr/bin/env python3
"""Quick script to verify Telegram bot token and chat ID are working."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
import os


def main() -> int:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token:
        print("FAIL: TELEGRAM_BOT_TOKEN not set in .env")
        return 1
    if not chat_id:
        print("FAIL: TELEGRAM_CHAT_ID not set in .env")
        return 1

    print(f"Token:   {token[:10]}...{token[-4:]}")
    print(f"Chat ID: {chat_id}")

    # Step 1: Test bot validity with getMe
    print("\n[1/2] Checking bot token with getMe...")
    try:
        url = f"https://api.telegram.org/bot{token}/getMe"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("ok"):
            bot = data["result"]
            print(f"  OK: Bot name = @{bot.get('username', '?')}")
        else:
            print(f"  FAIL: API returned ok=false: {data}")
            return 1
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        print(f"  FAIL: HTTP {e.code} — {body}")
        return 1
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        print("  (Check your network — can you reach api.telegram.org?)")
        return 1

    # Step 2: Send a test message
    print("\n[2/2] Sending test message...")
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": "Telegram alert test from Apex trading bot.",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("ok"):
            print("  OK: Message delivered. Check your Telegram.")
        else:
            print(f"  FAIL: API returned ok=false: {data}")
            return 1
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        print(f"  FAIL: HTTP {e.code} — {body}")
        if e.code == 400:
            print("  Likely cause: invalid chat_id or bot hasn't been started")
        elif e.code == 401:
            print("  Likely cause: invalid bot token")
        return 1
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        print("  (Check your network — can you reach api.telegram.org?)")
        return 1

    print("\nSUCCESS: Both token and chat_id are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
