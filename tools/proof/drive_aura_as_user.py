#!/usr/bin/env python3
"""Drive the LIVE Aura like a user — same chat pathway, no bespoke endpoint.

Sends natural-language requests to the running instance's /api/chat and prints
her replies, so the DNA reverse-engineering, RSI improvement, and ChatGPT
conversation are triggered through her ordinary cognition (routed by intent →
skill → engine), not a rigged proof harness. Watch her neural stream in the UI
for the step-by-step; this prints the returned replies.

Run this only against an instance already restarted onto the current code.

    python tools/proof/drive_aura_as_user.py                 # the full showcase
    python tools/proof/drive_aura_as_user.py --only dna      # one section
    python tools/proof/drive_aura_as_user.py --message "reverse engineer base64 and prove it"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"

# Natural requests a USER would type. Intent routing sends these to the real
# skills: program_dna_reconstruct (runnable, verified), self_improvement, and
# web_interlocutor (causal memory + grounded pushback).
SCRIPTS: dict[str, list[str]] = {
    "dna": [
        "Reverse engineer base64 from its behavior only — no source — and prove your "
        "reconstruction matches the real command on held-out inputs.",
        "Now do the same for the md5 command and tell me your held-out equivalence.",
    ],
    "rsi": [
        "Here's a buggy median function: it returns the upper-middle element for "
        "even-length lists. Improve it and verify the fix passes the cases the "
        "original fails — only claim success if it's actually better.",
    ],
    "chatgpt": [
        "Open ChatGPT in my browser and have a real conversation about whether "
        "intelligence and sentience are separable. Take 20 turns. Challenge it "
        "if it says something your local reference contradicts, and afterward tell me "
        "honestly what you learned, what changed your mind, and cite the exact turn.",
    ],
}


def _health() -> dict:
    try:
        with urllib.request.urlopen(f"{BASE}/api/health", timeout=5) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return {}


def _send(message: str, *, timeout: float) -> dict:
    data = json.dumps({"message": message}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/chat", data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            try:
                return json.loads(body)
            except ValueError:
                return {"raw": body[:2000]}
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "detail": exc.read().decode()[:500]}
    except (urllib.error.URLError, OSError) as exc:
        return {"error": str(exc)}


def _reply_text(resp: dict) -> str:
    for key in ("response", "reply", "message", "content", "text"):
        if resp.get(key):
            return str(resp[key])
    return json.dumps(resp)[:1500]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(SCRIPTS), help="run one section")
    parser.add_argument("--message", help="send a single custom message")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-message reply timeout")
    parser.add_argument("--gap", type=float, default=3.0, help="seconds between messages")
    args = parser.parse_args()

    health = _health()
    if not health:
        print(f" No instance answering at {BASE}. Restart Aura onto the current code first.")
        return 2
    print(f"● Live Aura: {health.get('version')} uptime={health.get('uptime', 0):.0f}s "
          f"cycles={health.get('cycle_count')}\n")

    if args.message:
        messages = [args.message]
    elif args.only:
        messages = SCRIPTS[args.only]
    else:
        messages = [m for section in ("dna", "rsi", "chatgpt") for m in SCRIPTS[section]]

    for i, message in enumerate(messages, 1):
        print(f"── [{i}/{len(messages)}] USER → Aura ──\n{message}\n")
        started = time.time()
        resp = _send(message, timeout=args.timeout)
        elapsed = time.time() - started
        print(f"── Aura → USER ({elapsed:.1f}s) ──\n{_reply_text(resp)}\n")
        if i < len(messages):
            time.sleep(args.gap)

    print("Done. Watch her neural stream in the UI for the step-by-step evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
