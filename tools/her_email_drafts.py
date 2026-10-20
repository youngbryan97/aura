#!/usr/bin/env python3
"""Her replies to her mail, held for Bryan: read them, and send the ones he lets go.

    her_email_drafts.py list          the replies she wrote and has not sent, each with what it answers
    her_email_drafts.py send ID       send that reply, from her account, in the thread it answers

Held by core/skills/her_replies_to_mail.py until AURA_EMAIL_SENDS_WITHOUT_ASKING=1 is in her launch environment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from core.skills.her_replies_to_mail import _drafts_path, drafts

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("list", "send"))
    parser.add_argument("id", nargs="?")
    args = parser.parse_args()
    waiting = [d for d in drafts() if not d.get("sent")]
    if args.step == "list":
        for draft in waiting:
            print(f"{draft['id']}  to {draft['to']}  re: {draft['subject']}\n{draft['reply']}\n")
        print(f"{len(waiting)} waiting")
        return 0
    chosen = next((d for d in waiting if d["id"] == args.id), None)
    if chosen is None:
        print(f"no waiting reply {args.id}")
        return 1
    from core.skills.email_adapter import EmailAdapterSkill, EmailInput

    sent = asyncio.run(EmailAdapterSkill()._handle_reply(EmailInput(mode="reply", uid=chosen["uid"], body=chosen["reply"])))
    print(json.dumps({k: v for k, v in sent.items() if k in ("ok", "error", "status")}))
    if sent.get("ok"):
        with _drafts_path().open("a", encoding="utf-8") as kept:
            kept.write(json.dumps({"id": chosen["id"], "sent": True, "sent_at": time.time()}) + "\n")
    return 0 if sent.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
