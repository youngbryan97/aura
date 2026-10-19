#!/usr/bin/env python3
"""Rows of her live learner's training buffer that no person said, moved out of it (quarantined, never deleted).

A row is kept when a person said its user turn and it passes the live learner's own contamination gate: the turn is a
message typed in her chat (``conversations.db``, role ``user``, kept from 20 September 2026), or the row was recorded
as a conversation turn (``_meta.source``, her chats of March to August, before that store began); every other row is moved, whole, to a quarantine file beside the
buffer, after the buffer is copied. On 19 October 2026 9,138 of 9,638 rows were her runtime's own requests with her
replies ("[SWARM PROTOCOL: You are 'The Critic'...", "You are fixing a bug in your own code").

    quarantine_what_no_person_said.py            # what would move, by the opening of its user turn
    quarantine_what_no_person_said.py --apply    # copy the buffer, then move the rows
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = Path.home() / ".aura/data"


def _normal(text: str) -> str:
    return " ".join(str(text or "").split())


def user_and_reply(row: dict) -> tuple[str, str]:
    """The user turn and her reply of a buffer row, in either of the two formats the buffer holds."""
    messages = row.get("messages")
    if isinstance(messages, list):
        user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        reply = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
        return str(user), str(reply)
    text = str(row.get("text") or "")
    after = text.split("<|user|>", 1)[-1]
    user, _, reply = after.partition("<|assistant|>")
    return user, reply


def what_people_typed(db: Path) -> set[str]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return {_normal(c) for (c,) in conn.execute("select content from turns where role = 'user'") if c}


def sort_rows(buffer: Path, db: Path) -> tuple[list[str], list[str], Counter]:
    from core.learning.live_learner import training_contamination_reasons

    people = what_people_typed(db)
    kept, moved, why = [], [], Counter()
    for line in buffer.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            moved.append(line)
            why["unreadable"] += 1
            continue
        user, reply = user_and_reply(row)
        said_in_chat = (row.get("_meta") or {}).get("source") == "conversation"
        if not said_in_chat and _normal(user) not in people:
            moved.append(line)
            why["no person said it: " + _normal(user)[:48]] += 1
        elif training_contamination_reasons(user, reply):
            moved.append(line)
            why["contaminated"] += 1
        else:
            kept.append(line)
    return kept, moved, why


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--buffer", type=Path, default=DATA / "learning/experience_buffer.jsonl")
    parser.add_argument("--conversations", type=Path, default=DATA / "conversations.db")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    kept, moved, why = sort_rows(args.buffer, args.conversations)
    print(json.dumps({"rows": len(kept) + len(moved), "kept": len(kept), "moved": len(moved)}))
    for reason, n in why.most_common(15):
        print(f"{n:6d}  {reason}")
    if args.apply and moved:
        stamp = time.strftime("%Y%m%d")
        shutil.copy2(args.buffer, args.buffer.with_name(f"{args.buffer.name}.before-quarantine-{stamp}"))
        quarantine = args.buffer.with_name(f"{args.buffer.stem}.no-person-said-{stamp}.jsonl")
        with quarantine.open("a", encoding="utf-8") as sink:
            sink.write("".join(line + "\n" for line in moved))
        tmp = args.buffer.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        tmp.replace(args.buffer)
        print(f"moved {len(moved)} rows to {quarantine}; kept {len(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
