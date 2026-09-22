#!/usr/bin/env python3
"""Cut a tape of what the person she talks to has said to her, for the battery to replay.

The conversation condition gave her one sentence three hundred times a run.
This reads the partner's turns out of her own conversation store, read only, in
the order they were said, and writes them where the battery can play them back
with `--conversation-tape`: the n-th conversation turn of a run hears the n-th
thing said. Only turns from sessions whose principal is the owner are taken,
because those are the turns she received as his.

The tape is written outside the repository by default and must stay there. A
run records its digest and length, never its words.

    python tools/record_conversation_tape.py
    python tools/record_conversation_tape.py --out /tmp/claude-501/conversation_tape.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Longest turn kept, in characters. The kernel's own ceiling on a message, so a
#: turn on the tape is one the desktop runtime would have accepted.
from core.kernel.kernel_interface import MAX_KERNEL_MESSAGE_CHARS  # noqa: E402


def partner_turns(store: Path, principal: str) -> tuple[list[str], list[int]]:
    """The owner's turns in the order they were said, and which session each was in."""
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT t.content, t.session_id, s.metadata FROM turns t "
            "JOIN sessions s ON s.id = t.session_id "
            "WHERE t.role = 'user' ORDER BY t.created_at, t.id"
        ).fetchall()
    finally:
        connection.close()
    turns: list[str] = []
    sessions: list[int] = []
    numbered: dict[str, int] = {}
    for content, session_id, metadata in rows:
        try:
            owner = json.loads(metadata or "{}").get("principal_id")
        except (TypeError, ValueError):
            owner = None
        text = str(content or "").strip()
        if owner != principal or not text:
            continue
        turns.append(text[:MAX_KERNEL_MESSAGE_CHARS])
        sessions.append(numbered.setdefault(str(session_id), len(numbered)))
    return turns, sessions


def main() -> int:
    from core.config import config

    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=config.paths.data_dir / "conversations.db")
    parser.add_argument("--principal", default="bryan")
    parser.add_argument(
        "--out", type=Path, default=Path.home() / ".aura" / "subject_core" / "conversation_tape.json"
    )
    args = parser.parse_args()
    if REPO in args.out.resolve().parents:
        raise SystemExit("a conversation tape holds her conversations and is never written inside the repository")

    from core.governance_context import local_internal_governed_scope
    from core.subject.archive import write_json
    from core.subject.conversation_tape import ConversationTape

    turns, sessions = partner_turns(args.store, args.principal)
    if not turns:
        raise SystemExit(f"no turns from principal {args.principal!r} in {args.store}")
    tape = ConversationTape(
        turns=tuple(turns),
        sessions=tuple(sessions),
        notes={"store": str(args.store), "principal": args.principal},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with local_internal_governed_scope("subject_core.record_conversation_tape"):
        write_json(args.out.parent, args.out.name, tape.as_dict())
    print(json.dumps({"out": str(args.out), **tape.provenance()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
