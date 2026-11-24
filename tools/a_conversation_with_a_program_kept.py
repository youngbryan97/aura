"""A conversation with a program that talks in lines, kept as it happened: everything said to it and by it.

Not part of her. A program another program drives (an engine a chess GUI runs, say) is told how to behave in lines it
is sent: its settings, and how long to think. What the driving program says it set is a claim; what it sent is the
fact. Put in the program's place, this passes every line through unchanged, both ways, and keeps each one with when it
was said and which way it went. The receipt for a game then reads what was sent during it (tools/board_game_receipts.py)
rather than what the program's defaults are or what a settings file says (suggestions of 23 November 2026: a receipt
recorded queried defaults and supplied settings, which does not show what the GUI used).

Usage, in the program's place (the GUI runs this instead of the program):
    python tools/a_conversation_with_a_program_kept.py --program BINARY --kept FILE.jsonl

FILE.jsonl gets one line per line said: {"at": ISO time, "said": "to" | "by", "line": ..., "session": n}, and a
"began" and an "ended" line for each run of the program.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class _Kept:
    """The record, a line at a time, written as it is said: a conversation cut off still has what was said."""

    def __init__(self, path: Path, program: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - held for the whole conversation
        self._lock = threading.Lock()
        self.session = f"{os.getpid()}-{_now()}"
        self.write("began", program)

    def write(self, said: str, line: str) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.write(json.dumps({"at": _now(), "said": said, "line": line, "session": self.session})
                                 + "\n")

    def close(self) -> None:
        self.write("ended", "")
        with self._lock:
            self._file.close()


def kept_conversation(program: list[str], kept_at: Path, given=sys.stdin, shown=sys.stdout, *,
                      ends_with_it: bool = False) -> int:
    """Runs ``program``, passing ``given``'s lines to it and its lines to ``shown``, each kept as it passes. With
    ``ends_with_it``, this process ends when the program does, as the program would have, though nothing has closed
    what it is given: a GUI that told its engine to quit waits for the engine's end, not for its own."""
    kept = _Kept(kept_at, " ".join(program))
    child = subprocess.Popen(program, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def from_it() -> None:
        for line in child.stdout:
            kept.write("by", line.rstrip("\n"))
            shown.write(line)
            shown.flush()
        if ends_with_it:
            code = child.wait()
            kept.close()
            os._exit(code)

    answering = threading.Thread(target=from_it, daemon=True)
    answering.start()
    try:
        for line in given:
            kept.write("to", line.rstrip("\n"))
            try:
                child.stdin.write(line)
                child.stdin.flush()
            # not a failure: the program has gone, and what it was told after is kept as said and not heard.
            except (BrokenPipeError, OSError):
                break
    finally:
        try:
            child.stdin.close()
        # not a failure: closing what is already closed.
        except (BrokenPipeError, OSError):
            pass
        code = child.wait()
        answering.join(timeout=5)
        kept.close()
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--program", required=True, help="the program to run in this one's place")
    parser.add_argument("--kept", type=Path, required=True, help="the JSON-lines file the conversation is kept in")
    asked, rest = parser.parse_known_args(argv)
    return kept_conversation([asked.program, *rest], asked.kept, ends_with_it=True)


if __name__ == "__main__":
    sys.exit(main())
