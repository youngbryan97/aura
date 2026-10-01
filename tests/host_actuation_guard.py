"""tests/host_actuation_guard.py — no test opens an app on the machine running it.

Bryan, 2026-10-01: "when we run computer skills, for some reason my notes app
always opens? and sometimes a browser? yesterday she did a search for
'robot'". The pointer was already kept off the real desktop in tests; the
other half of acting on a Mac — `open`, and AppleScript that tells an
application to do something — was not. A test of a desktop skill that reached
its executor launched Notes, or opened a search page, on whoever's desktop it
ran on.

What this refuses, during a test not marked ``live``, ``hardware`` or
``host_actuation``:

- ``open`` in any form: it launches an app, a file in its app, or a URL in a
  browser.
- ``osascript`` whose script tells an application other than System Events
  to do anything (telling an app launches it), activates or launches one,
  opens a location, or brings a process to the front.
- ``webbrowser.open`` and its two siblings.

Reading through System Events — which app is in front, the window titles — is
left alone: it changes nothing a person sees. A refused command runs as a
shell that exits 1 with the reason on stderr, so the caller takes its own
failure path. Each refusal is recorded with the test that asked, in
``$AURA_LOG_DIR/host_actuation_refused.jsonl``.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Markers under which a test is about the real desktop and may act on it.
ALLOWED_MARKERS = ("live", "hardware", "host_actuation")

#: In an AppleScript, what changes what a person sees.
_SCRIPT_ACTS = re.compile(
    r"\btell\s+application\s+(?:id\s+)?\"(?!System Events\")[^\"]+\""
    r"|\bactivate\b|\blaunch\b|\bopen\s+location\b"
    r"|\bset\s+frontmost\b",
    re.IGNORECASE,
)

#: In a shell line, `open` as a command.
_SHELL_OPENS = re.compile(r"(?:^|[;&|(]\s*|\bthen\s+|\bdo\s+)(?:/usr/bin/)?open\s", re.MULTILINE)

REASON = "refused in a test: this would act on the real desktop"


def _script_text(argv: Sequence[str]) -> str:
    """The AppleScript an ``osascript`` argv would run, as far as it can be read."""
    parts: list[str] = []
    rest = list(argv[1:])
    index = 0
    while index < len(rest):
        arg = str(rest[index])
        if arg == "-e" and index + 1 < len(rest):
            parts.append(str(rest[index + 1]))
            index += 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        path = Path(arg)
        try:
            if path.is_file() and path.stat().st_size < 1_000_000:
                parts.append(path.read_text(errors="replace"))
        except OSError:
            pass
        # Anything after the script path is its arguments, not script.
        break
    return "\n".join(parts)


def would_act_on_the_desktop(args: Any, *, shell: bool = False) -> bool:
    """Whether running ``args`` would launch, show or bring forward something."""
    if shell or isinstance(args, (str, bytes)):
        line = args.decode(errors="replace") if isinstance(args, bytes) else str(args)
        if _SHELL_OPENS.search(line):
            return True
        if "osascript" in line:
            try:
                words = shlex.split(line)
            except ValueError:
                return bool(_SCRIPT_ACTS.search(line))
            for start, word in enumerate(words):
                if os.path.basename(word) == "osascript":
                    return bool(_SCRIPT_ACTS.search(_script_text(words[start:])))
            return bool(_SCRIPT_ACTS.search(line))
        return False
    try:
        argv = [os.fspath(part) if not isinstance(part, bytes) else part.decode() for part in args]
    except TypeError:
        return False
    if not argv:
        return False
    program = os.path.basename(str(argv[0]))
    if program == "open":
        return True
    if program == "osascript":
        return bool(_SCRIPT_ACTS.search(_script_text(argv)))
    if program in {"sh", "bash", "zsh"} and "-c" in argv:
        at = argv.index("-c")
        if at + 1 < len(argv):
            return would_act_on_the_desktop(argv[at + 1], shell=True)
    return False


def record(nodeid: str, what: Any) -> None:
    """Put one refusal on the ledger, attributed to the test that asked."""
    log_dir = os.environ.get("AURA_LOG_DIR")
    if not log_dir:
        return
    try:
        path = Path(log_dir) / "host_actuation_refused.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as ledger:
            ledger.write(
                json.dumps({"at": time.time(), "test": nodeid, "asked": str(what)[:400]})
                + "\n"
            )
    except OSError:
        pass
