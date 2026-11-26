"""Her code writes no emoji (26 November 2026).

Bryan: "can we also remove the emojis from the code? if we want to create our own icons that's fine but emojis update
and break and it's just a failure point we can avoid." A log line, a report or a tab label that carries an emoji
depends on a font and an emoji version to show at all, and a parser that looks for the mark breaks when the line loses
it. The marks came out of 1,024 Python files and 21 others (the shell scripts, the Makefile, two workflows, the app)
on that day: a status mark became its word (ok, failed, warning) and the rest were dropped; the app's three emoji tabs
became line icons.

What may still hold one is code that reads them, in what a person writes or what a screen shows, and tests that hand
them in as input. Each is named here with what it reads.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]

#: Emoji, by code point. Kept out: the chess pieces, the card suits, the music notes, the check and cross marks
#: (U+2713, U+2715, U+2717) and the smiling face U+263A, which are text symbols her code reads and draws.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u2639\u263B-\u2653\u2668\u266C-\u2712\u2714\u2716\u2718-\u27BF\u2B00-\u2BFF"
    "\u231A\u231B\u2328\u23CF\u23E9-\u23F3\u23F8-\u23FA\u2139]")
_CODE = ("*.py", "*.js", "*.mjs", "*.html", "*.css", "*.sh", "*.swift", "*.yml", "*.yaml", "Makefile")
_NOT_HERS = ("archive/", "dev_archive/", "interface/static/vendor/")

#: The files that read emoji, and what each reads.
READERS = {
    "core/conversational/dynamics.py": "laughter in what a person writes",
    "core/conversational/humor_engine.py": "laughter in what a person writes",
    "core/social/conversational_profile.py": "how a person jokes, in what they write",
    "core/social/stance_inference.py": "a joking stance, in what a person writes",
    "core/perception/observation_evidence.py": "the close mark on a control she sees",
    "core/perception/general_terminal_parser.py": "a shell prompt's sigil on a terminal she reads",
    "core/making/pieces/apps.js": "a checklist in markdown, drawn as a checked box",
    "tools/agi_gauntlet/environments/pairs.py": "the shapes a puzzle is made of",
    "tests/test_chat_reliability_proof.py": "emoji in a message, as input",
    "tests/test_security_stress.py": "emoji in a request, as input",
    "tests/integration/chaos_fuzzer.py": "emoji in a request, as input",
    "tests/integration/verify_10_unique_messages.py": "emoji in a message, as input",
}


def _her_code() -> list[str]:
    listed = subprocess.run(["git", "-c", "core.fsmonitor=false", "ls-files", "--", *_CODE], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split("\n")
    return [name for name in listed if name and not name.startswith(_NOT_HERS)]


def test_her_code_writes_no_emoji():
    found = []
    for name in _her_code():
        if name in READERS:
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if _EMOJI.search(line):
                found.append(f"{name}:{number}: {line.strip()[:120]}")
    assert not found, ("emoji in her code; a status mark is its word (ok, failed, warning), an icon is drawn:\n"
                       + "\n".join(found[:40]))


def test_each_reader_named_still_reads_one():
    """A reader that no longer holds an emoji comes off the list, so the list says only what is true."""
    stale = [name for name in READERS
             if not (ROOT / name).is_file() or not _EMOJI.search((ROOT / name).read_text(encoding="utf-8"))]
    assert not stale, stale
