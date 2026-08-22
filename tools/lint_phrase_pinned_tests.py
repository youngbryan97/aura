#!/usr/bin/env python3
"""Tests that assert on production wording, counted and ratcheted.

Reading production source in a test is not automatically wrong. Three
different things use the same two calls, and only one of them is a defect:

* **structural** — "this call site exists", "this import is present", "the
  kernel reads the shared set rather than its own copy". No behavioural test
  can express those; they are the honest use, and this repository relies on
  them heavily.
* **fixture** — reading a tmp_path file the test itself wrote. Not about
  production source at all.
* **phrase pin** — asserting a sentence, a log line, a user-visible message
  or a docstring clause. This is the defect. It passes while the wording is
  frozen and fails the moment somebody improves it, which means it punishes
  exactly the work it should protect.

Two real cases: a test pinned the literal string "I'm here", which broke
when 718e46091 correctly grounded a recovery claim in current evidence; and
one pinned ``os.environ["AURA_MEDIA_SIDECAR_PROCESS"] = "1"`` in the sensory
client, which broke when child-process spawning was correctly centralised
behind the subprocess gateway. Both improvements. Both tests red.

Converting all of them at once would be a large mechanical change to the
test suite with no behavioural benefit, so this ratchets instead: the count
in ``config/phrase_pinned_test_baseline.json`` may only fall. A new test
that pins a phrase has to earn it by replacing an old one.

Run: ``python tools/lint_phrase_pinned_tests.py`` / ``--write-baseline``
(the refresh refuses to record growth without ``--accept-growth --reason``)
"""

from __future__ import annotations

import ast
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, TypedDict

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "phrase_pinned_test_baseline.json"

#: A read is of production source if it names a package or uses getsource.
_SOURCE_HINTS = ("core/", "interface/", "skills/", "getsource")
#: ...and is a fixture read if it names a temp or artifact location.
_FIXTURE_HINTS = ("tmp_path", "tmp_dir", "tmpdir", "fixture", "artifacts", "/tmp")

#: How far past the read to look for the assertions it feeds.
_WINDOW = 12

#: Keywords that open a Python statement. A literal beginning with one is a
#: code fragment a structural test is asserting the existence of, not prose.
_CODE_OPENERS = frozenset(
    {
        "def", "class", "async", "import", "from", "return", "raise", "yield",
        "await", "with", "if", "elif", "else", "for", "while", "try", "except",
        "finally", "assert", "lambda", "global", "nonlocal", "del", "pass",
        "@property", "@staticmethod", "@classmethod",
    }
)

_ASSERT_LITERAL = re.compile(r'assert[^\n]*?["\']([^"\']{4,})["\']')


def _is_structural(literal: str) -> bool:
    """Is this literal a fragment of CODE rather than a sentence?

    Parsing is the discriminator, and a much better one than counting words.
    ``from core.container import ServiceContainer`` is a layering invariant
    that no behavioural test can express — it parses. ``Bryan is kin.`` and
    ``I'm here`` are wording — they do not. Word counts got the first of
    those wrong and would have had this tool demanding the removal of the
    assertions that hold the architecture together.
    """
    candidate = literal.strip()
    if not candidate:
        return True
    try:
        ast.parse(candidate)
    except SyntaxError:
        pass
    else:
        return True
    # A fragment of code that does not stand alone. ``def _drain_phi_residual_ring``
    # is exactly what a structural test asserts — "this function exists" — and it
    # will never parse, because it is half a statement. Leading keyword, not a
    # sentence.
    first = candidate.split(maxsplit=1)[0].rstrip(":")
    if first in _CODE_OPENERS:
        return True
    # A call site named but not closed: ``_self_health_answer_or_empty(``,
    # ``getUserMedia({``, ``replace(``. A structural test asserting "this call
    # exists" writes exactly this, and it can never parse because it is half an
    # expression. Judged by shape rather than by parseability: an identifier
    # (dotted allowed) followed by an opening bracket, nothing else.
    return bool(re.fullmatch(r"[A-Za-z_][\w.]*\s*[({\[][\s({\[]*", candidate))


def _asserted_literals(window: str) -> list[str]:
    """String literals a test asserts ON, excluding each assert's own message.

    ``assert guard_index < register_index, "shutdown guard must precede service
    registration"`` pins nothing. The second operand of an assert is the
    explanation shown when it fails — the test's own prose, not production
    wording — and counting it made a well-documented structural test look like
    debt. The regex this replaced took the first literal on the line, which for
    that shape is the message.

    AST rather than another regex because the docstring above is already right
    about which is the better discriminator, and because only ``node.test``
    can be isolated correctly by parsing.
    """
    try:
        tree = ast.parse(textwrap.dedent(window))
    except SyntaxError:
        # A window clipped mid-statement. Fall back to the line-wise regex,
        # minus anything after a top-level comma on the assert.
        out: list[str] = []
        for line in window.splitlines():
            if "assert" not in line:
                continue
            head = line.split(",")[0]
            out.extend(_ASSERT_LITERAL.findall(head))
        return out

    literals: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for inner in ast.walk(node.test):
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                if len(inner.value) >= 4:
                    literals.append(inner.value)
    return literals


def _phrase_pins(source: str) -> int:
    """Assertions on production wording in one test file."""
    lines = source.splitlines()
    pins = 0
    for index, line in enumerate(lines):
        if "getsource(" not in line and "read_text(" not in line:
            continue
        if any(hint in line for hint in _FIXTURE_HINTS):
            continue
        if not any(hint in line for hint in _SOURCE_HINTS):
            continue
        window = "\n".join(lines[index : index + _WINDOW])
        for literal in _asserted_literals(window):
            if not _is_structural(literal):
                pins += 1
    return pins


#: This tool's own calibration fixtures. ``test_phrase_pin_classifier.py``
#: holds one example of every shape this scanner must catch and every shape it
#: must not, as string data — so scanning it counts the examples themselves and
#: the instrument reports its own test bench as debt. Excluded by name rather
#: than by contorting the fixtures into something the scanner cannot see, which
#: would make the test less readable to buy the same result.
_SELF_TEST = "tests/test_phrase_pin_classifier.py"


class Measurement(TypedDict):
    """What one run of this gate measured."""

    total: int
    files: int
    by_file: dict[str, int]


def measure() -> Measurement:
    by_file: dict[str, int] = {}
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        relative = str(path.relative_to(ROOT))
        if relative == _SELF_TEST:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        pins = _phrase_pins(source)
        if pins:
            by_file[str(path.relative_to(ROOT))] = pins
    return {"total": sum(by_file.values()), "files": len(by_file), "by_file": by_file}


def _assertion_counts(payload: dict[str, Any]) -> dict[str, int]:
    by_file = payload.get("by_file")
    if not isinstance(by_file, dict):
        return {}
    return {str(name): int(count) for name, count in by_file.items()}


def main(argv: list[str]) -> int:
    current = measure()
    total = int(current["total"])
    print(
        f"tests asserting on production wording: {total} "
        f"across {current['files']} file(s)"
    )

    if "--write-baseline" in argv:
        from tools.ratchet_baseline import guard_growth, load

        written: int = guard_growth(
            dict(current),
            load(BASELINE),
            BASELINE,
            argv,
            counts=_assertion_counts,
            tool="tools/lint_phrase_pinned_tests.py",
        )
        return written

    if "--list" in argv:
        for name, count in sorted(
            current["by_file"].items(), key=lambda kv: -kv[1]
        )[:30]:
            print(f"  {count:3d}  {name}")
        return 0

    if not BASELINE.is_file():
        print(f"❌ no baseline at {BASELINE.relative_to(ROOT)}; run --write-baseline")
        return 1

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    allowed = int(baseline.get("total", 0))
    if total > allowed:
        print(f"❌ phrase-pinned assertions rose: {allowed} -> {total}")
        previous = baseline.get("by_file") or {}
        for name, count in sorted(current["by_file"].items()):
            was = int(previous.get(name, 0))
            if count > was:
                print(f"    {name}: {was} -> {count}")
        print(
            "\nAssert the behaviour instead of the wording. A test that fails "
            "when someone improves a message is a test that punishes the work "
            "it exists to protect."
        )
        return 1

    if total < allowed:
        print(f"⬇️  phrase-pinned assertions fell: {allowed} -> {total}")
        print("    refresh with: python tools/lint_phrase_pinned_tests.py --write-baseline")
        return 1

    print("✅ phrase-pinned assertions held at baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
