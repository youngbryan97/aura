"""A helper cannot read a variable the turn has not reached yet.

LIVE, 2026-08-19. Asked to read an unfamiliar repository and find the failing
test, the turn died with:

    cannot access free variable 'preflight_context_message' where it is not
    associated with a value in enclosing scope

and the person got "I hit an error before a coherent answer formed."

``_try_serve_grounded_recovery`` is defined early in the chat handler and
closes over ``preflight_context_message``; the assignment sat roughly 1,400
lines further down. Any turn that reached the helper by an earlier path — and
a turn that goes to the tool loop does — read an unbound free variable.

Python binds a closure's free variables when the closure RUNS, not where it is
written, so the only safe place for the assignment is above every path that
can call it.
"""

from __future__ import annotations

import re
from pathlib import Path

CHAT = Path(__file__).resolve().parents[1] / "interface/routes/chat.py"


def _line_of(pattern: str, source: str) -> int:
    match = re.search(pattern, source, re.MULTILINE)
    assert match, f"anchor no longer present: {pattern}"
    return source[: match.start()].count("\n") + 1


def test_the_binding_comes_before_every_reader():
    source = CHAT.read_text()
    bound = _line_of(r"^    preflight_context_message = str\(body\.message or \"\"\)", source)
    readers = [
        source[:m.start()].count("\n") + 1
        for m in re.finditer(r"preflight_context_message=preflight_context_message", source)
    ]
    assert readers, "no reader found; this test is watching the wrong name"
    assert all(bound < reader for reader in readers), (
        f"bound at {bound}, read at {sorted(readers)} — a reader runs before the binding"
    )


def test_the_helper_that_died_is_defined_after_the_binding():
    """It closes over the variable, so being defined later is what makes it safe."""
    source = CHAT.read_text()
    bound = _line_of(r"^    preflight_context_message = str\(body\.message or \"\"\)", source)
    helper = _line_of(r"^        async def _try_serve_grounded_recovery\(", source)
    assert bound < helper


def test_no_second_binding_shadows_the_first():
    """A later re-assignment inside a branch is how this happened."""
    source = CHAT.read_text()
    assignments = re.findall(r"^\s*preflight_context_message = ", source, re.MULTILINE)
    assert len(assignments) == 1, (
        f"{len(assignments)} assignments; one of them is unreachable for some path"
    )
