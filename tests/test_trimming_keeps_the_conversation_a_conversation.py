"""Dropping messages must not reorder the ones that are kept.

The working-memory trimmer selected what to keep and then concatenated the
selections: user-and-large messages first, then the most recent non-user ones,
then the last four. That is not a conversation. On a plain alternating
exchange of eighteen turns it produced:

    U1 U2 U3 ... U16   A7 A8 ... A16   U17 A17 U18 A18

Sixteen consecutive user messages, then ten consecutive replies. Every answer
was torn away from the question it answered, and A1-A6 were dropped outright.
What she reasoned over was a transcript that never happened — each reply
appearing to respond to whichever question happened to precede it after the
shuffle.

It also reshapes the KV prefix every time it runs, so the prompt cache could
never reuse more than the leading system block. Measured live:
"prefix diverges at token 226 (9% of 2561 reused)".
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

SOURCE = (Path(__file__).resolve().parents[1] / "core/phases/memory_consolidation.py")


def _run_trim(messages: list[dict], max_working_memory: int = 30) -> list[dict]:
    """Execute the shipped trim block against a conversation."""
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "tail = wm[-4:]")
    end = next(i for i, l in enumerate(lines) if "Context trim pass 2" in l)
    code = textwrap.dedent("\n".join(lines[start:end]))
    namespace = {"max_working_memory": max_working_memory, "wm": list(messages)}
    exec(code, namespace)  # noqa: S102 — running the shipped block is the point
    return namespace["wm"]


def _conversation(turns: int) -> list[dict]:
    out: list[dict] = []
    for i in range(1, turns + 1):
        out.append({"role": "user", "content": f"U{i}"})
        out.append({"role": "assistant", "content": f"A{i}"})
    return out


def test_what_survives_is_still_in_the_order_it_happened():
    original = _conversation(18)
    kept = _run_trim(original)

    order = [m["content"] for m in original]
    positions = [order.index(m["content"]) for m in kept]
    assert positions == sorted(positions), (
        f"trimming reordered the conversation: {[m['content'] for m in kept]}"
    )


def test_answers_stay_attached_to_their_questions():
    """The specific harm: a reply relocated away from what it replied to."""
    kept = [m["content"] for m in _run_trim(_conversation(18))]
    paired = sum(1 for a, b in zip(kept, kept[1:]) if a.startswith("U") and b == f"A{a[1:]}")
    assert paired >= 10, f"only {paired} question/answer pairs survived intact: {kept}"


def test_no_run_of_same_role_longer_than_the_conversation_had():
    """Sixteen user messages in a row is not something that was ever said."""
    kept = _run_trim(_conversation(18))
    longest = run = 1
    for previous, current in zip(kept, kept[1:]):
        run = run + 1 if current["role"] == previous["role"] else 1
        longest = max(longest, run)
    assert longest <= 3, f"a run of {longest} same-role messages was manufactured"


def test_the_most_recent_turns_are_always_kept():
    kept = [m["content"] for m in _run_trim(_conversation(18))]
    for recent in ("U17", "A17", "U18", "A18"):
        assert recent in kept, recent


def test_it_still_trims_to_the_bound():
    kept = _run_trim(_conversation(40))
    assert len(kept) <= 30


def test_a_short_conversation_is_returned_untouched():
    original = _conversation(3)
    assert _run_trim(original) == original


@pytest.mark.parametrize("turns", [10, 18, 25, 40])
def test_order_holds_at_every_length(turns):
    original = _conversation(turns)
    kept = _run_trim(original)
    order = [m["content"] for m in original]
    positions = [order.index(m["content"]) for m in kept]
    assert positions == sorted(positions), turns
