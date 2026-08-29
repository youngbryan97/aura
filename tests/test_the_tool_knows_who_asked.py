"""A tool dispatched for a person's request should know it was theirs.

The conscience holds a skill whose worst case looks harmful unless a person
asked for it directly, in the foreground, on their own machine. It decides
that from the origin and the message on the execution context — and the tool
loop passed neither, so every dispatch arrived as origin=unknown and the
override could never fire.

Live on 2026-08-29: asked to use a library at a named path, the model called
code_repl with that path and was held at "worst-case harm 0.80". It then tried
sys, then importlib, then exec — each correctly refused — and came back to the
right call, which was held again. The person had asked for it in those words.
"""

from __future__ import annotations

from core.capability_engine import CapabilityEngine

_ASKED = (
    "Read the docs at /tmp/ledgerkit and use that library to record this: "
    "on 2026-03-01 we invoiced a customer 25000 cents for hosting."
)


def test_a_foreground_request_with_its_words_is_direct() -> None:
    assert (
        CapabilityEngine._directly_requested_by_the_user(
            {"origin": "user", "message": _ASKED}, ""
        )
        is True
    )


def test_an_unknown_origin_is_not() -> None:
    """What every tool dispatch used to look like."""

    assert (
        CapabilityEngine._directly_requested_by_the_user(
            {"origin": "unknown", "message": _ASKED}, ""
        )
        is False
    )


def test_an_autonomous_cycle_carrying_the_words_is_not() -> None:
    """The narrowness is the point: a loop that quotes a person is not a person."""

    assert (
        CapabilityEngine._directly_requested_by_the_user(
            {"origin": "autonomous_initiative_loop", "message": _ASKED}, ""
        )
        is False
    )


def test_a_foreground_origin_with_no_words_is_not() -> None:
    assert (
        CapabilityEngine._directly_requested_by_the_user({"origin": "user"}, "") is False
    )


def test_the_tool_loop_passes_both() -> None:
    """The wiring, which is what was missing."""

    from pathlib import Path

    source = Path("core/brain/inference_gate.py").read_text()
    start = source.index('"required_skills": list(required),')
    block = source[start : start + 1400]
    assert '"origin": origin or "user"' in block, block[:400]
    assert '"message": text,' in block, block[:400]
