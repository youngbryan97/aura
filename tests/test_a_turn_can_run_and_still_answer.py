"""A tool loop that uses every turn calling has none left to answer.

LIVE, 2026-08-28: "read the docs at <path>, then actually use it" was given two
tools and five turns. It spent one on a path that was denied, one on an
execution that was refused, and three on reads that worked — and the person got
a list of what ran instead of an answer.

Two separate faults met in that turn, and both are here.
"""

from __future__ import annotations

from pathlib import Path

_GATE = Path("core/brain/inference_gate.py")


def test_the_last_turn_is_free_to_write() -> None:
    """One more than the calls, so answering is not itself a call."""

    body = _GATE.read_text()
    assert "max_turns=max(4, 2 * len(tools) + 2)" in body
    # Two tools now leave a turn spare rather than exactly none.
    assert max(4, 2 * 2 + 2) > max(3, 2 * 2 + 1)


def test_the_self_service_ceiling_carries_its_own_authority() -> None:
    """It is defined as the most a turn may do WITHOUT anybody asking.

    "The most a turn may do without the person having asked for that effect.
    Sandboxed computation is the ceiling: it can calculate anything and change
    nothing outside its own sandbox." Something that by definition needs no
    permission was refused for want of one: "Permission denied: Requires user
    confirmation: Typed execution contract: scope=sandboxed_compute".
    """

    from core.brain.inference_gate import (
        _REQUESTED_ARTIFACT_EFFECT_CEILING,
        _SELF_SERVICE_EFFECT_CEILING,
    )
    from core.phases.response_contract import (
        _REQUESTED_ARTIFACT_CEILING,
        _SELF_SERVICE_CEILING,
    )

    # Read from the one place that decides them, not spelled twice.
    assert _SELF_SERVICE_EFFECT_CEILING == _SELF_SERVICE_CEILING
    assert _REQUESTED_ARTIFACT_EFFECT_CEILING == _REQUESTED_ARTIFACT_CEILING

    body = _GATE.read_text()
    start = body.index('"user_explicitly_authorized"')
    window = body[start : start + 300]
    assert "_SELF_SERVICE_EFFECT_CEILING" in window
    assert "_REQUESTED_ARTIFACT_EFFECT_CEILING" in window


def test_nothing_above_those_two_ceilings_is_authorised() -> None:
    """Deleting, sending and spending still need their own consent."""

    from core.brain.inference_gate import (
        _REQUESTED_ARTIFACT_EFFECT_CEILING,
        _SELF_SERVICE_EFFECT_CEILING,
    )

    carried = {_SELF_SERVICE_EFFECT_CEILING, _REQUESTED_ARTIFACT_EFFECT_CEILING}
    for scope in (
        "external_io",
        "privileged_mutation",
        "state_mutation",
        "foreground_desktop_control",
    ):
        assert scope not in carried, scope
