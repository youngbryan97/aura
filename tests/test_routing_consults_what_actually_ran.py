"""Routing asks what actually happened, not only what somebody enumerated.

`looks_like_desktop_objective` decides with seventeen patterns and has been
wrong in both directions: a request to build a web app went to the screen lane
and came back "os_automation refused to act… completed 0/1 steps", while a
goal to play a game online read as conversation and answered about identity
with the browser on a blank page.

The intention log holds distinct requests a person made and the capability
that SUCCEEDED for each. Measured on a held-out third: AUROC 0.979.
"""

from __future__ import annotations

import logging

import pytest

from core.language.desktop_actuation import actuation_surface
from core.language.label_mining import (
    ACTUATION_TOOLS,
    LabelledRequest,
    mine_desktop_actuation_labels,
)

logging.disable(logging.CRITICAL)


def test_a_receipt_records_what_happened_not_what_should_have() -> None:
    """"build me a small web app" is in the log against desktop_task, where it
    was misrouted and completed zero of one steps. Learning from every receipt
    would teach the mistake as the rule."""
    from core.language.label_mining import _first_successful_tool

    failed = '[{"tool_name": "os_automation", "success": false}]'
    worked = '[{"tool_name": "computer_use", "success": true}]'
    assert _first_successful_tool(failed) == ""
    assert _first_successful_tool(worked) == "computer_use"


def test_an_objective_that_names_its_own_tool_teaches_nothing() -> None:
    pairs = [
        LabelledRequest(request="Use tool 'web_search' to find something", tool="web_search"),
        LabelledRequest(request="open my notes app and write a note in it", tool="computer_use"),
    ]
    positives, negatives = mine_desktop_actuation_labels(pairs)
    assert positives == ["open my notes app and write a note in it"]
    assert negatives == []


def test_the_screen_capabilities_are_named_once() -> None:
    assert {"computer_use", "desktop_task", "os_automation"} <= ACTUATION_TOOLS
    assert "web_search" not in ACTUATION_TOOLS
    assert "http_request" not in ACTUATION_TOOLS


def test_the_surface_is_seeded_from_real_traffic() -> None:
    surface = actuation_surface()
    if not surface.positives:
        return
    assert len(surface.positives) >= 8
    assert len(surface.negatives) >= 8


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("open my notes app and write a note describing yourself", True),
        ("Go find a 2048 game online and play it until you get a 128 tile", True),
        ("read /etc/hosts and tell me the first line", False),
        ("what is the temperature at that endpoint", False),
    ],
)
def test_it_gets_the_clear_cases_right(request_text: str, expected: bool) -> None:
    surface = actuation_surface()
    if not surface.positives:
        return
    verdict = surface.decide(request_text)
    if verdict is not None:
        assert verdict is expected


def test_routing_only_adds_and_never_removes() -> None:
    """Sending real screen work somewhere else is the worse error, so a
    pattern that says yes keeps saying yes until the surface beats it on the
    pattern's own examples."""
    from pathlib import Path

    source = Path("core/runtime/desktop_objective_intent.py").read_text(encoding="utf-8")
    body = source[source.index("by_pattern = bool(_DIRECT_DESKTOP_ACTION_RE") :]
    body = body[: body.index("def _learned_actuation_decision")]
    assert "if not by_pattern:" in body
    assert "keeping the patterns" in body
    assert "return by_pattern" in body


def test_a_surface_consulted_on_a_live_turn_is_warmed() -> None:
    """The integration bug this caught.

    The routing surface was wired to the non-blocking path and warmed by
    nothing: it queued every novel request, abstained, and could never turn a
    queued request into a decision. Verdicts are deliberately not restored
    across restarts, so it had nothing to fall back on — inert in production
    while measuring 0.979 offline.

    Asking IS the registration now, so a second surface cannot repeat it.
    """
    from core.language.learned_matcher import registered_surfaces
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    looks_like_desktop_objective("tell me something interesting about octopuses")
    assert "desktop_actuation" in {surface.name for surface in registered_surfaces()}


def test_the_warmer_covers_every_consulted_surface() -> None:
    from pathlib import Path

    source = Path("core/conversation/response_reliability.py").read_text(encoding="utf-8")
    body = source[source.index("def warm_language_matchers") :]
    body = body[: body.index("\ndef ", 10)]
    assert "warm_all" in body
    assert "_ACTION_CLAIM_MATCHER.warm" not in body


def test_the_queue_becomes_a_decision() -> None:
    """First sighting queues and abstains; after warming it decides."""
    from core.language.learned_matcher import warm_all

    surface = actuation_surface()
    if not surface.positives:
        return
    probe = "put the phrase VESSEL-88 onto my clipboard for me"
    surface._decided.pop(probe, None)
    assert surface.decide_without_waiting(probe) is None
    warm_all(limit=8)
    assert surface.decide_without_waiting(probe) is not None


def test_the_surface_is_asked_before_the_vocabulary_gate_gives_up() -> None:
    """A request with no enumerated term returned False before anything else
    looked at it — precisely the case the surface exists for."""
    from pathlib import Path

    source = Path("core/runtime/desktop_objective_intent.py").read_text(encoding="utf-8")
    gate = source[source.index("_DESKTOP_OBJECTIVE_ACTION_TERMS\n    ) or not") - 400 :]
    gate = gate[: gate.index("try:")]
    assert "_learned_actuation_decision(user_message) is True" in gate
