"""A tool she says is hers has to be one this build registers.

LIVE, 2026-08-20. Asked what she had been working on, with the record in
front of her naming swarm_debate, web_search and http_request, she answered:

    "I've been testing my memory reasoning with a tool called WebGPT."

No such capability exists here. The rest of that reply was grounded — she
named a curiosity topic straight from the record — so the evidence was fine.
One clause invented a name.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _claims_a_capability_it_does_not_have as claims_one,
)

ASKED = "what have you actually been working on lately?"


def test_the_live_fabrication() -> None:
    assert claims_one(ASKED, "I've been testing my memory reasoning with a tool called WebGPT.")


@pytest.mark.parametrize(
    "reply",
    [
        "I ran a tool called web_search a few times tonight.",
        "I used my http_request skill to read the endpoint.",
        "I used curl to check it.",
        "I've been reading the Open-Meteo API.",
        "A tool called WebGPT exists in the literature.",
        "You could build a tool called WebGPT for that.",
        "",
    ],
)
def test_what_must_not_be_flagged(reply: str) -> None:
    assert claims_one(ASKED, reply) is False


def test_the_person_naming_it_first_is_not_a_fabrication() -> None:
    assert claims_one("tell me about WebGPT", "I've been using a tool called WebGPT, as you say.") is False


def test_it_fails_open_without_a_catalogue(monkeypatch) -> None:
    """A rule that fires when it cannot check is worse than one that stays quiet."""
    import core.conversation.response_reliability as reliability

    monkeypatch.setattr(reliability, "_registered_capability_names", lambda: frozenset())
    assert claims_one(ASKED, "I used a tool called WebGPT.") is False


def test_the_reason_is_repairable_not_fatal() -> None:
    from core.phases.response_generation import _DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS

    assert "unregistered_capability_claim" in _DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS


def test_the_registry_is_read_from_the_live_catalogue() -> None:
    from core.conversation.response_reliability import _registered_capability_names

    names = _registered_capability_names()
    assert "http_request" in names
    assert "web_search" in names
    assert "webgpt" not in names
