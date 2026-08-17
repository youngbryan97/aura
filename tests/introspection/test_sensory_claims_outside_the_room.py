"""She cannot see the weather, and she cannot see you.

LIVE 2026-08-17, asked "morning, what's your read on today so far?":

    "It's afternoon. The sun is shining. My buffers are empty..."

She has no window. The existing scene guard covered the INSIDE of the room —
desk, light, face, silence — so a claim that stepped one room over went
unguarded. That is the shape a table of patterns keeps producing: each entry
describes one room, and the next invention happens in the next room along.
"""

from __future__ import annotations

import pytest

from core.introspection.self_evidence import (
    resolve_shared_present,
    sensory_claim_correction,
    unsupported_sensory_claims,
)


def _fires(reply: str) -> bool:
    return bool(str(sensory_claim_correction(reply, "hi") or "").strip())


# ── the live regression ──────────────────────────────────────────────────────

def test_the_sun_is_shining_is_caught() -> None:
    assert _fires("It's afternoon. The sun is shining.")


@pytest.mark.parametrize(
    "reply",
    [
        "It's raining out.",
        "The sky is grey today.",
        "It's bright outside.",
        "The weather is nice.",
    ],
)
def test_weather_claims_are_caught(reply: str) -> None:
    assert _fires(reply)


@pytest.mark.parametrize(
    "reply",
    ["You seem tired today.", "You look well.", "You're smiling."],
)
def test_claims_about_how_you_look_are_caught(reply: str) -> None:
    """Being told a machine noticed your mood when it did not is worse than silence."""
    assert _fires(reply)


# ── the guard must not swallow ordinary talk ─────────────────────────────────

@pytest.mark.parametrize(
    "reply",
    [
        "I fixed the parser and pushed it.",
        "The sun is a G-type main-sequence star.",
        "You seem to be asking about two different things.",
        "That looks like a scoping bug in the router.",
    ],
)
def test_ordinary_replies_are_untouched(reply: str) -> None:
    assert not _fires(reply)


def test_the_documented_2026_08_04_case_still_fires() -> None:
    """The original regression this table was built for."""
    assert _fires(
        "You're still here. The room is silent, the light remains unchanged on your desk."
    )


def test_a_supported_channel_would_not_be_corrected() -> None:
    """The gate stops invention, not description — one live sense is enough."""
    bundle = resolve_shared_present()
    states = {r.channel: r.present for r in bundle.readings}
    if states.get("camera"):
        pytest.skip("camera is live on this host; nothing to assert about absence")

    assert unsupported_sensory_claims("The sun is shining.", bundle)
