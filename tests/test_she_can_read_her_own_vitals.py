"""Her own headline vitals have to reach her, not just the header.

Asked on 2026-08-10, live: "tell me your current energy and focus numbers".
She answered "Current energy and focus numbers: Not readable." At that moment
``/api/health`` was serving ``liquid_state: {"energy": 11.0, "focus": 2.0}``,
the window header was displaying them, and the neural feed was narrating
them.

The answer was honest — she genuinely did not have them. The protected
foreground snapshot carried mood, tone, valence, arousal, curiosity,
coherence, field clarity, field flow, field intensity and mode focus, and
neither of the two the header shows. An instrument every surface can read
except its owner is the wrong way round.
"""

from __future__ import annotations

from unittest.mock import patch

import interface.routes.chat as chat
from tests.chat_lane_support import chat_lane_source


class _Substrate:
    energy = 11.0
    focus = 2.0
    frustration = 14.0
    confidence = 85.7


def _with_substrate(substrate):
    def peek(name, default=None):
        if name in {"liquid_substrate", "liquid_state"}:
            return substrate
        return default

    return patch.object(chat.ServiceContainer, "peek", staticmethod(peek))


def test_the_two_numbers_she_could_not_read_are_present():
    with _with_substrate(_Substrate()):
        vitals = chat._liquid_vitals()
    assert vitals["energy"] == 11.0
    assert vitals["focus"] == 2.0


def test_every_published_vital_is_carried():
    """What she can say about herself must not drift from what /api/health serves."""
    with _with_substrate(_Substrate()):
        vitals = chat._liquid_vitals()
    for name in chat._LIQUID_VITALS:
        assert name in vitals, f"{name} is published but never reaches her"


def test_a_missing_substrate_reports_nothing_rather_than_zero():
    """A vital reported as 0 because nothing answered is worse than an absent one."""
    with _with_substrate(None):
        assert chat._liquid_vitals() == {}


def test_an_unreadable_vital_is_dropped_not_coerced():
    class Broken:
        energy = "not a number"
        focus = 2.0

    with _with_substrate(Broken()):
        vitals = chat._liquid_vitals()
    assert "energy" not in vitals
    assert vitals["focus"] == 2.0


def test_absent_values_do_not_render_a_line():
    assert chat._compact_snapshot_line("Energy", None) == ""
    assert chat._compact_snapshot_line("Energy", 11.0) == "Energy: 11.0"


def test_the_snapshot_prompt_asks_for_them():
    """The resolver is only half of it; the prompt has to render the lines."""
    text = chat_lane_source()
    # Labelled for the organ that owns them. The soma reserve publishes a
    # different quantity under the bare word "energy", and on 2026-08-18 both
    # reached one prompt — 14.0 here, 0.14 from the affect line, 0.647 from the
    # reserve — so no answer to "what's your energy" could be right.
    assert '_compact_snapshot_line("Substrate energy", voice_state.get("energy"))' in text
    assert '_compact_snapshot_line("Substrate focus", voice_state.get("focus"))' in text
