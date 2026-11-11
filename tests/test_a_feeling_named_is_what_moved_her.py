"""The mind page names the feelings something moved, not the mood her resting chemistry reads as.

L08, 29 September 2026: "How she feels" came from her mood vector, which at her chemicals' setpoints still reads
"calm" and "sociality", so a mood nothing had appraised was shown as one she had. The page now names feelings from
the mood's departure from the mood her resting chemistry gives.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import core.consciousness.adaptive_mood as adaptive_mood
from core.consciousness.neurochemical_system import NeurochemicalSystem

MIND_PAGE = Path(__file__).resolve().parent.parent / "interface" / "static" / "mind.html"
SHOWN = 0.05  # what the page shows as a chip


@pytest.fixture(autouse=True)
def _a_mood_learner_of_its_own(tmp_path):
    adaptive_mood.reset_singleton_for_test()
    adaptive_mood.get_adaptive_mood(db_path=tmp_path / "adaptive_mood.sqlite3")
    yield
    adaptive_mood.reset_singleton_for_test()


def test_at_rest_her_mood_reads_as_feelings_and_nothing_has_moved():
    system = NeurochemicalSystem()
    resting = system.get_mood_vector()
    assert any(value > SHOWN for value in resting.values()), "at rest the mood vector reads as feelings"
    assert all(abs(value) < 1e-9 for value in system.get_mood_departure().values())


def test_a_threat_moves_her_toward_stress():
    system = NeurochemicalSystem()
    system.on_threat(0.8)
    departure = system.get_mood_departure()
    assert departure["stress"] > SHOWN
    assert departure["stress"] == max(departure.values())


def test_reading_the_departure_leaves_the_learners_last_prediction_alone():
    system = NeurochemicalSystem()
    learner = adaptive_mood.get_adaptive_mood()
    system.get_mood_vector()
    kept = dict(learner._last_prediction)
    system.on_threat(0.8)
    system.get_mood_departure()
    assert learner._last_prediction == kept


def test_the_page_names_feelings_from_the_departure():
    page = MIND_PAGE.read_text(encoding="utf-8")
    chips = page[page.index("function renderMoodChips"):page.index("function renderAwake")]
    assert "mood_departure_from_rest" in chips
    assert "current_mood_vector" not in chips
