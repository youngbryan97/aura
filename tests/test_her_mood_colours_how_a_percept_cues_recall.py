"""A mood colours how a percept is appraised, and so what it brings back.

In a good mood a bad event is judged less bad (Forgas, 1995). On 24 October,
when she felt better, her internal state drew good and bad percepts 0.94
closer, and what they brought back did not move with it (per class pair, rho
-0.08 and -0.25 against the internal change in two runs inside her forks).
With each appraised emotion weighed by her mood against her usual, and her mood
a cue of its own, it read +0.61 and +0.67.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from core.memory.felt_at_encoding import encode, percept_carried
from core.phases.memory_retrieval import _appraised_in_her_mood

pytestmark = pytest.mark.unit

ERROR = ("fear", "sadness", "frustration")


def _affect(valence: float, level: float = 0.2, spread: float = 0.1) -> SimpleNamespace:
    return SimpleNamespace(valence=valence, markers={"valence_usual": {"n": 40, "level": level, "spread": spread}})


def test_in_a_good_mood_an_error_is_felt_less():
    weights = _appraised_in_her_mood(ERROR, _affect(0.3))
    assert all(value == pytest.approx(1.0 - math.tanh(1.0)) for value in weights.values())
    low = _appraised_in_her_mood(ERROR, _affect(0.1))
    assert all(value == pytest.approx(1.0 + math.tanh(1.0)) for value in low.values())


def test_at_her_usual_and_without_one_the_percept_cues_as_it_did():
    assert all(value == pytest.approx(1.0) for value in _appraised_in_her_mood(ERROR, _affect(0.2)).values())
    assert _appraised_in_her_mood(ERROR, SimpleNamespace(valence=0.3, markers={})) is None


def test_a_memory_of_the_error_s_feelings_is_cued_less_in_a_good_mood():
    lost = encode({"sadness": 0.3, "frustration": 0.2})
    plain = percept_carried(0.0, ERROR, lost)
    glad = percept_carried(0.0, ERROR, lost, weights=_appraised_in_her_mood(ERROR, _affect(0.3)))
    usual = percept_carried(0.0, ERROR, lost, weights=_appraised_in_her_mood(ERROR, _affect(0.2)))
    assert usual == pytest.approx(plain)
    assert glad < plain
