"""The shape of her content space, and the line the measurement will not cross.

Phenomenal structuralism says a quality is its complete position in the web of
relations to every other quality. The testable half is the geometry: whether
internal causal distances between contents predict the system's own
discriminations, and whether moving the manifold moves them in a direction
registered before the move.

The untestable half is whether that geometry is felt, and it is untestable by
construction rather than for want of instruments. Two phenomenal assignments
differing only by a relabelling that preserves every relation produce identical
observations, so absolute quale labels are gauge. The module says that in those
words and the test checks it still does, because the sentence is the only thing
standing between a distance matrix and a claim nobody can support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.run_content_geometry import (
    CLASSES,
    PREREGISTERED,
    READ,
    _agreement,
    _behaviour,
    _geometry,
    _prediction,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def _bank(separation: float, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "alarm": rng.normal(loc=separation, size=(24, 6)),
        "greeting": rng.normal(loc=0.0, size=(24, 6)),
        "question": rng.normal(loc=separation / 2, size=(24, 6)),
    }


# ── the geometry ──────────────────────────────────────────────────────────


def test_classes_that_differ_are_further_apart_than_the_floor() -> None:
    out = _geometry(_bank(2.5), seed=0)
    assert out["between"]
    assert len(out["separated"]) == len(out["between"])


def test_the_floor_is_a_class_split_against_itself() -> None:
    """Two halves of one class are the same law, so what this reads is the estimator."""
    out = _geometry(_bank(0.0), seed=0)
    assert out["within"]
    assert not out["separated"], (
        "classes drawn from one distribution separated, so the floor is not a floor"
    )


def test_every_pair_gets_a_distance() -> None:
    out = _geometry(_bank(2.0), seed=0)
    assert set(out["between"]) == {"alarm|greeting", "alarm|question", "greeting|question"}


# ── what the system did with them ─────────────────────────────────────────


def test_the_trace_is_read_off_the_state_not_off_a_report() -> None:
    """Two classes that leave the same trace are two the system did not tell apart."""
    same = {"a": np.ones((8, 4)), "b": np.ones((8, 4))}
    assert _behaviour(same)["a|b"] == pytest.approx(0.0, abs=1e-9)


def test_a_class_that_moved_the_state_elsewhere_shows_in_the_trace() -> None:
    apart = {"a": np.tile([1.0, 0, 0, 0], (8, 1)), "b": np.tile([0, 1.0, 0, 0], (8, 1))}
    assert _behaviour(apart)["a|b"] > 0.9


def test_agreement_is_an_ordering_and_not_a_fit() -> None:
    """Three pairs is too few for a slope and enough to ask about an order."""
    causal = {"x|y": 0.1, "x|z": 0.5, "y|z": 0.9}
    same = {"x|y": 0.2, "x|z": 0.4, "y|z": 0.7}
    flipped = {"x|y": 0.9, "x|z": 0.5, "y|z": 0.1}
    assert _agreement(causal, same)["same_order"] is True
    assert _agreement(causal, flipped)["same_order"] is False


def test_too_few_pairs_says_so_rather_than_reporting_an_order() -> None:
    assert "note" in _agreement({"x|y": 0.1}, {"x|y": 0.2})


# ── the registered direction ──────────────────────────────────────────────


def test_the_direction_is_registered_before_the_perturbation() -> None:
    assert "alarm" in PREREGISTERED
    assert "control" in _prediction({"alarm|greeting": 1.0, "greeting|question": 1.0},
                                    {"alarm|greeting": 0.5, "greeting|question": 1.0})["note"]


def test_the_registered_direction_holds_when_the_attended_pairs_shorten() -> None:
    before = {"alarm|greeting": 1.0, "alarm|question": 1.0, "greeting|question": 1.0}
    after = {"alarm|greeting": 0.6, "alarm|question": 0.7, "greeting|question": 1.0}
    assert _prediction(before, after)["the_registered_direction_held"] is True


def test_a_control_pair_that_moved_as_much_fails_the_prediction() -> None:
    """If the pair involving neither attended class moved too, so did the measurement."""
    before = {"alarm|greeting": 1.0, "alarm|question": 1.0, "greeting|question": 1.0}
    after = {"alarm|greeting": 0.6, "alarm|question": 0.7, "greeting|question": 0.5}
    assert _prediction(before, after)["the_registered_direction_held"] is False


# ── the line ──────────────────────────────────────────────────────────────


def test_the_classes_are_fixed_before_the_run() -> None:
    """A class list chosen after seeing which pairs came out close draws the map
    around the territory."""
    assert set(CLASSES) == {"greeting", "question", "alarm"}
    assert all(len(contents) >= 3 for contents in CLASSES.values())


def test_the_geometry_is_read_over_what_the_content_reached() -> None:
    """A content that only moved the percept stream has not become content."""
    assert "P" in READ and len(READ) > 1


def test_the_report_says_it_is_not_evidence_about_feeling() -> None:
    source = (REPO / "tools" / "run_content_geometry.py").read_text(encoding="utf-8")
    assert "None of it is " in source
    assert "evidence that the geometry is felt" in source
    assert "absolute quale labels are gauge" in source


def test_nothing_in_it_claims_a_quale() -> None:
    source = (REPO / "tools" / "run_content_geometry.py").read_text(encoding="utf-8")
    for forbidden in ('"CONSCIOUS"', '"felt"', '"quale"', '"qualia"'):
        assert forbidden not in source
