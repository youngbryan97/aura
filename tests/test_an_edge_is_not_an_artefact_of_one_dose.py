"""An edge measured at one displacement might only exist at that displacement.

Every channel in the battery is read at one intervention size. A channel that
appears at that size and nowhere near it is a threshold in the harness rather
than something the organism does, and one edge table cannot tell the two apart.
Turning the dial can.

Two shapes are named. An edge present at exactly one dose and absent either
side of it is the artefact this exists to catch. An edge the same size across a
fourfold change in dose is carrying a flag rather than a quantity, which is a
different thing to know and not automatically wrong.
"""

from __future__ import annotations

import pytest

from tools.run_dose_response import ABSENT, DELTAS, _curve, _direction, _verdict

pytestmark = pytest.mark.unit

BAR = 0.3


def _by_delta(**edges: list[float]) -> dict[float, dict[str, float]]:
    return {
        delta: {edge: values[index] for edge, values in edges.items()}
        for index, delta in enumerate(DELTAS)
    }


def test_the_doses_are_fixed_and_the_smallest_runs_first() -> None:
    """So the run cannot be stopped once a flattering one has been seen."""
    assert list(DELTAS) == sorted(DELTAS)
    assert len(DELTAS) >= 3


def test_an_edge_at_one_dose_only_is_named() -> None:
    out = _curve(_by_delta(spike=[0.0, 0.0, 0.9, 0.0]), BAR)
    assert out["edges_at_one_dose_only"] == ["spike"]
    assert out["edges"]["spike"]["present_at"] == [DELTAS[2]]


def test_an_edge_that_grows_with_the_dose_is_not_named() -> None:
    out = _curve(_by_delta(real=[0.2, 0.4, 0.6, 0.9]), BAR)
    assert out["edges_at_one_dose_only"] == []
    assert out["edges"]["real"]["grows_with_dose"] is True
    assert out["edges"]["real"]["dose_correlation"] > 0.9


def test_an_edge_flat_across_a_fourfold_change_is_named() -> None:
    """Present at every dose and the same size at all of them."""
    out = _curve(_by_delta(flag=[0.8, 0.8, 0.8, 0.8]), BAR)
    assert out["edges_flat_across_the_range"] == ["flag"]
    assert out["edges_at_one_dose_only"] == []


def test_an_edge_absent_everywhere_is_neither() -> None:
    out = _curve(_by_delta(quiet=[0.0, 0.01, 0.0, 0.02]), BAR)
    assert out["edges_at_one_dose_only"] == []
    assert out["edges_flat_across_the_range"] == []


def test_the_verdict_fails_when_any_edge_exists_at_one_dose_only() -> None:
    curves = {"positive": _curve(_by_delta(spike=[0.0, 0.0, 0.9, 0.0]), BAR)}
    assert _verdict({"curves": curves})["no_edge_exists_at_only_one_dose"] is False


def test_the_verdict_passes_when_every_edge_has_a_shape() -> None:
    curves = {"positive": _curve(_by_delta(real=[0.2, 0.4, 0.6, 0.9]), BAR)}
    assert _verdict({"curves": curves})["no_edge_exists_at_only_one_dose"] is True


def test_a_channel_answering_the_same_way_either_way_is_named() -> None:
    """A displacement producing the same effect pushed either way carries no direction."""
    both = _direction(
        {
            "positive": _by_delta(edge=[0.1, 0.4, 0.6, 0.8]),
            "negative": _by_delta(edge=[0.1, 0.4, 0.6, 0.8]),
        },
        BAR,
    )
    assert both["edge"]["kept_either_way"] is True
    assert both["edge"]["one_direction_only"] is False


def test_a_channel_that_only_answers_one_way_is_named_too() -> None:
    one = _direction(
        {
            "positive": _by_delta(edge=[0.1, 0.4, 0.6, 0.8]),
            "negative": _by_delta(edge=[0.0, 0.0, 0.01, 0.02]),
        },
        BAR,
    )
    assert one["edge"]["one_direction_only"] is True


def test_the_absent_bar_is_below_the_edge_bar() -> None:
    assert 0.0 < ABSENT < 1.0
