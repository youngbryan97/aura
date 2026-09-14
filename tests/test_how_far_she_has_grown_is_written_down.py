"""A score three things read and nothing ever wrote.

`identity.evolution_score` is read by the subject-core schema as a coordinate
of her self-state, held by the clamp for a lesion, and tested against 0.70 by
the gate on her governed self-modification proposal path. Nothing has ever
written it. It is initialised to zero and stays there, so the column never
moved and the gate never opened — her own route to proposing a change to
herself was closed on a value that could not rise.

The organ that measures her development is already running and already
calibrates the reading: `relative_displacement` is this step against how much
that reservoir usually moves, with a half as an ordinary step.
"""

from __future__ import annotations

import pytest

from core.self.growth import MIN_STEPS, ORDINARY_STEP, Growth, GrowthLedger


def test_an_ordinary_life_sits_at_the_ordinary_step() -> None:
    led = GrowthLedger()
    for _ in range(MIN_STEPS + 3):
        led.note(ORDINARY_STEP)
    reading = led.read()
    assert reading.measured
    assert reading.score == pytest.approx(ORDINARY_STEP)
    assert not reading.above_ordinary


def test_development_moving_more_than_usual_rises() -> None:
    led = GrowthLedger()
    for _ in range(MIN_STEPS + 3):
        led.note(0.85)
    reading = led.read()
    assert reading.above_ordinary
    assert reading.score > 0.70, "the gate that has never opened can open"


def test_a_quiet_stretch_stays_under_the_gate() -> None:
    led = GrowthLedger()
    for _ in range(MIN_STEPS + 3):
        led.note(0.2)
    assert led.read().score < 0.70


def test_with_too_few_steps_it_says_so_rather_than_scoring_zero() -> None:
    led = GrowthLedger()
    led.note(0.9)
    reading = led.read()
    assert not reading.measured
    assert not reading.above_ordinary
    assert "developmental steps needed" in reading.why


def test_a_reading_outside_the_scale_is_brought_onto_it() -> None:
    led = GrowthLedger()
    for _ in range(MIN_STEPS + 1):
        led.note(4.0)
    assert led.read().score == pytest.approx(1.0)


def test_an_unreadable_step_is_not_a_step() -> None:
    led = GrowthLedger()
    before = led.steps()
    led.note(float("nan"))
    led.note("far")  # type: ignore[arg-type]
    assert led.steps() == before


def test_the_phase_writes_the_score(monkeypatch) -> None:
    """The gate's input has a writer now, and it is the organ's own reading."""
    import ast
    import inspect

    from core.phases import affect_update

    tree = ast.parse(inspect.getsource(affect_update))
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "evolution_score"
    ]
    assert writes, "nothing writes the score"


def test_the_reading_serialises_what_a_reader_needs() -> None:
    led = GrowthLedger()
    for _ in range(MIN_STEPS + 1):
        led.note(0.6)
    row = led.read().as_dict()
    for key in ("score", "steps", "above_ordinary", "measured", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not Growth().measured
    assert not Growth().above_ordinary
