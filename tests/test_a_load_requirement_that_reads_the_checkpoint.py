"""A spawn refused for room the checkpoint did not need.

LIVE 2026-09-19, immediately after the delivery poll for a turn somebody was
waiting on:

    GET /api/chat/delivery/aura-chat-674d1899-... 202 Accepted
    ModelLoadAdmissionRefused: memory_pressure_refused_worker_spawn:
      model_load_headroom:15.1GB < required 24.0GB

The turn never got its worker, so the cortex never answered it, and the
reply came back from the fallback with the footer that says so. On the same
boot the brainstem lane was refused 222 times on the same gate.

Two reasons the number was wrong.

The 16GB floor is for a checkpoint whose size cannot be read. It was
applied where the size IS known, so a model measured at 8.0GB was held to
twice its footprint.

And when the directory genuinely cannot be measured the requirement falls
back to a flat 24.0 with nothing but a debug line to say why — a number
derived from nothing, deciding whether a waiting turn gets a model.
"""
from __future__ import annotations

from core.brain.llm.how_big_is_the_checkpoint import (
    _measured_model_footprint_gb,
    _model_load_min_available_gb,
)

BONSAI = "/Users/bryan/.aura/live-source/models/Ternary-Bonsai-2-27B-mlx-2bit"


def test_a_measured_checkpoint_is_not_held_to_the_unmeasured_floor():
    """The floor is for the case where nothing was measured."""
    measured = _measured_model_footprint_gb(BONSAI)
    if measured is None:  # pragma: no cover - the pack is not on this host
        import pytest

        pytest.skip("the measured checkpoint is not present here")

    required = _model_load_min_available_gb(BONSAI)
    assert required == measured * 1.20 + 1.0
    assert required < 16.0, (
        "an 8GB checkpoint held to a 16GB floor is the refusal this is about"
    )


def test_the_requirement_still_covers_the_load_with_room():
    """Relaxing it must not admit a model that does not fit."""
    measured = _measured_model_footprint_gb(BONSAI)
    if measured is None:  # pragma: no cover
        import pytest

        pytest.skip("the measured checkpoint is not present here")

    assert _model_load_min_available_gb(BONSAI) > measured, (
        "the requirement can never sit below the footprint it admits"
    )


def test_an_unmeasurable_checkpoint_keeps_the_conservative_default():
    """Unreadable is not a reason to wave a load through."""
    assert _model_load_min_available_gb("/nonexistent/Aura-Cortex-32B") >= 22.0


def test_an_unmeasurable_checkpoint_says_so():
    """A requirement derived from nothing is worth a record.

    The only trace of why it was 24.0 was a debug line about an unreadable
    directory, and a debug line is not something anybody reads while a turn
    is failing.
    """
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    _model_load_min_available_gb("/nonexistent/Aura-Cortex-32B")

    recorded = [
        record
        for record in tracker.recent()
        if "footprint unreadable" in str(getattr(record, "error_message", ""))
    ]
    assert recorded, "an undecidable requirement has to be on the record"


def test_a_measured_checkpoint_records_nothing():
    """The record is for the case that cannot be measured, not for every load."""
    from core.runtime.errors import get_degradation_tracker

    measured = _measured_model_footprint_gb(BONSAI)
    if measured is None:  # pragma: no cover
        import pytest

        pytest.skip("the measured checkpoint is not present here")

    tracker = get_degradation_tracker()
    tracker.reset()
    _model_load_min_available_gb(BONSAI)
    assert not [
        record
        for record in tracker.recent()
        if "footprint unreadable" in str(getattr(record, "error_message", ""))
    ]
