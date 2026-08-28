"""A generation is reading and then writing. Only the writing was timed.

The clock that extends a turn's deadline counted decoding at the measured rate
and said nothing about prefill, so a turn was given time to say its answer and
none to read the question. On the resident model a six-kilobyte prompt takes
about two minutes to read before a single token is produced.

LIVE, 2026-08-28: three files read correctly, then the answer generation
cancelled at 119.5 seconds of a 120-second prefill, having produced nothing.
Everything the loop found was right and the turn had never been given time to
look at it.

The decode rate was measured and the read rate was not, which is why one half
of every generation was invisible to every deadline built from these numbers.
"""

from __future__ import annotations

import pytest

from core.brain.llm.thinking_reserve import (
    forget,
    record_read_rate,
    seconds_to_read,
)


@pytest.fixture(autouse=True)
def _clean():
    forget()
    yield
    forget()


def test_an_unmeasured_rate_extends_nothing() -> None:
    """Silent in the same way the decode rate is, and for the same reason."""

    assert seconds_to_read(6298) == 0.0


def test_it_learns_what_reading_actually_costs() -> None:
    for _ in range(12):
        record_read_rate(prompt_chars=6000, elapsed_s=110.0)
    got = seconds_to_read(6298)
    assert 100.0 < got < 140.0, got


def test_only_comparable_prompts_count() -> None:
    """A window of short prompts must not report a rate a long one never sees.

    The same rule the decode rate follows: below half the wanted size the
    measurement is dominated by something else.
    """

    for _ in range(20):
        record_read_rate(prompt_chars=200, elapsed_s=0.5)
    assert seconds_to_read(6298) == 0.0


def test_nonsense_is_not_recorded() -> None:
    for bad in ({"prompt_chars": 0, "elapsed_s": 1.0},
                {"prompt_chars": 100, "elapsed_s": 0.0},
                {"prompt_chars": 100, "elapsed_s": -1.0},
                {"prompt_chars": "x", "elapsed_s": 1.0}):
        record_read_rate(**bad)
    assert seconds_to_read(100) == 0.0


def test_the_clock_adds_reading_to_decoding() -> None:
    from pathlib import Path

    body = Path("core/brain/inference_gate.py").read_text()
    assert "(_decode_s + _read_s) * _generations" in body
    # And the worker records the measurement the clock reads.
    worker = Path("core/brain/llm/mlx_worker.py").read_text()
    assert "_record_read_rate(" in worker
    assert "first_token_latency_s" in worker


def test_forgetting_forgets_the_read_rate_too() -> None:
    for _ in range(12):
        record_read_rate(prompt_chars=6000, elapsed_s=110.0)
    assert seconds_to_read(6298) > 0.0
    forget()
    assert seconds_to_read(6298) == 0.0


def test_the_reading_rate_survives_a_restart(monkeypatch, tmp_path) -> None:
    """A measurement that does not survive is a measurement nobody has.

    The recorder was added and its persistence was not, so the rate was learned
    in one process, could not be read from any other, and every restart began
    knowing nothing about how long a prompt takes to read — on a runtime that
    restarts whenever the model changes.
    """

    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    from core.brain.llm import thinking_reserve as reserve

    reserve.forget()
    for _ in range(12):
        reserve.record_read_rate(prompt_chars=6000, elapsed_s=110.0)
    learned = reserve.seconds_to_read(6298)
    assert learned > 0.0
    assert reserve.save()

    # A fresh process: memory empty, the file still there.
    reserve._read_rates.clear()
    reserve._restored = False
    assert reserve.seconds_to_read(6298) == pytest.approx(learned)
    reserve.forget()


def test_a_corrupt_row_on_disk_does_not_take_the_rest_with_it(
    monkeypatch, tmp_path
) -> None:
    """The file is read by a later process, so it is where corruption arrives."""

    import json

    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    from core.brain.llm import thinking_reserve as reserve

    reserve.forget()
    rows = [[6000, 54.5] for _ in range(12)]
    rows.insert(4, ["not a number", "nor this"])
    rows.insert(9, [0, -1.0])
    (tmp_path / "decode_measurements.json").write_text(
        json.dumps({"read_rates": rows, "rates": [], "reasoning_tokens": []})
    )

    reserve._read_rates.clear()
    reserve._restored = False
    assert reserve.seconds_to_read(6298) > 0.0
    reserve.forget()
