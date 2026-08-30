"""One rate window for every model let a 9B size a 27B's clock.

A 9B decodes at about fifteen tokens a second and a 27B at half that. Pooled,
a turn on the larger model was given a clock sized by the smaller one, ran past
it, and was aborted with everything it had written discarded. LIVE 2026-08-30,
three times on one question, at 166.8 seconds each.
"""

from __future__ import annotations

import pytest

from core.brain.llm import thinking_reserve


@pytest.fixture(autouse=True)
def _nothing_measured_yet(monkeypatch):
    monkeypatch.setattr(thinking_reserve, "_rates", {})
    monkeypatch.setattr(thinking_reserve, "_restore_once", lambda: None)
    monkeypatch.setattr(thinking_reserve, "_written_down", lambda: None)


def _measure(model: str, tokens: int, seconds: float, times: int = 12) -> None:
    for _ in range(times):
        thinking_reserve.record_decode_rate(
            generated_tokens=tokens, elapsed_s=seconds, model=model
        )


def test_a_slower_model_is_given_a_longer_clock():
    _measure("a 9B", 600, 40.0)
    _measure("a 27B", 600, 85.0)

    assert thinking_reserve.seconds_to_decode(600, "a 27B") > thinking_reserve.seconds_to_decode(
        600, "a 9B"
    )


def test_a_fast_model_cannot_shorten_a_slow_one_s_clock():
    """The defect, stated directly."""
    _measure("a 27B", 600, 85.0)
    slow_alone = thinking_reserve.seconds_to_decode(600, "a 27B")
    _measure("a 9B", 600, 10.0, times=200)

    assert thinking_reserve.seconds_to_decode(600, "a 27B") == slow_alone


def test_a_caller_that_names_no_model_still_gets_an_estimate():
    """Every caller did this before, and it must keep working."""
    _measure("a 9B", 600, 40.0)
    assert thinking_reserve.seconds_to_decode(600) > 0.0


def test_a_model_with_no_readings_of_its_own_falls_back_rather_than_refusing():
    _measure("a 9B", 600, 40.0)
    assert thinking_reserve.seconds_to_decode(600, "a model nobody has timed") > 0.0


def test_nothing_measured_anywhere_stays_unmeasured():
    assert thinking_reserve.seconds_to_decode(600, "a 27B") == 0.0


def test_readings_survive_a_round_trip_per_model(tmp_path, monkeypatch):
    monkeypatch.setattr(thinking_reserve, "_store_path", lambda: tmp_path / "rates.json")
    _measure("a 27B", 600, 85.0)
    slow = thinking_reserve.seconds_to_decode(600, "a 27B")
    thinking_reserve.save()

    monkeypatch.setattr(thinking_reserve, "_rates", {})
    thinking_reserve.load()
    assert thinking_reserve.seconds_to_decode(600, "a 27B") == pytest.approx(slow, rel=0.01)


def test_a_file_written_before_models_were_kept_still_reads(tmp_path, monkeypatch):
    import json

    target = tmp_path / "rates.json"
    target.write_text(json.dumps({"rates": [[600, 7.0]] * 12}))
    monkeypatch.setattr(thinking_reserve, "_store_path", lambda: target)

    thinking_reserve.load()
    assert thinking_reserve.seconds_to_decode(600) > 0.0
