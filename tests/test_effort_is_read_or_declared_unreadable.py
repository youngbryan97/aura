"""She either measures the effort or says she cannot see it.

LIVE 2026-08-18: "how hard was that last answer to produce?"

    It took about 1.2 seconds of wall time on the host system to generate that
    response. CPU utilization was around 3% during that period, and memory
    usage peaked at about 2.7GB.

The turn had taken fourteen seconds. Every figure was invented, in the
confident register of a reading, on a question that sounds like it should have
one.

The instruments block already names what it cannot see — cycle count, mood,
skill registry — precisely so an absence is spoken rather than filled. Effort
had no line at all: nothing to say, and something said anyway.
"""

from __future__ import annotations

import pytest

from core.brain.self_state_report import runtime_self_report


def test_effort_is_always_accounted_for() -> None:
    """Either a measurement or a stated inability — never nothing."""
    report = runtime_self_report()

    assert "Effort and duration" in report or "Measured cognitive phases" in report


def test_an_unreadable_measurement_forbids_estimating_it() -> None:
    report = runtime_self_report()
    if "Measured cognitive phases" in report:
        pytest.skip("phases are measured in this process")

    assert "do not estimate seconds or percentages" in report


def test_measured_phases_do_not_claim_to_be_wall_time(monkeypatch) -> None:
    """Phase time is not the same number as how long the answer took."""
    import core.brain.self_state_report as module

    class _Instrumentation:
        def report(self):
            return {
                "passes": {"draft": {"runs": 2, "total_s": 3.5, "max_s": 2.0}},
                "hottest": ["draft"],
            }

    monkeypatch.setattr(
        "core.pipeline.pass_manager.get_instrumentation", lambda: _Instrumentation()
    )

    line = module._turn_cost_line()

    assert "draft 3.50s over 2 run(s)" in line
    assert "not end-to-end wall time" in line


def test_a_broken_instrument_still_produces_the_honest_line(monkeypatch) -> None:
    def _explode():
        raise RuntimeError("instrumentation gone")

    monkeypatch.setattr("core.pipeline.pass_manager.get_instrumentation", _explode)

    import core.brain.self_state_report as module

    assert "not readable from this turn" in module._turn_cost_line()
