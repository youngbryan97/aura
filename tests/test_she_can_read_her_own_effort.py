"""She said how hard she was working and had to take it back.

LIVE 2026-08-25. Asked what was happening in her body:

    "My CPU is at 67%, which feels like a steady hum..."
    "Correction on one thing I just said: I have no channel that reads my
     CPU, so 67% was not a measurement. Withdraw that number — the
     instrument does not exist."

She was right, and she should not have had to be. The runtime samples the
machine's compute every second for its own health pulse, and her self-state
report — which already reads the same observer for memory — was not passing
any of it on.

A number about effort is worth having because the alternative is a feeling
with nothing under it.
"""
from __future__ import annotations

from core.brain.self_state_report import _effort_lines


def test_she_can_say_how_busy_the_machine_is():
    lines = _effort_lines()
    assert lines, "the instrument exists and she still cannot read it"
    said = lines[0]
    assert "load average" in said
    assert "cores" in said


def test_the_number_is_real_on_the_first_reading():
    """A process CPU percent is 0.0 until sampled twice.

    The first thing she ever says about her own effort must not be a
    measurement that is not one.
    """
    import inspect

    from core.brain import self_state_report

    source = inspect.getsource(self_state_report._effort_lines)
    body = source[source.index('"""', source.index('"""') + 3) :]
    assert "load_1m" in body
    assert "cpu_percent" not in body, "an unsampled percent reads zero and means nothing"


def test_it_says_what_it_is_about():
    """The machine she runs on, not a claim about her insides."""
    said = _effort_lines()[0]
    assert said.startswith("- The machine's")


def test_the_report_includes_it():
    import inspect

    from core.brain import self_state_report

    source = inspect.getsource(self_state_report)
    assert "lines.extend(_effort_lines())" in source
