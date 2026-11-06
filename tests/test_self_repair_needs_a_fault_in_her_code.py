"""Her self-repair turn needs evidence of a fault her code can be changed to fix (6 November 2026).

Of the 37 repair turns in her log, 27 were for model replies that were not valid JSON, 4 for tool and endpoint
timeouts, 1 for a model that returned no valid Python, and one, an email read that met its bound, started the turn that
stopped her state log on the third day run.
"""
from __future__ import annotations

from core import terminal_monitor as module
from core.terminal_monitor import ErrorEntry, TerminalMonitor

ROOT = module._OWN_ROOT


def _monitor():
    monitor = TerminalMonitor.__new__(TerminalMonitor)
    monitor._actionable_patterns = {
        r"KeyError|IndexError": "Fix a data access error in the code",
        r"JSONDecodeError|json.decoder": "Fix a JSON parsing error in data",
        r"TimeoutError|timed out": "Investigate a timeout",
        r"Foreground conversation lane returned no text": "Investigate foreground conversation lane blank output",
    }
    return monitor


def _traceback(path: str, ending: str) -> str:
    return (
        "Aura.Mind | tick failed\nTraceback (most recent call last):\n"
        f'  File "{ROOT}core/mind_tick.py", line 210, in tick\n    await self._step()\n'
        f'  File "{path}", line 88, in _step\n    value = table[key]\n'
        f"{ending}"
    )


def test_a_key_error_in_her_code_is_for_repair_and_names_where():
    goal = _monitor()._classify_error(ErrorEntry(
        message=_traceback(f"{ROOT}core/affect/what_it_was_worth.py", "KeyError: 'valence'"), level="ERROR", source="x"))
    assert goal == "Fix a data access error in the code at core/affect/what_it_was_worth.py:88"


def test_a_bound_that_held_or_a_reply_that_would_not_parse_is_not():
    monitor = _monitor()
    for ending in ("TimeoutError", "asyncio.exceptions.CancelledError", "ConnectionResetError: peer",
                   "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"):
        entry = ErrorEntry(message=_traceback(f"{ROOT}core/x.py", ending), level="ERROR", source="x")
        assert monitor._classify_error(entry) is None, ending


def test_log_lines_with_no_traceback_are_not():
    monitor = _monitor()
    for message in (
        "core.resilience.cognitive_governor | Task email_adapter timed out.",
        "Aura.Errors | [DEGRADATION] opinion_engine (critical): JSONDecodeError: Unterminated string",
        "Brain.HealthRouter | Endpoint Reflex timed out after 12.0s (force_aborted=False).",
    ):
        assert monitor._classify_error(ErrorEntry(message=message, level="ERROR", source="x")) is None, message


def test_a_traceback_only_through_libraries_is_not():
    entry = ErrorEntry(message=(
        "Traceback (most recent call last):\n"
        '  File "/opt/homebrew/lib/python3.12/site-packages/httpx/_client.py", line 10, in send\n    x()\n'
        "KeyError: 'h'"), level="ERROR", source="x")
    assert _monitor()._classify_error(entry) is None


def test_a_person_who_waited_is_still_looked_at():
    entry = ErrorEntry(message="Foreground conversation lane returned no text", level="ERROR", source="Aura.Chat")
    assert _monitor()._classify_error(entry) == "Investigate foreground conversation lane blank output"
