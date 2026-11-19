"""A reply that gives what her own program returned answers the question, whatever cue words it leaves out."""
from __future__ import annotations

from core.conversation import surface_disposition
from core.conversation.response_request_coverage import _instruction_coverage_reasons

_ASKED = ("Under core/brain in your own source, find the three largest Python files by line count. For each one, tell me "
          "its line count and how many functions it defines, counting methods and async functions too. Actually run "
          "something to get the numbers, then tell me which of the three has the most functions.")
_TABLE = "| File | Lines | Functions |\n|---|---|---|\n| `mlx_client.py` | 12,589 | 324 |\n\n`mlx_client.py` has the most."
_KEY = ("code_repl", "run", "", True, "done")


def _returned(monkeypatch, text, ok=True):
    key = (*_KEY[:3], ok, _KEY[4])
    monkeypatch.setattr(surface_disposition, "_this_turns_receipts", lambda: [(key, {"as_made": {"returned": text}})])


def test_a_table_of_what_her_program_printed_is_not_short(monkeypatch):
    _returned(monkeypatch, "core/brain/llm/mlx_client.py lines=12589 functions=324")
    assert "missing_requested_objective_facets" not in _instruction_coverage_reasons(_ASKED, _TABLE)


def test_the_same_table_with_no_work_behind_it_is_still_asked_about(monkeypatch):
    monkeypatch.setattr(surface_disposition, "_this_turns_receipts", list)
    assert "missing_requested_objective_facets" in _instruction_coverage_reasons(_ASKED, _TABLE)


def test_numbers_from_a_call_that_failed_are_not_work(monkeypatch):
    _returned(monkeypatch, "lines=12589 functions=324", ok=False)
    assert "missing_requested_objective_facets" in _instruction_coverage_reasons(_ASKED, _TABLE)
