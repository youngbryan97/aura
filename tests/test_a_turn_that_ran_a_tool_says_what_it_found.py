"""A tool ran and found something, and the turn apologised anyway.

LIVE, 2026-08-27, repeatedly: file_operation read a project's docs in 4ms,
diagnose_repo returned a complete finding in 276ms, code_repl executed — and
the turn still ended in "I couldn't get to an answer I'd stand behind", because
the model wrote nothing usable afterwards.

Every governance link was clear by then. What was missing was somebody saying
what the tool had returned. Saying "I couldn't get to an answer" on top of a
tool result is not honesty; it is losing the work.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _what_the_tools_found


def test_nothing_ran_says_nothing() -> None:
    """It reports receipts; with none, the honest refusal stands."""
    assert _what_the_tools_found() == ""


def test_it_reads_the_turns_own_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a tool that really executed, and only content it really observed."""
    monkeypatch.setattr(
        "core.conversation.surface_disposition.turn_tool_receipts",
        lambda: (
            {
                "tool": "diagnose_repo",
                "ok": True,
                "observed_content": "add_line answered differently at invoice.py:4",
            },
        ),
        raising=False,
    )
    told = _what_the_tools_found()
    assert "diagnose_repo returned" in told
    assert "invoice.py:4" in told


def test_a_tool_that_failed_is_not_reported_as_a_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.conversation.surface_disposition.turn_tool_receipts",
        lambda: ({"tool": "web_search", "ok": False, "observed_content": "nothing"},),
        raising=False,
    )
    assert _what_the_tools_found() == ""


def test_a_tool_that_observed_nothing_is_not_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A receipt with no content is a record that something ran, not a finding."""
    monkeypatch.setattr(
        "core.conversation.surface_disposition.turn_tool_receipts",
        lambda: ({"tool": "file_operation", "ok": True, "observed_content": ""},),
        raising=False,
    )
    assert _what_the_tools_found() == ""


def test_it_says_it_is_a_record_rather_than_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It reports; the interpreting is what failed."""
    monkeypatch.setattr(
        "core.conversation.surface_disposition.turn_tool_receipts",
        lambda: ({"tool": "code_repl", "ok": True, "observed_content": "125.0"},),
        raising=False,
    )
    told = _what_the_tools_found()
    assert "what I ran and what came back" in told
    assert "125.0" in told


def test_the_receipt_records_what_came_back_not_only_that_it_ran() -> None:
    """A receipt holding the arguments and no result supports no answer.

    LIVE, 2026-08-27: file_operation read a project's docs in 6ms, the turn had
    nothing to say afterwards, and the fallback that reports what the tools
    found had nothing to report — the record held the call and not the result.
    """
    import inspect

    from core.brain import inference_gate

    source = inspect.getsource(inference_gate)
    assert "observed_content=observed[:2000]" in source, (
        "the tool-loop receipt stopped carrying what the tool returned"
    )
    # And it reads the shapes a tool result actually arrives in.
    for key in ("content", "stdout", "summary", "text", "output"):
        assert f'returned.get("{key}")' in source, key


def test_both_giving_up_paths_ask_what_the_tools_found() -> None:
    """Two places build the refusal, and a fix applied to one leaves the other.

    LIVE, 2026-08-27: the rescue was added to the first, the second served the
    turn, and a successful file_operation was still followed by "I couldn't get
    to an answer I'd stand behind."
    """
    from pathlib import Path

    source = Path("interface/routes/chat.py").read_text()
    builds = source.count("failure_reply = THE_HONEST_FAILURE")
    asks = source.count("evidenced_reply = _what_the_tools_found()")
    assert builds >= 2, "the refusal is built somewhere this no longer counts"
    assert asks == builds, (
        f"{builds} paths build the refusal and {asks} ask what the tools returned"
    )
