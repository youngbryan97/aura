"""Ask the receipts before saying there is no grounded answer.

LIVE 2026-08-17: "prove to me you actually read that file and didn't just
pattern-match the answer" returned "I don't have a clean grounded answer on
that yet. I need to stick to the source instead of guessing."

The receipt for reading that file was on disk. The proof she was asked for was
the one thing she could have produced, and the fallback fired without ever
looking.

This is a reading, not a rescue: concise_past_action_answer returns text only
when receipts actually cover the question, so it cannot describe an action she
did not take.
"""

from __future__ import annotations

from pathlib import Path

ROUTES = Path(__file__).resolve().parents[2] / "interface" / "routes"


def test_the_canned_line_is_not_served_from_her_chat_routes() -> None:
    """13 November 2026 (0ff478280): no canned text replaces her draft. The line
    this test used to order receipts before is gone from every chat route, and
    it must not come back as a fallback."""
    for module in sorted(ROUTES.glob("*.py")):
        source = module.read_text(encoding="utf-8")
        assert "I don't have a clean grounded answer on that yet" not in source, module.name


def test_a_past_action_question_hands_her_the_receipts(monkeypatch) -> None:
    """The proof she was asked for reaches her turn asked again, and her reply is not rewritten."""
    import core.introspection.self_evidence as self_evidence
    from core.brain.her_turn_as_asked import a_turn_is_asked, what_the_record_says
    from interface.routes.chat_turn_recall import _append_past_action_record

    monkeypatch.setattr(self_evidence, "asks_about_past_actions", lambda _message: True)
    monkeypatch.setattr(
        self_evidence,
        "concise_past_action_answer",
        lambda _message: "I read core/agency/goal_engine.py at 14:02 (tool_execution receipt).",
    )
    reply = "I just matched the pattern, I think."

    with a_turn_is_asked("receipts"):
        served = _append_past_action_record("prove you actually read that file", reply)
        record = what_the_record_says()

    assert served == reply
    assert record and "goal_engine.py" in record[0]


def test_receipts_that_cover_nothing_yield_nothing() -> None:
    """It must not invent an action to avoid the fallback."""
    from core.introspection.self_evidence import concise_past_action_answer

    assert concise_past_action_answer("what is the capital of France") == ""


def test_garbage_input_is_safe() -> None:
    from core.introspection.self_evidence import concise_past_action_answer

    for value in (None, "", "   ", 0):
        assert concise_past_action_answer(value) == ""
