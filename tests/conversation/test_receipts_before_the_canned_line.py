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

import inspect


def test_the_search_fallback_consults_receipts_first() -> None:
    from interface.routes import chat

    source = inspect.getsource(chat)
    index_receipts = source.find("concise_past_action_answer(user_message)")
    index_canned = source.find(
        "I don't have a clean grounded answer on that yet. I need to stick"
    )

    assert index_receipts != -1, "receipts are never consulted"
    assert index_canned != -1
    assert index_receipts < index_canned, "the canned line runs before the reading"


def test_receipts_that_cover_nothing_yield_nothing() -> None:
    """It must not invent an action to avoid the fallback."""
    from core.introspection.self_evidence import concise_past_action_answer

    assert concise_past_action_answer("what is the capital of France") == ""


def test_garbage_input_is_safe() -> None:
    from core.introspection.self_evidence import concise_past_action_answer

    for value in (None, "", "   ", 0):
        assert concise_past_action_answer(value) == ""
