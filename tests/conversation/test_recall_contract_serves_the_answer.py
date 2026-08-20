"""When the runtime has composed the recall, serve it rather than refuse.

LIVE 2026-08-17: "what was the first thing I said to you in this conversation?"
returned "I couldn't get to an answer I'd stand behind on that one."

The transcript reading had been taken and delivered — the grounding line shows
present,receipts,transcript surviving to dispatch — and the route had already
built expected_recall_reply from the conversation's own turns. It then compared
the model's answer against that, found it inadequate, and threw BOTH away.

expected_recall_reply is a reading of what was said, not model prose
substituted for model prose, so serving it cannot invent an exchange. This is
the same lesson written three other places in the same file: a computed
arithmetic result, a measured file count, and a receipt-backed past action all
beat an apology.
"""

from __future__ import annotations

import inspect


def test_the_recall_contract_serves_instead_of_refusing() -> None:
    from interface.routes import chat

    source = inspect.getsource(chat)
    marker = "serving the transcript-composed recall instead of refusing"

    assert marker in source, "the recall contract still refuses"
    index = source.find(marker)
    tail = source[index : index + 900]

    assert "return expected_recall_reply" in tail
    assert "return None" not in tail.split("return expected_recall_reply")[0]


def test_the_served_path_is_named_in_the_trace() -> None:
    """A substituted answer must be attributable, not silently authored."""
    from interface.routes import chat

    source = inspect.getsource(chat)

    assert 'response_path="conversation_recall_from_transcript"' in source


def test_a_recall_question_with_no_draft_at_all_still_gets_the_answer() -> None:
    """Generation returning nothing is the case the contract could not reach.

    LIVE 2026-08-17: "what was the first thing I said to you in this
    conversation?" produced "compact desktop generation returned no usable
    text" three attempts running. The recall contract compares a draft against
    the composed answer, so with no draft it never ran, and the person got the
    apology while the transcript sat read and delivered in the same turn.
    """
    from interface.routes import chat

    source = inspect.getsource(chat)
    marker = "Serving the transcript-composed recall"

    assert marker in source
    index = source.find(marker)
    end = source.find("# A refusal is not an answer", index)
    recall_branch = source[index:end]

    assert "return await _finalize_fastpath(" in recall_branch
    assert "composed_recall," in recall_branch


def test_the_composed_recall_is_only_used_when_nothing_was_salvaged() -> None:
    """A real draft still wins; this is a floor, not an override."""
    from interface.routes import chat

    source = inspect.getsource(chat)
    index = source.find("if not salvaged_no_reply:")

    assert index != -1
    assert "_build_conversation_recall_reply" in source[index : index + 600]
