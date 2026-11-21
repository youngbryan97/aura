"""A request to work on code goes to her code runner, not her desktop lane.

LIVE 20 November 2026, typed into her window: "Find the two largest Python files under core/brain by line count, count
their lines by actually running something, and tell me by what percentage the larger one is bigger than the smaller."
The words "find" and "files" made it screen work; her desktop lane refused it ("no complete observable acceptance
contract"), the whole message was searched on the web, and she served no answer of hers. Her reading of whether a
request asks for work on code (core/intent/asks_for_code_work.py) read it as not code work too: nothing it declared ran
code to measure anything.
"""
from __future__ import annotations

import pytest

LIVE = ("Find the two largest Python files under core/brain by line count, count their lines by actually running "
        "something, and tell me by what percentage the larger one is bigger than the smaller.")


@pytest.mark.unit
def test_a_request_read_as_work_on_code_is_not_a_desktop_objective(monkeypatch):
    import core.intent.asks_for_code_work as reader
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    monkeypatch.setattr(reader, "asks_for_code_work", lambda _text: None)
    assert looks_like_desktop_objective(LIVE) is True, "the words alone call it screen work"
    monkeypatch.setattr(reader, "asks_for_code_work", lambda _text: True)
    assert looks_like_desktop_objective(LIVE) is False


@pytest.mark.unit
def test_a_named_application_is_still_opened(monkeypatch):
    import core.intent.asks_for_code_work as reader
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    monkeypatch.setattr(reader, "asks_for_code_work", lambda _text: True)
    assert looks_like_desktop_objective("Open Terminal and run make test.") is True


@pytest.mark.model
def test_on_her_encoder_the_live_request_is_work_on_code_and_not_screen_work():
    from core.cognition.evidence_relevance import prewarm_evidence_relevance, semantic_routing_ready

    try:
        prewarm_evidence_relevance()
    except RuntimeError as exc:
        pytest.skip(f"her encoder is not available here: {exc}")
    if not semantic_routing_ready():
        pytest.skip("her encoder did not load here")
    from core.intent.asks_for_code_work import asks_for_code_work
    from core.runtime.desktop_objective_intent import looks_like_desktop_objective

    assert asks_for_code_work(LIVE) is True
    assert looks_like_desktop_objective(LIVE) is False


#: Screen work the routing tests hold, none of them among the reader's declared examples.
SCREEN_WORK = ["increase the brightness", "create a folder called Orca on my desktop",
               "Create a Word document with two bullet points and save it.", "close the front window",
               "when I close the window, save my notes to Notes first"]


@pytest.mark.model
@pytest.mark.parametrize("request_text", SCREEN_WORK)
def test_on_her_encoder_screen_work_is_not_work_on_code(request_text):
    """Once this reading decided which lane a request goes to, on her encoder it read these as work on code: it knew
    two kinds, code and questions about the world, and screen work sat nearer code."""
    from core.cognition.evidence_relevance import prewarm_evidence_relevance, semantic_routing_ready
    from core.intent import asks_for_code_work as reader

    try:
        prewarm_evidence_relevance()
    except RuntimeError as exc:
        pytest.skip(f"her encoder is not available here: {exc}")
    if not semantic_routing_ready():
        pytest.skip("her encoder did not load here")
    assert request_text not in {*reader._CODE_WORK, *reader._ABOUT_THE_WORLD, *reader._ON_THE_SCREEN}
    assert reader.asks_for_code_work(request_text) is not True
