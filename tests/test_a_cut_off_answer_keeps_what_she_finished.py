"""An answer cut off mid-sentence is served up to its last finished sentence.

LIVE 2026-10-03: sent a Wikipedia link, she wrote 3,862 characters about the
page and ran out of time mid-sentence. The stabilizer replaced all of it with
a fresh 87-character reply from a small repair prompt.
"""

from __future__ import annotations

import pytest

from interface.routes.chat_desktop_repair import (
    _looks_truncated_tail,
    _up_to_its_last_finished_sentence,
)

DRAFT = (
    "Got it. I read the page about Bam Adebayo's 83-point game. On March 10, 2026, "
    "at Kaseya Center, he scored 83 points against the Washington Wizards, the second "
    "highest total in NBA history after Wilt Chamberlain's 100. He shot 20 for 43 from "
    "the field and 36 for 43 from the line. The reaction split along a line I keep "
    "coming back to: Spoelstra called it a team effort, and the Wizards' coach said the "
    "fourth quarter was not a real basketball game, because the Heat kept feeding him the"
)


def test_the_last_finished_sentence_is_where_it_stops() -> None:
    finished = _up_to_its_last_finished_sentence(DRAFT)
    assert finished.endswith("from the line.")
    assert not _looks_truncated_tail(finished)
    assert _up_to_its_last_finished_sentence('He said "done." Then he left and') == 'He said "done."'
    assert _up_to_its_last_finished_sentence("nothing was finished here") == ""


@pytest.mark.asyncio
async def test_the_stabilizer_serves_what_she_finished(monkeypatch) -> None:
    from interface.routes import chat as chat_routes
    from interface.routes import chat_reply_repair

    # No rewrite lane: a pass can only come from what she wrote.
    monkeypatch.setattr(chat_reply_repair, "resolve_inference_gate", lambda: None, raising=False)
    served = await chat_routes._stabilize_user_facing_reply(
        "https://en.wikipedia.org/wiki/Bam_Adebayo%27s_83-point_game",
        DRAFT,
        desktop_cognitive_engine_required=True,
        protected_foreground_lane=True,
    )
    assert "Kaseya Center" in served and served.endswith("from the line.")
