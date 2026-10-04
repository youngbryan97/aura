"""Talking about a page someone sent is not unrequested commentary.

LIVE 2026-10-03: sent a bare Wikipedia link, she wrote about the article, and
the stabilizer rejected the reply as an unrequested review because the
person's message named no "article". Her regenerated reply said the link held
only navigation.
"""

from __future__ import annotations

from interface.routes.chat import _looks_like_unrequested_content_review

REVIEW = (
    "The article is a strong piece: the narrative of the game report follows his 83 points "
    "quarter by quarter, and the passage on the reaction is the best part."
)


def test_a_review_of_a_page_she_was_sent_is_what_was_asked() -> None:
    link = "https://en.wikipedia.org/wiki/Bam_Adebayo%27s_83-point_game"
    assert _looks_like_unrequested_content_review(link, REVIEW) == (False, "")


def test_a_review_nobody_asked_for_is_still_caught() -> None:
    assert _looks_like_unrequested_content_review("How was your day?", REVIEW)[0] is True
