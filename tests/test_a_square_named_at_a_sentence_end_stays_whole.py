"""A square named at the end of a sentence stays whole: "f3. It's ..." is not "f" and a list item "3."

LIVE 21 November 2026 her account of a game against Stockfish reached Bryan as "Knight g1 → f" and, on its own line,
"3. It's the piece that leaps two-and-one", at every move. A rule for a list marker welded onto a word ("12 marbles2.
We want to find…") took each square's number for one. A marker continues a list, so the one before it is there.
"""
from __future__ import annotations

from core.conversation.response_reliability import _normalize_prose_format


def test_a_square_at_a_sentence_end_is_not_a_list_marker() -> None:
    said = "**Move 1 (mine):** Knight g1 → f3. It's the piece that leaps. **Move 1 (theirs):** Pawn d7 → d5. They push."
    assert _normalize_prose_format(said) == said


def test_a_list_marker_welded_onto_a_word_still_gets_its_own_line() -> None:
    said = "So 4 + 8 = 12 marbles1. Count them. Then 12 marbles2. We want to find more."
    assert "\n2. We want to find more." in _normalize_prose_format(said)
