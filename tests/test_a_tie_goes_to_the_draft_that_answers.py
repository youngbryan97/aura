"""A second voice whose job was to carry what the first could not.

`core/expression/antiphony.py` measures whether a reply answers a statement
rather than continuing it, and had no caller anywhere in the tree. The drafts
competition had the matching hole: when the lead is smaller than the spread
among the drafts that lost, the win means nothing and the first of the tie was
taken.

The council argues, which is voices holding positions against each other, and
the drafts compete, which is alternatives for one slot where only one survives.
Neither kept a voice whose job was to answer. On a tie — only on a tie, because
a draft that won on coherence won — the one that answers takes it.
"""

from __future__ import annotations

from core.expression.antiphony import APART, answer, answers
from core.expression.register import read


def test_a_statement_with_no_shape_has_no_answer() -> None:
    assert answer(read("hm")).measured is False


def test_an_answer_is_the_complement_of_the_statement() -> None:
    statement = read(
        "I think the parser is wrong and I want you to tell me exactly why it is wrong"
    )
    reading = answer(statement)
    if reading.measured:
        for move in reading.moves:
            assert move.theirs == 1.0 - move.mine


def test_answering_is_distance_not_direction() -> None:
    statement = read("I have been carrying this for weeks and I do not know what to do")
    echo = read("I have been carrying this for weeks and I do not know what to do")
    assert answers(statement, echo) is False, "an echo is not an answer"


def test_the_bar_is_the_packages_own() -> None:
    assert 0.0 < APART <= 1.0


def test_the_drafts_competition_asks_on_a_tie() -> None:
    from pathlib import Path

    source = Path("core/consciousness/multiple_drafts.py").read_text(encoding="utf-8")
    assert "if not decisive:" in source
    assert "self._one_that_answers(" in source
    assert "self._statement_text = text" in source


def test_a_decisive_win_is_left_alone() -> None:
    """The tie-break is inside the not-decisive branch and nowhere else."""
    from pathlib import Path

    source = Path("core/consciousness/multiple_drafts.py").read_text(encoding="utf-8")
    at = source.index("winner = self._one_that_answers(")
    before = source[:at].rsplit("\n", 3)[-3:]
    assert any("if not decisive:" in line for line in before), before


def test_the_tie_break_prefers_the_one_that_answers() -> None:
    from core.consciousness.multiple_drafts import MultipleDraftsEngine

    model = MultipleDraftsEngine.__new__(MultipleDraftsEngine)

    class _D:
        def __init__(self, content):
            self.content = content

    statement = (
        "I have been carrying this for weeks and I really do not know what I should do now"
    )
    echo = _D(statement)
    reply = _D("Name one thing you could try tomorrow, and tell me what would stop you")
    model._current_drafts = [echo, reply]

    chosen = model._one_that_answers(statement, echo)
    assert chosen in (echo, reply)
    if answers(read(statement), read(reply.content)):
        assert chosen is reply
