"""Her voice is one witness; what she read is another.

Asked to consult her encyclopedia as well as her own voice, and to check the
two against each other: a thing two independent sources say is worth leaning
on, and a thing only one says is held knowing that.
"""

from __future__ import annotations

from core.agency.task_knowledge import Finding, TaskKnowledge
from core.agency.what_agrees import borne_out, say_the_same, witnesses


def test_two_sources_saying_the_same_thing_in_different_words_agree():
    assert say_the_same(
        "Keep your largest tile in a corner and build around it",
        "Players keep the largest tile in one corner of the board",
    )


def test_a_number_they_disagree_on_is_a_disagreement():
    assert not say_the_same("Merge tiles to reach 2048", "Merge tiles to reach 4096")


def test_nothing_in_common_is_nothing_in_common():
    assert not say_the_same("Keep the largest tile in a corner", "The game was released in 2014")


def test_a_source_does_not_second_itself():
    found = witnesses(
        [
            Finding("Keep the largest tile in a corner", "the encyclopedia"),
            Finding("Keep your largest tile in the corner", "the encyclopedia"),
            Finding("Players keep the largest tile in a corner", "what I read on the web"),
        ]
    )
    assert found[0].by == ("what I read on the web",)
    assert found[2].by == ("the encyclopedia",)


def test_what_her_voice_says_is_checked_against_what_she_read():
    read = [Finding("Keep the largest tile in a corner of the board", "the encyclopedia")]
    borne = borne_out("keep the largest tile in the bottom-left corner", read)
    assert borne.second_witness and "the encyclopedia agree" in borne.says_so()
    alone = borne_out("alternate left and right as fast as possible", read)
    assert not alone.second_witness and alone.says_so() == "only my own voice says so"


def test_she_says_when_what_she_read_is_seconded():
    known = TaskKnowledge(
        goal="play 2048",
        findings=[
            Finding("Keep the largest tile in a corner", "the encyclopedia"),
            Finding("Players keep the largest tile in one corner", "what I read on the web"),
        ],
    )
    assert "the encyclopedia and what I read on the web agree" in known.narrate()


def test_the_line_she_takes_is_checked_against_what_she_read():
    from screen_pursuit_support import pursuit_source
    from source_contract import in_order

    in_order(pursuit_source(), "said = fresh.narrate()", "borne_out(fresh.approach, read)", "borne.says_so()")
