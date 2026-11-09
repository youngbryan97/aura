"""A question asked in a video she watches is not the person's, in her remarks or her replies.

LIVE 8 November 2026, the demo: a question asked in the Allen Pan video, "do you think this snake enjoys having legs?",
became the person's in her remark: "And then you ask him". "Ask" was not in the speech-act class that check attributions
to the person, and her turns while watching were not checked at all.
"""
from __future__ import annotations

import pytest

from core.dialogue.shared_history import fabricated_shared_history
from core.skills.watching_along import _without_words_put_in_their_mouth

ASKED = ("Now let's do Allen Pan. Go to his channel, look through his videos, and pick one under ten minutes that "
         "interests you most. Same again.")
REMARK = ("And he just covered his face. And then you ask him, \"do you think this snake enjoys having legs?\" and he "
          "goes for sure.\n\nAnd now you can walk your snake.")


@pytest.mark.parametrize(("said", "theirs"), [
    ("And then you ask him, \"do you think this snake enjoys having legs?\" and he goes for sure.", False),
    ("If you ask me, the noodle part was the whole video.", True),
    ("Unless you say otherwise, I will keep the snake one.", True),
    ("You asked me to pick one under ten minutes, so I went with the snake legs.", True),
])
def test_what_is_put_in_the_persons_mouth(said, theirs):
    assert (not fabricated_shared_history(said, ASKED)) is theirs


def test_her_remark_keeps_all_but_the_sentence_that_gives_the_video_s_question_to_the_person():
    kept = _without_words_put_in_their_mouth(REMARK, ASKED)
    assert "you ask him" not in kept
    assert kept.startswith("And he just covered his face.")
    assert kept.endswith("And now you can walk your snake.")


def test_what_the_person_said_while_watching_is_theirs():
    exchanges = [{"user": "Do you think this snake enjoys having legs?", "aura": "Maybe."}]
    assert _without_words_put_in_their_mouth(REMARK, ASKED, exchanges) == REMARK
