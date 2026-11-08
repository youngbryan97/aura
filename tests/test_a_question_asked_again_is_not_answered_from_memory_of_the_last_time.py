"""Recall does not bring back an earlier sitting's answer to the same words: it was about what was the case then.

LIVE 8 November 2026, dress rehearsal: asked "So what did you think of all three?" half an hour after the sitting before
had asked it, her answer gave that sitting's second video and closing line as this sitting's.
"""

import time

from core.memory.rag_bridge import _asked_before_in_these_words

ASKED = "So what did you think of all three?"
NOW = time.time()


def _memory(text, ago):
    return {"text": text, "timestamp": NOW - ago}


def test_an_earlier_sittings_answer_to_these_words_is_left_out():
    memory = _memory("User: so what did you think of all three? Aura: Fish Tank was the reset.", 3600)
    assert _asked_before_in_these_words(memory, ASKED, sitting_began_at=NOW - 1800)


def test_this_sittings_memories_and_other_words_are_not():
    assert not _asked_before_in_these_words(_memory("so what did you think of all three", 60), ASKED, NOW - 1800)
    assert not _asked_before_in_these_words(_memory("Bryan said his sister moved to Denver.", 3600), ASKED, NOW - 1800)
    assert not _asked_before_in_these_words({"text": "so what did you think of all three"}, ASKED, NOW - 1800)


def test_its_time_may_be_in_its_metadata():
    memory = {"text": "So what did you think of all three? The lamp one.", "metadata": {"timestamp": NOW - 7200}}
    assert _asked_before_in_these_words(memory, ASKED, sitting_began_at=NOW - 1800)
