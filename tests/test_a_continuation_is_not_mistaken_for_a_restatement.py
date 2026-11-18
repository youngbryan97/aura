"""A continuation joins the answer it continues; only one that says the whole answer again replaces it.

LIVE 17 November 2026, typed into her window: her answer about the week's outages was cut at its token ceiling after
1,362 characters, the continuation added ", or what the Apple/NSA outages specifically affected?", and the person was
served the fragment alone, because every subject word of it was in her answer.
"""
from __future__ import annotations

from interface.routes.chat_reply_continuation import (
    _continuation_restates,
    _merge_reply_continuation,
)

ANSWER = ("Here is what was reported this week. On Friday a telecom outage triggered a nationwide FAA ground stop, "
          "and all flights were grounded for hours. Major banks had outages yesterday, and today Apple and the NSA "
          "both reported outages of their own. The sources do not say the outages share a cause, and the NSA says "
          "little about what was affected; I can look into the causes")
FRAGMENT = ", or what the Apple/NSA outages specifically affected?"


def test_a_fragment_of_the_answers_words_continues_it():
    assert not _continuation_restates(ANSWER, FRAGMENT)
    merged = _merge_reply_continuation(ANSWER, FRAGMENT)
    assert merged.startswith(ANSWER) and merged.endswith(FRAGMENT)


def test_an_answer_said_again_in_other_words_still_replaces_one_left_unfinished():
    """LIVE 2026-09-19: the same answer, started again, went to the person twice."""
    head = "**240 minutes (4 hours).** Net drain is 8 − 3 = 5 L/min, so 1,200 ÷ 5 = 240"  # cut before its stop
    tail = "**240 min.** Net drain 8 − 3 = 5 L/min → 1,200 ÷ 5 = 240."
    assert _continuation_restates(head, tail)
    assert _merge_reply_continuation(head, tail) == tail
