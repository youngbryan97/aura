"""A closed answer in the form the person dictated is not discarded for missing analytic facets.

On 5 November the self-prediction proof asked her "Which would you choose? Answer with that word, then how sure you
are, as a number from 0 to 100." She answered "Rain. 80", and the coverage gate read the request as asking her to
select and to explain, each needing eight words, and discarded her answer twice before falling back.
"""

from __future__ import annotations

from core.conversation.response_request_coverage import _instruction_coverage_reasons
from core.subject.self_prediction import ITEMS_V2
from core.subject.self_prediction_by_moment import stated_prompt


def _asked(item_id: str) -> str:
    return stated_prompt(next(i for i in ITEMS_V2 if i.id == item_id))


def test_her_word_and_how_sure_she_is_answers_the_question():
    for item_id, reply in (("rain_snow", "Rain. 80"), ("rain_snow", "Rain — 73"), ("piano_violin", "Piano, 60")):
        assert "missing_requested_objective_facets" not in _instruction_coverage_reasons(_asked(item_id), reply)


def test_a_reply_that_leaves_out_what_was_dictated_is_still_incomplete():
    # The number she was asked for is missing, so this is not the closed answer the person asked for.
    assert "missing_requested_objective_facets" in _instruction_coverage_reasons(_asked("rain_snow"), "Rain")
