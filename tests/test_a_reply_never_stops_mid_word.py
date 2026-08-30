"""A reply cut off by the clock is not served as though it had finished.

The ladder has a fixed slice of time and the model spends it, so a long answer
stops wherever the clock did. LIVE 2026-08-30, a good answer about a memory
leak ended at "2. **Acc". A reader cannot tell a truncated thought from a
confused one, and the half-word says the machine broke rather than that time
ran out.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _ends_where_it_meant_to


def test_a_reply_cut_mid_word_is_trimmed_to_its_last_finished_sentence():
    said, cut = _ends_where_it_meant_to("First point is done. Second is **Acc")
    assert cut is True
    assert said == "First point is done."


def test_the_generator_is_believed_over_the_shape_of_the_text():
    """A list whose last item has no full stop is finished, and looks cut."""
    listed = "Steps:\n1. do this\n2. do that"
    assert _ends_where_it_meant_to(listed, "eos") == (listed, False)
    assert _ends_where_it_meant_to(listed, "")[1] is True


@pytest.mark.parametrize("why", ["configured_stop", "eos", "semantic_contract_satisfied"])
def test_a_reason_that_means_finished_leaves_the_reply_alone(why):
    text = "Anything at all, ending however it likes **bold"
    assert _ends_where_it_meant_to(text, why) == (text, False)


@pytest.mark.parametrize("why", ["max_tokens", "deadline_exceeded", "soft_cancelled"])
def test_a_reason_that_means_it_ran_out_is_reported(why):
    _said, cut = _ends_where_it_meant_to("One done. Two half", why)
    assert cut is True


def test_a_closed_code_fence_is_a_finished_thought():
    text = "Here is code:\n```python\nx = 1\n```"
    assert _ends_where_it_meant_to(text) == (text, False)


def test_an_open_code_fence_is_not():
    _said, cut = _ends_where_it_meant_to("Talking, then:\n```python\nx = 1")
    assert cut is True


def test_trimming_never_throws_away_most_of_the_answer():
    """Losing the bulk to tidy the end is worse than an untidy end."""
    text = "Short. " + "a long unfinished clause that never terminates " * 6
    said, cut = _ends_where_it_meant_to(text)
    assert cut is True
    assert said == text.rstrip()


def test_an_empty_reply_is_not_called_truncated():
    assert _ends_where_it_meant_to("") == ("", False)
