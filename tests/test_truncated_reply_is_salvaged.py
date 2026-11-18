"""An answer cut off at the token budget is trimmed, not thrown away.

Live on the desktop surface 2026-07-26, in this order:

    ✅ Cortex response received (len=366)
    FAULT ... RuntimeError: desktop_quick_reply_midsentence_cutoff
    CognitiveEngine desktop chat reply failed reliability gate (truncated_tail)
    Skipping CognitiveEngine desktop repair retry; the foreground model owner
        already produced work for this turn.
    → "I couldn't get to an answer I'd stand behind on that one..."

She produced 366 characters. The token budget cut the last clause. The gate
read the dangling tail as `truncated_tail`, the retry was skipped because the
turn's work was already spent, and the person was handed a refusal instead of
the answer that existed.

`_trim_midsentence_cutoff` was supposed to prevent exactly this, but it only
cut when a sentence boundary fell in the last 40% of the draft — measuring the
salvage against what was GENERATED rather than what would SURVIVE. A reply that
answered early and then ran long kept its dangling clause and was discarded
whole.
"""
from __future__ import annotations

import pytest

from core.brain.cognitive_engine import (
    _MIN_SALVAGEABLE_REPLY_CHARS,
    _trim_midsentence_cutoff,
)


def test_the_live_failure_shape_is_salvaged() -> None:
    """A complete answer followed by a severed clause keeps the answer."""
    text, trimmed = _trim_midsentence_cutoff(
        "17 - 8 = 9. Then 9 x 3 = 27. So the answer is 27. Weighted against"
    )
    assert trimmed is True
    assert text == "17 - 8 = 9. Then 9 x 3 = 27. So the answer is 27."


def test_an_early_answer_survives_a_long_severed_tail() -> None:
    """The old 40%-of-draft rule discarded this entire reply."""
    text, trimmed = _trim_midsentence_cutoff(
        "The answer is 27. Now let me also mention that in general when you "
        "subtract and then multiply the order matters quite a lot and"
    )
    assert trimmed is True
    assert text == "The answer is 27."


@pytest.mark.parametrize(
    "reply",
    [
        "Complete sentence here.",
        "Everything finished cleanly!",
        "Did it work?",
    ],
)
def test_finished_replies_are_left_alone(reply: str) -> None:
    text, trimmed = _trim_midsentence_cutoff(reply)
    assert trimmed is False
    assert text == reply


@pytest.mark.parametrize(
    "reply",
    [
        "No boundary at all just a long dangling fragment that never ends",
        "Hi. and then",
        "Sure. blah",
        "",
    ],
)
def test_nothing_worth_keeping_is_left_untouched(reply: str) -> None:
    """A partial answer still beats an empty one — never trim to a stub."""
    text, trimmed = _trim_midsentence_cutoff(reply)
    assert trimmed is False
    assert text == reply.rstrip()


def test_the_floor_admits_a_real_short_answer() -> None:
    """"The answer is 27." must clear the floor; "Sure." must not."""
    assert _MIN_SALVAGEABLE_REPLY_CHARS <= len("The answer is 27.")
    assert _MIN_SALVAGEABLE_REPLY_CHARS > len("Sure.")


def test_code_blocks_are_never_cut() -> None:
    reply = "Here:\n```python\nprint(27)\n```"
    text, trimmed = _trim_midsentence_cutoff(reply)
    assert trimmed is False
    assert text == reply


def test_chat_last_resort_will_not_serve_a_known_partial_answer() -> None:
    """A preserved draft is evidence for retry, not a finished user answer."""
    from interface.routes.chat import _servable_draft_or_none

    partial = (
        "Let's reason through the code. It initializes a dictionary, updates each "
        "balance, and if a balance becomes zero it is deleted from the"
    )

    assert _servable_draft_or_none(partial, "Explain what this code does") == ""


_ASKED = ("Under core/brain in your own source, find the three largest Python files by line count. For each one, tell me "
          "its line count and how many functions it defines, counting methods and async functions too. Actually run "
          "something to get the numbers, then tell me which of the three has the most functions.")
_THE_TABLE = """Here are the results from actually running an AST walk over every `.py` file under `core/brain`:

| # | File | Lines | Functions |
|---|------|-------|-----------|
| 1 | `core/brain/llm/mlx_client.py` | 12,589 | 324 |
| 2 | `core/brain/inference_gate.py` | 11,227 | 199 |
| 3 | `core/brain/llm/latent_cortex/engine.py` | 10,397 | 133 |

`core/brain/llm/mlx_client.py` has the most functions of the three, with 324."""
_HER_WORDS = {"engine_authored_the_text": True, "final_requested_output_contract_proven": True}


def test_her_right_table_stands_at_the_last_exit():
    """LIVE 18 November 2026: she ran the code and wrote this table, and it was held back for cue words it did not use."""
    from interface.routes.chat_lane_bookkeeping import _only_cue_words_stand_against_her

    assert _only_cue_words_stand_against_her(_HER_WORDS, _ASKED, _THE_TABLE)


def test_words_that_are_not_hers_do_not_stand():
    from interface.routes.chat_lane_bookkeeping import _only_cue_words_stand_against_her

    assert not _only_cue_words_stand_against_her({**_HER_WORDS, "legacy_fallback_used": True}, _ASKED, _THE_TABLE)
    assert not _only_cue_words_stand_against_her({}, _ASKED, _THE_TABLE)


def test_a_voice_nobody_spoke_does_not_stand():
    from interface.routes.chat_lane_bookkeeping import _only_cue_words_stand_against_her

    heard = "The voices. The small ones. They're whispering in my ear. Telling me things."
    assert not _only_cue_words_stand_against_her(_HER_WORDS, "Hey Aura", heard)
