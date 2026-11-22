"""A reply that ends on a quotation keeps its closing quote, and is not read as cut off for losing it (U07).

LIVE 22 November 2026, 00:26, typed into her window: "Aura, you freaking genius…. I think we can work with this!"
Her answer, 7,389 characters on how a system could out-strategize Stockfish, ended
`... rather than just "playing well."` and her model stopped there on its own. Her runtime's last clean-up of a reply
stripped every straight quote from both of its ends, so the closing quote went. That left 49 straight quotes, an odd
number, and her reliability gate read the answer as cut off mid-quotation (truncated_tail). Three more tries and two
smaller models later, 512 s after the message, Bryan was told "I lost the reply lane for a moment." The quotes around a
whole reply are taken off only when they wrap all of it.
"""
from __future__ import annotations

import pytest

from core.conversation.response_surface_checks import _has_truncated_tail
from core.synthesis import stabilize_user_facing_response, strip_role_artifacts
from core.utils.text_after_removal import without_wrapping_quotes

pytestmark = pytest.mark.unit

HER_ENDING = (
    "The trajectory layer feels like the most novel piece, but the opponent model is probably what makes it *feel* "
    'like "out-strategizing" rather than just "playing well."'
)


@pytest.mark.parametrize(("written", "kept"), [
    ('She said it was "fine."', 'She said it was "fine."'),
    ('"Fine," she said, and left.', '"Fine," she said, and left.'),
    ("I watched every move of the players'", "I watched every move of the players'"),
    ('"Paris"', "Paris"),
    ("'Paris'", "Paris"),
    ('  "The answer is 42."  ', "The answer is 42."),
    ('"Yes," I said. "Fine."', '"Yes," I said. "Fine."'),
])
def test_only_quotes_around_the_whole_reply_come_off(written, kept):
    assert without_wrapping_quotes(written) == kept
    assert strip_role_artifacts(written) == kept


def test_her_answer_keeps_its_closing_quote_and_is_not_read_as_cut_off():
    cleaned = stabilize_user_facing_response(HER_ENDING, "Aura, you freaking genius…. I think we can work with this!")
    assert cleaned.endswith('"playing well."'), cleaned[-40:]
    assert not _has_truncated_tail(cleaned, generation_stop_reason="eos")


def test_a_thought_of_hers_keeps_its_closing_quote():
    from core.consciousness.hot_engine import HigherOrderThoughtEngine

    engine = HigherOrderThoughtEngine.__new__(HigherOrderThoughtEngine)
    assert engine._coerce_hot_text('I keep coming back to "playing well."').endswith('"playing well."')
