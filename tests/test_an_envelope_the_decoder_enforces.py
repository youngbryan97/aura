"""An answer inside a tag, made structural instead of requested.

The strict-answer recovery asked — "Put your final answer strictly inside
<answer>...</answer> tags. Do not include any conversational preamble" — and
then checked whether the envelope had arrived, failing the turn when it had
not. A model asked for a delimiter produces one most of the time, and most of
the time is what becomes a retry and then a person told the runtime could not
get to an answer.

Both mechanisms were already in the tree: the assistant turn is prefilled with
the opening marker, the way `_build_operator_evidence_prompt` prefills its
own, and `stop_sequences` ends the generation at the closing one.
"""

from __future__ import annotations

import inspect

from core.brain.llm.an_envelope_the_decoder_enforces import (
    AN_ANSWER,
    AnEnvelope,
    the_request_for,
)


def test_the_model_starts_inside_the_envelope():
    request = the_request_for(AN_ANSWER, [{"role": "user", "content": "2+2?"}])
    assert request["messages"][-1] == {"role": "assistant", "content": "<answer>"}
    assert request["stop_sequences"] == ["</answer>"]
    # The caller's own turns survive.
    assert request["messages"][0] == {"role": "user", "content": "2+2?"}


def test_what_comes_back_is_the_contents_and_gets_its_envelope():
    assert AN_ANSWER.around("42") == "<answer>42</answer>"
    assert AN_ANSWER.holds(AN_ANSWER.around("42"))


def test_a_model_that_wrote_its_own_markers_does_not_get_two():
    assert AN_ANSWER.around("<answer>42</answer>") == "<answer>42</answer>"
    assert AN_ANSWER.around("<answer>42") == "<answer>42</answer>"
    assert AN_ANSWER.around("42</answer> and some trailing noise") == "<answer>42</answer>"


def test_nothing_generated_is_not_an_empty_envelope():
    """An envelope around nothing would pass the caller's check and carry no
    answer, which is the failure this replaces wearing a different hat."""
    assert AN_ANSWER.around("") == ""
    assert AN_ANSWER.around("   ") == ""
    assert AN_ANSWER.around(None) == ""
    assert AN_ANSWER.around("<answer></answer>") == ""


def test_holds_wants_both_ends():
    assert AN_ANSWER.holds("<answer>42</answer>") is True
    assert AN_ANSWER.holds("<answer>42") is False
    assert AN_ANSWER.holds("42</answer>") is False
    assert AN_ANSWER.holds("") is False


def test_it_is_not_specific_to_one_marker():
    other = AnEnvelope(opens="[[", closes="]]")
    assert other.around("x") == "[[x]]"
    assert other.stop_sequences == ("]]",)


def test_the_recovery_no_longer_asks_for_the_tags():
    from core.brain import cognitive_engine

    source = inspect.getsource(cognitive_engine)
    # The code, not the note recording what it used to ask for.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "Put your final answer strictly inside" not in code
    assert "Do not include any conversational preamble" not in code
    at = source.index("Last-resort direct structured recovery succeeded.")
    window = source[max(0, at - 3000) : at]
    assert "the_request_for(" in window
    assert "AN_ANSWER.around(content)" in window
    assert "AN_ANSWER.holds(cleaned)" in window
