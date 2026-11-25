"""A question is left out of a reply where neither its words nor what it means answer it, and two replies are never put together as one."""
from __future__ import annotations

import pytest

from core.conversation import left_unanswered
from core.conversation.request_coverage import unanswered_question_parts
from core.runtime.structured_input import analyze_prompt_shape

#: What each text is about, as a direction: a stand-in for her encoder, with the answers on record a little off their questions.
_ABOUT = {"enjoy": 0, "more than i expected": 0, "opening": 1, "castled": 1, "dough": 2}


def _vectors(texts):
    out = []
    for text in texts:
        row = [0.02] * 6
        low = str(text).lower()
        for key, at in _ABOUT.items():
            if key in low:
                row[at] = 1.0
        if max(row) < 1.0:
            row[4] = 0.6
            row[5] = 0.4
        out.append(row)
    return out


@pytest.fixture
def by_meaning(monkeypatch):
    monkeypatch.setattr(left_unanswered, "_vectors", _vectors)
    monkeypatch.setattr(left_unanswered, "ANSWERED", (("did you enjoy it?", "Yes, more than I expected."), ("which opening?", "I castled early.")))
    left_unanswered._BOUND.clear()
    yield
    left_unanswered._BOUND.clear()


ASKED = "Did you enjoy playing it? Which opening did you choose?"
REPLY = "Yeah, more than I expected. I castled on the queen side early."


def test_a_question_answered_in_other_words_is_not_left_out(by_meaning):
    shape = analyze_prompt_shape(ASKED)
    assert len(shape.question_segments) == 2 and shape.requires_single_reply_coverage
    assert unanswered_question_parts(REPLY, shape) == []
    assert len(unanswered_question_parts("The dough rests overnight.", shape)) == 2, "what answers neither still leaves both out"


def test_where_it_cannot_be_read_by_meaning_the_words_decide(monkeypatch):
    monkeypatch.setattr(left_unanswered, "_vectors", lambda texts: [])
    left_unanswered._BOUND.clear()
    assert left_unanswered.left_out("did you enjoy it?", REPLY) is None
    assert unanswered_question_parts(REPLY, analyze_prompt_shape(ASKED)), "the words share nothing with either question"


def test_two_whole_replies_are_not_put_together(by_meaning):
    """LIVE 24 November 2026, 23:27: he was given both, one under the other, as one."""
    from interface.routes.chat_common import _merge_obligation_completion, _UserSurfaceObligation

    part = _UserSurfaceObligation(segment_index=0, segment="Did you enjoy playing it?", numbered_label=None)
    again = "Yeah. Quietly, actually, more than I expected. What about you?"
    assert _merge_obligation_completion(REPLY, again, part) == REPLY
    short = "I castled on the queen side early."
    assert _merge_obligation_completion(short, again, part) == f"{short}\n\n{again}", "a part that was left out is still answered"


def test_a_workers_verdict_is_put_right_and_only_of_what_was_left_out(by_meaning):
    def receipt(**over):
        return {"semantic_completion_missing_part_indexes": [0], "semantic_completion_missing_part_count": 1, "semantic_completion_incomplete": True, "semantic_completion_satisfied": False,
                "semantic_completion_eos_boundary": True, "semantic_completion_quality_reasons": ["unanswered_question_part"], "surface_quality_gate_reasons": ["unanswered_question_part"], **over}

    told = receipt()
    assert left_unanswered.settle_by_meaning(told, ASKED, REPLY)
    assert not told["semantic_completion_incomplete"] and told["semantic_completion_satisfied"] and told["surface_quality_gate_reasons"] == []
    assert not left_unanswered.settle_by_meaning(receipt(semantic_completion_eos_boundary=False), ASKED, REPLY), "what was cut off stays cut off"
    assert not left_unanswered.settle_by_meaning(receipt(semantic_completion_quality_reasons=["truncated_tail"]), ASKED, REPLY)
    assert not left_unanswered.settle_by_meaning(receipt(), ASKED, "The dough rests overnight.")
    assert not left_unanswered.settle_by_meaning({"semantic_completion_incomplete": True}, ASKED, REPLY)


def test_a_worker_is_not_made_to_load_an_encoder_to_judge_by():
    """In a worker each loading took the time of the words being written: 4.3 s of one reply, 24 November 2026."""
    import inspect

    assert "vectors_if_ready" in inspect.getsource(left_unanswered._vectors)
    assert "acquire_shared_embedding_engine" not in inspect.getsource(left_unanswered)
