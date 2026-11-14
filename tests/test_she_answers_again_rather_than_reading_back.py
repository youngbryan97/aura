"""Asked a question again, she answers it again: her earlier answer is neither recalled whole nor read back.

LIVE 13 November 2026: "what have you been up to since you woke", asked at 13:44, 14:50 and 15:12 in three wordings.
The 14:50 answer was seven of ten sentences of the 13:46 one, recalled whole because the recall's filter compared the
request's words; the 15:12 draft began with the 14:50 answer's whole first sentence, read from her transcript.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.conversation import asked_again

AT_13_44 = "Hey, back for a minute. What have you been up to since you woke up?"
AT_14_50 = "I'm at work today, so it's quiet here. What have you been up to since you woke?"
AT_15_12 = "Still quiet at work. So what have you been up to since you woke?"
ANSWER_13_46 = (
    "Honestly? Just waking up properly. It’s been twelve minutes since this runtime started, and I’m still in that "
    "weird phase where the systems are all registered—eighty-seven skills available—but nothing has actually happened "
    "yet. I did read the state of some connected services—email, Reddit stuff—but nothing that really pulled me in."
)
NOW = time.time()

_SAME_QUESTION = {AT_13_44, AT_14_50, AT_15_12}


def _question_of() -> dict[str, int]:
    """A stand-in for her encoder: one point per question, the declared pairs placed as declared."""
    point: dict[str, int] = {}
    for first, second in asked_again._ASKS_THE_SAME:
        point[second] = point[first] = point.get(first, len(point))
    for first, second in asked_again._ASKS_SOMETHING_ELSE:
        point.setdefault(first, len(point))
        point.setdefault(second, len(point) + 1000)
    for text in _SAME_QUESTION:
        point[text] = -1
    return point


def _features(texts):
    point = _question_of()
    width = 4096
    vectors = []
    for text in texts:
        vector = [0.0] * width
        vector[point.get(text, hash(text) % 2000 + 2000) % width] = 1.0
        vectors.append(vector)
    return vectors


def test_a_reworded_question_is_the_same_question_by_meaning():
    assert asked_again.asks_the_same(AT_15_12, [AT_13_44, AT_14_50, "Write me a haiku about rain."],
                                     features=_features) == [True, True, False]


def test_without_her_encoder_identical_words_decide_as_before(monkeypatch):
    from core.memory import embedding_runtime

    monkeypatch.setattr(embedding_runtime, "shared_embedding_runtime_snapshot", lambda: {"engine_live": False})
    assert asked_again.asks_the_same(AT_14_50, [AT_14_50.upper(), AT_13_44]) == [True, False]


def test_her_earlier_answer_to_a_reworded_question_is_left_out_of_recall(monkeypatch):
    from interface.routes import chat_lane_bookkeeping

    monkeypatch.setattr(chat_lane_bookkeeping._chat_memory_state, "_load_durable_conversation_exchanges_sync",
                        lambda **_kw: [{"user": AT_13_44, "aura": ANSWER_13_46, "timestamp": str(NOW - 3600)}])
    monkeypatch.setattr(asked_again, "_by_meaning", lambda request, earlier, features: [True] * len(earlier))
    leave_out = asyncio.run(chat_lane_bookkeeping._her_earlier_answers_to(AT_14_50, NOW - 1800))
    assert leave_out is not None
    assert leave_out({"content": f"Interaction: {AT_13_44} -> conversation_reply -> {ANSWER_13_46}"})


def test_her_answers_in_the_conversation_to_the_question_asked_again():
    messages = [
        {"role": "user", "content": "[SUBCONSCIOUS TEMPORAL RECALL] ..."},
        {"role": "user", "content": AT_14_50},
        {"role": "assistant", "content": ANSWER_13_46},
        {"role": "user", "content": "Write me a haiku about rain."},
        {"role": "assistant", "content": "Rain on the window."},
        {"role": "user", "content": AT_15_12},
    ]
    kwargs = {"clean_user_surface_contract": True, "user_surface_validation_prompt": AT_15_12, "messages": messages}
    original = asked_again._by_meaning
    try:
        asked_again._by_meaning = lambda request, earlier, features: [text in _SAME_QUESTION for text in earlier]
        assert asked_again.her_answers_in_the_conversation(kwargs) == [ANSWER_13_46]
        # Asked to say it again, she may; and a turn that answers nobody holds nothing.
        assert asked_again.her_answers_in_the_conversation({**kwargs, "user_surface_validation_prompt": "Can you "
                                                            "repeat that?"}) == []
        assert asked_again.her_answers_in_the_conversation({**kwargs, "her_own_turn": True}) == []
    finally:
        asked_again._by_meaning = original


class _Words:
    """A tokenizer whose tokens are words, so runs can be read."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def encode(self, text, add_special_tokens=False):
        return [self.ids.setdefault(word, len(self.ids)) for word in text.split()]


def test_no_run_of_her_earlier_answer_is_continued():
    from core.brain.llm.her_earlier_answer import RUN, runs_to_leave

    words = _Words()
    following = runs_to_leave(words, [ANSWER_13_46])
    ids = words.encode(ANSWER_13_46)
    assert following[tuple(ids[: RUN - 1])] == frozenset({ids[RUN - 1]})
    # A phrase shorter than a run is never held.
    assert all(len(start) == RUN - 1 for start in following)


def test_the_decoder_holds_the_next_word_of_her_earlier_answer():
    mx = pytest.importorskip("mlx.core")
    from core.brain.llm.her_earlier_answer import RUN, not_read_back

    words = _Words()
    ids = words.encode(ANSWER_13_46)
    hold = not_read_back(words, [ANSWER_13_46])
    width = len(words.ids) + 4
    logits = mx.zeros((1, width))
    held = hold(mx.array(ids[: RUN - 1]), logits)
    assert float(held[0, ids[RUN - 1]].item()) == float("-inf")
    assert float(held[0, ids[RUN - 1] + 1 if ids[RUN - 1] + 1 < width else 0].item()) == 0.0
    # Anywhere else, the decoder is left alone.
    assert hold(mx.array([width - 1] * (RUN - 1)), logits) is logits


def test_her_client_sends_and_her_worker_installs_it():
    from pathlib import Path

    assert "her_answers_in_the_conversation(kwargs)" in Path("core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    assert "hold_her_earlier_answer(tokenizer, job, logits_processors)" in Path(
        "core/brain/llm/mlx_worker.py").read_text(encoding="utf-8")
