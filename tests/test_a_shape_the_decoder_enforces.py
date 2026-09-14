"""JSON out of the decoder because the decoder cannot produce anything else.

"Return ONLY JSON" is a request, and every parser on the other side of one
has a fallback for when it was not honoured. The shape is held by a logits
processor over the real tokenizer's vocabulary: prose before the value is not
available to sample, a string cannot go unclosed, a brace cannot go
unbalanced, and after the value the only token left is the end of the turn.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.brain.llm.a_shape_the_decoder_enforces import (
    JsonState,
    allowed_token_ids,
    enforce_json,
    feed,
    is_complete,
)

_MODEL = Path(
    os.environ.get(
        "AURA_TEST_TOKENIZER",
        "~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15",
    )
).expanduser()


class TestTheGrammar:
    @pytest.mark.parametrize(
        "text",
        ['{"a": 1}', '[1, 2.5e3, "x", true, null, {"k": [false]}]', '  {"s": "a\\"b"}',
         '-0.5', '"str"', 'true', '{"nested": {"deep": [[]]}}'],
    )
    def test_valid_json_is_accepted_and_complete(self, text: str) -> None:
        state = feed(JsonState(), text)
        assert state is not None and is_complete(state), text

    @pytest.mark.parametrize(
        "text",
        ['{"a": 1', '{a: 1}', '[1,]', '01', 'tru', '{"a":1}}', 'yes {"a":1}',
         '{"a": "unterminated', '{"a":1} ', "{'a': 1}"],
    )
    def test_anything_else_is_refused_or_unfinished(self, text: str) -> None:
        state = feed(JsonState(), text)
        assert state is None or not is_complete(state), text

    def test_a_caller_may_require_an_object_or_an_array(self) -> None:
        assert feed(JsonState(literal="object"), '"x"') is None
        assert feed(JsonState(literal="object"), ' {"a":1}') is not None
        assert feed(JsonState(literal="array"), '{"a":1}') is None
        assert feed(JsonState(literal="array"), '[1]') is not None


@pytest.mark.skipif(not _MODEL.is_dir(), reason="the resident tokenizer is not on this machine")
class TestAgainstTheRealVocabulary:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(_MODEL))

    def test_no_prose_token_is_available_at_the_start(self, tokenizer) -> None:
        from core.brain.llm.a_shape_the_decoder_enforces import _vocabulary_for

        vocabulary = _vocabulary_for(tokenizer)
        allowed = set(allowed_token_ids(vocabulary, JsonState(literal="object")))
        for word in ("Sure", "Here", " the", "```", "Answer"):
            for token_id in tokenizer.encode(word, add_special_tokens=False):
                assert token_id not in allowed, f"{word!r} could start a JSON object"
        assert tokenizer.encode("{", add_special_tokens=False)[0] in allowed

    def test_a_prose_loving_model_is_held_to_an_object_and_ended(self, tokenizer) -> None:
        import mlx.core as mx

        held = enforce_json(tokenizer, require="object")
        assert held is not None
        prose = tokenizer.encode("Sure! Here is the JSON:", add_special_tokens=False)
        wanted = prose + tokenizer.encode('{"answer": 42, "ok": true}', add_special_tokens=False)
        # A model that wants its prose, and when refused falls back the way a
        # trained one does: close the string, close the object, end the turn.
        # The grammar constrains what can be produced; finishing is the
        # model's, and this is the least a model that knows JSON would do.
        quote = tokenizer.encode('"', add_special_tokens=False)[0]
        close = tokenizer.encode("}", add_special_tokens=False)[0]
        tokens: list[int] = []
        vocab = len(tokenizer)
        for step_i in range(80):
            logits = mx.zeros((vocab,), dtype=mx.float32)
            if step_i < len(wanted):
                logits[wanted[step_i]] = 10.0
            logits[quote] = 3.0
            logits[close] = 2.0
            logits[tokenizer.eos_token_id] = 1.0
            logits = held(mx.array(tokens, dtype=mx.int32), logits)
            chosen = int(mx.argmax(logits).item())
            tokens.append(chosen)
            if chosen == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        import json

        parsed = json.loads(text)
        assert isinstance(parsed, dict), text
        assert not text.startswith("Sure"), "prose before the value was available to sample"
        assert tokens[-1] == tokenizer.eos_token_id, "the value ended and nothing else was allowed"
        assert held.state["refused"] == 0

    def test_the_shape_waits_for_the_private_channel_to_close(self, tokenizer) -> None:
        import mlx.core as mx

        closing = tokenizer.convert_tokens_to_ids("</think>")
        held = enforce_json(tokenizer, after_token=closing, require="object")
        assert held is not None
        thinking = tokenizer.encode("Let me think about this.", add_special_tokens=False)
        vocab = len(tokenizer)
        # While thinking, prose is untouched.
        logits = mx.zeros((vocab,), dtype=mx.float32)
        logits[thinking[0]] = 10.0
        out = held(mx.array([], dtype=mx.int32), logits)
        assert int(mx.argmax(out).item()) == thinking[0]
        # After the channel closes, only JSON.
        logits = mx.zeros((vocab,), dtype=mx.float32)
        logits[thinking[0]] = 10.0
        out = held(mx.array(thinking + [closing], dtype=mx.int32), logits)
        assert int(mx.argmax(out).item()) != thinking[0]
