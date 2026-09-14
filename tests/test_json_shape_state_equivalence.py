"""The decoder's cached grammar must agree with complete-token parsing."""

import json
from itertools import product

import pytest

from core.brain.llm.a_shape_the_decoder_enforces import JsonState, feed, is_complete


@pytest.mark.parametrize("prefix", ['"', '{"'])
@pytest.mark.parametrize("escape", [r"\uZZZZ", r"\u123", r"\u12x4", r"\u", r"\u000"])
def test_unicode_escapes_require_four_hex_digits(prefix, escape):
    text = prefix + escape + ('":0}' if prefix == '{"' else '"')
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    parsed = feed(JsonState(), text)
    assert parsed is None or not is_complete(parsed)


@pytest.mark.parametrize("escape", [r"\u1234", r"\u0000", r"\uD834\uDD1E", r"\uabcd"])
def test_unicode_escapes_work_for_keys_values_and_token_boundaries(escape):
    text = '{"' + escape + '":"' + escape + '"}'
    expected = json.loads(text)
    for split in range(len(text) + 1):
        state = feed(JsonState(), text[:split])
        assert state is not None
        state = feed(state, text[split:])
        assert state is not None and is_complete(state)
    assert isinstance(expected, dict)


@pytest.mark.parametrize("left,right,continuation", [
    ("[[", "[", "]]"),
    ('{"a":[', '[[', ']}'),
    ("[1e", "[1e+", "+1]"),
    ('"\\u0', '"\\u00', '00"'),
])
def test_states_with_different_token_languages_cannot_share_a_mask(left, right, continuation):
    a, b = feed(JsonState(), left), feed(JsonState(), right)
    assert a is not None and b is not None
    assert (feed(a, continuation) is None) != (feed(b, continuation) is None)
    assert a.cache_key() != b.cache_key()


def test_short_numeric_language_matches_the_standard_library():
    for length in range(1, 6):
        for chars in product("01.e+-", repeat=length):
            text = "".join(chars)
            try:
                json.loads(text)
                accepted = True
            except json.JSONDecodeError:
                accepted = False
            state = feed(JsonState(), text)
            assert (state is not None and is_complete(state)) == accepted, text


def test_complete_token_masks_remain_sound_after_cache_reuse():
    mx = pytest.importorskip("mlx.core")
    from core.brain.llm.a_shape_the_decoder_enforces import enforce_json

    class Tokenizer:
        texts = ["[", "]]", "]", ",", "1", "1e", "+", "1e+", "[1e", "[1e+", ""]
        eos_token_id = 10
        all_special_ids = [10]

        def __len__(self):
            return len(self.texts)

        def decode(self, ids):
            return "".join(self.texts[i] for i in ids)

        def convert_tokens_to_ids(self, name):
            return None

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        tokenizer = Tokenizer()
        held = enforce_json(tokenizer, require="array")
        tokens = []
        # The same generation visits both one and two open-array states.
        for token in (0, 0, 2, 3, 8, 6, 4, 2, 2):
            out = held(mx.array(tokens, dtype=mx.int32), mx.zeros((len(tokenizer),)))
            state = held.state["json"]
            for candidate, text in enumerate(tokenizer.texts[:-1]):
                assert bool(mx.isfinite(out[candidate]).item()) == (feed(state, text) is not None)
            assert bool(mx.isfinite(out[token]).item())
            tokens.append(token)
        held(mx.array(tokens, dtype=mx.int32), mx.zeros((len(tokenizer),)))
        assert held.state["refused"] == 0
        assert json.loads(tokenizer.decode(tokens)) == [[], [10.0]]
    finally:
        mx.set_default_device(previous)


class _BoundaryTokenizer:
    texts = ['prompt prose', '{', '"x"', ':', '1', '}', '[', ']', '</think>', 'thinking', '']
    eos_token_id = 10
    all_special_ids = [8, 10]

    def __len__(self):
        return len(self.texts)

    def decode(self, ids):
        return ''.join(self.texts[i] for i in ids)

    def convert_tokens_to_ids(self, name):
        return None


@pytest.fixture
def cpu_mlx():
    mx = pytest.importorskip('mlx.core')
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield mx
    finally:
        mx.set_default_device(previous)


def test_prompt_history_is_not_parsed_as_the_current_answer(cpu_mlx):
    from core.brain.llm.a_shape_the_decoder_enforces import enforce_json

    mx = cpu_mlx
    tokenizer = _BoundaryTokenizer()
    held = enforce_json(tokenizer, require='object')
    logits = mx.zeros((len(tokenizer),))
    masked = held(mx.array([0]), logits)
    assert bool(mx.isfinite(masked[1]).item())
    assert not bool(mx.isfinite(masked[0]).item())
    assert held.state['refused'] == 0


def test_an_old_private_close_in_the_prompt_does_not_start_the_current_shape(cpu_mlx):
    from core.brain.llm.a_shape_the_decoder_enforces import enforce_json

    mx = cpu_mlx
    tokenizer = _BoundaryTokenizer()
    held = enforce_json(tokenizer, after_token=8, require='object')
    logits = mx.zeros((len(tokenizer),))
    prompt = [0, 8, 0]
    assert bool(mx.isfinite(held(mx.array(prompt), logits)[9]).item())
    assert bool(mx.isfinite(held(mx.array(prompt + [9]), logits)[9]).item())
    masked = held(mx.array(prompt + [9, 8]), logits)
    assert not bool(mx.isfinite(masked[9]).item())
    assert bool(mx.isfinite(masked[1]).item())
    assert held.state['refused'] == 0


def test_speculative_rewind_and_same_length_replacement_replay_the_real_prefix(cpu_mlx):
    from core.brain.llm.a_shape_the_decoder_enforces import enforce_json

    mx = cpu_mlx
    tokenizer = _BoundaryTokenizer()
    held = enforce_json(tokenizer)
    logits = mx.zeros((len(tokenizer),))
    held(mx.array([0]), logits)
    held(mx.array([0, 6, 7]), logits)  # draft proposes []
    masked = held(mx.array([0, 1, 2]), logits)  # verifier replaces it by {"x"
    assert bool(mx.isfinite(masked[3]).item())
    assert not bool(mx.isfinite(masked[10]).item())
    masked = held(mx.array([0, 1]), logits)  # verifier rewinds further
    assert bool(mx.isfinite(masked[2]).item())
    assert held.state['refused'] == 0


def test_installed_mlx_generate_step_keeps_the_shape_after_prefill(cpu_mlx):
    from mlx_lm.generate import generate_step

    from core.brain.llm.a_shape_the_decoder_enforces import enforce_json

    mx = cpu_mlx
    tokenizer = _BoundaryTokenizer()

    class Model:
        layers = []

        def __call__(self, tokens, cache=None):
            # Prefer prose, then an empty object when the grammar forbids prose.
            logits = mx.zeros((*tokens.shape, len(tokenizer)))
            logits[..., 0] = 10
            logits[..., 5] = 4
            logits[..., 1] = 3
            logits[..., 10] = 2
            return logits

    held = enforce_json(tokenizer, require='object')
    actual = []
    for token, _ in generate_step(mx.array([0, 0, 0]), Model(), max_tokens=4,
                                 prompt_cache=[], logits_processors=[held]):
        actual.append(int(token))
        if actual[-1] == tokenizer.eos_token_id:
            break
    assert actual == [1, 5, 10]
    assert json.loads(tokenizer.decode(actual[:-1])) == {}
    assert held.state['refused'] == 0


def test_vocabulary_cache_retains_identity_and_is_bounded(monkeypatch):
    from collections import OrderedDict

    import core.brain.llm.a_shape_the_decoder_enforces as shape

    monkeypatch.setattr(shape, '_VOCABULARIES', OrderedDict())
    owners = [_BoundaryTokenizer() for _ in range(3)]
    first = shape._vocabulary_for(owners[0])
    assert shape._VOCABULARIES[id(owners[0])][0] is owners[0]
    assert shape._vocabulary_for(owners[0]) is first
    for owner in owners[1:]:
        shape._vocabulary_for(owner)
    assert len(shape._VOCABULARIES) == 2
    assert id(owners[0]) not in shape._VOCABULARIES
    assert shape._vocabulary_for(owners[0]).texts == first.texts


def test_mask_eviction_changes_storage_not_the_grammar(cpu_mlx, monkeypatch):
    import core.brain.llm.a_shape_the_decoder_enforces as shape

    mx = cpu_mlx
    tokenizer = _BoundaryTokenizer()
    monkeypatch.setattr(shape, '_MASK_CACHE_BYTES', len(tokenizer) * 4)
    held = shape.enforce_json(tokenizer, require='object')
    tokens = [0]
    for token in (1, 2, 3, 4, 5, 10):
        out = held(mx.array(tokens), mx.zeros((len(tokenizer),)))
        assert bool(mx.isfinite(out[token]).item())
        assert len(held.masks) == 1
        tokens.append(token)
    assert held.state['refused'] == 0
