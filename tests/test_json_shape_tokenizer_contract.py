"""Grammar masks use declared token identities across runtime wrappers."""

import pytest

from core.brain.llm.a_shape_the_decoder_enforces import JsonState, _Vocabulary, allowed_token_ids


def test_real_mlx_wrapper_includes_added_tokens_and_all_declared_eos():
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    base = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "{": 2, "}": 3}, unk_token="<unk>")),
        unk_token="<unk>", eos_token="<eos>",
    )
    base.add_tokens(["[]"])
    base.add_special_tokens({"additional_special_tokens": ["<second-end>"]})
    added = base.convert_tokens_to_ids("[]")
    second = base.convert_tokens_to_ids("<second-end>")
    wrapped = TokenizerWrapper(base, eos_token_ids=[1, second])
    with pytest.raises(TypeError):
        len(wrapped)
    vocabulary = _Vocabulary(wrapped)
    assert vocabulary.size == len(base)
    assert added in allowed_token_ids(vocabulary, JsonState())
    assert vocabulary.ends == {1, second}
    assert allowed_token_ids(vocabulary, JsonState(mode="done")) == [1, second]


@pytest.mark.parametrize("ends", [7, [7], {7}, None])
def test_sparse_token_ids_preserve_width_without_inventing_tokens(ends):
    class Sparse:
        eos_token_id = 7
        eos_token_ids = ends
        all_special_ids = [7]
        def get_vocab(self):
            return {"[]": 2, "end": 7}
        def decode(self, ids):
            return "[]"  # Some decoders substitute text for undeclared ids.
        def convert_tokens_to_ids(self, name):
            return 0  # An unknown marker must not become an end token.
    vocabulary = _Vocabulary(Sparse())
    assert vocabulary.size == 8
    assert allowed_token_ids(vocabulary, JsonState()) == [2]
    assert allowed_token_ids(vocabulary, JsonState(mode="done")) == [7]


@pytest.mark.parametrize("vocab", [{}, {"negative": -1}, {"fraction": 1.5}, {"bool": True}])
def test_invalid_declared_vocabulary_is_explicit(vocab):
    class Invalid:
        def get_vocab(self):
            return vocab
    with pytest.raises(ValueError, match="vocabulary"):
        _Vocabulary(Invalid())
