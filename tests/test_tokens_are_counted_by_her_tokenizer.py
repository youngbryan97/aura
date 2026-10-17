"""A prompt's tokens are counted by her own tokenizer, read in this process, before a lane is chosen.

Live 17 October: a conversation of date arithmetic, 10,354 characters, was 8,421 tokens to her model, which
writes a number one digit a token. Counted at about three characters a token, it went to a lane admitting
8,192, and the worker refused the turn.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

tokenizers = pytest.importorskip("tokenizers")


@pytest.fixture
def her_folder(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["Total", ":", "+", "=", "days"] + [str(d) for d in range(10)]
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="Total"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(), pre_tokenizers.Digits(individual_digits=True)]
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    from core.brain import memory_guard
    from core.brain.llm import model_registry

    monkeypatch.setattr(model_registry, "get_active_cortex_spec", lambda **_k: SimpleNamespace(model_path=tmp_path))
    monkeypatch.setattr(memory_guard, "_COUNTING_TOKENIZERS", {})
    return tmp_path


def test_digits_are_counted_one_a_token_as_her_model_reads_them(her_folder):
    from core.brain.living_mind_context import estimate_context_tokens
    from core.brain.memory_guard import _heuristic_tokens

    sums = "Total : 245 + 366 + 365 = 976 days " * 20
    counted = estimate_context_tokens(sums)
    assert counted == 20 * 18
    assert _heuristic_tokens(sums) < counted


def test_without_her_tokenizer_the_count_is_the_heuristic(her_folder):
    from core.brain.living_mind_context import estimate_context_tokens
    from core.brain.memory_guard import _heuristic_tokens

    (her_folder / "tokenizer.json").unlink()
    text = "Total : 245 + 366 days"
    assert estimate_context_tokens(text) == _heuristic_tokens(text)
