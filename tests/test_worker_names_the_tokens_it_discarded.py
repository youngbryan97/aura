"""A generation that yields no text says which tokens it produced.

LIVE, 2026-08-20. "Generation produced 1 token(s) but no text survived to the
caller" was logged on the reply lane of a turn whose tool had already
succeeded. A count alone cannot separate a decoder fault from a model that
ended its turn immediately, and those need opposite fixes.
"""

from __future__ import annotations

from pathlib import Path

from fixtures.source_loading import load_function

WORKER = Path(__file__).resolve().parents[1] / "core" / "brain" / "llm" / "mlx_worker.py"


def _load_helper():
    """The worker's helper, without importing the worker.

    Importing mlx_worker pulls in MLX and a Metal device. The shared loader in
    tests/fixtures/source_loading.py compiles the one function out of the
    file; it is the only reviewed exec in the test tree.
    """
    return load_function(WORKER, "_name_tokens")


class _Tokenizer:
    def decode(self, ids):
        return {1: "hello", 2: "", 3: " world"}.get(ids[0], "")

    def convert_ids_to_tokens(self, token_id):
        return {2: "<|im_end|>"}.get(token_id, f"<unk{token_id}>")


def test_a_special_token_is_named_not_blank() -> None:
    assert _load_helper()(_Tokenizer(), [2]) == "<|im_end|>"


def test_ordinary_tokens_show_their_text() -> None:
    assert _load_helper()(_Tokenizer(), [1, 2, 3]) == "'hello', <|im_end|>, ' world'"


def test_nothing_and_nonsense_are_distinguishable() -> None:
    name_tokens = _load_helper()
    assert name_tokens(_Tokenizer(), []) == "none"
    assert name_tokens(_Tokenizer(), ["not an id"]) == "unreadable"


def test_a_tokenizer_that_raises_does_not_break_the_log() -> None:
    class _Broken:
        def decode(self, ids):
            raise RuntimeError("no vocabulary")

        def convert_ids_to_tokens(self, token_id):
            raise RuntimeError("no vocabulary")

    assert _load_helper()(_Broken(), [7]) == "id:7"


def test_the_warning_reports_the_generated_tail_not_the_prompt() -> None:
    """`tokens` is the encoded prompt with generated ids appended.

    The first version of this diagnostic printed tokens[:8] and reported
    "'<|im_start|>', 'system', '#', ' Tools'" — the start of the prompt — for
    a generation that had produced one token. A diagnostic that names the
    wrong thing is worse than the count it replaced.
    """
    source = WORKER.read_text(encoding="utf-8")
    assert "_name_tokens(tokenizer, tokens[-token_count:][:8])" in source
    assert "_name_tokens(tokenizer, tokens[:8])" not in source
