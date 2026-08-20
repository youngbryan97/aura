"""A generation that yields no text says which tokens it produced.

LIVE, 2026-08-20. "Generation produced 1 token(s) but no text survived to the
caller" was logged on the reply lane of a turn whose tool had already
succeeded. A count alone cannot separate a decoder fault from a model that
ended its turn immediately, and those need opposite fixes.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "core" / "brain" / "llm" / "mlx_worker.py"


def _load_helper():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_name_tokens"
    )
    namespace: dict[str, object] = {"Any": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<worker>", "exec"), namespace)
    return namespace["_name_tokens"]


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


def test_the_warning_reports_them() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "_name_tokens(tokenizer, tokens[:8])" in source
