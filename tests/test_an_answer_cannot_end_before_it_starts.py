"""A generation may not open with a control token.

LIVE, 2026-08-20. A conversational turn whose tool had already fetched the
right document produced exactly one token — ``<|im_start|>`` — which is a stop
sequence, so the decode halted with no text at all. The runtime logged
"produced 1 token(s) but no text survived", retried once, and served "I
couldn't get to an answer I'd stand behind on that one" while the answer sat
in working memory.

The guard for this already existed, fitted only to strict contracts.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "core" / "brain" / "llm" / "mlx_worker.py"


def _suppression_ids():
    """Load the id builder without importing the worker's mlx dependencies."""
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_first_token_suppression_ids"
    )
    namespace: dict[str, object] = {"Any": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<worker>", "exec"), namespace)
    return namespace["_first_token_suppression_ids"]


class _Tokenizer:
    eos_token_id = 151645
    pad_token_id = 151643
    _vocab = {
        "<|endoftext|>": 151643,
        "<|im_end|>": 151645,
        "<|im_start|>": 151644,
    }

    def encode(self, text, add_special_tokens=False):
        return [self._vocab[text]] if text in self._vocab else [1, 2]


def test_the_token_that_broke_the_live_turn_is_banned() -> None:
    banned = _suppression_ids()(_Tokenizer())
    assert 151644 in banned, "<|im_start|> must not be able to open an answer"
    assert 151645 in banned
    assert 151643 in banned


def test_ordinary_tokens_are_untouched() -> None:
    banned = _suppression_ids()(_Tokenizer())
    assert 1 not in banned and 2 not in banned


def test_the_guard_is_not_limited_to_strict_contracts() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "nonempty_start_processor" in source
    assert "elif not _expected_empty_warmup_precompile(job):" in source


def test_a_warmup_precompile_may_still_produce_nothing() -> None:
    """Warmup deliberately generates one token and expects no visible text."""
    source = WORKER.read_text(encoding="utf-8")
    guard = source[source.index("elif not _expected_empty_warmup_precompile(job):") :]
    guard = guard[: guard.index("# Foreground non-parametric memory")]
    assert "logits_processors.append(nonempty_start_processor)" in guard


def test_the_guard_only_constrains_the_first_position() -> None:
    """After one real token the model must be free to end its turn."""
    source = WORKER.read_text(encoding="utf-8")
    guard = source[source.index("def nonempty_start_processor(") :]
    guard = guard[: guard.index("logits_processors.append(nonempty_start_processor)")]
    assert "if len(tokens) == 0:" in guard
