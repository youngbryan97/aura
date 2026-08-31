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

from pathlib import Path

import pytest
from fixtures.source_loading import load_function, load_functions

WORKER = Path(__file__).resolve().parents[1] / "core" / "brain" / "llm" / "mlx_worker.py"


def _suppression_ids():
    """Load the id builder without importing the worker's mlx dependencies."""
    return load_function(WORKER, "_first_token_suppression_ids")


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

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "unfinished continuation" if token_ids else ""


def test_the_token_that_broke_the_live_turn_is_banned() -> None:
    banned = _suppression_ids()(_Tokenizer())
    assert 151644 in banned, "<|im_start|> must not be able to open an answer"
    assert 151645 in banned
    assert 151643 in banned


def test_ordinary_tokens_are_untouched() -> None:
    banned = _suppression_ids()(_Tokenizer())
    assert 1 not in banned and 2 not in banned


def test_every_decode_path_installs_the_guard() -> None:
    """The worker assembles logits processors twice, once per decode path.

    The guard went into the generate path and the streaming path never saw
    it, so the same turn failed the same way an hour later. Both now call one
    function, and this counts the assemblies against the installs.
    """
    source = WORKER.read_text(encoding="utf-8")
    assemblies = source.count("logits_processors = []")
    installs = source.count("guard = build_nonempty_start_processor(tokenizer)")
    assert installs >= assemblies, (
        f"{assemblies} processor assemblies but only {installs} install the guard"
    )


def test_a_warmup_precompile_may_still_produce_nothing() -> None:
    """Warmup deliberately generates one token and expects no visible text."""
    source = WORKER.read_text(encoding="utf-8")
    for index, _ in enumerate(source.split("guard = build_nonempty_start_processor(tokenizer)")[:-1]):
        preceding = source.split("guard = build_nonempty_start_processor(tokenizer)")[index]
        assert "_expected_empty_warmup_precompile(job)" in preceding[-600:]


def test_the_guard_only_constrains_the_opening_positions() -> None:
    """After a real token the model must be free to end its turn."""
    source = WORKER.read_text(encoding="utf-8")
    body = source[source.index("def build_nonempty_start_processor(") :]
    body = body[: body.index("def _schema_root_openers(")]
    assert "generated_count >= limit" in body
    assert "positions: int = 1" in body
    assert "return None" in body


def _build_guard():
    """The real processor, built without importing the worker's job loop."""
    loaded = load_functions(
        WORKER, {"_first_token_suppression_ids", "build_nonempty_start_processor"}
    )
    return loaded["build_nonempty_start_processor"](_Tokenizer())


def test_the_mask_reaches_the_vocabulary_on_either_logits_shape() -> None:
    """The defect that made the guard a no-op while logging ACTIVE.

    mlx_lm hands the processor logits sometimes as (1, vocab) and sometimes as
    (vocab,). ``mask[:, id]`` on the one-dimensional case neither raises nor
    writes, so every banned id was silently skipped — installed, logged
    active, banning nothing. Both shapes are checked because only one of them
    was ever wrong.
    """
    mx = pytest.importorskip("mlx.core")
    guard = _build_guard()
    # mlx_lm includes the final prompt token on its first processor call.
    prompt_boundary = mx.array([99], dtype=mx.int32)

    two_dimensional = guard(prompt_boundary, mx.zeros((1, 152000)))
    assert float(two_dimensional[0, 151644]) == float("-inf")
    assert float(two_dimensional[0, 5]) == 0.0

    guard = _build_guard()
    one_dimensional = guard(prompt_boundary, mx.zeros((152000,)))
    assert float(one_dimensional[151644]) == float("-inf")
    assert float(one_dimensional[5]) == 0.0


def test_the_guard_lets_go_after_the_first_token() -> None:
    mx = pytest.importorskip("mlx.core")
    guard = _build_guard()
    guard(mx.array([99], dtype=mx.int32), mx.zeros((152000,)))
    after = guard(mx.array([99, 7], dtype=mx.int32), mx.zeros((152000,)))
    assert float(after[151644]) == 0.0


def test_the_guard_resets_at_the_prompt_boundary_for_a_replacement_attempt() -> None:
    mx = pytest.importorskip("mlx.core")
    guard = _build_guard()
    guard(mx.array([99], dtype=mx.int32), mx.zeros((152000,)))
    released = guard(mx.array([99, 7], dtype=mx.int32), mx.zeros((152000,)))
    assert float(released[151644]) == 0.0

    replacement = guard(mx.array([42], dtype=mx.int32), mx.zeros((152000,)))
    assert float(replacement[151644]) == float("-inf")


def _build_semantic_guard(job):
    loaded = load_functions(
        WORKER,
        {"_first_token_suppression_ids", "build_semantic_completion_terminal_guard"},
        namespace={"_semantic_surface_stop_ready": lambda *_a, **_k: False},
    )
    return loaded["build_semantic_completion_terminal_guard"](_Tokenizer(), job)


def test_simple_initial_semantic_contract_keeps_natural_eos() -> None:
    """A one-part answer must not be forced past its natural ending."""

    assert (
        _build_semantic_guard(
            {
                "clean_user_surface_contract": True,
                "semantic_completion_contract": True,
                "user_surface_validation_prompt": "What is Dijkstra's algorithm?",
            }
        )
        is None
    )


def test_multipart_initial_semantic_contract_keeps_natural_eos() -> None:
    """Coverage is observed without turning rejected EOS into repetition."""

    assert _build_semantic_guard(
        {
            "clean_user_surface_contract": True,
            "semantic_completion_contract": True,
            "user_surface_validation_prompt": (
                "Explain Dijkstra in one response. Include: (1) its invariant, "
                "(2) pseudocode, and (3) a worked example."
            ),
        }
    ) is None


def test_incomplete_append_only_continuation_keeps_natural_eos() -> None:
    assert _build_semantic_guard(
        {
            "clean_user_surface_contract": True,
            "semantic_completion_contract": True,
            "user_surface_continuation_contract": True,
        }
    ) is None


def test_explicit_semantic_terminal_hold_contract_masks_terminal_tokens() -> None:
    mx = pytest.importorskip("mlx.core")
    guard = _build_semantic_guard(
        {
            "clean_user_surface_contract": True,
            "semantic_completion_contract": True,
            "semantic_terminal_hold_contract": True,
            "user_surface_continuation_contract": True,
        }
    )
    assert guard is not None
    masked = guard(mx.array([99], dtype=mx.int32), mx.zeros((152000,)))
    assert float(masked[151644]) == float("-inf")


@pytest.mark.parametrize(
    "job",
    [
        {},
        {"clean_user_surface_contract": True},
        {"semantic_completion_contract": True},
        {
            "clean_user_surface_contract": True,
            "semantic_completion_contract": True,
            "user_surface_validation_prompt": "Give one definition.",
        },
    ],
)
def test_semantic_terminal_guard_requires_all_typed_continuation_contracts(job) -> None:
    assert _build_semantic_guard(job) is None


def test_neither_decode_loop_lets_coverage_heuristics_terminate_the_answer() -> None:
    import ast

    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "response"
    ]
    assert len(loops) >= 2
    for loop in loops:
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_semantic_surface_stop_ready"
            for node in ast.walk(loop)
        ), "A covered prefix is not evidence that the model has finished."


def test_no_mask_indexes_by_the_second_axis() -> None:
    """All three processors in the worker had the same assumption."""
    source = WORKER.read_text(encoding="utf-8")
    assert "mask[:, token_id]" not in source
