"""The prompt budget's chars-per-token ratio, and what it admits about itself.

Every budget in the assembler is a character count converted from a token
window by one number. It was `* 4`, annotated "Rough estimation" — the English
prose average, applied to prompts made of code, JSON receipts and file paths,
which run nearer two to three characters per token. A prompt built to fit could
be half again over the real window, and overflow is not the symmetric failure:
the backend drops from the head, and the head is the identity lock and the
structural constraint block.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import queue
from pathlib import Path

import pytest

from core.brain.llm import token_budget_evidence as tbe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_ratio():
    tbe.reset_for_test()
    yield
    tbe.reset_for_test()


def test_an_unobserved_ratio_says_it_is_assumed():
    ratio = tbe.chars_per_token()

    assert ratio.source is tbe.RatioSource.ASSUMED
    assert ratio.measured is False
    assert ratio.observations == 0


def test_the_assumption_is_below_the_prose_average():
    """Under-filling wastes context; over-filling deletes the constraints.
    Those costs are not comparable, so the guess leans to the survivable one."""
    assert tbe.ASSUMED_CHARS_PER_TOKEN < 4.0


def test_enough_observations_make_it_measured():
    for _ in range(tbe.MIN_OBSERVATIONS):
        assert tbe.observe_prompt_tokenization(300, 100) is True

    ratio = tbe.chars_per_token()
    assert ratio.source is tbe.RatioSource.MEASURED
    assert ratio.measured is True
    assert ratio.ratio == pytest.approx(3.0)
    assert ratio.observations == tbe.MIN_OBSERVATIONS


def test_worker_init_calibrates_with_the_already_loaded_tokenizer():
    from core.brain.llm.mlx_worker import _token_budget_calibration_evidence

    class _Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return list(range(max(1, len(text) // 3)))

    payload = _token_budget_calibration_evidence(_Tokenizer())

    assert payload["schema"] == tbe.CALIBRATION_SCHEMA
    assert len(payload["observations"]) == tbe.MIN_OBSERVATIONS
    assert not tbe.calibration_batch_errors(payload)


def test_worker_init_calibration_crosses_before_the_first_prompt():
    from core.brain.llm.mlx_client import _observe_worker_token_budget_calibration
    from core.brain.llm.mlx_worker import _token_budget_calibration_evidence

    class _Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return list(range(max(1, len(text) // 3)))

    admitted = _observe_worker_token_budget_calibration(
        {
            "status": "ok",
            "action": "init",
            "token_budget_calibration": _token_budget_calibration_evidence(
                _Tokenizer()
            ),
        }
    )

    assert admitted == tbe.MIN_OBSERVATIONS
    ratio = tbe.chars_per_token()
    assert ratio.source is tbe.RatioSource.MEASURED
    assert ratio.observations == tbe.MIN_OBSERVATIONS


def test_malformed_calibration_is_atomic_not_partly_admitted():
    payload = {
        "schema": tbe.CALIBRATION_SCHEMA,
        "observations": [
            *(
                {"chars": 300, "tokens": 100}
                for _ in range(tbe.MIN_OBSERVATIONS - 1)
            ),
            {"chars": 100_000, "tokens": 1},
        ],
    }

    assert tbe.observe_calibration_batch(payload) == 0
    assert tbe.chars_per_token().observations == 0


def test_one_prompt_is_not_a_measurement():
    tbe.observe_prompt_tokenization(300, 100)

    assert tbe.chars_per_token().source is tbe.RatioSource.ASSUMED


def test_an_impossible_ratio_is_refused_not_averaged():
    """A ratio outside the plausible band means the caller paired a string with
    a token count from a different string. Averaging that in corrupts every
    later budget, so it is discarded and recorded."""
    assert tbe.observe_prompt_tokenization(100_000, 3) is False
    assert tbe.observe_prompt_tokenization(3, 100_000) is False
    assert tbe.chars_per_token().observations == 0


def test_zero_and_negative_counts_are_refused():
    assert tbe.observe_prompt_tokenization(0, 10) is False
    assert tbe.observe_prompt_tokenization(10, 0) is False
    assert tbe.observe_prompt_tokenization(-5, 10) is False


def test_the_assembler_no_longer_multiplies_by_four():
    """A char limit derived from tokens goes through the measurement.

    This required EVERY assignment to char_limit to mention tokens_to_chars,
    which also forbids clamping one char limit against another — and that is
    what went red when a prefill ceiling, already expressed in characters,
    was applied here. Converting and clamping are different operations and
    only one of them can invent a chars-per-token constant.
    """

    source = (ROOT / "core" / "brain" / "llm" / "context_assembler.py").read_text("utf-8")
    tree = ast.parse(source)
    converted = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "char_limit" not in targets:
            continue
        if "tokens_to_chars" in ast.dump(node.value):
            converted += 1
            continue
        # Not a conversion, so it must not be one in disguise: a literal
        # multiplication is the four this test is named for.
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult):
                literal = any(
                    isinstance(side, ast.Constant) and isinstance(side.value, int | float)
                    for side in (sub.left, sub.right)
                )
                assert not literal, ast.unparse(node)

    assert converted, "nothing converts tokens to characters through the measurement"


def test_the_worker_reports_what_it_already_tokenized():
    """Nothing in the evidence module tokenizes. It is fed by the process that
    encoded the prompt anyway — the assembler must not load a tokenizer in the
    process that serves conversation."""
    worker = (ROOT / "core" / "brain" / "llm" / "mlx_worker.py").read_text("utf-8")
    module = (ROOT / "core" / "brain" / "llm" / "token_budget_evidence.py").read_text("utf-8")

    assert "observe_prompt_tokenization" in worker
    assert '"prompt_tokenization"' in worker
    assert "prompt_token_count = len(tokens)" in worker
    assert '"tokens": prompt_token_count' in worker
    assert "tokenizer" not in module.split('"""', 2)[2], "the evidence module tokenizes"


def test_terminal_worker_measurements_cross_into_the_parent_budget() -> None:
    from core.brain.llm.mlx_client import _observe_worker_prompt_tokenization

    for index in range(tbe.MIN_OBSERVATIONS):
        assert _observe_worker_prompt_tokenization(
            {
                "id": f"request-{index}",
                "status": "ok",
                "action": "generate",
                "prompt_tokenization": {"chars": 300, "tokens": 100},
            }
        )

    ratio = tbe.chars_per_token()
    assert ratio.source is tbe.RatioSource.MEASURED
    assert ratio.observations == tbe.MIN_OBSERVATIONS
    assert ratio.ratio == pytest.approx(3.0)


def test_nonterminal_or_implausible_worker_measurements_are_not_admitted() -> None:
    from core.brain.llm.mlx_client import _observe_worker_prompt_tokenization

    assert not _observe_worker_prompt_tokenization(
        {
            "id": "heartbeat",
            "status": "heartbeat",
            "action": "generate",
            "prompt_tokenization": {"chars": 300, "tokens": 100},
        }
    )
    assert not _observe_worker_prompt_tokenization(
        {
            "id": "malformed",
            "status": "ok",
            "action": "generate",
            "prompt_tokenization": {"chars": 100_000, "tokens": 3},
        }
    )
    assert tbe.chars_per_token().observations == 0


@pytest.mark.asyncio
async def test_the_parent_listener_consumes_correlated_worker_measurements() -> None:
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient(model_path="/tmp/aura-token-evidence-test-model")
    response_queue: queue.Queue[dict[str, object]] = queue.Queue()
    client._res_q = response_queue
    client._response_queue_generation = 17
    request_id = "token-evidence-listener"
    future = asyncio.get_running_loop().create_future()
    client._pending_generations[request_id] = future
    client._current_request_id = request_id
    client._current_gen_future = future

    listener = asyncio.create_task(
        client._response_listener_loop(response_queue, 17)
    )
    try:
        response_queue.put(
            {
                "id": request_id,
                "status": "ok",
                "action": "generate",
                "text": "done",
                "prompt_tokenization": {"chars": 330, "tokens": 100},
            }
        )
        result = await asyncio.wait_for(asyncio.shield(future), timeout=2.0)
    finally:
        listener.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener

    assert result["text"] == "done"
    assert tbe.chars_per_token().observations == 1
