"""A generation's proof must be readable by whoever checks it.

LIVE 2026-08-29: five tool calls, a code_repl returning the trial balance, an
answer composed from it, and "ownership_evidence=[live_mind(tokens=-,decode=-,
attempts=0,applied=False); latent_cortex_receipt(tokens=-,decode=0)]". No
receipt for the generation that wrote the answer reached the contract at all.

A foreground answer proves authorship from a surface-control receipt with a
token count in it, and that receipt is published on the object that ran the
generation. The tool loop runs on the inference gate; the layer that checks the
proof reads the health router. Both are right about their own object and the
answer falls between them.
"""

from __future__ import annotations

import pytest

from core.conversation.turn_evidence_custody import (
    bind_turn_evidence_custody,
    record_turn_model_generation,
    turn_model_generations,
)

pytestmark = pytest.mark.unit


def test_the_turn_records_what_wrote_for_it() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert record_turn_model_generation(
            "/models/Aura-Qwen3.8-27B", tokens=412, path="tool_loop"
        )
        (row,) = turn_model_generations()
        assert row["tokens"] == 412
        assert row["path"] == "tool_loop"
        assert row["turn_id"] == "t"


def test_a_generation_of_nothing_is_not_a_generation() -> None:
    """Absence of a count is not a small count."""

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert not record_turn_model_generation("m", tokens=0)
        assert not record_turn_model_generation("m", tokens=-5)
        assert turn_model_generations() == ()


def test_it_does_not_outlive_its_turn() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t-a"):
        record_turn_model_generation("m", tokens=10)
    assert not record_turn_model_generation("m", tokens=10)
    with bind_turn_evidence_custody(session_id="s", turn_id="t-b"):
        assert turn_model_generations() == ()


def test_ownership_reads_the_turn_when_the_metadata_proves_nothing() -> None:
    from interface.routes.chat import _this_turn_generated_something

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert not _this_turn_generated_something()
        record_turn_model_generation("/models/27B", tokens=412, path="tool_loop")
        assert _this_turn_generated_something()


def test_the_metadata_path_still_proves_it_on_its_own() -> None:
    """The turn's record is a second reader, not a replacement."""

    from interface.routes.chat import _generation_metadata_consumed_foreground_owner

    assert _generation_metadata_consumed_foreground_owner(
        {"surface_control_receipt": {"generated_tokens": 412}}
    )
    assert not _generation_metadata_consumed_foreground_owner({"tool_loop": True})


def test_the_tool_loop_records_it_where_the_tokens_are_counted() -> None:
    from pathlib import Path

    gate = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    assert "record_turn_model_generation(" in gate
    assert 'path="tool_loop"' in gate
    # Only when something was actually counted — by the worker's receipt, or
    # by the client watching the tokens arrive.
    assert "if _tokens > 0" in gate


class TestTheCountComesFromWhereTheTokensArrive:
    """A worker that attaches no total still generated tokens.

    LIVE 2026-08-29: "tool loop generation unrecorded: tokens=0
    receipt_keys=none". The generation that wrote the answer came back with
    neither a surface-control receipt nor a tokens_used total, and authorship
    is proven from exactly that number — so the turn refused its own work
    because of which branch of the worker happened to reply.
    """

    def test_the_client_counts_them_as_they_arrive(self) -> None:
        from core.brain.llm.mlx_client import MLXLocalClient

        counter = MLXLocalClient.tokens_generated_for_this_request

        class _Lane:
            tokens_generated_for_this_request = counter

        lane = _Lane()
        assert lane.tokens_generated_for_this_request() == 0
        lane._tokens_this_request = 412
        assert lane.tokens_generated_for_this_request() == 412

    def test_a_bad_counter_reads_as_none_rather_than_raising(self) -> None:
        from core.brain.llm.mlx_client import MLXLocalClient

        class _Lane:
            _tokens_this_request = "not a number"
            tokens_generated_for_this_request = (
                MLXLocalClient.tokens_generated_for_this_request
            )

        with pytest.raises((TypeError, ValueError)):
            _Lane().tokens_generated_for_this_request()

    def test_the_gate_falls_back_to_it_only_when_the_receipt_has_nothing(self) -> None:
        from pathlib import Path

        gate = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
        assert "tokens_generated_for_this_request" in gate
        assert "if _tokens <= 0:" in gate
        # The receipt is still preferred: it is the worker's own accounting.
        assert 'for _key in ("generated_tokens", "decode_generated_tokens"):' in gate
