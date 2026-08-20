"""The tool loop is told what the turn has already read.

LIVE, 2026-08-20. The evidence step fetched the document the person named and
the tool loop received the objective alone — "system=0 chars, objective=137
chars" — so the model fetched it again from a URL it rebuilt from memory, got
a 400, and told the person about the failure before giving the answer.
"""

from __future__ import annotations

import inspect

from core.brain.llm.mlx_client import MLXLocalClient, _tool_loop_evidence_messages


def test_only_what_a_skill_produced_is_carried() -> None:
    """The conversational scaffold belongs to the reply, not to a tool call."""
    carried = _tool_loop_evidence_messages(
        [
            {"role": "system", "content": "READ https://x: 11.7", "metadata": {"type": "skill_result"}},
            {"role": "system", "content": "persona and present moment", "metadata": {"type": "identity"}},
            {"role": "user", "content": "the request"},
        ]
    )
    assert carried == [{"role": "system", "content": "READ https://x: 11.7"}]


def test_nothing_in_means_nothing_out() -> None:
    for value in (None, [], "a string", 7, [{"role": "system"}], [{"metadata": {}}]):
        assert _tool_loop_evidence_messages(value) == []


def test_a_large_document_is_bounded() -> None:
    from core.brain.llm.mlx_client import _TOOL_LOOP_EVIDENCE_CHARS

    carried = _tool_loop_evidence_messages(
        [
            {"role": "system", "content": "x" * 50_000, "metadata": {"type": "skill_result"}},
            {"role": "system", "content": "y" * 50_000, "metadata": {"type": "skill_result"}},
        ]
    )
    assert sum(len(m["content"]) for m in carried) <= _TOOL_LOOP_EVIDENCE_CHARS


def test_the_loop_accepts_evidence() -> None:
    assert "evidence" in inspect.signature(MLXLocalClient.think_and_act).parameters


def test_the_gate_hands_the_turn_messages_over() -> None:
    from pathlib import Path

    gate = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    assert "evidence=messages," in gate
    handoff = gate[gate.index("async def _tool_grounded_answer") :]
    handoff = handoff[: handoff.index("\n    async def ", 10)]
    assert "evidence=evidence," in handoff
