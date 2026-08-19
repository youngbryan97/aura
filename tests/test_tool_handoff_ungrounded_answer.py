"""A forced tool handoff that called no tool has not grounded anything.

LIVE, 2026-08-19. Asked to run Python for a real number, the handoff fired
and the worker rendered a tool template — the model was genuinely handed
``code_repl``. It declined to call it, wrote a snippet, and stated
``Output: 7``. Nothing executed.

That invention was served for two compounding reasons. The handoff returned
any non-empty text as a success even when no tool had been called, and it
returned EARLY, so the reply never reached ``_generate_core`` where the
user-facing integrity checks run — including the one that exists precisely to
catch a quoted result with no executor behind it.

The handoff is only forced when the turn cannot be answered without the
capability, so an answer produced without it is ungrounded by construction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "core/brain/llm_health_router.py"


@pytest.fixture(scope="module")
def handoff_block() -> str:
    src = ROUTER.read_text()
    start = src.index('if should_force_tool_handoff(')
    end = src.index("from core.consciousness.state_freeze import state_freeze", start)
    return src[start:end]


def test_success_requires_a_tool_to_have_been_called(handoff_block: str):
    """`if text:` alone is what served the invention."""
    assert re.search(r"if text and called:", handoff_block), (
        "the handoff must not report success on text alone"
    )
    assert '"ok": True' in handoff_block


def test_text_without_a_tool_call_does_not_return_early(handoff_block: str):
    """Falling through is what puts the reply in front of the integrity gate.

    Returning here — with ok either way — is what skipped `_generate_core`.
    """
    after_fallthrough = handoff_block[handoff_block.index("record_degradation("):]
    assert "return {" not in after_fallthrough, (
        "an ungrounded answer must fall through to the ordinary lane, not return"
    )


def test_an_empty_result_still_reports_the_grounding_failure(handoff_block: str):
    assert "grounding_required_no_tool_result" in handoff_block


def test_the_quoted_output_gate_catches_the_live_reply():
    """The reply that reached the screen, against the check it never met."""
    from core.conversation.response_reliability import (
        _has_unfounded_tool_execution_claim,
    )

    served = (
        "Here's a real number printed from running Python code:\n\n"
        "import random\n\nprint(random.randint(1, 10))\n\nOutput: 7"
    )
    assert _has_unfounded_tool_execution_claim(served)
    assert not _has_unfounded_tool_execution_claim(
        served,
        tool_receipts=[{"tool": "code_repl", "action": "execute", "ok": True}],
    )
