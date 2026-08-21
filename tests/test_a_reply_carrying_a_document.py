"""A reply that must contain a program is not conversational prose.

LIVE, 2026-08-20. "build me a small web app… one self-contained file" was
answered with the page written into the reply, and the reply stopped
mid-attribute at `<script type=`. The 4096-token default for a user-facing
turn had been scaled to 970 by Phi control and memory pressure — a fair size
for prose, and half an HTML page.

The plan lane already has a floor for the same reason: a turn that must emit
a plan cannot be shrunk below the plan.
"""

from __future__ import annotations

import pytest

from core.brain.inference_gate import _asks_for_a_document
from core.brain.llm import mlx_client


class _Pressure:
    """Warning level, which a resident 32B produces as a steady state."""

    max_token_cap = 384


@pytest.mark.parametrize(
    "request_text",
    [
        "build me a small web app: a single HTML page that tracks my sitting time",
        "write me a python script that renames files by date",
        "create an html page with a countdown timer",
    ],
)
def test_a_request_for_a_program_is_recognised(request_text: str) -> None:
    assert _asks_for_a_document(request_text) is True


@pytest.mark.parametrize(
    "request_text", ["how are you today?", "what is 2 + 2", "open safari", ""]
)
def test_conversation_is_not(request_text: str) -> None:
    assert _asks_for_a_document(request_text) is False


def test_the_document_reply_is_not_clamped_to_prose() -> None:
    document = {"document_output_contract": True, "max_tokens": 4096}
    prose = {"max_tokens": 4096}
    mlx_client._apply_memory_pressure_generation_controls(
        document, _Pressure(), default_max_tokens=4096
    )
    mlx_client._apply_memory_pressure_generation_controls(
        prose, _Pressure(), default_max_tokens=4096
    )
    assert int(document["max_tokens"]) == 4096
    assert int(prose["max_tokens"]) < int(document["max_tokens"])


def test_phi_scaling_exempts_it() -> None:
    from pathlib import Path

    gate = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    block = gate[gate.index("phi_val < 0.8") :]
    block = block[: block.index("phi_scale =")]
    assert "not document_output_contract" in block


def test_one_notion_of_a_request_that_produces_a_file() -> None:
    """The gate asks the same question the desktop router does."""
    from pathlib import Path

    gate = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    assert "asks_to_build_software" in gate
