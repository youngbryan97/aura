"""The pass that writes the answer counted its tokens without telling anyone.

A thinking model that spends its allowance writes the visible answer in a
second pass. That loop incremented the token count and emitted no progress
frame, so the parent — which watches token progress to tell a working
generation from a wedged one — saw the whole answer as silence. At two
thousand tokens on a 27B that is minutes of it.

LIVE 2026-08-29: "Token progress stalled during generation (>40.0s)" beside
"Cortex still sending heartbeats (2.2s ago)", twice, on a question about why
turns were slow.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

WORKER = Path("core/brain/llm/mlx_worker.py")


def _the_shared_pass_loop() -> ast.For:
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        call = node.iter
        if isinstance(call, ast.Name) and call.id == "generation_passes":
            return node
    raise AssertionError("generation passes no longer share their response consumer")


def test_the_answer_pass_emits_progress() -> None:
    body = ast.unparse(_the_shared_pass_loop())
    # ast.unparse normalises string quotes, so match on the content.
    assert "_should_emit_generation_progress" in body
    assert "status" in body and "progress" in body
    assert "tokens_generated" in body and "token_count" in body
    assert "ipc_writer.put" in body


def test_it_uses_the_same_emitter_as_the_first_pass() -> None:
    """One rule for when a generation says it is alive, not two."""

    body = ast.unparse(_the_shared_pass_loop())
    assert "generation_passes.continue_with" in body
    assert "tokens.append(response.token)" in body
    assert "soft_cancel_requested" in body
    assert "sentinel.feed(response.text)" in body


def test_it_advances_the_shared_emit_clock() -> None:
    """Otherwise every token past the fourth would emit, flooding the pipe."""

    body = ast.unparse(_the_shared_pass_loop())
    assert "last_progress_emit_at" in body


def test_the_first_pass_still_emits() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert '"tokens_generated": token_count' in source
    assert "last_progress_emit_at = progress_now" in source
