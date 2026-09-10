"""A degradation with no recovery path is not a signal, it is a fuse.

`identity.stability` had exactly one writer anywhere in the tree — the loop
detector in memory consolidation — and it only ever subtracted. One repeated
sentence early in a session dropped it by three tenths, and nothing raised it
again for the life of the state. The phi estimate, the executive closure, the
causal self-state and the workspace's self bid all read it from wherever it had
fallen to.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.state.aura_state import AuraState

REPO = Path(__file__).resolve().parents[1]


def _phase():
    from core.container import ServiceContainer
    from core.phases.memory_consolidation import MemoryConsolidationPhase

    return MemoryConsolidationPhase(ServiceContainer)


def _said(state: AuraState, text: str) -> None:
    state.cognition.working_memory.append({"role": "user", "content": "and then?"})
    state.cognition.working_memory.append({"role": "assistant", "content": text})


def test_the_only_writer_used_to_only_subtract() -> None:
    source = (REPO / "core" / "phases" / "memory_consolidation.py").read_text()
    assert "_LOOP_STABILITY_STEP * (1.0 - current)" in source, "the recovery is gone again"


def test_repeating_herself_costs_stability() -> None:
    phase = _phase()
    state = AuraState.default()
    line = "the same sentence, at more than twenty characters long"
    _said(state, line)
    _said(state, line)
    out = asyncio.run(phase.execute(state))
    assert out.identity.stability < 1.0


def test_and_it_comes_back_when_she_stops() -> None:
    phase = _phase()
    state = AuraState.default()
    state.identity.stability = 0.1
    _said(state, "one thing she said")
    _said(state, "a different thing entirely, also long enough")
    out = asyncio.run(phase.execute(state))
    assert out.identity.stability > 0.1

    for _ in range(6):
        _said(out, f"another distinct sentence number {_} of some length")
        out = asyncio.run(phase.execute(out))
    assert out.identity.stability > 0.9


def test_a_genuine_loop_holds_it_down() -> None:
    phase = _phase()
    state = AuraState.default()
    line = "she keeps saying exactly this, over and over again"
    for _ in range(5):
        _said(state, line)
        state = asyncio.run(phase.execute(state))
    assert state.identity.stability == pytest.approx(0.1)
