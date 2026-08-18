"""The one reply that must never be reachable was the least diagnosable.

LIVE 2026-08-18. Asked to open 2048 and play it, the person got "I couldn't
get to an answer I'd stand behind on that one." The chain was:

    Cortex exhausted its worker-owned semantic quality retries
    InferenceGate refused generation: kind=lanes_exhausted
      reason=worker_semantic_quality_retries_exhausted
    CognitiveEngine desktop chat produced a failure envelope
    -> the canned apology

The model generated, a surface-quality gate rejected the text, it retried, and
every attempt was rejected. WHICH check objected was computed by
_surface_quality_failure_reasons, carried on the receipt as
surface_quality_gate_reasons — and written down nowhere. That key appears ZERO
times in a 20,000-record log full of these refusals.

The gate keeps only INTEGRITY failures: leaks, corruption, prompt artefacts,
text that is not language. So the reason is exactly what separates a model
producing garbage from a gate that is too strict, and those two want opposite
fixes. Without it, every occurrence of the banned reply said "a gate said no"
and nothing said what it objected to.
"""
from __future__ import annotations

import inspect
import re

from core.brain import inference_gate


def _exhaustion_site() -> str:
    source = inspect.getsource(inference_gate)
    start = source.index("worker_semantic_quality_retries_exhausted")
    return source[max(0, start - 2200) : start + 500]


def test_the_refusal_log_names_the_rejecting_check():
    block = _exhaustion_site()

    assert "surface_quality_gate_reasons" in block
    assert "rejected_for" in block


def test_the_refusal_detail_carries_the_reasons_for_receipts():
    """A log line is for a person; the detail is for the ledger."""
    block = _exhaustion_site()
    detail = block[block.index("detail={") :][:400]

    assert "surface_quality_gate_reasons" in detail


def test_an_empty_reason_set_is_still_reported():
    """Silence must not read as "no reason" — it reads as a missing channel."""
    block = _exhaustion_site()

    assert "no_reasons_reported" in block


def test_the_reasons_come_from_the_receipt_the_gate_writes():
    """Wiring: the gate computes them, so the refusal must read that key."""
    from core.brain.llm import mlx_worker

    gate_source = inspect.getsource(mlx_worker._surface_quality_failure_reasons)

    assert "assess_user_facing_reply" in gate_source
    assert "integrity_failures" in gate_source


def test_the_gate_still_keeps_only_integrity_failures():
    """Completeness shortfalls are real content and must not be suppressed.

    If this ever widens to completeness, a merely-short answer starts being
    destroyed instead of delivered, which is the opposite of the fix.
    """
    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker._surface_quality_failure_reasons)

    assert "integrity_failures(reasons)" in source
    assert re.search(r"COMPLETENESS is different", source), "the rationale must stay"
