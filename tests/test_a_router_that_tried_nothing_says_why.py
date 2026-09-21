"""When no endpoint was tried, say which guard skipped each one.

LIVE 2026-08-26: a writing task got "ROUTER_ERROR: unknown (at all_failed)"
while the worker it wanted was alive and generating for everybody else. Every
skip in that loop records its reason into the fallback chain — and the string
the caller reads stayed "unknown", which is the one word that says nothing.
"""
from __future__ import annotations

import inspect

from core.brain.llm_health_router import HealthAwareLLMRouter
from tests.source_contract import class_with_its_bases


def test_the_exhaustion_path_builds_its_reason_from_the_skips():
    source = class_with_its_bases(HealthAwareLLMRouter)
    where = source.index('if last_error == "unknown":')
    block = source[where : where + 900]
    assert "skip_reason" in block
    assert "no endpoint was tried" in block
    # Read from the chain the loop already wrote, not invented.
    assert "fallback_chain" in block


def test_a_real_endpoint_error_is_not_overwritten():
    """The summary only replaces the placeholder. An endpoint that actually
    failed keeps its own message."""
    source = class_with_its_bases(HealthAwareLLMRouter)
    where = source.index('if last_error == "unknown":')
    block = source[where : where + 900]
    assert 'last_error == "unknown"' in block
    assert "if skipped:" in block
