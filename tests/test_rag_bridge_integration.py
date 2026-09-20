"""The Invisible RAG Bridge is WIRED (July external review defect).

The bridge (core/memory/rag_bridge.py) — silent per-turn semantic recall
with telemetry and temporal reranking — existed with zero callers on the
turn path. These contracts pin the integration: the chat context builder
fetches it (bounded), and BOTH prompt assemblers render it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.chat_lane_support import chat_lane_source

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestBridgeFetch:
    def test_bridge_returns_reranked_context_from_facade(self, monkeypatch):
        from core.container import ServiceContainer
        from core.memory.rag_bridge import fetch_deep_context

        ServiceContainer.clear()
        try:
            import time as _time

            now = _time.time()

            async def _search(query, limit=10):
                # The facade contract: async, normalized to text+content.
                return [
                    {
                        "text": "bryan prefers worktree checkpoints pushed to main",
                        "content": "bryan prefers worktree checkpoints pushed to main",
                        "timestamp": now,
                    },
                    {
                        "text": "the soak artifacts live under artifacts/reliability",
                        "content": "the soak artifacts live under artifacts/reliability",
                        "timestamp": now,
                    },
                ]

            facade = SimpleNamespace(search=_search)
            ServiceContainer.register_instance("memory_facade", facade)
            out = asyncio.run(fetch_deep_context("what did we decide about pushing checkpoints"))
            assert "worktree checkpoints" in out
        finally:
            ServiceContainer.clear()

    def test_bridge_skips_trivial_queries(self):
        from core.container import ServiceContainer
        from core.memory.rag_bridge import fetch_deep_context

        ServiceContainer.clear()
        try:
            assert asyncio.run(fetch_deep_context("hey")) == ""
        finally:
            ServiceContainer.clear()

    def test_bridge_forwards_exact_principal_scope(self):
        from core.container import ServiceContainer
        from core.memory.rag_bridge import fetch_deep_context

        observed = {}

        async def _search(query, limit=10, **kwargs):
            observed.update(kwargs)
            return []

        ServiceContainer.clear()
        try:
            ServiceContainer.register_instance(
                "memory_facade",
                SimpleNamespace(search=_search),
            )
            asyncio.run(
                fetch_deep_context(
                    "what did we discuss in the prior conversation",
                    principal_id="paired-device:a",
                    principal_surface="paired_device",
                )
            )
            assert observed == {
                "principal_id": "paired-device:a",
                "principal_surface": "paired_device",
            }
        finally:
            ServiceContainer.clear()


class TestTurnPathWiring:
    """Source-level pins: deleting the integration fails the build."""

    def test_chat_context_builder_fetches_deep_memory(self):
        source = chat_lane_source()
        assert "_fetch_deep_memory_context" in source
        assert 'context["deep_memory_context"]' in source
        assert "fetch_deep_context" in source, "the bridge must be the fetcher"

    def test_fetch_is_time_bounded(self):
        source = chat_lane_source()
        helper = source.split("async def _fetch_deep_memory_context", 1)[1][:1500]
        assert "wait_for" in helper, "a slow vault must never stall a live turn"

    def test_cognitive_engine_renders_the_block(self):
        from source_contract import family_text

        from core.brain import cognitive_engine

        # the engine and the modules lifted out of it
        source = family_text(cognitive_engine)
        assert "[DEEP MEMORY RECALL]" in source
        assert 'context.get("deep_memory_context")' in source

    def test_response_generation_renders_the_block(self):
        source = (REPO_ROOT / "core" / "phases" / "response_generation.py").read_text(encoding="utf-8")
        assert "DEEP MEMORY RECALL" in source
        assert 'runtime_context.get("deep_memory_context")' in source
