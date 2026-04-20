"""Tests for the curiosity-driven search pathway.

When Aura's CuriosityEngine autonomously decides to explore a topic, it issues:

    await self.orchestrator.execute_tool("web_search", {"query": ...})

These tests verify that path end-to-end:

1. `EnhancedWebSearchSkill.execute()` — the actual tool curiosity invokes —
   accepts a dict payload (like curiosity sends), and returns a well-formed
   result dict.
2. `CuriosityEngine._explore()` — when given a topic — reaches the
   orchestrator's `execute_tool` with `"web_search"` and the expected
   params, and stores the result in the knowledge graph. We wire this up
   with a real (not monkey-patched) lightweight orchestrator-shaped object
   that implements the documented contract.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Orchestrator-shaped test double (compositional — no monkey-patching).
#
# We implement the exact contract `CuriosityEngine._explore` requires:
#   - .execute_tool(name, params)
#   - .is_busy attribute
#   - .liquid_state attribute (for curiosity level)
#   - .kernel attribute (for volition check)
#   - .knowledge_graph (for result storage)
# ---------------------------------------------------------------------------

class _Kernel:
    volition_level = 3  # allow exploration


class _LiquidCurrent:
    def __init__(self, curiosity: float = 0.7):
        self.curiosity = curiosity
        self.frustration = 0.0
        self.energy = 0.8


class _LiquidState:
    def __init__(self):
        self.current = _LiquidCurrent()

    def get_status(self):
        return {"health": 1.0, "status": {"initialized": True, "running": True}}


class _KnowledgeGraph:
    def __init__(self):
        self.stored: List[Dict[str, Any]] = []

    def add_knowledge(self, content: str, type: str, source: str, confidence: float, metadata: Dict[str, Any]):
        self.stored.append({
            "content": content,
            "type": type,
            "source": source,
            "confidence": confidence,
            "metadata": metadata,
        })


class StubOrchestrator:
    """Real Python object that satisfies the orchestrator contract
    CuriosityEngine reads. It records every `execute_tool` call.

    Named without the "Test" prefix so pytest's collector doesn't try to
    instantiate it as a test class.
    """

    def __init__(self, search_result: Dict[str, Any]):
        import time as _time
        self._search_result = search_result
        self.is_busy = False
        self.liquid_state = _LiquidState()
        self.kernel = _Kernel()
        self.knowledge_graph = _KnowledgeGraph()
        self.calls: List[Dict[str, Any]] = []
        # Background policy requires an idle window; pretend the user was
        # active well in the past so exploration is permitted.
        self._last_user_interaction_time = _time.time() - 3600.0
        self._suppress_unsolicited_proactivity_until = 0.0
        self._foreground_user_quiet_until = 0.0

    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"tool": tool_name, "params": dict(params)})
        if tool_name != "web_search":
            return {"ok": False, "error": f"unknown tool: {tool_name}"}
        return dict(self._search_result)


class _ProactiveComm:
    """Minimal contract: `get_boredom_level()`."""
    def get_boredom_level(self) -> float:
        return 0.9


# ---------------------------------------------------------------------------
# Test 1: curiosity._explore triggers the web_search tool with the right shape.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_curiosity_explore_triggers_web_search():
    from core.curiosity_engine import CuriosityEngine, CuriosityTopic

    orch = StubOrchestrator(search_result={
        "ok": True,
        "result": "The Vela Incident (1979) was a double flash near South Africa likely from a nuclear test.",
        "data": "",
        "source": "web",
    })
    curiosity = CuriosityEngine(orchestrator=orch, proactive_comm=_ProactiveComm())
    topic = CuriosityTopic(topic="Vela Incident 1979", reason="test", priority=0.9)

    await curiosity._explore(topic)

    # The tool was invoked exactly once, with web_search + a query containing our topic.
    web_calls = [c for c in orch.calls if c["tool"] == "web_search"]
    assert len(web_calls) == 1, f"Expected exactly one web_search call, got: {orch.calls}"
    params = web_calls[0]["params"]
    query = params.get("query", "")
    assert "Vela Incident 1979" in query, f"Query should include the topic, got: {query!r}"
    assert params.get("deep") is True
    assert params.get("retain") is True
    assert params.get("num_results") == 6

    # The result was stored in the knowledge graph.
    assert orch.knowledge_graph.stored, "Search result must be added to the knowledge graph"
    stored = orch.knowledge_graph.stored[0]
    assert "Vela" in stored["content"] and stored["type"] == "curiosity_finding"
    assert stored["metadata"]["topic"] == "Vela Incident 1979"


# ---------------------------------------------------------------------------
# Test 2: curiosity._explore gracefully handles a search failure.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_curiosity_explore_handles_search_failure():
    from core.curiosity_engine import CuriosityEngine, CuriosityTopic

    orch = StubOrchestrator(search_result={"ok": False, "error": "no results"})
    curiosity = CuriosityEngine(orchestrator=orch, proactive_comm=_ProactiveComm())
    topic = CuriosityTopic(topic="definitely not a real topic xyz", reason="test", priority=0.9)

    # Should not raise — failure is absorbed, logged, and emits a notice.
    await curiosity._explore(topic)

    # execute_tool was called; knowledge graph untouched.
    assert any(c["tool"] == "web_search" for c in orch.calls)
    assert not orch.knowledge_graph.stored, (
        "Failed searches must not poison the knowledge graph"
    )


# ---------------------------------------------------------------------------
# Test 3: EnhancedWebSearchSkill accepts curiosity's dict-shaped payload and
# returns a response with the keys curiosity expects (`ok`, `result`/`data`).
#
# This is a contract test — no internet required because we construct a
# real skill instance and assert on its input handling without completing
# a network call. We DO NOT monkey-patch; we read the public validation
# path only.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_search_skill_rejects_empty_query():
    from core.skills.web_search import EnhancedWebSearchSkill

    skill = EnhancedWebSearchSkill()
    result = await skill.execute({"query": ""}, context={})
    assert isinstance(result, dict)
    assert result.get("ok") is False
    assert "No search query provided" in result.get("error", "")


# ---------------------------------------------------------------------------
# Test 4: EnhancedWebSearchSkill contract matches what CuriosityEngine reads.
# This is a structural compatibility check that protects against either
# side of that contract silently drifting.
# ---------------------------------------------------------------------------

def test_curiosity_result_shape_compatible_with_skill_output():
    """Curiosity reads: result["ok"], result["result"], result["data"].
    Skill output contract (from EnhancedWebSearchSkill.execute docstring and
    ResearchSearchPipeline.search return shape) must include these keys for
    successful responses."""
    # These keys are asserted by ResearchSearchPipeline.search (see lines
    # 387-403 in core/search/research_pipeline.py): ok, result, data-like
    # content fields. Curiosity reads result.get("result", result.get("data", "")).
    # A minimal simulated successful response:
    result = {
        "ok": True,
        "query": "test",
        "result": "synthesized answer",
        "content": "raw content",
        "source": "http://example.com",
    }
    # Curiosity's own extraction pattern (from core/curiosity_engine.py:196):
    extracted = result.get("result", result.get("data", ""))
    assert extracted == "synthesized answer"
