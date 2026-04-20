"""End-to-end self-heal-via-search tests for ReActLoop.

Verifies the full loop Aura is expected to perform when she hits an internal
problem:

    encounter error  →  search the web for a fix  →  apply fix  →
    succeed  →  retain the lesson in episodic memory  →  next encounter,
    recall the lesson and apply it directly.

These tests wire together the real ReActLoop, real EpisodicMemory, real
sandbox, and a real `EnhancedWebSearchSkill` stub that short-circuits DDGS
with a deterministic response — we never monkey-patch the ReActLoop itself.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain.react_loop import (  # noqa: E402
    Action,
    ActionExecutor,
    ActionType,
    Observation,
    ReActLoop,
)
from core.container import ServiceContainer  # noqa: E402
from core.memory.episodic_memory import EpisodicMemory  # noqa: E402


# ---------------------------------------------------------------------------
# Test double types — all compositional (no monkey-patching of loaded code).
# ---------------------------------------------------------------------------

@dataclass
class _ThoughtEnvelope:
    content: str


class ScriptedBrain:
    """Deterministic brain that steps through a pre-written script of
    Thought/Action/ActionInput blocks. Real object, real duck-type."""

    def __init__(self, script: List[str], name: str = "scripted"):
        self.script = list(script)
        self.name = name
        self.calls: List[str] = []

    async def think(self, prompt: str, **kwargs):
        self.calls.append(prompt[-200:])
        if not self.script:
            return _ThoughtEnvelope(content='Thought: script empty.\nAction: FINAL_ANSWER\nActionInput: {"text": "done"}')
        return _ThoughtEnvelope(content=self.script.pop(0))


class DeterministicBrowser:
    """A *real* object registered as `sovereign_browser` in the
    ServiceContainer. No monkey-patching: we're composing the real executor
    with a real dependency that happens to return fixed content for the test.
    """

    def __init__(self, content: str):
        self._content = content
        self.queries: List[str] = []

    async def search(self, query: str):
        self.queries.append(query)
        return self._content


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_container(tmp_path: Path):
    """Clean ServiceContainer with a real per-test EpisodicMemory."""
    ServiceContainer.clear()
    mem = EpisodicMemory(db_path=str(tmp_path / "episodic.db"))
    ServiceContainer.register_instance("episodic_memory", mem)
    yield mem
    ServiceContainer.clear()


# ---------------------------------------------------------------------------
# Scenario 1: Full self-heal flow within a single run.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_heal_within_single_run(fresh_container: EpisodicMemory, monkeypatch):
    """Aura tries code → gets an error → searches → retries → succeeds, in one
    ReAct trace. The episode must capture both the error and the recovery.

    We intentionally do NOT monkey-patch anything on react_loop itself; we
    register a deterministic `sovereign_browser` (a documented, supported
    integration point) to keep the web_search path offline-deterministic.
    """
    from core.brain.cognitive_engine import ThinkingMode

    # Remove Gemini key for this test so sovereign_browser is used before DDGS.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    browser = DeterministicBrowser(
        content=(
            "SEARCH RESULT: In Python's math module, factorial is spelled "
            "`math.factorial(n)` — not `math.fact(n)`. Use the correct name."
        )
    )
    ServiceContainer.register_instance("sovereign_browser", browser)

    brain = ScriptedBrain([
        # Step 1 — try an incorrect function call.
        'Thought: I need 5! — I will call math.fact.\n'
        'Action: PYTHON_SANDBOX\n'
        'ActionInput: {"code": "import math\\nprint(math.fact(5))"}',
        # Step 2 — observation will say AttributeError or similar. Decide to search.
        'Thought: That errored. I do not know the right name for the factorial function.\n'
        'Action: WEB_SEARCH\n'
        'ActionInput: {"query": "python math module factorial function name"}',
        # Step 3 — armed with search result, retry correctly.
        'Thought: Search says math.factorial. Retry.\n'
        'Action: PYTHON_SANDBOX\n'
        'ActionInput: {"code": "import math\\nprint(math.factorial(5))"}',
        # Step 4 — finalize.
        'Thought: Got 120.\n'
        'Action: FINAL_ANSWER\n'
        'ActionInput: {"text": "5! = 120. I learned math.factorial is the correct call."}',
    ])

    loop = ReActLoop(
        brain=brain, max_steps=8, think_mode=ThinkingMode.FAST,
        simple_threshold=0, timeout_seconds=30.0,
    )
    trace = await loop.run("Please compute 5 factorial for me.")

    # Structural assertions on the trace.
    action_sequence = [s.action.action_type for s in trace.steps]
    assert action_sequence[:4] == [
        ActionType.PYTHON_SANDBOX,
        ActionType.WEB_SEARCH,
        ActionType.PYTHON_SANDBOX,
        ActionType.FINAL_ANSWER,
    ], f"Unexpected action sequence: {[a.value for a in action_sequence]}"

    # The first sandbox attempt must surface an error observation.
    first_sandbox = trace.steps[0].observation
    assert not first_sandbox.success, (
        f"First sandbox attempt should fail with AttributeError, got: {first_sandbox.content}"
    )

    # The web search must have actually run — verify our real browser saw it.
    assert browser.queries, "sovereign_browser.search must have been invoked"
    assert "factorial" in browser.queries[0].lower()

    # The second sandbox attempt must succeed and print 120.
    second_sandbox = trace.steps[2].observation
    assert second_sandbox.success
    assert "120" in second_sandbox.content

    assert "120" in trace.final_answer

    # Retention: episode must carry lessons covering both the error and fix.
    await asyncio.sleep(0.05)
    episodes = await fresh_container.recall_similar_async("factorial python math", limit=5)
    assert episodes, "Self-heal trace must persist an episode"
    ep = episodes[0]
    assert "python_sandbox" in ep.tools_used and "web_search" in ep.tools_used, (
        f"Expected tools_used to include python_sandbox and web_search, got: {ep.tools_used}"
    )
    lessons_text = " ".join(ep.lessons).lower()
    assert lessons_text, f"Expected at least one lesson, got empty; episode={ep.to_dict()}"
    assert "error" in lessons_text or "attribute" in lessons_text, (
        f"Expected an error-lesson, got: {ep.lessons}"
    )
    assert "recovered" in lessons_text or "python_sandbox" in lessons_text, (
        f"Expected a recovery-lesson, got: {ep.lessons}"
    )


# ---------------------------------------------------------------------------
# Scenario 2: Retention across two separate runs — Aura learns and reuses.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retention_across_runs(fresh_container: EpisodicMemory, monkeypatch):
    """Run 1 learns via search. Run 2 (fresh brain) recalls the lesson from
    episodic memory WITHOUT needing to search again. This is the core of
    "retain and use what she learned" — covered end-to-end."""
    from core.brain.cognitive_engine import ThinkingMode

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    browser = DeterministicBrowser(
        content=(
            "SEARCH RESULT: Python's math module exposes `math.factorial(n)` "
            "for integer factorials. There is no `math.fact`."
        )
    )
    ServiceContainer.register_instance("sovereign_browser", browser)

    # ---- Run 1: discover the fix via search ----
    brain_1 = ScriptedBrain([
        'Thought: Try math.fact.\n'
        'Action: PYTHON_SANDBOX\n'
        'ActionInput: {"code": "import math\\nprint(math.fact(6))"}',
        'Thought: Errored. Search for the correct name.\n'
        'Action: WEB_SEARCH\n'
        'ActionInput: {"query": "python math factorial function"}',
        'Thought: It is math.factorial. Retry.\n'
        'Action: PYTHON_SANDBOX\n'
        'ActionInput: {"code": "import math\\nprint(math.factorial(6))"}',
        'Thought: Got 720.\n'
        'Action: FINAL_ANSWER\n'
        'ActionInput: {"text": "6! = 720 via math.factorial."}',
    ], name="run1")
    loop1 = ReActLoop(
        brain=brain_1, max_steps=8, think_mode=ThinkingMode.FAST,
        simple_threshold=0, timeout_seconds=30.0,
    )
    trace1 = await loop1.run("Please compute 6 factorial.")
    assert "720" in trace1.final_answer

    # ---- Run 2: fresh brain, different query, but first consults memory ----
    # Swap in a NEW browser that would fail loudly if called — we want to
    # prove that run 2 never needs the web.
    class NoCallBrowser:
        async def search(self, query):  # pragma: no cover
            raise AssertionError("Run 2 must not call web search — memory recall should suffice.")

    ServiceContainer.register_instance("sovereign_browser", NoCallBrowser())

    brain_2 = ScriptedBrain([
        'Thought: First check if I learned anything relevant before.\n'
        'Action: MEMORY_QUERY\n'
        'ActionInput: {"query": "python math factorial function"}',
        'Thought: I have a prior episode — math.factorial is the right call.\n'
        'Action: PYTHON_SANDBOX\n'
        'ActionInput: {"code": "import math\\nprint(math.factorial(7))"}',
        'Thought: 5040. Done.\n'
        'Action: FINAL_ANSWER\n'
        'ActionInput: {"text": "7! = 5040. Recalled from prior learning."}',
    ], name="run2")
    loop2 = ReActLoop(
        brain=brain_2, max_steps=6, think_mode=ThinkingMode.FAST,
        simple_threshold=0, timeout_seconds=20.0,
    )
    trace2 = await loop2.run("And 7 factorial?")

    assert "5040" in trace2.final_answer, f"Expected 5040 in final, got: {trace2.final_answer}"

    # The MEMORY_QUERY observation in run 2 must surface content from run 1's episode.
    memory_steps = [s for s in trace2.steps if s.action.action_type == ActionType.MEMORY_QUERY]
    assert memory_steps, "Run 2 must query memory"
    recalled_content = memory_steps[0].observation.content.lower()
    assert "factorial" in recalled_content, (
        f"Run 2's memory query must surface the factorial lesson, got: {memory_steps[0].observation.content!r}"
    )

    # Run 2 must not include WEB_SEARCH — proving retention-driven reuse.
    run2_actions = [s.action.action_type for s in trace2.steps]
    assert ActionType.WEB_SEARCH not in run2_actions, (
        f"Run 2 must not need web search; actions were: {[a.value for a in run2_actions]}"
    )


# ---------------------------------------------------------------------------
# Scenario 3: Silent internal failure — max_steps triggers synthesis fallback
# and still records an episode (we learn even from getting stuck).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stuck_loop_still_retains_episode(fresh_container: EpisodicMemory):
    from core.brain.cognitive_engine import ThinkingMode

    # Brain that endlessly reflects without terminating — drives into max_steps.
    brain = ScriptedBrain([
        'Thought: step 1\nAction: SELF_REFLECT\nActionInput: {}',
        'Thought: step 2\nAction: SELF_REFLECT\nActionInput: {}',
        'Thought: step 3\nAction: SELF_REFLECT\nActionInput: {}',
        # Only 3 scripted; script fallback produces FINAL_ANSWER on the 4th call,
        # but we cap max_steps=3 so we exit via max_steps path.
    ], name="stuck")
    loop = ReActLoop(
        brain=brain, max_steps=3, think_mode=ThinkingMode.FAST,
        simple_threshold=0, timeout_seconds=15.0,
    )
    trace = await loop.run("Ponder indefinitely.")
    assert trace.terminated_reason in ("max_steps", "final_answer"), (
        f"Expected max_steps (or synthesized final_answer), got: {trace.terminated_reason}"
    )

    await asyncio.sleep(0.05)
    episodes = await fresh_container.recall_similar_async("ponder indefinitely", limit=3)
    assert episodes, "Even stuck runs should produce an episode (learning from getting stuck)"
    ep = episodes[0]
    assert ep.importance >= 0.7, (
        f"Stuck runs should carry elevated importance to aid future avoidance; got {ep.importance}"
    )
