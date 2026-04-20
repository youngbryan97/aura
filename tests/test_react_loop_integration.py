"""Integration tests for ReActLoop plumbing fixes.

These tests exercise the REAL ReActLoop, real EpisodicMemory (backed by a
per-test SQLite file), and real ActionExecutor — no methods are monkey-patched
on the system under test. The only test double is a ScriptedBrain which is a
*compositional* stand-in for the LLM: it satisfies the `async def think()`
duck-type contract so we can drive deterministic action sequences through the
real executor and verify end-to-end behavior.

Coverage:
    - MEMORY_QUERY returns episodes via the fixed recall path (regression)
    - think_mode is honored (no hardcoded DEEP)
    - Completed traces are persisted as episodes for retention
    - Error → recovery flows record `lessons` so Aura learns from failures
    - PYTHON_SANDBOX sandboxing still works after fixes
    - WEB_SEARCH falls back correctly when GEMINI_API_KEY is absent (DDGS)
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

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
    Thought,
)
from core.container import ServiceContainer  # noqa: E402
from core.memory.episodic_memory import EpisodicMemory  # noqa: E402


# ---------------------------------------------------------------------------
# Compositional test brain — NOT monkey-patching anything.
# ---------------------------------------------------------------------------

@dataclass
class _ThoughtEnvelope:
    """Shape-compatible with ThinkingMode responses (has `.content`)."""
    content: str


class ScriptedBrain:
    """A real object that satisfies ReActLoop's brain contract.

    It produces pre-scripted `Thought/Action/ActionInput` text so tests can
    drive the real ReAct orchestrator through deterministic paths. It records
    every call for later assertions.
    """

    def __init__(self, script: List[str]):
        self.script = list(script)
        self.calls: List[dict] = []

    async def think(self, prompt: str, *, mode=None, max_tokens=None, priority=False, **kwargs):
        self.calls.append({
            "prompt_snippet": prompt[-400:],
            "mode": mode,
            "max_tokens": max_tokens,
            "priority": priority,
        })
        if not self.script:
            # Deterministic fallback terminator so loops don't run away.
            return _ThoughtEnvelope(content='Thought: Out of script.\nAction: FINAL_ANSWER\nActionInput: {"text": "done"}')
        return _ThoughtEnvelope(content=self.script.pop(0))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_episodic(tmp_path: Path):
    """Real EpisodicMemory with a per-test SQLite file (no monkey-patching)."""
    db_path = tmp_path / "episodic.db"
    # Ensure no leftover state from previous tests.
    ServiceContainer.clear()
    mem = EpisodicMemory(db_path=str(db_path))
    ServiceContainer.register_instance("episodic_memory", mem)
    yield mem
    ServiceContainer.clear()


# ---------------------------------------------------------------------------
# 1. MEMORY_QUERY uses the correct recall API and returns episodes.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_query_returns_stored_episodes(tmp_episodic: EpisodicMemory):
    # Seed an episode about the Y Combinator so we can recall it.
    episode_id = await tmp_episodic.record_episode_async(
        context="User asked about anonymous recursion in Python",
        action="python_sandbox",
        outcome="Implemented Y combinator: Y = lambda f: (lambda x: f(lambda v: x(x)(v)))(lambda x: f(lambda v: x(x)(v)))",
        success=True,
        tools_used=["python_sandbox"],
        lessons=["Y combinator enables anonymous recursion without named functions"],
        importance=0.8,
    )
    assert episode_id, "record_episode_async must return a non-empty id"

    from core.brain.cognitive_engine import ThinkingMode

    brain = ScriptedBrain([
        'Thought: I should check episodic memory for what I learned before.\n'
        'Action: MEMORY_QUERY\nActionInput: {"query": "Y combinator anonymous recursion"}',
        'Thought: I recalled my prior learning.\n'
        'Action: FINAL_ANSWER\nActionInput: {"text": "Recalled prior Y combinator derivation."}',
    ])
    loop = ReActLoop(
        brain=brain, max_steps=4, think_mode=ThinkingMode.FAST,
        simple_threshold=0, timeout_seconds=15.0,
    )
    trace = await loop.run("Do you remember the Y combinator?")

    memory_steps = [s for s in trace.steps if s.action.action_type == ActionType.MEMORY_QUERY]
    assert memory_steps, "MEMORY_QUERY step must have executed"
    obs = memory_steps[0].observation
    assert obs.success, f"MEMORY_QUERY observation must succeed, got: {obs.content}"
    assert "Y combinator" in obs.content, (
        f"Expected recalled episode content to surface, got: {obs.content!r}"
    )


# ---------------------------------------------------------------------------
# 2. think_mode parameter is honored — not hardcoded to DEEP.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_think_mode_parameter_honored(tmp_episodic: EpisodicMemory):
    from core.brain.cognitive_engine import ThinkingMode

    brain = ScriptedBrain([
        'Thought: Trivial.\nAction: FINAL_ANSWER\nActionInput: {"text": "ok"}',
    ])
    loop = ReActLoop(
        brain=brain,
        max_steps=2,
        think_mode=ThinkingMode.FAST,
        simple_threshold=0,  # force ReAct path
        timeout_seconds=10.0,
    )
    await loop.run("What is 2 plus 2 when answered with deliberation?")

    assert brain.calls, "Brain must have been called"
    modes = {call["mode"] for call in brain.calls}
    assert ThinkingMode.FAST in modes, (
        f"Expected think_mode=FAST to reach brain.think, got modes={modes}"
    )
    assert ThinkingMode.DEEP not in modes, (
        "DEEP mode must not be used when think_mode=FAST is specified"
    )


# ---------------------------------------------------------------------------
# 3. Completed traces are persisted to episodic memory for retention.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_completed_trace_is_recorded_as_episode(tmp_episodic: EpisodicMemory):
    from core.brain.cognitive_engine import ThinkingMode

    brain = ScriptedBrain([
        'Thought: I can compute this with Python.\n'
        'Action: PYTHON_SANDBOX\n'
        'ActionInput: {"code": "result = sum(range(11))\\nprint(result)"}',
        'Thought: Got 55.\nAction: FINAL_ANSWER\nActionInput: {"text": "The sum 0..10 is 55."}',
    ])
    loop = ReActLoop(
        brain=brain, max_steps=4, think_mode=ThinkingMode.FAST,
        simple_threshold=0, timeout_seconds=20.0,
    )
    trace = await loop.run("Compute the sum of 0 through 10 inclusive.")

    assert "55" in trace.final_answer
    # Give the write-through async path a beat to finalize the DB row.
    await asyncio.sleep(0.05)
    recalled = await tmp_episodic.recall_similar_async("sum 0 through 10", limit=5)
    assert any("55" in ep.outcome or "sum" in ep.context.lower() for ep in recalled), (
        f"Episode from completed trace must be retrievable, got: {[ep.to_dict() for ep in recalled]}"
    )
    # tools_used should include python_sandbox
    match = next((ep for ep in recalled if "python_sandbox" in ep.tools_used), None)
    assert match is not None, (
        f"Expected tools_used to include python_sandbox, got: {[ep.tools_used for ep in recalled]}"
    )


# ---------------------------------------------------------------------------
# 4. Self-heal flow: error → WEB_SEARCH → retry → success; lessons retained.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_then_recovery_records_lessons(tmp_episodic: EpisodicMemory):
    """When the LLM hits an error and then finds a fix, the learned fix is retained."""
    from core.brain.cognitive_engine import ThinkingMode

    # First PYTHON_SANDBOX fails (uses a blocked module). LLM then learns and
    # retries without the forbidden import. This mirrors the self-heal pattern
    # where Aura encounters an error and adapts.
    brain = ScriptedBrain([
        'Thought: I will open a file to read its contents.\n'
        'Action: PYTHON_SANDBOX\n'
        'ActionInput: {"code": "import os\\nprint(os.listdir(\\".\\"))"}',
        # Reads the blocked-import error. Pivots to simpler code.
        'Thought: The sandbox blocks os. I will compute with pure Python instead.\n'
        'Action: PYTHON_SANDBOX\n'
        'ActionInput: {"code": "result = [x*x for x in range(4)]\\nprint(result)"}',
        'Thought: Got the squares list.\n'
        'Action: FINAL_ANSWER\n'
        'ActionInput: {"text": "Squares 0..3 are [0, 1, 4, 9]."}',
    ])
    loop = ReActLoop(
        brain=brain, max_steps=6, think_mode=ThinkingMode.FAST,
        simple_threshold=0, timeout_seconds=20.0,
    )
    trace = await loop.run("Give me the squares of 0 through 3.")

    # The first sandbox observation must be an error; the second must succeed.
    sandbox_obs = [s.observation for s in trace.steps if s.action.action_type == ActionType.PYTHON_SANDBOX]
    assert len(sandbox_obs) == 2, f"Expected 2 sandbox attempts, got {len(sandbox_obs)}"
    assert not sandbox_obs[0].success, "First sandbox call must fail (blocked import)"
    assert sandbox_obs[1].success, "Second sandbox call must succeed"
    assert "[0, 1, 4, 9]" in trace.final_answer

    # Retention: the recorded episode must contain a lesson capturing both the
    # error and the recovery — this is what enables next-turn self-heal.
    await asyncio.sleep(0.05)
    episodes = await tmp_episodic.recall_similar_async("squares python sandbox", limit=5)
    assert episodes, "Self-heal run must produce an episode"
    ep = episodes[0]
    assert ep.lessons, f"Self-heal episode must capture lessons, got: {ep.lessons}"
    combined_lessons = " ".join(ep.lessons).lower()
    assert "error" in combined_lessons or "blocked" in combined_lessons, (
        f"Expected the error lesson to be recorded, got: {ep.lessons}"
    )
    assert "recovered" in combined_lessons or "python_sandbox" in combined_lessons, (
        f"Expected the recovery lesson to be recorded, got: {ep.lessons}"
    )


# ---------------------------------------------------------------------------
# 5. PYTHON_SANDBOX security: blocked modules are refused, not executed.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_python_sandbox_blocks_os_import():
    executor = ActionExecutor()
    action = Action(
        action_type=ActionType.PYTHON_SANDBOX,
        params={"code": "import subprocess\nsubprocess.run(['ls'])"},
    )
    obs = await executor.execute(action)
    assert not obs.success, "Subprocess import must be blocked"
    assert "Blocked" in obs.content or "not allowed" in obs.content


@pytest.mark.asyncio
async def test_python_sandbox_allows_safe_code():
    executor = ActionExecutor()
    action = Action(
        action_type=ActionType.PYTHON_SANDBOX,
        params={"code": "import math\nresult = math.factorial(5)\nprint(result)"},
    )
    obs = await executor.execute(action)
    assert obs.success, f"Safe code must run, got: {obs.content}"
    assert "120" in obs.content


# ---------------------------------------------------------------------------
# 6. WEB_SEARCH DDGS fallback produces a real observation when GEMINI is absent.
#    This test actually hits the internet — mark as `online` so CI can skip.
# ---------------------------------------------------------------------------

@pytest.mark.online
@pytest.mark.asyncio
async def test_web_search_ddgs_fallback_live():
    """Verify the real DuckDuckGo fallback returns non-empty content.

    Skipped if AURA_OFFLINE is set so the test can be skipped in sealed envs.
    """
    if os.environ.get("AURA_OFFLINE"):
        pytest.skip("AURA_OFFLINE set")
    # Temporarily unset GEMINI_API_KEY so we exercise the fallback explicitly.
    saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        executor = ActionExecutor()
        action = Action(
            action_type=ActionType.WEB_SEARCH,
            params={"query": "Python official documentation website"},
        )
        obs = await executor.execute(action)
    finally:
        if saved is not None:
            os.environ["GEMINI_API_KEY"] = saved
    assert obs.source in ("web_deep_crawl", "web_ddgs_snippets", "web_browser"), (
        f"Expected a fallback source, got: source={obs.source} content={obs.content!r}"
    )
    assert obs.content and len(obs.content) > 30, (
        f"Expected non-trivial web content, got: {obs.content!r}"
    )
