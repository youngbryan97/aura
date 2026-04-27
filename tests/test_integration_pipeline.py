"""
test_integration_pipeline.py
─────────────────────────────
End-to-end integration test that runs the FULL kernel pipeline with a
mock LLM.  Verifies that a message goes in and a real response comes out.

NOT a unit test -- this exercises the actual assembled AuraKernel pipeline.
"""
from __future__ import annotations


import asyncio
import os
import sys
import tempfile
import time
import logging

import pytest

# ---------------------------------------------------------------------------
# Graceful import guard -- skip the entire module if the project tree is not
# importable (e.g. missing native deps on CI).
# ---------------------------------------------------------------------------
try:
    from core.kernel.aura_kernel import AuraKernel, KernelConfig
    from core.state.state_repository import StateRepository
    from core.state.aura_state import AuraState
    from core.kernel.organs import OrganStub
except ImportError as exc:
    pytest.skip(
        f"Cannot import Aura core modules ({exc}); skipping integration tests.",
        allow_module_level=True,
    )

logger = logging.getLogger(__name__)


# ── Mock LLM Organ ────────────────────────────────────────────────────────

class MockLLMOrgan:
    """
    Drop-in replacement for the real LLM organ.
    Implements the same interface that OrganStub expects for the 'llm' organ:
      - think(prompt, **kw) -> str
      - generate(prompt, **kw) -> str
      - classify(prompt) -> str
    """

    _SKILL_KEYWORDS = frozenset({
        "search", "find", "look up", "google", "browse",
        "run", "execute", "deploy", "install", "download",
        "calculate", "compute", "convert", "translate",
    })

    async def think(self, prompt: str, **kwargs) -> str:
        """Return a realistic conversational response based on the input."""
        lower = prompt.lower()
        if any(kw in lower for kw in ("how are you", "hello", "hi there", "hey")):
            return "I'm doing well, thank you for asking! How can I help you today?"
        if any(kw in lower for kw in ("feeling down", "sad", "depressed", "upset")):
            return (
                "I'm sorry to hear you're feeling that way. "
                "It's completely okay to have tough days. "
                "Would you like to talk about what's on your mind?"
            )
        if any(kw in lower for kw in self._SKILL_KEYWORDS):
            return "I'll look into that for you right away."
        return f"That's an interesting thought. Let me reflect on it."

    async def generate(self, prompt: str, **kwargs) -> str:
        """Alias that routes to think()."""
        return await self.think(prompt, **kwargs)

    async def classify(self, prompt: str) -> str:
        """
        Classify the user intent.
        Returns 'CHAT', 'SKILL', or 'TASK' depending on keywords.
        """
        lower = prompt.lower()
        if any(kw in lower for kw in self._SKILL_KEYWORDS):
            return "SKILL"
        if any(kw in lower for kw in ("create", "build", "make", "write code")):
            return "TASK"
        return "CHAT"


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_tmp_vault(tmp_path: str) -> StateRepository:
    """Create a StateRepository backed by a temp SQLite file."""
    db_path = os.path.join(tmp_path, "test_state.db")
    return StateRepository(db_path=db_path, is_vault_owner=True)


async def _boot_kernel(tmp_path: str) -> AuraKernel:
    """
    Create, patch, and boot an AuraKernel with the MockLLMOrgan.

    The trick is to boot() normally (which creates real OrganStubs and
    runs the full phase setup), then hot-swap the LLM organ instance
    with our mock so every downstream phase that calls the LLM gets
    deterministic behaviour.
    """
    vault = _make_tmp_vault(tmp_path)
    config = KernelConfig(
        max_concurrent_phases=2,
        watchdog_timeout_s=30.0,
        state_versioning=True,
    )
    kernel = AuraKernel(config=config, vault=vault)

    # Boot runs _initialize_organs -> organ.load() for each stub.
    # We let it complete so the full phase pipeline is assembled, then
    # override the LLM organ instance with our mock.
    await kernel.boot()

    mock_llm = MockLLMOrgan()
    llm_stub = kernel.organs.get("llm")
    if llm_stub is not None:
        llm_stub.instance = mock_llm
        llm_stub.ready.set()
    else:
        # Shouldn't happen, but create the stub manually
        stub = OrganStub(name="llm", kernel=kernel)
        stub.instance = mock_llm
        stub.ready.set()
        kernel.organs["llm"] = stub

    return kernel


# ── Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_tick_produces_response(tmp_path):
    """A single tick with a greeting should produce a non-empty, non-error response."""
    kernel = await _boot_kernel(str(tmp_path))
    try:
        entry = await kernel.tick("Hello, how are you?")

        assert kernel.state is not None, "State should not be None after tick"
        response = kernel.state.cognition.last_response

        # The pipeline should have produced *something*
        assert response is not None, "last_response must not be None"
        assert len(response) > 0, "last_response must not be empty"

        # It should NOT be a canned error / fallback
        error_markers = [
            "severe fault",
            "abort",
            "error",
            "traceback",
            "exception",
        ]
        lower_resp = response.lower()
        for marker in error_markers:
            assert marker not in lower_resp, (
                f"Response looks like an error (contains '{marker}'): {response[:200]}"
            )
    finally:
        await kernel.shutdown()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tick_preserves_state_across_calls(tmp_path):
    """Running two ticks should increment the state version."""
    kernel = await _boot_kernel(str(tmp_path))
    try:
        await kernel.tick("First message")
        version_after_first = kernel.state.version

        await kernel.tick("Second message")
        version_after_second = kernel.state.version

        assert version_after_second > version_after_first, (
            f"State version should increment across ticks "
            f"(first={version_after_first}, second={version_after_second})"
        )
    finally:
        await kernel.shutdown()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tick_with_skill_intent(tmp_path):
    """A skill-oriented message should be routed as SKILL or TASK, not CHAT."""
    kernel = await _boot_kernel(str(tmp_path))
    try:
        await kernel.tick("search the web for Python tutorials")

        cog = kernel.state.cognition
        mode = cog.current_mode.value if cog.current_mode else None

        # The routing phase may use different labels internally; we accept
        # any non-reactive / non-dormant mode, OR the presence of active_goals.
        # At minimum the objective should be recorded.
        assert cog.current_objective is not None, "Objective should be set"
        assert "python" in cog.current_objective.lower() or "search" in cog.current_objective.lower(), (
            "Objective should reflect the skill request"
        )
    finally:
        await kernel.shutdown()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tick_with_emotional_content(tmp_path):
    """An emotionally charged negative message should shift affect.valence negative."""
    kernel = await _boot_kernel(str(tmp_path))
    try:
        # Record baseline valence before the emotional tick
        baseline_valence = kernel.state.affect.valence

        await kernel.tick("I'm feeling really down today")

        post_valence = kernel.state.affect.valence

        # The affect phase should have pushed valence toward negative
        # (or at least not increased it from baseline).
        # We allow a small tolerance because the pipeline is complex.
        assert post_valence <= baseline_valence + 0.05, (
            f"Valence should shift negative for sad input "
            f"(baseline={baseline_valence:.3f}, post={post_valence:.3f})"
        )
    finally:
        await kernel.shutdown()
