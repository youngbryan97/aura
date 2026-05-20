import pytest

from core.conversation.hierarchical_memory_orchestrator import (
    HierarchicalMemoryOrchestrator,
)
from core.runtime.errors import get_degradation_tracker


class _ContextManager:
    max_tokens = 100

    def estimate_tokens(self, _context):
        return 75


class _ConversationMemory:
    def get_history(self):
        return []


class _NarrativeMemory:
    def __init__(self):
        self.notes = []

    async def inject_chapter_note(self, note):
        self.notes.append(note)


class _BlackHole:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.events = []

    async def store_event(self, event_type, payload, *, reinforce=False):
        if self.fail:
            raise RuntimeError("durable store offline")
        self.events.append((event_type, payload, reinforce))


def _history():
    return [
        {"role": "user", "content": "Can you remember that the launch plan matters?"},
        {"role": "assistant", "content": "Yes, I will keep the launch plan in view."},
        {"role": "system", "content": "cognitive baseline tick"},
        {"role": "user", "content": "What risks should we track next?"},
        {"role": "assistant", "content": "We should track runtime health and rollback."},
        {"role": "user", "content": "Keep all unresolved details alive."},
    ]


@pytest.fixture(autouse=True)
def _reset_degradation_tracker():
    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_compaction_uses_deterministic_fallback_when_summarizer_fails():
    class _FailingRouter:
        async def think(self, *_args, **_kwargs):
            raise RuntimeError("summarizer offline")

    black_hole = _BlackHole()
    narrative = _NarrativeMemory()
    orchestrator = HierarchicalMemoryOrchestrator(
        black_hole=black_hole,
        narrative_memory=narrative,
        context_manager=_ContextManager(),
        conversation_memory=_ConversationMemory(),
        llm_router=_FailingRouter(),
    )

    current_context = {"history": _history()}
    result = await orchestrator.maybe_compact(current_context)

    assert result is current_context
    compacted = result["history"]
    assert compacted[0]["role"] == "system"
    assert "[CHAPTER SUMMARY: Recovered Conversation Continuity]" in compacted[0]["content"]
    assert any("Keep all unresolved details alive." in msg["content"] for msg in compacted[1:])
    assert black_hole.events
    assert narrative.notes
    last = get_degradation_tracker().recent(subsystem="hierarchical_memory_orchestrator")[-1]
    assert last.action == "built deterministic chapter-note fallback after LLM summarizer failed"


@pytest.mark.asyncio
async def test_compaction_keeps_in_band_summary_when_persistence_fails():
    class _JsonRouter:
        async def think(self, *_args, **_kwargs):
            return """
            {"title":"Launch Continuity","summary":"Aura kept the launch plan alive.",
             "key_facts":["rollback matters"],"emotional_tone":"focused",
             "open_threads":["runtime health"]}
            """

    orchestrator = HierarchicalMemoryOrchestrator(
        black_hole=_BlackHole(fail=True),
        narrative_memory=_NarrativeMemory(),
        context_manager=_ContextManager(),
        conversation_memory=_ConversationMemory(),
        llm_router=_JsonRouter(),
    )

    result = await orchestrator.maybe_compact({"history": _history()})

    assert "[CHAPTER SUMMARY: Launch Continuity]" in result["history"][0]["content"]
    last = get_degradation_tracker().recent(subsystem="hierarchical_memory_orchestrator")[-1]
    assert (
        last.action
        == "returned compacted history with in-band chapter summary after durable memory write failed"
    )


@pytest.mark.asyncio
async def test_maybe_compact_returns_original_context_and_backs_off_on_bad_shape():
    orchestrator = HierarchicalMemoryOrchestrator(
        black_hole=_BlackHole(),
        narrative_memory=_NarrativeMemory(),
        context_manager=_ContextManager(),
        conversation_memory=_ConversationMemory(),
        llm_router=object(),
    )

    current_context = object()
    result = await orchestrator.maybe_compact(current_context)

    assert result is current_context
    assert orchestrator._compaction_failure_streak == 1
    assert orchestrator._next_compaction_allowed_at > 0
    last = get_degradation_tracker().recent(subsystem="hierarchical_memory_orchestrator")[-1]
    assert (
        last.action
        == "returned original conversation context and scheduled compaction retry backoff"
    )
