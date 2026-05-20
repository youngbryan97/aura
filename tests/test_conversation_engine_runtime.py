import asyncio
from types import SimpleNamespace

from core.conversation.engine import MAX_CONVERSATIONS, ConversationEngine


class ReplyBrain:
    async def think(self, **_kwargs):
        return SimpleNamespace(content="Here is the answer.", reasoning=["checked context"])


class FailingBrain:
    async def think(self, **_kwargs):
        self.calls = getattr(self, "calls", 0) + 1
        raise RuntimeError("brain lane unavailable")


class SyncMemory:
    def __init__(self):
        self.episodes = []

    def record_episode_async(self, **kwargs):
        self.episodes.append(kwargs)
        return None


class FailingCompactor:
    async def maybe_compact(self, payload):
        self.calls = getattr(self, "calls", 0) + 1
        raise RuntimeError(f"cannot compact {len(payload.get('history', []))} entries")


def test_conversation_engine_delivers_response_and_sync_memory_write():
    memory = SyncMemory()
    engine = ConversationEngine(ReplyBrain(), memory)
    engine.hierarchical_memory = None

    reply = asyncio.run(engine.process_message("Can you help debug this?", "session-a"))

    assert "Here is the answer." in reply
    assert memory.episodes
    context = engine.get_context("session-a")
    assert context.history[-1].role == "aura"
    assert any(msg.type.value == "internal_thought" for msg in context.history)


def test_conversation_engine_returns_bounded_recovery_on_brain_failure():
    engine = ConversationEngine(FailingBrain(), SyncMemory())
    engine.hierarchical_memory = None

    reply = asyncio.run(engine.process_message("Please answer this", "session-b"))

    assert "recoverable fault" in reply
    context = engine.get_context("session-b")
    assert context.history[-2].type.value == "system_error"
    assert context.history[-1].role == "aura"


def test_conversation_engine_keeps_chat_alive_after_compaction_failure():
    engine = ConversationEngine(ReplyBrain(), SyncMemory())
    engine.hierarchical_memory = FailingCompactor()

    reply = asyncio.run(engine.process_message("Teach me the architecture", "session-c"))

    assert "Here is the answer." in reply
    assert engine.hierarchical_memory.calls == 1


def test_conversation_engine_bounds_conversation_cache():
    engine = ConversationEngine(ReplyBrain(), SyncMemory())
    engine.hierarchical_memory = None

    for idx in range(MAX_CONVERSATIONS + 5):
        engine.get_context(f"session-{idx}")

    assert len(engine.conversations) == MAX_CONVERSATIONS
    assert "session-0" not in engine.conversations
    assert f"session-{MAX_CONVERSATIONS + 4}" in engine.conversations
