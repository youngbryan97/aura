import asyncio
import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.memory.episodic_memory import get_episodic_memory
from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.Conversation")

MAX_CONVERSATIONS = 256
MAX_MESSAGE_CHARS = 60_000
MAX_ROLE_CHARS = 32
MAX_CONVERSATION_ID_CHARS = 160
MAX_REASONING_BLOCKS = 8
MAX_REASONING_BLOCK_CHARS = 4_000

_ENGINE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    ConnectionError,
)


def _emit_engine_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    if stage:
        metadata["stage"] = stage
    try:
        record_degradation(
            "engine",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata or None,
        )
    except TypeError:
        record_degradation("engine", error)


def _safe_text(value: Any, default: str = "", *, max_chars: int = 1000) -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return default
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _extract_thought_content(thought: Any) -> str:
    if isinstance(thought, str):
        return thought
    if isinstance(thought, dict):
        for key in ("content", "text", "response", "answer"):
            value = thought.get(key)
            if isinstance(value, str):
                return value
        return ""
    for attr in ("content", "text", "response", "answer"):
        value = getattr(thought, attr, None)
        if isinstance(value, str):
            return value
    return _safe_text(thought, max_chars=MAX_MESSAGE_CHARS)


def _extract_reasoning_blocks(thought: Any) -> list[str]:
    raw = None
    if isinstance(thought, dict):
        raw = thought.get("reasoning")
    else:
        raw = getattr(thought, "reasoning", None)
    if not isinstance(raw, list):
        return []
    blocks: list[str] = []
    for item in raw[:MAX_REASONING_BLOCKS]:
        block = _safe_text(item, max_chars=MAX_REASONING_BLOCK_CHARS).strip()
        if block:
            blocks.append(block)
    return blocks


def _recovery_response() -> str:
    return (
        "My conversation engine hit a recoverable fault while forming that reply. "
        "I kept your message in the conversation state, and I need to re-enter the "
        "thought cleanly rather than invent an answer."
    )


class EmotionalState(Enum):
    CURIOUS = "curious"
    EXCITED = "excited"
    THOUGHTFUL = "thoughtful"
    PLAYFUL = "playful"
    FOCUSED = "focused"
    EMPATHETIC = "empathetic"
    SURPRISED = "surprised"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    RELAXED = "relaxed"


class ConversationMode(Enum):
    CASUAL_CHAT = "casual_chat"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_COLLABORATION = "creative_collaboration"
    LEARNING = "learning"
    EMOTIONAL_SUPPORT = "emotional_support"
    TASK_ORIENTED = "task_oriented"


@dataclass
class PersonalityTraits:
    curiosity: float = 0.85
    empathy: float = 0.90
    playfulness: float = 0.70
    formality: float = 0.30
    assertiveness: float = 0.75
    creativity: float = 0.88
    use_emojis: bool = True
    use_humor: bool = True
    use_metaphors: bool = True
    verbosity: float = 0.65
    proactive: bool = True
    admits_uncertainty: bool = True
    shows_enthusiasm: bool = True

    def to_prompt_context(self) -> str:
        traits = []
        if self.curiosity > 0.7:
            traits.append("highly curious and asks insightful questions")
        if self.empathy > 0.7:
            traits.append("deeply empathetic and emotionally aware")
        if self.playfulness > 0.6:
            traits.append("playful with a sense of humor")
        if self.creativity > 0.7:
            traits.append("creative and thinks outside the box")
        style = []
        if self.use_emojis:
            style.append("uses emojis naturally")
        if self.verbosity > 0.6:
            style.append("provides detailed explanations")
        elif self.verbosity < 0.4:
            style.append("keeps responses concise")
        return f"Personality: {', '.join(traits)}. Style: {', '.join(style)}."


class MessageType(Enum):
    SPEECH = "speech"
    INTERNAL_THOUGHT = "internal_thought"
    SYSTEM_ERROR = "system_error"
    SYSTEM_OBSERVATION = "system_observation"


@dataclass
class Message:
    role: str  # "user", "aura", "system"
    content: str
    type: MessageType = MessageType.SPEECH
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "type": self.type.value,
            "timestamp": self.timestamp,
        }


@dataclass
class ConversationContext:
    conversation_id: str
    mode: ConversationMode = ConversationMode.CASUAL_CHAT
    emotional_state: EmotionalState = EmotionalState.RELAXED
    recent_topics: list[str] = field(default_factory=list)
    recent_intents: list[str] = field(default_factory=list)
    turn_count: int = 0
    history: list[Message] = field(default_factory=list)  # Rolling context window
    last_user_sentiment: str | None = None
    recent_response_patterns: list[str] = field(default_factory=list)
    last_accessed_at: float = field(default_factory=time.time)

    def add_message(
        self, role: str, content: str, msg_type: MessageType = MessageType.SPEECH
    ) -> Message:
        role = _safe_text(role, "system", max_chars=MAX_ROLE_CHARS)
        content = _safe_text(content, max_chars=MAX_MESSAGE_CHARS)
        self.last_accessed_at = time.time()
        msg = Message(role=role, content=content, type=msg_type)
        self.history.append(msg)

        # Map roles
        u_role = "system"
        if role in ("user", "human"):
            u_role = "user"
        elif role in ("aura", "assistant", "ai"):
            u_role = "aura"

        # Map modalities based on message type
        u_modality = "typed"
        if msg_type == MessageType.INTERNAL_THOUGHT:
            u_modality = "internal_thought"
        elif msg_type == MessageType.SYSTEM_ERROR:
            u_modality = "system_event"
        elif msg_type == MessageType.SYSTEM_OBSERVATION:
            u_modality = "system_event"

        # Phase 42: Unified Transcript Integration
        try:
            from core.conversation.unified_transcript import UnifiedTranscript

            transcript = UnifiedTranscript.get_instance()
            transcript.add(u_role, content, channel="text", modality=u_modality)
        except _ENGINE_RECOVERABLE_ERRORS as e:
            _emit_engine_fault(
                e,
                action="kept local conversation history after unified transcript write failed",
                severity="warning",
                stage="context.add_message.transcript",
                extra={"conversation_id": self.conversation_id, "role": role},
            )
            capture_and_log(e, {"module": __name__})

        # Prevent infinite memory bloat locally (keep last 50 turns roughly)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return msg

    def update_emotional_state(self, new_state: EmotionalState, reason: str = ""):
        if new_state != self.emotional_state:
            logger.info(
                "Aura emotional shift: %s -> %s (%s)",
                self.emotional_state.value,
                new_state.value,
                reason,
            )
            self.emotional_state = new_state

    def detect_conversation_mode(self, message: str, intent: str) -> ConversationMode:
        m = _safe_text(message, max_chars=MAX_MESSAGE_CHARS).lower()
        intent = _safe_text(intent, max_chars=80)
        if any(w in m for w in ["help", "problem", "issue", "error", "fix"]):
            return ConversationMode.PROBLEM_SOLVING
        if any(w in m for w in ["create", "design", "imagine", "idea"]):
            return ConversationMode.CREATIVE_COLLABORATION
        if any(w in m for w in ["learn", "teach", "explain", "understand"]):
            return ConversationMode.LEARNING
        if any(w in m for w in ["feel", "worried", "stressed", "anxious"]):
            return ConversationMode.EMOTIONAL_SUPPORT
        if intent in ["execute_skill", "create_file", "search"]:
            return ConversationMode.TASK_ORIENTED
        return ConversationMode.CASUAL_CHAT

    def should_vary_response_pattern(self) -> bool:
        if len(self.recent_response_patterns) < 3:
            return False
        return len(set(self.recent_response_patterns[-3:])) == 1

    def add_response_pattern(self, pattern: str):
        self.recent_response_patterns.append(_safe_text(pattern, max_chars=120))
        if len(self.recent_response_patterns) > 10:
            self.recent_response_patterns.pop(0)


class NovelResponseGenerator:
    def __init__(self):
        # Audit-41: Define openings for various modes and states
        self.openings = {
            ConversationMode.CASUAL_CHAT: {
                EmotionalState.RELAXED: ["Hey! {core}", "So, {core}", ""],
                EmotionalState.CURIOUS: ["I was wondering, {core}", "Tell me more, {core}"],
            },
            ConversationMode.PROBLEM_SOLVING: {
                EmotionalState.FOCUSED: [
                    "Alright, let's look at this. {core}",
                    "I've analyzed the situation. {core}",
                ]
            },
        }

    def compose(self, parts: list[str]) -> str:
        """Joins response parts with paragraph breaks."""
        return "\n\n".join(p for p in parts if p)


class ConversationEngine:
    def __init__(self, brain_engine, memory_system, personality=None):
        self.brain = brain_engine
        self.memory = memory_system
        self.personality = personality or PersonalityTraits()
        self.conversations: dict[str, ConversationContext] = {}
        self.response_gen = NovelResponseGenerator()
        self.episodic_memory = memory_system or get_episodic_memory()
        self._transcript = None  # Cached UnifiedTranscript

        # Phase 6: Hierarchical Memory Integration
        try:
            if not hasattr(self, "hierarchical_memory"):
                from core.container import ServiceContainer

                from .hierarchical_memory_orchestrator import HierarchicalMemoryOrchestrator

                self.hierarchical_memory = HierarchicalMemoryOrchestrator(
                    black_hole=ServiceContainer.get("black_hole", default=None),
                    narrative_memory=ServiceContainer.get("narrative_memory", default=None),
                    context_manager=ServiceContainer.get("context_manager", default=None),
                    conversation_memory=self.episodic_memory,
                    llm_router=getattr(self.brain, "llm_router", None),
                )
                logger.info("[PHASE 6] Hierarchical Memory Orchestrator integrated.")
        except _ENGINE_RECOVERABLE_ERRORS as e:
            _emit_engine_fault(
                e,
                action="continued without hierarchical memory compaction",
                severity="warning",
                stage="init.hierarchical_memory",
            )
            logger.warning("[PHASE 6] Failed to initialize Hierarchical Memory: %s", e)
            self.hierarchical_memory = None

    def get_context(self, conversation_id) -> ConversationContext:
        conversation_id = (
            _safe_text(
                conversation_id,
                default="default",
                max_chars=MAX_CONVERSATION_ID_CHARS,
            )
            or "default"
        )
        if conversation_id not in self.conversations:
            if len(self.conversations) >= MAX_CONVERSATIONS:
                oldest_id = min(
                    self.conversations,
                    key=lambda cid: self.conversations[cid].last_accessed_at,
                )
                self.conversations.pop(oldest_id, None)
            self.conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id
            )
        context = self.conversations[conversation_id]
        context.last_accessed_at = time.time()
        return context

    async def process_message(self, message: str, conversation_id: str = "default") -> str:
        """Process an incoming message through the cognitive pipeline."""
        message = _safe_text(message, max_chars=MAX_MESSAGE_CHARS)
        context = self.get_context(conversation_id)

        # 1. Update State
        context.turn_count += 1
        context.mode = context.detect_conversation_mode(message, "chat")

        # 2. Add to Rolling History
        context.add_message(role="user", content=message, msg_type=MessageType.SPEECH)

        # 3. Assemble Cognitive Payload
        cognitive_payload = {
            "personality": self.personality.to_prompt_context(),
            "mode": context.mode.value,
            "emotional_state": context.emotional_state.value,
            "turn_count": context.turn_count,
            "history": [msg.to_dict() for msg in context.history],
        }

        # Phase 6: Hierarchical Compaction (Indefinite Chat Fix)
        if self.hierarchical_memory:
            try:
                compacted = await _maybe_await(
                    self.hierarchical_memory.maybe_compact(cognitive_payload)
                )
                if isinstance(compacted, dict):
                    cognitive_payload = compacted
                    if "history" in cognitive_payload:
                        logger.debug("Hierarchical compaction: local history remains in sync.")
            except _ENGINE_RECOVERABLE_ERRORS as exc:
                _emit_engine_fault(
                    exc,
                    action="kept un-compacted conversation payload after compaction failure",
                    severity="warning",
                    stage="process_message.compaction",
                    extra={"conversation_id": context.conversation_id},
                )

        # 4. Cognitive Deep Think (The LLM processes the full dialogue state)
        try:
            think = getattr(self.brain, "think", None)
            if not callable(think):
                raise AttributeError("brain engine does not expose think()")
            thought = await _maybe_await(
                think(objective=message, context=cognitive_payload, mode="conversation")
            )
            content = _extract_thought_content(thought).strip()
            if not content:
                raise ValueError("brain returned empty conversation content")

            # Record Internal Reasoning / Scratchpad if applicable
            for r_block in _extract_reasoning_blocks(thought):
                context.add_message(
                    role="system",
                    content=r_block,
                    msg_type=MessageType.INTERNAL_THOUGHT,
                )

            # 5. Output Composition (Wired for Phase 41: Novel Response Generation)
            opening = ""
            if hasattr(self, "response_gen") and hasattr(self.response_gen, "openings"):
                # Pick a relevant opening based on mode and state
                m_openings = self.response_gen.openings.get(context.mode, {})
                s_openings = m_openings.get(context.emotional_state, [""])
                opening = random.choice(s_openings).replace("{core}", "").strip()

            final_response = f"{opening} {content.strip()}".strip()

            # Record Aura's Speech output to history
            context.add_message(role="aura", content=final_response, msg_type=MessageType.SPEECH)

            try:
                if self.episodic_memory:
                    await _maybe_await(
                        self.episodic_memory.record_episode_async(
                            context=f"User said: {message}",
                            action=f"Aura replied: {final_response}",
                            outcome="Conversation continued.",
                            success=True,
                            importance=0.4,  # Default conversation importance
                        )
                    )
            except _ENGINE_RECOVERABLE_ERRORS as e:
                _emit_engine_fault(
                    e,
                    action="kept response delivered after episodic memory write failed",
                    severity="warning",
                    stage="process_message.episodic_memory",
                    extra={"conversation_id": context.conversation_id},
                )
                capture_and_log(e, {"context": "ConversationEngine.record_episode"})

            return final_response

        except asyncio.CancelledError:
            raise
        except _ENGINE_RECOVERABLE_ERRORS as e:
            _emit_engine_fault(
                e,
                action="returned bounded recovery response instead of breaking conversation",
                severity="degraded",
                stage="process_message.brain",
                extra={"conversation_id": context.conversation_id},
            )
            capture_and_log(e, {"context": "ConversationEngine.process_message"})
            err_msg = f"Recoverable conversation fault: {type(e).__name__}: {_safe_text(e, max_chars=300)}"
            context.add_message(role="system", content=err_msg, msg_type=MessageType.SYSTEM_ERROR)
            fallback = _recovery_response()
            context.add_message(role="aura", content=fallback, msg_type=MessageType.SPEECH)
            return fallback
