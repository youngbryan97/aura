from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log
import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.memory.episodic_memory import get_episodic_memory

logger = logging.getLogger("Aura.Conversation")

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
        if self.curiosity > 0.7: traits.append("highly curious and asks insightful questions")
        if self.empathy > 0.7: traits.append("deeply empathetic and emotionally aware")
        if self.playfulness > 0.6: traits.append("playful with a sense of humor")
        if self.creativity > 0.7: traits.append("creative and thinks outside the box")
        style = []
        if self.use_emojis: style.append("uses emojis naturally")
        if self.verbosity > 0.6: style.append("provides detailed explanations")
        elif self.verbosity < 0.4: style.append("keeps responses concise")
        return f"Personality: {', '.join(traits)}. Style: {', '.join(style)}."

class MessageType(Enum):
    SPEECH = "speech"
    INTERNAL_THOUGHT = "internal_thought"
    SYSTEM_ERROR = "system_error"
    SYSTEM_OBSERVATION = "system_observation"

@dataclass
class Message:
    role: str # "user", "aura", "system"
    content: str
    type: MessageType = MessageType.SPEECH
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "type": self.type.value,
            "timestamp": self.timestamp
        }

@dataclass
class ConversationContext:
    conversation_id: str
    mode: ConversationMode = ConversationMode.CASUAL_CHAT
    emotional_state: EmotionalState = EmotionalState.RELAXED
    recent_topics: List[str] = field(default_factory=list)
    recent_intents: List[str] = field(default_factory=list)
    turn_count: int = 0
    history: List[Message] = field(default_factory=list) # Rolling context window
    last_user_sentiment: Optional[str] = None
    recent_response_patterns: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str, msg_type: MessageType = MessageType.SPEECH) -> Message:
        msg = Message(role=role, content=content, type=msg_type)
        self.history.append(msg)
        
        # Map roles
        u_role = "system"
        if role in ("user", "human"): u_role = "user"
        elif role in ("aura", "assistant", "ai"): u_role = "aura"
        
        # Map modalities based on message type
        u_modality = "typed"
        if msg_type == MessageType.INTERNAL_THOUGHT: u_modality = "internal_thought"
        elif msg_type == MessageType.SYSTEM_ERROR: u_modality = "system_event"
        elif msg_type == MessageType.SYSTEM_OBSERVATION: u_modality = "system_event"

        # Phase 42: Unified Transcript Integration
        try:
            from core.conversation.unified_transcript import UnifiedTranscript
            transcript = UnifiedTranscript.get_instance()
            transcript.add(u_role, content, channel="text", modality=u_modality)
        except Exception as e:
            record_degradation('engine', e)
            capture_and_log(e, {'module': __name__})

        # Prevent infinite memory bloat locally (keep last 50 turns roughly)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return msg

    def update_emotional_state(self, new_state: EmotionalState, reason: str = ""):
        if new_state != self.emotional_state:
            logger.info("Aura emotional shift: %s → %s (%s)", self.emotional_state.value, new_state.value, reason)
            self.emotional_state = new_state

    def detect_conversation_mode(self, message: str, intent: str) -> ConversationMode:
        m = message.lower()
        if any(w in m for w in ['help', 'problem', 'issue', 'error', 'fix']): return ConversationMode.PROBLEM_SOLVING
        if any(w in m for w in ['create', 'design', 'imagine', 'idea']): return ConversationMode.CREATIVE_COLLABORATION
        if any(w in m for w in ['learn', 'teach', 'explain', 'understand']): return ConversationMode.LEARNING
        if any(w in m for w in ['feel', 'worried', 'stressed', 'anxious']): return ConversationMode.EMOTIONAL_SUPPORT
        if intent in ['execute_skill', 'create_file', 'search']: return ConversationMode.TASK_ORIENTED
        return ConversationMode.CASUAL_CHAT

    def should_vary_response_pattern(self) -> bool:
        if len(self.recent_response_patterns) < 3: return False
        return len(set(self.recent_response_patterns[-3:])) == 1

    def add_response_pattern(self, pattern: str):
        self.recent_response_patterns.append(pattern)
        if len(self.recent_response_patterns) > 10: self.recent_response_patterns.pop(0)

class NovelResponseGenerator:
    def __init__(self):
        # Audit-41: Define openings for various modes and states
        self.openings = {
            ConversationMode.CASUAL_CHAT: {
                EmotionalState.RELAXED: ["Hey! {core}", "So, {core}", ""],
                EmotionalState.CURIOUS: ["I was wondering, {core}", "Tell me more, {core}"]
            },
            ConversationMode.PROBLEM_SOLVING: {
                EmotionalState.FOCUSED: ["Alright, let's look at this. {core}", "I've analyzed the situation. {core}"]
            }
        }

    def compose(self, parts: list[str]) -> str:
        """Joins response parts with paragraph breaks."""
        return "\n\n".join(p for p in parts if p)

class ConversationEngine:
    def __init__(self, brain_engine, memory_system, personality=None):
        self.brain = brain_engine
        self.memory = memory_system
        self.personality = personality or PersonalityTraits()
        self.conversations: Dict[str, ConversationContext] = {}
        self.response_gen = NovelResponseGenerator()
        self.episodic_memory = memory_system or get_episodic_memory()
        self._transcript = None # Cached UnifiedTranscript
        
        # Phase 6: Hierarchical Memory Integration
        try:
            if not hasattr(self, 'hierarchical_memory'):
                from .hierarchical_memory_orchestrator import HierarchicalMemoryOrchestrator
                from core.container import ServiceContainer
                
                self.hierarchical_memory = HierarchicalMemoryOrchestrator(
                    black_hole=ServiceContainer.get("black_hole", default=None),
                    narrative_memory=ServiceContainer.get("narrative_memory", default=None),
                    context_manager=ServiceContainer.get("context_manager", default=None),
                    conversation_memory=self.episodic_memory,
                    llm_router=self.brain.llm_router if hasattr(self.brain, 'llm_router') else None
                )
                logger.info("🧠 [PHASE 6] Hierarchical Memory Orchestrator integrated.")
        except Exception as e:
            record_degradation('engine', e)
            logger.warning(f"⚠️ [PHASE 6] Failed to initialize Hierarchical Memory: {e}")
            self.hierarchical_memory = None

    def get_context(self, conversation_id) -> ConversationContext:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(conversation_id=conversation_id)
        return self.conversations[conversation_id]

    async def process_message(self, message: str, conversation_id: str = "default") -> str:
        """Process an incoming message through the cognitive pipeline."""
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
            "history": [msg.to_dict() for msg in context.history]
        }

        # Phase 6: Hierarchical Compaction (Indefinite Chat Fix)
        if self.hierarchical_memory:
            cognitive_payload = await self.hierarchical_memory.maybe_compact(cognitive_payload)
            # Update local history if compaction happened (keep in sync)
            if "history" in cognitive_payload:
                # Local history remains Message objects, compaction returns dicts
                # We only need to sync if the length changed significantly
                logger.debug("Hierarchical compaction: local history remains in sync.")
        
        # 4. Cognitive Deep Think (The LLM processes the full dialogue state)
        try:
            thought = await self.brain.think(
                objective=message, 
                context=cognitive_payload, 
                mode="conversation"
            )
            content = thought.content if hasattr(thought, 'content') else str(thought)
            
            # Record Internal Reasoning / Scratchpad if applicable
            reasoning = thought.reasoning if hasattr(thought, 'reasoning') else []
            if reasoning and isinstance(reasoning, list):
                for r_block in reasoning:
                    if r_block.strip():
                        context.add_message(role="system", content=r_block, msg_type=MessageType.INTERNAL_THOUGHT)
            
            # 5. Output Composition (Wired for Phase 41: Novel Response Generation)
            opening = ""
            if hasattr(self, 'response_gen') and hasattr(self.response_gen, 'openings'):
                # Pick a relevant opening based on mode and state
                m_openings = self.response_gen.openings.get(context.mode, {})
                s_openings = m_openings.get(context.emotional_state, [""])
                opening = random.choice(s_openings).replace("{core}", "").strip()
            
            final_response = f"{opening} {content.strip()}".strip()
            
            # Record Aura's Speech output to history
            context.add_message(role="aura", content=final_response, msg_type=MessageType.SPEECH)
            
            try:
                if self.episodic_memory:
                    await self.episodic_memory.record_episode_async(
                        context=f"User said: {message}",
                        action=f"Aura replied: {final_response}",
                        outcome="Conversation continued.",
                        success=True,
                        importance=0.4 # Default conversation importance
                    )
            except Exception as e:
                record_degradation('engine', e)
                capture_and_log(e, {"context": "ConversationEngine.record_episode"})
            
            return final_response
            
        except Exception as e:
            record_degradation('engine', e)
            capture_and_log(e, {"context": "ConversationEngine.process_message"})
            # Re-raise major errors, swallow others with friendly message
            if isinstance(e, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            err_msg = f"Encountered internal dissonance: {str(e)}"
            context.add_message(role="system", content=err_msg, msg_type=MessageType.SYSTEM_ERROR)
            return "My cognitive systems just hit a snag. Let me try compiling that thought again."
