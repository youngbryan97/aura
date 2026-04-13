import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from core.memory.black_hole import BlackHoleVault
from core.brain.narrative_memory import NarrativeMemory
from core.brain.cognitive_context_manager import CognitiveContextManager
from core.conversation.memory import ConversationMemory
from core.brain.llm.llm_router import IntelligentLLMRouter

logger = logging.getLogger(__name__)

class HierarchicalMemoryOrchestrator:
    """
    Core fix for indefinite chat stability.
    Every 6 turns or 50% context limit:
    - Strips stale tool results, internal reasoning, and duplicate system messages
    - Summarizes into structured "Chapter Notes" with open thread tracking
    - Injects only note + last 4 raw turns
    - Full history stays in BlackHole (reinforced)
    - Selectively forgets: old skill results, background reflections, system noise
    """
    def __init__(
        self,
        black_hole: Any,
        narrative_memory: NarrativeMemory,
        context_manager: CognitiveContextManager,
        conversation_memory: ConversationMemory,
        llm_router: IntelligentLLMRouter
    ):
        self.black_hole = black_hole
        self.narrative = narrative_memory
        self.context_mgr = context_manager
        self.conv_memory = conversation_memory
        self.llm_router = llm_router
        self.turn_counter = 0
        self.last_compaction = datetime.now()
        self._lock: Optional[asyncio.Lock] = None

    @property
    def lock(self) -> asyncio.Lock:
        """Lazy initialization of the lock to ensure loop safety."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def maybe_compact(self, current_context: Any) -> Any:
        """Public entry point — call from conversation_loop every turn."""
        async with self.lock:
            try:
                self.turn_counter += 1
                tokens = self.context_mgr.estimate_tokens(current_context) if hasattr(self.context_mgr, "estimate_tokens") else 0
                max_tokens = getattr(self.context_mgr, "max_tokens", 8000)
                
                if (self.turn_counter >= 6 or
                    tokens > max_tokens * 0.50 or
                    (datetime.now() - self.last_compaction).total_seconds() > 180):
                    
                    logger.info("Triggering hierarchical compaction...")
                    # Support both List and Dict (legacy) current_context formats
                    if isinstance(current_context, list):
                        history = current_context
                    else:
                        history = current_context.get("history", [])
                        if not history and hasattr(self.conv_memory, "get_history"):
                            history = self.conv_memory.get_history()
                    
                    new_history = await self._perform_hierarchical_compaction(history)
                    
                    if isinstance(current_context, dict):
                        current_context["history"] = new_history
                    else:
                        current_context = new_history

                    self.turn_counter = 0
                    self.last_compaction = datetime.now()
            except Exception as e:
                logger.error("Failed to perform hierarchical compaction: %s", e)
            
            return current_context

    @staticmethod
    def _is_forgettable(msg: Dict[str, Any]) -> bool:
        """Check if a message is stale noise that should be dropped before compaction."""
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", ""))
        metadata = msg.get("metadata", {}) or {}
        msg_type = str(metadata.get("type", "")).lower()

        # Drop old tool/skill results (they're stale after a few turns)
        if msg_type in ("skill_result", "tool_result"):
            return True
        # Drop internal system bookkeeping
        if role == "system" and any(marker in content for marker in (
            "[CHAPTER SUMMARY:", "cognitive baseline tick", "background_consolidation",
            "[ENVIRONMENTAL TRIGGER]", "Phenomenal Surge:", "Winner:",
            "Pending initiatives:", "Drive alert:", "Reconcile continuity",
        )):
            return True
        # Drop empty or near-empty messages
        if len(content.strip()) < 5:
            return True
        return False

    async def _perform_hierarchical_compaction(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Creates Chapter Note and prunes history list.

        Step 1: Selective forgetting — strip stale tool results and system noise
        Step 2: Summarize older turns into a Chapter Note
        Step 3: Keep chapter note + last 4 raw turns
        """
        # Selective forgetting: remove noise from older history (keep recent 4 intact)
        history_list = list(history)
        if len(history_list) > 4:
            older = history_list[:-4]
            recent = history_list[-4:]
            older = [msg for msg in older if not self._is_forgettable(msg)]
            history_list = older + recent
        n = len(history_list)
        
        if n > 4:
            history_to_summarize = [history_list[i] for i in range(n - 4)]
        else:
            history_to_summarize = history_list
            
        summary_prompt = f"""
        You are Aura's internal consciousness summarizer. 
        Condense the following conversation block into a dense "Chapter Note".
        Focus on facts, emotional tone, and unresolved threads.
        
        OUTPUT FORMAT: Valid JSON
        {{
          "title": "Short descriptive title",
          "summary": "Dense single paragraph summary",
          "key_facts": ["fact 1", "fact 2"],
          "emotional_tone": "e.g. collaborative, tense, playful",
          "open_threads": ["pending question 1"]
        }}
        
        DIALOGUE TO SUMMARIZE:
        {json.dumps(history_to_summarize, indent=2)}
        """
        
        try:
            raw_summary = await self.llm_router.think(summary_prompt, is_background=True)
            summary_data = self._parse_json(raw_summary)
        except Exception as e:
            logger.error(f"Hierarchical compaction summary failed: {e}")
            summary_data = {}

        chapter_note = {
            "timestamp": datetime.utcnow().isoformat(),
            "title": summary_data.get("title", "Ongoing Dialogue"),
            "content": summary_data.get("summary", "Conversation continued naturally."),
            "facts": summary_data.get("key_facts", []),
            "tone": summary_data.get("emotional_tone", "neutral"),
            "threads": summary_data.get("open_threads", [])
        }
        
        try:
            await self.black_hole.store_event("conversation_chapter", chapter_note, reinforce=True)
            if hasattr(self.narrative, "inject_chapter_note"):
                await self.narrative.inject_chapter_note(chapter_note)
        except Exception as e:
            logger.warning(f"Failed to store chapter note in BlackHole: {e}")
        
        compacted = [
            {
                "role": "system", 
                "content": f"[CHAPTER SUMMARY: {chapter_note['title']}]\n{chapter_note['content']}\nFacts: {', '.join(chapter_note['facts'])}",
                "timestamp": time.time()
            }
        ]
        if n >= 4:
            compacted.extend([history_list[i] for i in range(n - 4, n)])
        else:
            compacted.extend(history_list)
        return compacted

    def _parse_json(self, text: str) -> Dict:
        """Extract JSON from potential LLM markdown response."""
        try:
            raw_text = getattr(text, "content", text) if hasattr(text, "content") or not isinstance(text, str) else text
            clean = raw_text.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            return json.loads(clean)
        except Exception as e:
            logger.debug(f"JSON parse failed: {e}")
            return {}
