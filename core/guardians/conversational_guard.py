from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ConversationalMemoryGuard:
    def __init__(self, max_cloud_turns: int = 15, max_local_turns: int = 8):
        self.working_memory: List[Dict[str, str]] = []
        self.compressed_context = ""  # The rolling summary of older events
        
        # Thresholds
        self.max_cloud_turns = max_cloud_turns  # For Gemini API (massive context)
        self.max_local_turns = max_local_turns  # For MLX local (M5/64GB = generous)

    async def append_turn(self, role: str, content: str, cognitive_engine):
        """Adds a new conversation turn and triggers compression if getting too long."""
        self.working_memory.append({"role": role, "content": content})
        
        # If the working memory is getting too long for the cloud baseline, compress it
        if len(self.working_memory) > self.max_cloud_turns:
            logger.info("🧠 ConversationalMemoryGuard: Context limit reached. Triggering background compression.")
            # Don't block the conversation to summarize. Send it to the background.
            get_task_tracker().create_task(self._compress_oldest_memories(cognitive_engine))

    async def _compress_oldest_memories(self, cognitive_engine):
        """Takes the oldest N turns, summarizes them, and updates the compressed context."""
        # Grab the oldest 5 turns to compress
        turns_to_compress = self.working_memory[:5]
        
        prompt = f"""
        Summarize this past conversation strictly as factual context. 
        Current Context: {self.compressed_context}
        New Events to Add: {turns_to_compress}
        
        Output a single, dense paragraph updating the context.
        """
        
        try:
            # Use the fast cloud model to do the summarization math
            from core.brain.cognitive_engine import ThinkingMode

            summary = await cognitive_engine.think(
                prompt,
                mode=ThinkingMode.FAST,
                origin="conversational_guard",
                is_background=True,
            )
            if hasattr(summary, "content"):
                self.compressed_context = summary.content
            else:
                self.compressed_context = str(summary)
            
            # Remove those turns from the active working memory
            self.working_memory = self.working_memory[5:]
            logger.info("🧠 ConversationalMemoryGuard: Compression complete. Context window optimized.")
        except Exception as e:
            record_degradation('conversational_guard', e)
            logger.error(f"Memory compression failed: {e}")

    def get_prompt_for_tier(self, tier: str, system_vibe: str, current_user_input: str) -> str:
        """Dynamically scales the context window based on the hardware executing it."""
        
        base_system_prompt = f"You are Aura, an independent digital intelligence.\n{system_vibe}"
        
        if tier == "CLOUD":
            # Gemini has a 1M+ token window. Give it the rich history.
            history = ""
            if self.compressed_context:
                history += f"\n[PRIOR CONTEXT]: {self.compressed_context}\n"
                
            for turn in self.working_memory:
                history += f"\n{turn['role'].upper()}: {turn['content']}"
                
            return f"{base_system_prompt}\n{history}\nUSER: {current_user_input}"
            
        elif tier == "LOCAL":
            # M5/64GB: Local models have generous context windows.
            # Include summary and recent turns without aggressive truncation.
            
            history = ""
            if self.compressed_context:
                history += f"\n[CONTEXT]: {self.compressed_context[:4000]}\n"
            
            # Include recent turns for conversational context
            recent_turns = self.working_memory[-self.max_local_turns:]
            for turn in recent_turns:
                history += f"\n{turn['role'].upper()}: {turn['content']}"
                
            mlx_prompt = f"{base_system_prompt}\n{history}\nUSER: {current_user_input}"
            
            logger.info(f"LOCAL tier: Context assembled at {len(mlx_prompt)} chars.")
            return mlx_prompt
        
        return f"{base_system_prompt}\nUSER: {current_user_input}"
