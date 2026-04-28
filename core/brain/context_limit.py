"""
core/brain/context_limit.py
───────────────────────────
Implements rolling memory compaction to prevent Context Window Collapse.
"""
from core.runtime.errors import record_degradation
import logging
from typing import List, Dict, Any

logger = logging.getLogger("Aura.ContextLimit")

async def compact_working_memory(chat_history: List[Dict[str, Any]], max_raw_turns: int = 4) -> List[Dict[str, Any]]:
    """
    Keeps the most recent N messages raw, but compresses older history into a semantic summary.
    This keeps the token count flat indefinitely.
    """
    # Each "turn" is usually User + Assistant (2 messages)
    # We keep max_raw_turns * 2 messages raw
    max_messages = max_raw_turns * 2
    
    if len(chat_history) <= max_messages:
        return chat_history

    logger.info("🧠 CONTEXT LIMIT: Compacting memory (%s messages -> %s + summary)", len(chat_history), max_messages)
    
    # Split into messages to compress and messages to keep
    to_compress = chat_history[:-max_messages]
    to_keep = chat_history[-max_messages:]
    
    # Identify the current system prompt if it's the first message
    system_prompt = None
    if to_compress and to_compress[0]["role"] == "system":
        system_prompt = to_compress.pop(0)

    # Convert messages to a text block for the summarizer
    content_block = ""
    for msg in to_compress:
        content_block += f"{msg['role'].upper()}: {msg['content']}\n\n"

    try:
        from core.container import ServiceContainer
        llm = ServiceContainer.get("llm_router", default=None)
        
        summary_prompt = (
            "Condense the following conversation into a dense, factual summary of what was discussed, "
            "decided, and the current state of tasks. Do not use dialogue. Be extremely concise."
        )
        
        # We use a fast call to summarize
        # Note: We append the history to the prompt for summarization
        full_summarization_request = f"{summary_prompt}\n\nCONVERSATION:\n{content_block}"
        
        from core.brain.types import ThinkingMode
        summary_response = await llm.think(
            full_summarization_request, 
            system_prompt="You are a memory consolidation sub-process.",
            mode=ThinkingMode.FAST
        )
        summary_text = summary_response.strip() if isinstance(summary_response, str) else str(summary_response)
        
        # Build the condensed history
        # Always preserve the primary system prompt if it exists
        new_history = []
        if system_prompt:
            new_history.append(system_prompt)
            
        new_history.append({
            "role": "system", 
            "content": f"[PRIOR CONTEXT SUMMARY]: {summary_text}"
        })
        new_history.extend(to_keep)
        
        return new_history
        
    except Exception as e:
        record_degradation('context_limit', e)
        logger.error("Memory compaction failed: %s", e)
        # Fallback: Just drop oldest if summarization fails (better than crashing)
        return chat_history[-10:]