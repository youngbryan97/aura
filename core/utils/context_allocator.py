import logging
import tiktoken
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger("Aura.ContextAllocator")

class ContextPriority(Enum):
    CRITICAL = 10  # Identity, Persona, Hardening Core
    GOAL = 8       # Current active objectives
    RELEVANT = 5   # Recent interaction context
    EPHEMERAL = 2  # Sensory noise, small logs
    METADATA = 1   # Timestamps, non-essential attributes

@dataclass
class ContextBlock:
    id: str
    content: str
    priority: ContextPriority
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class TokenGovernor:
    """Manages the allocation of tokens across different cognitive buckets."""
    
    def __init__(self, model_name: str = "gpt-4", max_tokens: int = 8192):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        self.max_tokens = max_tokens
        # Reservations (percentage of total window)
        self.reservations = {
            ContextPriority.CRITICAL: 0.20,
            ContextPriority.GOAL: 0.15,
            ContextPriority.RELEVANT: 0.40,
            ContextPriority.EPHEMERAL: 0.15,
            ContextPriority.METADATA: 0.10
        }

    def count_tokens(self, text: str) -> int:
        # Treat literal special-token text as ordinary content so background
        # cognition degrades cleanly instead of raising tokenizer exceptions.
        return len(self.encoding.encode(text, disallowed_special=()))

    def allocate(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Prioritizes and prunes blocks to fit the token window."""
        # Calculate tokens for each block
        for block in blocks:
            if block.tokens <= 0:
                block.tokens = self.count_tokens(block.content)

        # Sort by priority (descending) then by recency (descending)
        sorted_blocks = sorted(
            blocks, 
            key=lambda x: (x.priority.value, x.metadata.get("timestamp", 0)), 
            reverse=True
        )

        total_tokens = 0
        allocated = []

        for block in sorted_blocks:
            if total_tokens + block.tokens <= self.max_tokens:
                allocated.append(block)
                total_tokens += block.tokens
            else:
                logger.debug(f"⚠️ ContextAllocator: Pruning block '{block.id}' (Priority: {block.priority.name})")
        
        # Sort allocated blocks back to original chronological order (if timestamps exist)
        return sorted(allocated, key=lambda x: x.metadata.get("timestamp", 0))

    def wrap_messages(self, messages: List[Dict[str, Any]], priority: ContextPriority = ContextPriority.RELEVANT) -> List[ContextBlock]:
        """Helper to convert standard message dicts to ContextBlocks."""
        blocks = []
        for i, m in enumerate(messages):
            content = m.get("content", "")
            if not content: continue
            
            blocks.append(ContextBlock(
                id=f"msg_{i}",
                content=content,
                priority=m.get("priority", priority),
                metadata=m
            ))
        return blocks

def get_token_governor(model: str = "gpt-4", max_tokens: int = 8192):
    return TokenGovernor(model_name=model, max_tokens=max_tokens)
