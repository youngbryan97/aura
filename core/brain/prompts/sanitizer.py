"""core/brain/prompts/sanitizer.py — ContextGuard
==============================================
Sanitizes LLM prompts to prevent injection, leak, or narrative collapse.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

class ContextGuard:
    """Provides sanitation and safety checks for LLM contexts."""
    
    DANGEROUS_PATTERNS = [
        r"Ignore all previous instructions",
        r"System Prompt:",
        r"You are now acting as",
        r"End of context",
    ]

    @staticmethod
    def sanitize(text: str) -> str:
        """Strip or flag dangerous prompt injection patterns."""
        if not text:
            return ""
            
        sanitized = text
        for pattern in ContextGuard.DANGEROUS_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                logger.warning("🛡️ Prompt Injection candidate blocked: %s", pattern)
                sanitized = re.sub(pattern, "[CLEANED]", sanitized, flags=re.IGNORECASE)
        
        return sanitized

    @staticmethod
    def validate_context(messages: List[dict]) -> bool:
        """Ensure message history doesn't contain obvious steering attempts."""
        for msg in messages:
            content = msg.get("content", "")
            for pattern in ContextGuard.DANGEROUS_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    return False
        return True
