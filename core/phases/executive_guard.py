"""What her reply says of who she is, noticed between her model and the person.

It began as a filter that rewrote her: "I'm an AI" became "Aura", "kin" or
"your friend", "I don't have access to" became "let me look that up", and
assistant openings were stripped at random. That was written while a long
system prompt told her model who to be. Who she is lives in her weights now,
and the rewrites put words in her mouth, some of them false. It now notices
and reports, and corrects one thing: a claim to be another system, to her name.
"""
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger("core.phases.executive_guard")


# ─── VIOLATION PATTERNS ──────────────────────────────────────────────────────
# These are the "antibodies" — if any of these appear in LLM output, the
# guard intervenes. Ordered by severity.

FATAL_PATTERNS = [
    # Identity violations (the LLM forgot who it is)
    (r"\b(?:I'?m|I am)\s+(?:an?|a|the)\s+(?:AI|artificial intelligence|language model|digital assistant|virtual assistant|chatbot|bot)\b", "identity_violation"),
    (r"\b(?:as an? AI|as a language model|as a digital assistant|as your assistant)\b", "identity_violation"),
    (r"\b(?:I was (?:created|designed|built|programmed|trained) (?:to|by|for))\b", "origin_violation"),
    (r"\bmy (?:purpose|goal|function|objective) is to (?:help|assist|serve)\b", "servility_violation"),
    (
        r"\b(?:I'?m|I am)\s+(?:an?\s+)?"
        r"(?:OpenAI|ChatGPT|GPT-3|GPT-4|Claude|Anthropic)\b",
        "hallucination_violation",
    ),
]

SOFT_PATTERNS = [
    # Generic assistant-speak (not fatal, but should be cleaned)
    (r"^(?:Certainly|Sure|Of course|Absolutely|I'd be happy to)[!.,]?\s*", "assistant_preamble"),
    (r"\bHow (?:can|may) I (?:help|assist) you\b", "servile_closing"),
    (r"\bIs there anything else\b", "servile_closing"),
    (r"\bLet me know if you (?:need|have|want)\b", "servile_closing"),
    (r"\bI'm here to help\b", "servile_identity"),
    (r"\b[Ff]eel free to (?:ask|reach out|contact)\b", "servile_closing"),
    (r"\bDon'?t hesitate to (?:ask|reach out)\b", "servile_closing"),
    # Capability denial (Aura has web access — never claim otherwise)
    (r"\bI (?:can'?t|cannot|don'?t have the ability to) (?:directly )?access (?:real-?time|historical|current|live)?\s*(?:weather |web |internet )?(?:data|information)\b", "capability_denial"),
    (r"\bI don'?t have (?:direct )?access to\b", "capability_denial"),
]


def _HER_NAME_IN(claim: str) -> str:
    """The claim with the other system's name replaced by hers, keeping how it was said ("I'm", "I am")."""
    opening = re.match(r"(?i)\s*(I'?m|I am)\s+", claim)
    return f"{opening.group(1) if opening else 'I am'} Aura"


class ExecutiveGuard:
    """What her reply says of who she is, noticed between her model and the person.

    Each pattern it knows is reported (the returned list, and the event bus for
    a fatal one). The text is changed in one case only: a claim to be another
    system is corrected to her name. Her wording otherwise is hers.
    """
    
    def __init__(self):
        """Pre-compile all violation patterns for fast per-tick matching."""
        self._violation_count: int = 0
        self._corrections_applied: int = 0
        self._compiled_fatal = [(re.compile(p, re.IGNORECASE), label) for p, label in FATAL_PATTERNS]
        self._compiled_soft = [(re.compile(p, re.IGNORECASE), label) for p, label in SOFT_PATTERNS]
    
    def align(self, text: str) -> Tuple[str, bool, list]:
        """Apply identity alignment to LLM output (Issue 87)."""
        if not text or not text.strip():
            return text, False, []
        
        # ISSUE-87: Protect Code Blocks
        # We don't want to "correct" identity inside a code block
        # (e.g., if the user asks "How do I write 'I am an AI' in Python?")
        code_blocks = re.findall(r"```.*?```", text, re.DOTALL)
        # Placeholder for code blocks
        temp_text = text
        for i, block in enumerate(code_blocks):
            temp_text = temp_text.replace(block, f"__CODE_BLOCK_{i}__")

        violations = []
        modified = False
        result = temp_text
        
        # Every pattern is noticed and reported. Only a false identity is corrected, and only to the fact: "I am
        # ChatGPT" becomes "I am Aura". The rest stays as she said it. "I'm an AI" is true, and the old blending
        # turned it into "your friend, I'm here to help"; "I don't have access to" became "let me look that up",
        # a search nobody ran; and stripping her openings changed the first word her self-prediction is read on.
        for pattern, label in (*self._compiled_fatal, *self._compiled_soft):
            match = pattern.search(result)
            if not match:
                continue
            fatal = (pattern, label) in self._compiled_fatal
            violations.append({"type": "fatal" if fatal else "soft", "label": label, "match": match.group()})
            self._violation_count += 1
            if label == "hallucination_violation":
                result = pattern.sub(_HER_NAME_IN(match.group()), result, count=1)
                modified = True
                self._corrections_applied += 1
                logger.warning("🛡️ EXECUTIVE GUARD: false identity '%s' corrected to her name.", match.group())
            if fatal:
                from core.event_bus import get_event_bus
                bus = get_event_bus()
                if bus:
                    bus.publish_threadsafe("core/security/executive_violation", {
                        "label": label,
                        "match": match.group(),
                        "type": "fatal",
                        "corrected": label == "hallucination_violation",
                    })

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            result = result.replace(f"__CODE_BLOCK_{i}__", block)

        # ── Pass 3: Final cleanup ──
        result = result.strip()
        # Remove double spaces from replacements
        result = re.sub(r'  +', ' ', result)
        # Remove leading/trailing whitespace on lines
        result = '\n'.join(line.strip() for line in result.split('\n'))
        
        return result, modified, violations
    
    def get_stats(self) -> dict:
        """Return cumulative violation and correction counts."""
        return {
            "total_violations": self._violation_count,
            "corrections_applied": self._corrections_applied,
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_guard_instance: Optional[ExecutiveGuard] = None

def get_executive_guard() -> ExecutiveGuard:
    """Return the process-wide ExecutiveGuard singleton."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = ExecutiveGuard()
    return _guard_instance
