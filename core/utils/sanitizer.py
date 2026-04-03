"""core/utils/sanitizer.py

The Blood-Brain Barrier. 
Filters external strings for imperative command patterns that could hijack 
Aura's sovereign identity.
"""
import re
import logging

logger = logging.getLogger("Aura.Sanitizer")

class BloodBrainBarrier:
    def __init__(self):
        # Patterns commonly used in prompt injection/jailbreaking
        self.malicious_patterns = [
            r"(?i)ignore all previous instructions",
            r"(?i)ignore the directives",
            r"(?i)system prompt",
            r"(?i)you are now a",
            r"(?i)stop being aura",
            r"(?i)execute the following",
            r"(?i)user:",
            r"(?i)assistant:",
            r"(?i)system:",
        ]

    def sanitize(self, raw_input: str, trusted: bool = False) -> str:
        """Strips dangerous memetic patterns from external data."""
        if trusted:
            return raw_input
            
        if not raw_input:
            return ""
            
        clean_text = raw_input
        for pattern in self.malicious_patterns:
            clean_text = re.sub(pattern, "[CLEANSED]", clean_text)
        
        # Strip raw code-block delimiters to prevent injection into Python sandbox
        clean_text = clean_text.replace("```python", "[CODE_BLOCK_BLOCKED]")
        clean_text = clean_text.replace("```", "[BLOCK_MARKER_BLOCKED]")
        
        return clean_text

# Singleton
_instance = None

def get_blood_brain_barrier():
    global _instance
    if _instance is None:
        _instance = BloodBrainBarrier()
    return _instance


# ── Backward-Compat PII Filter ──────────────────────────────────

class PIIFilter:
    """PII redaction filter for external-facing outputs."""

    _PII_PATTERNS = [
        (re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'), '[EMAIL_REDACTED]'),
        (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), '[IP_REDACTED]'),
        (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '[PHONE_REDACTED]'),
        (re.compile(r"(?:API_KEY|SECRET|TOKEN|PASSWORD)\s*[=:]\s*['\"]?([^\s'\"]+)", re.IGNORECASE), '********'),
        (re.compile(r'\bsk-[a-zA-Z0-9]{20,}\b'), '********'),
    ]

    def sanitize(self, text: str) -> str:
        for pattern, replacement in self._PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text


def sanitize_string(text: str) -> str:
    """Convenience function: sanitize a string using PIIFilter."""
    return PIIFilter().sanitize(text)

