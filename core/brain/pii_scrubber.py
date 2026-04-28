"""PII Scrubber — Removes personal identifiers before cloud routing.

When Aura falls back to cloud inference (Gemini, etc.), the system prompt
may contain personal information from biography_private.json: real names,
trust scores, relationship labels, and other PII that should not leave
the local machine.

This module strips that data before the prompt is sent to external
endpoints, replacing it with generic placeholders that preserve the
conversational context without leaking identity information.

The scrubber is intentionally aggressive — it's better to lose some
personality context in cloud responses than to transmit real PII to
third-party inference infrastructure.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("Aura.PIIScrubber")

__all__ = ["scrub_pii_for_cloud", "get_pii_patterns"]

# Patterns that indicate PII-bearing content in system prompts
_PII_SECTION_MARKERS = (
    "CORE IDENTITY:",
    "SHARED HISTORY:",
    "KINSHIP:",
    "biography_private",
    "FamilyLegacy",
)

# Regex patterns for common PII structures in Aura's prompts
_PII_PATTERNS = [
    # Trust scores: "trust": 0.92, trust=0.92, trust: 0.92
    (re.compile(r'"?trust"?\s*[:=]\s*\d+\.\d+', re.IGNORECASE), '"trust": [REDACTED]'),
    # Relationship labels: "relation": "Architect / Friend / Equal"
    (re.compile(r'"?relation"?\s*[:=]\s*"[^"]*"', re.IGNORECASE), '"relation": "[REDACTED]"'),
    # Known entities blocks: known_entities["name"] = {...}
    (re.compile(r'known_entities\[["\'][^"\']+["\']\]\s*=\s*\{[^}]*\}', re.IGNORECASE), 'known_entities["user"] = {[REDACTED]}'),
    # Relationship graph blocks
    (re.compile(r'relationship_graph\[["\'][^"\']+["\']\]\s*=\s*\{[^}]*\}', re.IGNORECASE), 'relationship_graph["user"] = {[REDACTED]}'),
    # "name: warm" style trust indicators in prompts
    (re.compile(r'\b\w+:\s*(?:warm|trusted|sovereign|friend|equal|architect)\b', re.IGNORECASE), '[user]: [REDACTED]'),
]


def _load_private_names() -> list[str]:
    """Load real names from biography_private.json for targeted redaction."""
    try:
        import json
        from core.config import config
        config_path = config.paths.home_dir / "biography_private.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            names = []
            creator_name = data.get("creator_name", "")
            if creator_name and len(creator_name) > 1:
                names.append(creator_name)
            for kin in data.get("kin", []):
                name = kin.get("name", "")
                if name and len(name) > 1:
                    names.append(name)
            return names
    except Exception:
        pass  # no-op: intentional
    return []


_cached_names: Optional[list[str]] = None


def _get_private_names() -> list[str]:
    """Cached loader for private names."""
    global _cached_names
    if _cached_names is None:
        _cached_names = _load_private_names()
    return _cached_names


def scrub_pii_for_cloud(text: str) -> str:
    """Remove personal identifiers from text before sending to cloud.

    Replaces:
    - Real names from biography_private.json with "the user"
    - Trust scores with [REDACTED]
    - Relationship labels with [REDACTED]
    - Known entity blocks with generic placeholders
    - Entire CORE IDENTITY / SHARED HISTORY / KINSHIP sections with a
      generic "You have a positive relationship with the user" line

    Args:
        text: The system prompt or message content to scrub.

    Returns:
        Scrubbed text safe for cloud transmission.
    """
    if not text:
        return text

    scrubbed = text

    # Replace real names with "the user"
    for name in _get_private_names():
        if name in scrubbed:
            scrubbed = scrubbed.replace(name, "the user")
            # Also replace lowercase/title variants
            scrubbed = scrubbed.replace(name.lower(), "the user")
            scrubbed = scrubbed.replace(name.title(), "the user")

    # Apply regex patterns
    for pattern, replacement in _PII_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)

    # Replace entire PII sections with a generic summary
    for marker in _PII_SECTION_MARKERS:
        if marker in scrubbed:
            # Find the line containing the marker and replace it
            lines = scrubbed.split("\n")
            cleaned_lines = []
            skip_section = False
            for line in lines:
                if marker in line:
                    skip_section = True
                    cleaned_lines.append(
                        "CONTEXT: You have a positive working relationship with the user."
                    )
                    continue
                if skip_section and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    skip_section = False
                if not skip_section:
                    cleaned_lines.append(line)
            scrubbed = "\n".join(cleaned_lines)

    return scrubbed


def get_pii_patterns() -> list[tuple[re.Pattern, str]]:
    """Return the PII patterns for external testing/validation."""
    return list(_PII_PATTERNS)
