"""core/brain/prompt_registry.py — Centralized Prompt Management

All system prompts in one place with version tracking.
Prevents prompt drift by making it easy to audit, version, and refactor
all prompts that are injected into LLM calls.

Usage:
    from core.brain.prompt_registry import prompt_registry
    
    system_prompt = prompt_registry.get("aura_identity")
    prompt_registry.update("aura_identity", new_text)  # Auto-increments version
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.PromptRegistry")


@dataclass
class PromptEntry:
    """A versioned prompt entry."""
    name: str
    text: str
    version: int = 1
    created_at: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    category: str = "system"  # "system", "reflection", "tool", "persona"
    description: str = ""


class PromptRegistry:
    """Centralized prompt management with version tracking."""

    def __init__(self):
        self._prompts: Dict[str, PromptEntry] = {}
        self._history: Dict[str, List[Dict]] = {}  # name -> [{version, text, timestamp}]

    def register(self, name: str, text: str, category: str = "system",
                 description: str = "") -> None:
        """Register a prompt. If already registered, this is a no-op."""
        if name in self._prompts:
            return
        self._prompts[name] = PromptEntry(
            name=name, text=text, category=category, description=description,
        )
        self._history[name] = [{"version": 1, "text": text[:200], "timestamp": time.time()}]
        logger.debug("Registered prompt: %s (category=%s)", name, category)

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get the current text of a named prompt."""
        entry = self._prompts.get(name)
        if entry:
            return entry.text
        return default

    def update(self, name: str, text: str) -> int:
        """Update a prompt's text. Auto-increments version.
        
        Returns:
            New version number
        """
        entry = self._prompts.get(name)
        if not entry:
            self.register(name, text)
            return 1

        entry.version += 1
        entry.text = text
        entry.last_modified = time.time()

        # Record in history
        if name not in self._history:
            self._history[name] = []
        self._history[name].append({
            "version": entry.version,
            "text": text[:200],
            "timestamp": time.time(),
        })
        # Keep only last 20 versions
        self._history[name] = self._history[name][-20:]

        logger.info("Prompt '%s' updated to v%d", name, entry.version)
        return entry.version

    def list_all(self) -> List[Dict]:
        """List all registered prompts with metadata."""
        return [
            {
                "name": e.name,
                "category": e.category,
                "version": e.version,
                "description": e.description,
                "text_preview": e.text[:100] + "..." if len(e.text) > 100 else e.text,
                "last_modified": e.last_modified,
            }
            for e in self._prompts.values()
        ]

    def get_history(self, name: str) -> List[Dict]:
        """Get version history for a prompt."""
        return self._history.get(name, [])

    def search(self, query: str) -> List[str]:
        """Search prompt texts for a query string."""
        results = []
        for name, entry in self._prompts.items():
            if query.lower() in entry.text.lower():
                results.append(name)
        return results


# Singleton
prompt_registry = PromptRegistry()

# ── Pre-register known system prompts ────────────────────────────────
def _bootstrap_prompts():
    """Register all known system prompts on first import."""
    try:
        from core.brain.aura_persona import AURA_IDENTITY, REFLECTION_PROMPT, AUTONOMOUS_THOUGHT_PROMPT
        prompt_registry.register(
            "aura_identity", AURA_IDENTITY,
            category="persona",
            description="Primary identity prompt injected into every LLM call",
        )
        prompt_registry.register(
            "reflection", REFLECTION_PROMPT,
            category="reflection",
            description="Used when Aura reflects on recent conversations",
        )
        prompt_registry.register(
            "autonomous_thought", AUTONOMOUS_THOUGHT_PROMPT,
            category="reflection",
            description="Drives autonomous thinking when no one is talking to Aura",
        )
    except ImportError:
        logger.debug("aura_persona not available for prompt bootstrap")

_bootstrap_prompts()