"""formal interface for cognitive augmentation.
Defines the contract for extending the CognitiveEngine's reasoning process.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("Aura.CognitiveInterface")

class AbstractCognitiveAugmentor(ABC):
    """Abstract base class for all cognitive augmentors.
    Augmentors plug into the CognitiveEngine to provide additional context,
    system prompt enrichment, and mid-inference hooks.
    """

    @abstractmethod
    def prepare_context(self, objective: str, context: dict[str, Any]) -> dict[str, Any]:
        """Hook to inject data into the thinking context before prompt assembly."""
        return context

    @abstractmethod
    def enrich_prompt(self, system_prompt: str, context: dict[str, Any]) -> str:
        """Hook to append content to the system prompt."""
        return system_prompt