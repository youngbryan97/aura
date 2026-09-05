from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
        """Generate a text response from the LLM."""
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON response from the LLM."""
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Generate a stream of ChatStreamEvent objects."""
        raise NotImplementedError

    def check_health(self) -> bool:
        """Whether this provider is available and working.

        Returns False on the base class, and that is the point. It used to
        return True unconditionally, so every provider that never overrode
        it — `NucleusManager`, the primary local model lane — reported
        healthy without probing configuration, dependencies, model identity,
        readiness, or a single successful call. `FallbackLLMClient` selects
        providers on exactly this answer, so a lane with nothing loaded
        stayed at the front of the chain.

        A provider that has not implemented a health check has not
        established that it is healthy. Overriding this is how a provider
        says otherwise.
        """
        logger.warning(
            "%s does not implement check_health(); reporting unhealthy "
            "rather than certifying a provider that was never probed.",
            type(self).__name__,
        )
        return False
