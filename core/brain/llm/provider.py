from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
        """Generate a text response from the LLM."""
        raise RuntimeError(f"{type(self).__name__}.generate_text must be implemented by a provider")

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON response from the LLM."""
        raise RuntimeError(f"{type(self).__name__}.generate_json must be implemented by a provider")

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Generate a stream of ChatStreamEvent objects."""
        raise RuntimeError(f"{type(self).__name__}.generate_stream must be implemented by a provider")

    def check_health(self) -> bool:
        """Check if the provider is available and working."""
        return True
