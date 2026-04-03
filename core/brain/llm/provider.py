from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        """Generate a text response from the LLM."""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Generate a structured JSON response from the LLM."""
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs):
        """Generate a stream of ChatStreamEvent objects."""
        pass

    def check_health(self) -> bool:
        """Check if the provider is available and working."""
        return True