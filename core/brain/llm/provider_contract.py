"""core/brain/llm/provider_contract.py — Model Provider Contract.

Exposes standard specifications, metadata, capabilities, resource costs,
and health checks for model providers used by Aura's routed inference lanes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelCapabilities:
    """Modality and steering feature support matrix."""
    supports_vision: bool = False
    supports_tool_calling: bool = False
    supports_steering: bool = False
    supports_recurrent_depth: bool = False
    modalities: tuple[str, ...] = ("text",)


@dataclass
class ProviderContract:
    """Standard specifications that every model provider must expose."""
    provider_name: str
    model_name: str
    context_limit: int
    latency_estimate_ms: float
    memory_cost_gb: float
    is_local: bool
    privacy_level: str              # "local_isolated", "private_api", "public_api"
    failure_mode: str               # "fail_closed", "fail_over_degraded", "fallback"
    health_status: str              # "healthy", "degraded", "offline"
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "context_limit": self.context_limit,
            "latency_estimate_ms": self.latency_estimate_ms,
            "memory_cost_gb": self.memory_cost_gb,
            "is_local": self.is_local,
            "privacy_level": self.privacy_level,
            "failure_mode": self.failure_mode,
            "health_status": self.health_status,
            "capabilities": {
                "supports_vision": self.capabilities.supports_vision,
                "supports_tool_calling": self.capabilities.supports_tool_calling,
                "supports_steering": self.capabilities.supports_steering,
                "supports_recurrent_depth": self.capabilities.supports_recurrent_depth,
                "modalities": list(self.capabilities.modalities),
            }
        }


class ContractedLLMProvider(ABC):
    """Abstract interface enforcing the Model-Agnostic Provider Contract."""

    @abstractmethod
    def get_contract(self) -> ProviderContract:
        """Expose the contract specifications."""
        raise NotImplementedError

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Execute text generation conforming to contract constraints."""
        raise NotImplementedError
