"""Environment adapter registry."""
from __future__ import annotations

from typing import Any


class EnvironmentRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}

    def register(self, adapter: Any) -> None:
        environment_id = getattr(adapter, "environment_id", "")
        if not environment_id:
            raise ValueError("adapter_missing_environment_id")
        self._adapters[environment_id] = adapter

    def get(self, environment_id: str) -> Any:
        return self._adapters[environment_id]

    def list_ids(self) -> list[str]:
        return sorted(self._adapters)


_REGISTRY = EnvironmentRegistry()


def get_environment_registry() -> EnvironmentRegistry:
    return _REGISTRY


__all__ = ["EnvironmentRegistry", "get_environment_registry"]
