"""Environment adapter registry."""
from __future__ import annotations

from typing import Any


class EnvironmentRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}
        self._state_compilers: dict[str, Any] = {}

    def register(self, adapter: Any) -> None:
        environment_id = getattr(adapter, "environment_id", "")
        if not environment_id:
            raise ValueError("adapter_missing_environment_id")
        self._adapters[environment_id] = adapter
        compiler = getattr(adapter, "state_compiler", None)
        if compiler is not None:
            self.register_state_compiler(environment_id, compiler)

    def register_state_compiler(self, environment_id: str, compiler_or_factory: Any) -> None:
        if not environment_id:
            raise ValueError("environment_id_required")
        if compiler_or_factory is None:
            raise ValueError("compiler_required")
        self._state_compilers[str(environment_id)] = compiler_or_factory

    def get(self, environment_id: str) -> Any:
        return self._adapters[environment_id]

    def get_state_compiler(self, environment_id: str) -> Any | None:
        compiler = self._state_compilers.get(environment_id)
        if compiler is None:
            family = environment_id.split(":", 1)[0] if ":" in environment_id else environment_id
            compiler = self._state_compilers.get(family)
        if compiler is None:
            return None
        return compiler() if callable(compiler) and not hasattr(compiler, "compile") else compiler

    def list_ids(self) -> list[str]:
        return sorted(self._adapters)


_REGISTRY = EnvironmentRegistry()


def get_environment_registry() -> EnvironmentRegistry:
    return _REGISTRY


__all__ = ["EnvironmentRegistry", "get_environment_registry"]
