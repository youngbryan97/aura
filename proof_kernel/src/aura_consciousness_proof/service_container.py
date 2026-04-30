from __future__ import annotations
"""A tiny service registry for the standalone proof kernel."""


from typing import Any


class ServiceContainer:
    _services: dict[str, Any] = {}

    @classmethod
    def register_instance(cls, name: str, instance: Any) -> None:
        cls._services[name] = instance

    @classmethod
    def get(cls, name: str, default: Any = None) -> Any:
        return cls._services.get(name, default)

    @classmethod
    def clear(cls) -> None:
        cls._services.clear()
