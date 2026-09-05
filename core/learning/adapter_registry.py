"""core/learning/adapter_registry.py
Maintains registered candidate and active model adapters.
"""


class AdapterRegistry:
    """Tracks path references of active and candidate model adapters."""

    def __init__(self):
        self._adapters: dict[str, str] = {}
        self._active: str = "baseline"

    def register_adapter(self, name: str, file_path: str) -> None:
        self._adapters[name] = file_path

    def activate_adapter(self, name: str) -> None:
        if name in self._adapters or name == "baseline":
            self._active = name

    def get_active_adapter_path(self) -> str:
        return self._adapters.get(self._active, "baseline")
