"""core/morphogenesis/state_migrator.py — Version Registry and State Migrator Proxy

Manages dynamically reloaded or mutated morphogenetic cell components, ensuring that
active object state payloads are marshalled cleanly onto mutated schemas to prevent 
in-flight type mismatch drift during AST hot-reloading.
"""
import logging
from typing import Dict, Any, Type, Optional

logger = logging.getLogger("Aura.Morphogenesis.Migrator")


class VersionRegistry:
    """Manages versioned allocations of autopoietic components to prevent in-flight type drift."""
    _registry: Dict[str, Type] = {}
    _active_versions: Dict[str, int] = {}

    @classmethod
    def register_cell_version(cls, base_name: str, version: int, class_impl: Type):
        key = f"{base_name}_v{version}"
        cls._registry[key] = class_impl
        cls._active_versions[base_name] = max(cls._active_versions.get(base_name, 0), version)
        logger.info(f"🧬 Registered morphic variant: {key}")

    @classmethod
    def get_latest(cls, base_name: str) -> Optional[Type]:
        latest_ver = cls._active_versions.get(base_name)
        if latest_ver is None:
            return None
        return cls._registry.get(f"{base_name}_v{latest_ver}")

    @classmethod
    def clear(cls):
        """Clears registry (mostly for test cleanliness)."""
        cls._registry.clear()
        cls._active_versions.clear()


class MorphicStateProxy:
    """Transparent proxy that intercepts active calls and marshals state onto mutated schemas."""
    
    def __init__(self, base_name: str, initial_state: Dict[str, Any]):
        self._base_name = base_name
        self._underlying_state = initial_state
        self._current_version = 1

    def _sync_state(self) -> tuple[Optional[Type], int]:
        latest_class = VersionRegistry.get_latest(self._base_name)
        latest_version = VersionRegistry._active_versions.get(self._base_name, 1)
        
        if latest_class is None:
            return None, 1
            
        if latest_version > self._current_version:
            logger.warning(f"🔄 Migrating instance state from v{self._current_version} -> v{latest_version}")
            if hasattr(latest_class, "__migrate_state__"):
                self._underlying_state = latest_class.__migrate_state__(self._underlying_state)
            self._current_version = latest_version
            
        return latest_class, latest_version

    def __getattr__(self, name: str) -> Any:
        latest_class, _ = self._sync_state()
        if latest_class is None:
            if name in self._underlying_state:
                return self._underlying_state[name]
            raise AttributeError(f"MorphicStateProxy '{self._base_name}' has no attribute '{name}'")
            
        instance = latest_class(**self._underlying_state)
        attr = getattr(instance, name)
        
        if callable(attr):
            def wrapper(*args, **kwargs):
                latest_class_wrapper, _ = self._sync_state()
                if latest_class_wrapper is None:
                    raise RuntimeError(f"Latest version not registered for cell '{self._base_name}' during dynamic call.")
                instance_wrapper = latest_class_wrapper(**self._underlying_state)
                method = getattr(instance_wrapper, name)
                result = method(*args, **kwargs)
                
                # Sync structural mutations back to tracking payload
                if hasattr(instance_wrapper, "__dump_state__"):
                    self._underlying_state = instance_wrapper.__dump_state__()
                else:
                    self._underlying_state = instance_wrapper.__dict__.copy()
                return result
            return wrapper
        else:
            return attr
            
    def __setattr__(self, name: str, value: Any):
        if name in ("_base_name", "_underlying_state", "_current_version"):
            super().__setattr__(name, value)
        else:
            self._underlying_state[name] = value

    def __repr__(self) -> str:
        return f"<MorphicStateProxy for {self._base_name} (v{self._current_version})>"
