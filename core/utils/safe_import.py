import importlib
import logging
from types import SimpleNamespace
from typing import Any, Union

logger = logging.getLogger("aura.safe_import")

def safe_import(name: str, optional: bool = False) -> Any:
    """
    Try to import `name`. Returns module object if found,
    otherwise returns a dummy module with `__missing__` metadata.
    """
    try:
        mod = importlib.import_module(name)
        return mod
    except ImportError as e:
        logger.warning(f"safe_import: missing '{name}': {e}")
        if optional:
            # return a simple dummy to avoid attribute errors
            dummy = SimpleNamespace(__missing__=True, __name__=name)
            return dummy
        raise ImportError(f"Critical dependency '{name}' is missing and not optional.") from e

async def async_safe_import(name: str, optional: bool = False) -> Any:
    """Async wrapper for safe_import to prevent event loop blocking."""
    import asyncio
    # Use run_in_executor to avoid blocking the event loop during heavy imports
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, safe_import, name, optional)
def is_missing(module: Any) -> bool:
    """Check if a module returned by safe_import is actually missing."""
    return hasattr(module, "__missing__")
