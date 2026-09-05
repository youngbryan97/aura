"""Lazy PyAutoGUI loading for UI-control skills.

Avoid importing PyAutoGUI at module import time so skills can be registered,
listed, and instantiated even when display access is unavailable.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



from typing import Any

_PYAUTOGUI_MODULE: Any | None = None
_PYAUTOGUI_ERROR: Exception | None = None


def get_pyautogui() -> tuple[Any | None, Exception | None]:
    global _PYAUTOGUI_MODULE, _PYAUTOGUI_ERROR

    if _PYAUTOGUI_MODULE is not None or _PYAUTOGUI_ERROR is not None:
        return _PYAUTOGUI_MODULE, _PYAUTOGUI_ERROR

    try:
        from core.security.native_desktop_bridge import get_native_pyautogui

        native = get_native_pyautogui()
        if native is not None:
            _PYAUTOGUI_MODULE = native
            return _PYAUTOGUI_MODULE, None
    except (ImportError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        # Native bridge is preferred on macOS because its TCC identity matches
        # Aura.app.  PyAutoGUI remains the cross-platform fallback.
        _PYAUTOGUI_ERROR = exc

    try:
        import pyautogui as module

        module.FAILSAFE = True
        pause = float(getattr(module, "PAUSE", 0.0) or 0.0)
        if pause < 0.1:
            module.PAUSE = 0.1
        _PYAUTOGUI_MODULE = module
        _PYAUTOGUI_ERROR = None
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('_pyautogui_runtime', exc)
        _PYAUTOGUI_MODULE = None
        _PYAUTOGUI_ERROR = exc

    return _PYAUTOGUI_MODULE, _PYAUTOGUI_ERROR


def reset_pyautogui_cache() -> None:
    global _PYAUTOGUI_MODULE, _PYAUTOGUI_ERROR
    _PYAUTOGUI_MODULE = None
    _PYAUTOGUI_ERROR = None
