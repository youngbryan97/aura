"""core/capabilities/clipboard_manager.py — Clipboard Operations with History
==============================================================================
Manages clipboard state for paste operations with undo/restore.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_registry import register_runtime_service
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.ClipboardManager")


class ClipboardManager:
    """Clipboard operations with history and undo."""

    def __init__(self, max_history: int = 20) -> None:
        self._history: deque[str] = deque(maxlen=max_history)
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        register_runtime_service("clipboard_manager", self, required=False)
        self._started = True
        logger.info("ClipboardManager ONLINE")

    async def get(self) -> str:
        """Get current clipboard content."""
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["pbpaste"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="clipboard_manager.get",
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            return stdout.decode("utf-8", errors="replace") if stdout else ""
        except (TimeoutError, OSError, RuntimeError) as exc:
            record_degradation("clipboard.get", exc)
            return ""

    async def set(self, text: str) -> bool:
        """Set clipboard content, saving current for undo."""
        try:
            # Save current to history
            current = await self.get()
            if current:
                self._history.append(current)

            proc = await get_subprocess_gateway().spawn_async(
                ["pbcopy"],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="clipboard_manager.set",
                accelerator_capability="none",
            )
            await asyncio.wait_for(
                proc.communicate(input=text.encode("utf-8")),
                timeout=2.0,
            )
            return proc.returncode == 0
        except (TimeoutError, OSError, RuntimeError) as e:
            record_degradation("clipboard.set", e)
            return False

    async def paste(self) -> bool:
        """Simulate Cmd+V paste."""
        try:
            from core.capabilities.host_automation import AppleScriptRunner
            receipt = await AppleScriptRunner.run(
                'tell application "System Events" to keystroke "v" using command down',
                timeout=3.0,
            )
            return receipt.success
        except (ImportError, RuntimeError) as exc:
            record_degradation("clipboard.paste", exc)
            return False

    async def set_and_paste(self, text: str) -> bool:
        """Set clipboard and immediately paste."""
        if not await self.set(text):
            return False
        await asyncio.sleep(0.1)
        return await self.paste()

    async def restore_previous(self) -> bool:
        """Restore the previous clipboard content."""
        if not self._history:
            return False
        previous = self._history.pop()
        return await self.set(previous)

    def get_status(self) -> dict[str, Any]:
        return {
            "history_size": len(self._history),
        }


_instance: ClipboardManager | None = None


def get_clipboard_manager() -> ClipboardManager:
    global _instance
    if _instance is None:
        _instance = ClipboardManager()
    return _instance


__all__ = ["ClipboardManager", "get_clipboard_manager"]
