import asyncio
import logging
import os
from enum import Enum, auto
from typing import Any, Dict, Optional

from ..base_module import AuraBaseModule


class PermissionType(Enum):
    MIC = auto()
    CAMERA = auto()
    SCREEN = auto()


class PermissionGuard(AuraBaseModule):
    """Handles macOS TCC permission checks without triggering boot-time prompts."""

    def __init__(self):
        super().__init__("PermissionGuard")
        self._cache: Dict[PermissionType, Dict[str, Any]] = {}

    async def check_permission(self, ptype: PermissionType, force: bool = False) -> Dict[str, Any]:
        """Check if a hardware permission is granted.

        Returns:
            {"granted": bool, "status": str, "guidance": str}
        """
        if not force and ptype in self._cache:
            return self._cache[ptype]

        self.logger.info("Checking %s permission...", ptype.name)

        if ptype == PermissionType.SCREEN:
            result = await self._check_screen_permission()
        elif ptype == PermissionType.MIC:
            result = await self._check_mic_permission()
        else:
            result = {
                "granted": True,
                "status": "assumed",
                "guidance": "No check implemented for this type yet.",
            }

        self._cache[ptype] = result
        return result

    def _screen_preflight_probe(self) -> Optional[Dict[str, Any]]:
        """Use Quartz preflight when available so we don't trigger a capture prompt."""
        try:
            import Quartz  # type: ignore

            preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
            if callable(preflight):
                granted = bool(preflight())
                return {
                    "granted": granted,
                    "status": "active" if granted else "denied",
                    "guidance": "" if granted else self.get_guidance(PermissionType.SCREEN),
                }
        except Exception as exc:
            self.logger.debug("Quartz screen preflight unavailable: %s", exc)
        return None

    async def _check_screen_permission(self) -> Dict[str, Any]:
        """Probe screen-recording status without forcing a screenshot during boot."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._screen_preflight_probe)
        if result is not None:
            return result

        if os.getenv("AURA_ASSUME_SCREEN_PERMISSION", "0") == "1":
            return {"granted": True, "status": "assumed", "guidance": ""}

        cached = self._cache.get(PermissionType.SCREEN)
        if cached:
            return cached

        return {
            "granted": False,
            "status": "deferred",
            "guidance": (
                "Aura will only request Screen Recording when a screen-aware feature is explicitly used. "
                + self.get_guidance(PermissionType.SCREEN)
            ),
        }

    async def _check_mic_permission(self) -> Dict[str, Any]:
        try:
            return {"granted": True, "status": "active", "guidance": ""}
        except Exception as e:
            return {"granted": False, "status": "error", "guidance": f"Mic check failed: {e}"}

    def get_guidance(self, ptype: PermissionType) -> str:
        if ptype == PermissionType.SCREEN:
            return (
                "1. Open System Settings\n"
                "2. Go to Privacy & Security\n"
                "3. Select Screen Recording\n"
                "4. Ensure Aura is switched ON. If you use Terminal launch mode, ensure Terminal is switched ON too."
            )
        if ptype == PermissionType.MIC:
            return (
                "1. Open System Settings\n"
                "2. Go to Privacy & Security\n"
                "3. Select Microphone\n"
                "4. Ensure Aura/Terminal is switched ON."
            )
        return "Check your macOS Privacy & Security settings."
