import asyncio
import logging
import os
import subprocess
import time
from enum import Enum, auto
from typing import Any, Dict, Optional

from ..base_module import AuraBaseModule


class PermissionType(Enum):
    MIC = auto()
    CAMERA = auto()
    SCREEN = auto()
    ACCESSIBILITY = auto()
    AUTOMATION = auto()


class PermissionGuard(AuraBaseModule):
    """Handles macOS TCC permission checks without triggering boot-time prompts."""

    def __init__(self):
        super().__init__("PermissionGuard")
        self._cache: Dict[PermissionType, Dict[str, Any]] = {}
        self._cache_ts: Dict[PermissionType, float] = {}
        # Granted TCC permissions don't get revoked silently, so we cache for
        # 5 minutes once active. Denied/deferred entries still re-probe in 15s.
        self._cache_ttl_s: float = 15.0
        self._cache_ttl_granted_s: float = 300.0
        self._force_refresh_floor_s: float = 20.0

    async def check_permission(self, ptype: PermissionType, force: bool = False) -> Dict[str, Any]:
        """Check if a hardware permission is granted.

        Returns:
            {"granted": bool, "status": str, "guidance": str}
        """
        now = time.monotonic()
        cached = self._cache.get(ptype)
        cached_at = float(self._cache_ts.get(ptype, 0.0) or 0.0)
        if cached is not None:
            cache_age = max(0.0, now - cached_at)
            ttl = (
                self._cache_ttl_granted_s
                if cached.get("granted")
                else self._cache_ttl_s
            )
            if force:
                if cache_age < self._force_refresh_floor_s:
                    return cached
            elif cache_age < ttl:
                return cached

        # Env-override path: when the user has granted permission to a parent
        # process Aura can't see (different terminal host), let them assert it
        # with AURA_ASSUME_*_PERMISSION=1 so vision/automation aren't blocked
        # by an inherited TCC identity.
        env_override = self._env_override(ptype)
        if env_override is not None:
            self._cache[ptype] = env_override
            self._cache_ts[ptype] = now
            return env_override

        self.logger.debug("Checking %s permission...", ptype.name)

        if ptype == PermissionType.SCREEN:
            result = await self._check_screen_permission()
        elif ptype == PermissionType.MIC:
            result = await self._check_mic_permission()
        elif ptype == PermissionType.ACCESSIBILITY:
            result = await self._check_accessibility_permission()
        elif ptype == PermissionType.AUTOMATION:
            result = await self._check_automation_permission()
        else:
            result = {
                "granted": True,
                "status": "assumed",
                "guidance": "No check implemented for this type yet.",
            }

        self._cache[ptype] = result
        self._cache_ts[ptype] = now
        return result

    _ENV_OVERRIDE_KEYS: Dict[PermissionType, str] = {
        PermissionType.SCREEN: "AURA_ASSUME_SCREEN_PERMISSION",
        PermissionType.ACCESSIBILITY: "AURA_ASSUME_ACCESSIBILITY_PERMISSION",
        PermissionType.AUTOMATION: "AURA_ASSUME_AUTOMATION_PERMISSION",
        PermissionType.MIC: "AURA_ASSUME_MIC_PERMISSION",
        PermissionType.CAMERA: "AURA_ASSUME_CAMERA_PERMISSION",
    }

    def _env_override(self, ptype: PermissionType) -> Optional[Dict[str, Any]]:
        """Return a granted result if the user has explicitly asserted the
        permission via env var. Lets users bypass TCC parent-process
        inheritance when launching from a non-standard host."""
        key = self._ENV_OVERRIDE_KEYS.get(ptype)
        if not key:
            return None
        if os.getenv(key, "0") != "1":
            return None
        return {"granted": True, "status": "asserted_env", "guidance": ""}

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

    def _accessibility_preflight_probe(self) -> Optional[Dict[str, Any]]:
        """Use AXIsProcessTrusted without prompting so desktop-control checks stay passive."""
        try:
            import ctypes

            framework = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
            app_services = ctypes.CDLL(framework)
            probe = app_services.AXIsProcessTrusted
            probe.restype = ctypes.c_bool
            granted = bool(probe())
            return {
                "granted": granted,
                "status": "active" if granted else "denied",
                "guidance": "" if granted else self.get_guidance(PermissionType.ACCESSIBILITY),
            }
        except Exception as exc:
            self.logger.debug("Accessibility preflight unavailable: %s", exc)
        return None

    def _automation_preflight_probe(self) -> Dict[str, Any]:
        """Probe Apple Events access to System Events with a harmless frontmost-app query."""
        script = 'tell application "System Events" to get name of first application process whose frontmost is true'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:
            return {
                "granted": False,
                "status": "error",
                "guidance": f"Automation probe failed: {exc}",
            }

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode == 0:
            payload: Dict[str, Any] = {
                "granted": True,
                "status": "active",
                "guidance": "",
            }
            if stdout:
                payload["detail"] = stdout[:160]
            return payload

        normalized = stderr.lower()
        if "not authorized to send apple events" in normalized or "(-1743)" in normalized:
            return {
                "granted": False,
                "status": "denied",
                "guidance": self.get_guidance(PermissionType.AUTOMATION),
                "detail": stderr[:240],
            }
        return {
            "granted": False,
            "status": "error",
            "guidance": stderr[:240] or "System Events automation probe failed.",
        }

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

    async def _check_accessibility_permission(self) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._accessibility_preflight_probe)
        if result is not None:
            return result
        return {
            "granted": False,
            "status": "deferred",
            "guidance": self.get_guidance(PermissionType.ACCESSIBILITY),
        }

    async def _check_automation_permission(self) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._automation_preflight_probe)

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
        if ptype == PermissionType.ACCESSIBILITY:
            return (
                "1. Open System Settings\n"
                "2. Go to Privacy & Security\n"
                "3. Select Accessibility\n"
                "4. Ensure Aura is switched ON. If you launched from Terminal or Codex, enable that host app too."
            )
        if ptype == PermissionType.AUTOMATION:
            return (
                "1. Open System Settings\n"
                "2. Go to Privacy & Security\n"
                "3. Select Automation\n"
                "4. Allow Aura/Terminal/Codex to control System Events if you want desktop text and menu-bar access."
            )
        return "Check your macOS Privacy & Security settings."


_SHARED_PERMISSION_GUARD: Optional[PermissionGuard] = None


def get_permission_guard() -> PermissionGuard:
    """Return a shared permission guard so passive probes can reuse cache state."""
    global _SHARED_PERMISSION_GUARD

    try:
        from core.container import ServiceContainer

        existing = ServiceContainer.get("permission_guard", default=None)
        if existing is not None:
            return existing
    except Exception:
        ServiceContainer = None  # type: ignore[assignment]

    if _SHARED_PERMISSION_GUARD is None:
        _SHARED_PERMISSION_GUARD = PermissionGuard()

    try:
        if ServiceContainer is not None:
            ServiceContainer.register_instance("permission_guard", _SHARED_PERMISSION_GUARD, required=False)
    except Exception:
        pass

    return _SHARED_PERMISSION_GUARD
