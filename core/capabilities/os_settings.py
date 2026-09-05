"""core/capabilities/os_settings.py — General OS Settings Adapter
==================================================================
Not just wallpaper — any reversible OS setting.

Every change saves previous state for rollback. All operations go
through HostAutomationProvider for receipts and governance.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

if TYPE_CHECKING:
    from core.capabilities.host_automation import AutomationReceipt

logger = logging.getLogger("Aura.OSSettings")


@dataclass
class SettingChange:
    """Record of a setting change for rollback."""
    setting: str
    previous_value: str
    new_value: str
    success: bool
    timestamp: float = field(default_factory=time.time)


class OSSettingsAdapter:
    """General system settings manipulation.

    Every change saves the previous state so it can be rolled back.
    """

    def __init__(self) -> None:
        self._changes: list[SettingChange] = []
        self._max_changes = 100
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("os_settings", self, required=False)
        self._started = True
        logger.info("OSSettingsAdapter ONLINE")

    # ------------------------------------------------------------------
    # Wallpaper
    # ------------------------------------------------------------------

    async def get_wallpaper(self) -> str:
        """Get the current desktop wallpaper path."""
        try:
            proc = await get_subprocess_gateway().spawn_async(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get picture of first desktop',
                ],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="os_settings.get_wallpaper",
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode("utf-8", errors="replace").strip() if stdout else ""
        except (OSError, RuntimeError, asyncio.TimeoutError) as e:
            record_degradation("os_settings.wallpaper_get", e)
            return ""

    async def set_wallpaper(self, image_path: str) -> "AutomationReceipt":
        """Set the desktop wallpaper.

        Saves previous wallpaper for rollback.
        Verifies the change after setting.
        """
        from core.capabilities.host_automation import AutomationReceipt, AppleScriptRunner

        start = time.time()
        path = Path(image_path).expanduser().resolve()

        # Validate the image exists
        if not path.exists():
            return AutomationReceipt(
                action="set_wallpaper", target=str(path),
                adapter="osascript", success=False,
                error=f"Image not found: {path}",
                duration_ms=(time.time() - start) * 1000,
            )

        # Save current wallpaper for rollback
        previous = await self.get_wallpaper()

        # Set wallpaper via AppleScript. The POSIX file form is the
        # reliable one on modern macOS (Sonoma+/Tahoe); the bare path
        # string form silently no-ops on some versions.
        escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
        script = (
            'tell application "System Events" to set picture of every desktop '
            f'to POSIX file "{escaped}"'
        )
        receipt = await AppleScriptRunner.run(script, timeout=10.0)
        receipt.action = "set_wallpaper"
        receipt.target = str(path)

        if receipt.success:
            # Record for rollback
            self._changes.append(SettingChange(
                setting="wallpaper",
                previous_value=previous,
                new_value=str(path),
                success=True,
            ))
            if len(self._changes) > self._max_changes:
                self._changes = self._changes[-self._max_changes:]

            # Verify — read-back is racy on modern macOS (the wallpaper
            # store reports `missing value` for a moment after a set), so
            # poll until it names our file or the bounded budget elapses.
            current = ""
            confirmed = False
            for _ in range(8):
                await asyncio.sleep(0.5)
                current = await self.get_wallpaper()
                if str(path) in current or path.name in current:
                    confirmed = True
                    break
            if confirmed:
                receipt.result = f"Wallpaper set to {path.name}"
            else:
                receipt.success = False
                receipt.error = f"Wallpaper read-back did not confirm (current={current})"

        # Log receipt
        try:
            from core.runtime.life_trace import get_life_trace
            get_life_trace().record(
                event_type="action_executed",
                origin="os_settings",
                action_taken={"action": "set_wallpaper", "path": str(path)[:200]},
                result={"success": receipt.success, "previous": previous[:200]},
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("os_settings.life_trace", exc)

        return receipt

    async def restore_wallpaper(self) -> bool:
        """Restore the wallpaper to its previous value."""
        wallpaper_changes = [c for c in reversed(self._changes) if c.setting == "wallpaper"]
        if not wallpaper_changes:
            return False
        previous = wallpaper_changes[0].previous_value
        if previous:
            receipt = await self.set_wallpaper(previous)
            return receipt.success
        return False

    # ------------------------------------------------------------------
    # Appearance mode
    # ------------------------------------------------------------------

    async def get_appearance_mode(self) -> str:
        """Get current appearance mode: 'light', 'dark', or 'auto'."""
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="os_settings.get_appearance_mode",
                accelerator_capability="none",
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            if proc.returncode == 0 and stdout:
                return "dark" if "dark" in stdout.decode().lower() else "light"
            return "light"  # Default if key doesn't exist
        except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
            record_degradation("os_settings.appearance_get", exc)
            return "unknown"

    async def set_appearance_mode(self, mode: str) -> bool:
        """Set appearance mode ('light' or 'dark')."""
        previous = await self.get_appearance_mode()
        try:
            if mode.lower() == "dark":
                script = '''
                    tell application "System Events"
                        tell appearance preferences
                            set dark mode to true
                        end tell
                    end tell
                '''
            else:
                script = '''
                    tell application "System Events"
                        tell appearance preferences
                            set dark mode to false
                        end tell
                    end tell
                '''

            from core.capabilities.host_automation import AppleScriptRunner
            receipt = await AppleScriptRunner.run(script, timeout=5.0)
            if receipt.success:
                self._changes.append(SettingChange(
                    setting="appearance_mode",
                    previous_value=previous,
                    new_value=mode,
                    success=True,
                ))
            return receipt.success
        except (ImportError, RuntimeError) as e:
            record_degradation("os_settings.appearance", e)
            return False

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    async def get_volume(self) -> int:
        """Get current system volume (0-100)."""
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["osascript", "-e", "output volume of (get volume settings)"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="os_settings.get_volume",
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            return int(stdout.decode().strip()) if stdout else 50
        except (OSError, RuntimeError, asyncio.TimeoutError, ValueError) as exc:
            record_degradation("os_settings.volume_get", exc)
            return 50

    async def set_volume(self, level: int) -> bool:
        """Set system volume (0-100)."""
        level = max(0, min(100, level))
        previous = await self.get_volume()
        try:
            from core.capabilities.host_automation import AppleScriptRunner
            receipt = await AppleScriptRunner.run(
                f'set volume output volume {level}', timeout=3.0
            )
            if receipt.success:
                self._changes.append(SettingChange(
                    setting="volume",
                    previous_value=str(previous),
                    new_value=str(level),
                    success=True,
                ))
            return receipt.success
        except (ImportError, RuntimeError) as exc:
            record_degradation("os_settings.volume_set", exc)
            return False

    # ------------------------------------------------------------------
    # General rollback
    # ------------------------------------------------------------------

    async def rollback_last(self) -> Dict[str, Any]:
        """Rollback the last setting change."""
        if not self._changes:
            return {"success": False, "error": "No changes to rollback"}

        change = self._changes.pop()
        if change.setting == "wallpaper":
            success = await self.restore_wallpaper()
        elif change.setting == "appearance_mode":
            success = await self.set_appearance_mode(change.previous_value)
        elif change.setting == "volume":
            success = await self.set_volume(int(change.previous_value))
        else:
            success = False

        return {
            "success": success,
            "setting": change.setting,
            "restored_to": change.previous_value,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "changes": len(self._changes),
            "recent": [
                {"setting": c.setting, "value": c.new_value[:50]}
                for c in self._changes[-5:]
            ],
        }


_instance: Optional[OSSettingsAdapter] = None


def get_os_settings() -> OSSettingsAdapter:
    global _instance
    if _instance is None:
        _instance = OSSettingsAdapter()
    return _instance


__all__ = ["OSSettingsAdapter", "get_os_settings"]
