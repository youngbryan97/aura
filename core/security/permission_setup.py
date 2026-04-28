"""macOS permission setup helper.

Closes the "dispatch says the thing, permission denies it silently" path.
This module gives the runtime (and the user) a single place to see which
TCC permissions Aura needs, which ones are granted, and how to fix the
missing ones — including a deep-link into the right System Settings pane.

Everything is cached for the lifetime of the process, with an explicit
``refresh`` option for UI callers. Non-macOS platforms report all
permissions as N/A.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.security.permission_guard import PermissionType, get_permission_guard

logger = logging.getLogger(__name__)


# URL schemes that open the right pane in macOS System Settings.
_PANE_URLS: Dict[PermissionType, str] = {
    PermissionType.ACCESSIBILITY: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    PermissionType.AUTOMATION: "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
    PermissionType.SCREEN: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    PermissionType.MIC: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    PermissionType.CAMERA: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
}


@dataclass(frozen=True)
class PermissionStatus:
    name: str
    granted: bool
    available: bool
    guidance: str
    settings_url: Optional[str] = None
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "granted": self.granted,
            "available": self.available,
            "guidance": self.guidance,
            "settings_url": self.settings_url,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PermissionReport:
    platform: str
    supported: bool
    all_granted: bool
    statuses: List[PermissionStatus] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "supported": self.supported,
            "all_granted": self.all_granted,
            "missing": list(self.missing),
            "statuses": [s.as_dict() for s in self.statuses],
        }


async def check_all_permissions(*, refresh: bool = False) -> PermissionReport:
    system = platform.system()
    if system != "Darwin":
        return PermissionReport(platform=system, supported=False, all_granted=True, statuses=[], missing=[])

    guard = get_permission_guard()
    statuses: List[PermissionStatus] = []
    for ptype in (
        PermissionType.ACCESSIBILITY,
        PermissionType.AUTOMATION,
        PermissionType.SCREEN,
        PermissionType.MIC,
        PermissionType.CAMERA,
    ):
        try:
            result = await guard.check_permission(ptype, force=refresh)
            granted = bool(result.get("granted"))
            available = bool(result.get("available", True))
            detail = str(result.get("detail") or "")
        except Exception as exc:
            record_degradation('permission_setup', exc)
            logger.debug("Permission check failed for %s: %s", ptype, exc)
            granted = False
            available = False
            detail = f"check_failed: {type(exc).__name__}"
        statuses.append(
            PermissionStatus(
                name=ptype.name,
                granted=granted,
                available=available,
                guidance=guard.get_guidance(ptype),
                settings_url=_PANE_URLS.get(ptype),
                detail=detail,
            )
        )
    missing = [s.name for s in statuses if not s.granted and s.available]
    return PermissionReport(
        platform=system,
        supported=True,
        all_granted=not missing,
        statuses=statuses,
        missing=missing,
    )


def open_settings_pane(permission: str) -> bool:
    """Open System Settings at the pane required for ``permission``.

    Returns True on macOS when ``open`` was found and executed, False
    otherwise. Never raises.
    """
    if platform.system() != "Darwin":
        return False
    try:
        ptype = PermissionType[permission.upper()]
    except KeyError:
        logger.warning("Unknown permission name: %s", permission)
        return False
    url = _PANE_URLS.get(ptype)
    if not url:
        return False
    opener = shutil.which("open")
    if not opener:
        return False
    try:
        subprocess.Popen([opener, url])
        return True
    except Exception as exc:
        record_degradation('permission_setup', exc)
        logger.debug("Failed to open settings pane for %s: %s", permission, exc)
        return False


def format_report(report: PermissionReport) -> str:
    if not report.supported:
        return f"Platform {report.platform}: TCC permissions not applicable."
    if report.all_granted:
        return "All required macOS permissions are granted."
    lines = ["Missing macOS permissions:"]
    for status in report.statuses:
        if not status.granted and status.available:
            lines.append(f"  - {status.name}: not granted")
            lines.append(f"    Fix: {status.guidance}")
            if status.settings_url:
                lines.append(f"    Settings: {status.settings_url}")
    return "\n".join(lines)


async def diagnose_and_offer_fix(*, open_panes: bool = False) -> Dict[str, Any]:
    """One-shot helper for UIs: check, log, optionally open missing panes."""
    report = await check_all_permissions(refresh=True)
    logger.info("Permission diagnosis: %s", format_report(report).replace("\n", " | "))
    opened: List[str] = []
    if open_panes:
        for missing in report.missing:
            if open_settings_pane(missing):
                opened.append(missing)
    return {"report": report.as_dict(), "opened_panes": opened}
