"""core/capabilities/capability_discovery.py — Machine Capability Scan
========================================================================
Discovers what this machine can do BEFORE attempting actions.

Produces a CapabilityReport that the TaskDecomposer uses to plan
realistic task graphs. Before attempting any workflow, Aura can say:
"I have access to microphone, screen, Chrome, Finder, file writes,
 and wallpaper settings. Google Docs may require login."
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.runtime.task_ownership import create_tracked_task
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.CapabilityDiscovery")


# How a capability field came to hold its value. A bare boolean cannot say
# this, and that gap is the whole defect: a False that means "probed and
# absent" and a False that means "nobody ever looked" are the same value to
# every consumer, and an optimistic default made the second one indelible.
# The capability fields a probe is responsible for establishing. Anything
# here that is still UNPROBED after a scan is reported, not assumed.
_DISCOVERABLE_FIELDS = (
    "has_browser",
    "has_text_editor",
    "has_terminal",
    "has_accessibility",
    "has_screen_recording",
    "has_microphone",
    "has_camera",
    "has_network",
    "has_screencapture",
    "has_osascript",
    "has_pbcopy",
    "has_say",
)

UNPROBED = "unprobed"
PROBED = "probed"
INFERRED = "inferred"
PROBE_FAILED = "probe_failed"


@dataclass
class CapabilityReport:
    """Structured report of machine capabilities.

    CP126 (critical): terminal, screencapture, osascript, pbcopy and say all
    defaulted to ``True``. A freshly constructed report therefore ASSERTED
    those capabilities before any probe had run — and ``start`` installs a
    fresh report immediately, keeping it if scan scheduling fails, while
    ``discover`` gathers its sub-scans with ``return_exceptions=True`` so a
    raising ``_discover_tools`` silently leaves the optimistic values in
    place. Planners could receive fabricated positive capabilities at
    precisely the moment discovery had not executed.

    Every field now starts at the value that claims nothing, and
    ``provenance`` records whether a probe actually established it. Callers
    that must not act on a guess ask ``established``; callers that only want
    a hint keep reading the booleans.
    """
    # Apps
    installed_apps: list[str] = field(default_factory=list)
    has_browser: bool = False
    preferred_browser: str = ""
    has_text_editor: bool = False
    has_terminal: bool = False

    # Permissions
    has_accessibility: bool = False
    has_screen_recording: bool = False
    has_microphone: bool = False
    has_camera: bool = False
    has_full_disk_access: bool = False

    # System
    has_network: bool = False
    has_python_packages: dict[str, bool] = field(default_factory=dict)
    writable_directories: list[str] = field(default_factory=list)
    available_models: list[str] = field(default_factory=list)

    # Tools
    has_screencapture: bool = False
    has_osascript: bool = False
    has_pbcopy: bool = False
    has_say: bool = False

    # field name -> UNPROBED | PROBED | INFERRED | PROBE_FAILED. Absent means
    # UNPROBED; nothing has to remember to populate it to stay honest.
    provenance: dict[str, str] = field(default_factory=dict)
    # field name -> why its probe failed, for the health surface.
    probe_failures: dict[str, str] = field(default_factory=dict)

    timestamp: float = field(default_factory=time.time)

    def mark(self, *fields_: str, state: str = PROBED, detail: str = "") -> None:
        """Record how these fields were established."""
        for name in fields_:
            self.provenance[name] = state
            if state == PROBE_FAILED and detail:
                self.probe_failures[name] = detail[:300]
            else:
                self.probe_failures.pop(name, None)

    def state_of(self, name: str) -> str:
        return self.provenance.get(name, UNPROBED)

    def established(self, name: str) -> bool:
        """True only when a probe actually ran and the capability is present.

        This is the accessor a privileged caller wants: an inferred or
        unprobed capability answers False, so "we never checked" can never
        be spent as "yes".
        """
        return self.state_of(name) == PROBED and bool(getattr(self, name, False))

    @property
    def unprobed_fields(self) -> list[str]:
        return sorted(
            name
            for name in _DISCOVERABLE_FIELDS
            if self.state_of(name) in {UNPROBED, PROBE_FAILED}
        )

    @property
    def fully_discovered(self) -> bool:
        return not self.unprobed_fields

    def summary(self) -> str:
        """Human-readable capability summary."""
        parts = ["I have access to:"]
        if self.has_microphone:
            parts.append("microphone")
        if self.has_screen_recording:
            parts.append("screen recording")
        if self.has_accessibility:
            parts.append("accessibility")
        if self.has_browser:
            parts.append(f"{self.preferred_browser} browser")
        if self.has_text_editor:
            parts.append("text editor")
        if self.writable_directories:
            parts.append("file writes")
        if self.has_network:
            parts.append("network")
        capabilities = ", ".join(parts[1:]) if len(parts) > 1 else "limited capabilities"
        result = f"I have access to: {capabilities}."

        # Warnings. "Not detected" and "not checked" are different claims,
        # and saying the first when the second is true is the bug.
        if self.state_of("has_accessibility") != PROBED:
            result += " Accessibility permission has not been checked."
        elif not self.has_accessibility:
            result += " Accessibility permission not detected — UI automation may be limited."
        if self.state_of("has_network") != PROBED:
            result += " Network reachability has not been checked."
        elif not self.has_network:
            result += " Network appears unavailable."
        unprobed = self.unprobed_fields
        if unprobed:
            result += f" {len(unprobed)} capabilities are still unverified."
        return result


class CapabilityDiscovery:
    """Discovers what this machine can do."""

    def __init__(self) -> None:
        self._report: CapabilityReport | None = None
        self._started = False
        self._scan_task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("capability_discovery", self, required=False)
        self._started = True
        self._report = self._report or CapabilityReport()
        try:
            self._scan_task = create_tracked_task(
                self._run_initial_scan(),
                name="capability_discovery.initial_scan",
                owner="capability_discovery",
                bounded=True,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "capability_discovery.initial_scan",
                exc,
                action="left capability report at explicit unknown defaults after scan scheduling failed",
            )
            self._scan_task = None
        if self._scan_task is None:
            logger.warning("CapabilityDiscovery ONLINE — initial scan was not scheduled")
        else:
            logger.info("CapabilityDiscovery ONLINE — initial scan scheduled")

    async def _run_initial_scan(self) -> None:
        try:
            report = await self.discover()
        except (ImportError, AttributeError, RuntimeError, OSError, TimeoutError) as exc:
            record_degradation("capability_discovery.initial_scan", exc)
            logger.warning("CapabilityDiscovery initial scan failed: %s", exc)
            return
        self._report = report
        logger.info("CapabilityDiscovery scan complete — %s", report.summary()[:120])

    # Sub-scan -> the fields it is responsible for establishing. A scan that
    # raises leaves its fields PROBE_FAILED rather than at their defaults,
    # which is what keeps a swallowed exception from reading as a verdict.
    _SCAN_FIELDS: dict[str, tuple[str, ...]] = {
        "apps": ("has_browser", "has_text_editor", "has_terminal"),
        "permissions": (
            "has_accessibility",
            "has_screen_recording",
            "has_microphone",
            "has_camera",
        ),
        "network": ("has_network",),
        "tools": ("has_screencapture", "has_osascript", "has_pbcopy", "has_say"),
        "python_packages": (),
        "writable_dirs": (),
        "models": (),
    }

    async def discover(self) -> CapabilityReport:
        """Run full capability scan."""
        report = CapabilityReport()

        scans = {
            "apps": self._discover_apps(report),
            "permissions": self._discover_permissions(report),
            "network": self._discover_network(report),
            "tools": self._discover_tools(report),
            "python_packages": self._discover_python_packages(report),
            "writable_dirs": self._discover_writable_dirs(report),
            "models": self._discover_models(report),
        }
        names = list(scans)
        # return_exceptions keeps one failing scan from cancelling the rest —
        # but the results must then be READ. Gathering exceptions and
        # discarding them was how a raising _discover_tools left four
        # capabilities asserted on nothing.
        outcomes = await asyncio.gather(*scans.values(), return_exceptions=True)
        for name, outcome in zip(names, outcomes, strict=True):
            if not isinstance(outcome, BaseException):
                continue
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            report.mark(
                *self._SCAN_FIELDS.get(name, ()),
                state=PROBE_FAILED,
                detail=f"{type(outcome).__name__}: {outcome}",
            )
            record_degradation(
                f"capability_discovery.{name}",
                outcome,
                action="marked its capabilities unverified rather than leaving defaults",
            )

        self._report = report
        return report

    async def _discover_apps(self, report: CapabilityReport) -> None:
        """Discover installed applications."""
        try:
            registry = ServiceContainer.get("app_registry", default=None)
            if registry:
                apps = registry.all_apps()
                report.installed_apps = [a.name for a in apps]
                pref_browser = registry.get_preferred_browser()
                if pref_browser:
                    report.has_browser = True
                    report.preferred_browser = pref_browser.name
                pref_editor = registry.get_preferred_text_editor()
                if pref_editor:
                    report.has_text_editor = True
                report.mark("has_browser", "has_text_editor")
                report.mark(
                    "has_terminal",
                    state=INFERRED,
                    detail="app registry does not enumerate terminals",
                )
                return
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("capability_discovery.app_registry", exc)
            report.mark(
                "has_browser",
                "has_text_editor",
                state=PROBE_FAILED,
                detail=f"app registry unavailable: {type(exc).__name__}: {exc}",
            )

        # Fallback: scan /Applications directly (directory listing is
        # blocking I/O — keep it off the event loop).
        try:
            apps = await asyncio.to_thread(self._scan_applications_dir_sync)
            report.installed_apps = apps
            for browser in ["Google Chrome", "Safari", "Firefox"]:
                if browser in apps:
                    report.has_browser = True
                    report.preferred_browser = browser
                    break
            for editor in ["TextEdit", "Notes", "Visual Studio Code"]:
                if editor in apps:
                    report.has_text_editor = True
                    break
            report.has_terminal = any(
                terminal in apps for terminal in ("Terminal", "iTerm", "iTerm2", "Warp")
            )
            report.mark("has_browser", "has_text_editor", "has_terminal")
        except (OSError, PermissionError) as exc:
            record_degradation("capability_discovery.app_scan", exc)
            report.mark(
                "has_browser",
                "has_text_editor",
                "has_terminal",
                state=PROBE_FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )

    def _scan_applications_dir_sync(self) -> list[str]:
        app_dir = Path("/Applications")
        if not app_dir.exists():
            return []
        return sorted(e.stem for e in app_dir.iterdir() if e.suffix == ".app")

    async def _discover_permissions(self, report: CapabilityReport) -> None:
        """Check system permissions."""
        # Accessibility
        try:
            report.has_accessibility = await asyncio.wait_for(
                asyncio.to_thread(self._probe_accessibility_sync),
                timeout=1.0,
            )
            report.mark("has_accessibility")
        except (OSError, TimeoutError, RuntimeError) as exc:
            record_degradation("capability_discovery.accessibility_probe", exc)
            report.has_accessibility = False
            report.mark(
                "has_accessibility",
                state=PROBE_FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )

        # Screen recording
        try:
            from core.security.permission_guard import PermissionType, get_permission_guard

            guard = get_permission_guard()
            res = await guard.check_permission(PermissionType.SCREEN)
            report.has_screen_recording = res.get("granted", False)
            report.mark("has_screen_recording")
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("capability_discovery.screen_recording_probe", exc)
            report.has_screen_recording = False
            report.mark(
                "has_screen_recording",
                state=PROBE_FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )

        # Microphone and camera. TCC.db is not readable without Full Disk
        # Access, so its EXISTENCE is all we have — and that means "the
        # permission system is active on this machine", not "the permission
        # is granted". Marked INFERRED so `established()` refuses it: a
        # caller about to open the microphone must ask the real gate, and
        # this heuristic must never be the thing that authorizes it.
        tcc_db = Path.home() / "Library" / "Application Support" / "com.apple.TCC" / "TCC.db"
        report.has_microphone = tcc_db.exists()
        report.has_camera = report.has_microphone
        report.mark(
            "has_microphone",
            "has_camera",
            state=INFERRED,
            detail="TCC.db presence only; grant state is not readable",
        )

    @staticmethod
    def _probe_accessibility_sync() -> bool:
        """Read macOS Accessibility trust without prompting or scripting UI.

        AppleScript through System Events can wait indefinitely behind a TCC
        prompt and previously stalled startup discovery. AXIsProcessTrusted is
        the native, non-prompting status API and returns the permission attached
        to the signed Aura process identity.
        """

        if sys.platform != "darwin":
            return False
        try:
            from ApplicationServices import AXIsProcessTrusted
        except ImportError as exc:
            raise RuntimeError(
                "macOS ApplicationServices accessibility API is unavailable"
            ) from exc
        return bool(AXIsProcessTrusted())

    async def _discover_network(self, report: CapabilityReport) -> None:
        """Check network connectivity."""
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["ping", "-c", "1", "-W", "2", "8.8.8.8"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="capability_discovery.network_probe",
                accelerator_capability="none",
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
            report.has_network = proc.returncode == 0
            report.mark("has_network")
        except (OSError, TimeoutError, RuntimeError) as exc:
            record_degradation("capability_discovery.network_probe", exc)
            report.has_network = False
            report.mark(
                "has_network",
                state=PROBE_FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )

    async def _discover_tools(self, report: CapabilityReport) -> None:
        """Check for required CLI tools (PATH stats are blocking I/O)."""
        tools = {
            "screencapture": "has_screencapture",
            "osascript": "has_osascript",
            "pbcopy": "has_pbcopy",
            "say": "has_say",
        }
        try:
            found = await asyncio.to_thread(
                lambda: {tool: shutil.which(tool) is not None for tool in tools}
            )
        except (OSError, RuntimeError) as exc:
            record_degradation("capability_discovery.tool_probe", exc)
            report.mark(
                *tools.values(),
                state=PROBE_FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )
            return
        for tool, attr in tools.items():
            setattr(report, attr, found.get(tool, False))
        report.mark(*tools.values())

    async def _discover_python_packages(self, report: CapabilityReport) -> None:
        """Check for useful Python packages.

        find_spec answers "is it installed?" without executing module init
        code — importing numpy/PIL on the event loop at boot costs seconds
        of loop stall and loads megabytes nothing asked for yet.
        """
        packages = [
            "fpdf", "reportlab", "pytesseract", "PIL", "pyautogui",
            "psutil", "httpx", "numpy",
        ]

        def _availability() -> dict[str, bool]:
            import importlib.util

            status: dict[str, bool] = {}
            for pkg in packages:
                try:
                    status[pkg] = importlib.util.find_spec(pkg) is not None
                except (ImportError, ValueError, AttributeError):
                    status[pkg] = False
            return status

        report.has_python_packages = await asyncio.to_thread(_availability)

    @staticmethod
    def _probe_writable_dir(d: Path) -> None:
        """Blocking write probe — must run OFF the event loop.

        Every one of the 12 recorded live loop-wedge crashes (20-minute
        event-loop freezes ending in liveness-sentinel SIGKILL) had this
        probe's mkdir/write/unlink syscalls on the loop while the disk was
        thrashing. Probe files are worthless after a crash, so the write is
        atomic but non-durable (no fsync).
        """
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".aura_write_test"
        with local_internal_governed_scope(
            "capability_discovery.writable_dir_probe",
            receipt_prefix="capability-write-probe",
        ):
            get_file_write_gateway().write_text(
                test_file,
                "test",
                source="capability_discovery.writable_dir_probe",
                durable=False,
            )
        test_file.unlink()

    async def _discover_writable_dirs(self, report: CapabilityReport) -> None:
        """Check which directories are writable."""
        candidates = [
            Path.home() / "Documents" / "Aura",
            Path.home() / "Desktop" / "Aura",
            Path.home() / "Downloads",
            state_root() / "data",
            Path(tempfile.gettempdir()),
        ]
        for d in candidates:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._probe_writable_dir, d),
                    timeout=30.0,
                )
                report.writable_directories.append(str(d))
            except (OSError, PermissionError, RuntimeError, TimeoutError) as exc:
                record_degradation("capability_discovery.writable_dir_probe", exc)

    async def _discover_models(self, report: CapabilityReport) -> None:
        """Check for available LLM models."""
        try:
            router = ServiceContainer.get("llm_router", default=None)
            if router and hasattr(router, "list_models"):
                models = router.list_models()
                report.available_models = [str(m) for m in models[:10]]
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("capability_discovery.models", exc)

    def get_report(self) -> CapabilityReport:
        """Get the latest capability report."""
        if self._report is None:
            self._report = CapabilityReport()
        return self._report

    def get_status(self) -> dict[str, Any]:
        r = self._report or CapabilityReport()
        return {
            # "discovered" used to mean only "a report object exists", which
            # was true one line into start(). It now means a scan actually
            # established the capabilities, and the unverified list says what
            # is still missing when it has not.
            "discovered": r.fully_discovered,
            "report_present": self._report is not None,
            "unverified": r.unprobed_fields,
            "probe_failures": dict(r.probe_failures),
            "apps": len(r.installed_apps),
            "browser": r.preferred_browser,
            "accessibility": r.has_accessibility,
            "screen_recording": r.has_screen_recording,
            "network": r.has_network,
            "writable_dirs": len(r.writable_directories),
        }


_instance: CapabilityDiscovery | None = None


def get_capability_discovery() -> CapabilityDiscovery:
    global _instance
    if _instance is None:
        _instance = CapabilityDiscovery()
    return _instance


__all__ = ["CapabilityDiscovery", "CapabilityReport", "get_capability_discovery"]
