"""core/somatic/capability_discovery.py

Capability Discovery Daemon -- periodic scanning for new/lost capabilities.

Crossing the Rubicon (Technological Autonomy):
    A static body schema is insufficient for a truly autonomous agent.
    Environments change: new tools are installed, peripherals are connected
    or disconnected, network interfaces come and go, and OS permissions
    shift.  This daemon runs a lightweight sweep every 60 seconds (tunable)
    to detect changes and update the BodySchema accordingly.

    Discoveries and losses are logged to the neural stream so that
    higher-level cognition (the volition stack, curiosity engine, etc.)
    can react -- e.g. "I just gained access to Docker, let me see what
    I can do with it."

Design:
    - Runs as an ``asyncio.Task`` managed by the orchestrator lifecycle.
    - Consults $PATH, psutil, and OS permission checks.
    - All mutations go through ``BodySchema.update_limb / add_limb``.
    - Registered in ServiceContainer as ``capability_discovery``.
"""

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import psutil

from core.base_module import AuraBaseModule
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Somatic.CapabilityDiscovery")


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class CapabilityDiscoveryDaemon(AuraBaseModule):
    """Periodic scanner that keeps the BodySchema accurate over time.

    Registered as ``capability_discovery`` in ServiceContainer.
    """

    # Executables to watch for on $PATH.  This list is intentionally small
    # and security-conscious -- we only track tools Aura knows how to use.
    TRACKED_EXECUTABLES: Dict[str, str] = {
        "git": "Git version control",
        "docker": "Docker container runtime",
        "python3": "Python 3 interpreter",
        "node": "Node.js runtime",
        "curl": "HTTP client (curl)",
        "wget": "HTTP client (wget)",
        "ssh": "SSH remote access",
        "scp": "Secure file copy",
        "rsync": "Incremental file sync",
        "ffmpeg": "FFmpeg media processing",
        "nmap": "Network scanner (nmap)",
        "pip3": "Python package manager",
        "npm": "Node.js package manager",
        "brew": "Homebrew package manager",
        "apt": "APT package manager",
        "code": "VS Code CLI",
        "tmux": "Terminal multiplexer",
    }

    # Sensor libraries to probe (import name, human description)
    TRACKED_SENSORS: Dict[str, str] = {
        "cv2": "Camera capture via OpenCV",
        "sounddevice": "Microphone input via sounddevice",
        "mss": "Screen capture via mss",
        "pyttsx3": "Text-to-speech via pyttsx3",
        "speech_recognition": "Speech recognition library",
    }

    def __init__(self, interval: float = 60.0) -> None:
        super().__init__("CapabilityDiscovery")
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._known_executables: Set[str] = set()
        self._known_interfaces: Set[str] = set()
        self._known_sensors: Set[str] = set()
        self._scan_count = 0

        # Snapshot initial state so first delta is meaningful
        self._snapshot_current_state()
        logger.info(
            "Capability Discovery Daemon initialised (interval=%.0fs).", interval
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin the periodic scan loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scan_loop(), name="capability_discovery")
        logger.info("Capability Discovery Daemon started.")

    async def stop(self) -> None:
        """Gracefully stop the daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Capability Discovery Daemon stopped.")

    # Orchestrator hooks
    async def on_start_async(self) -> None:
        await self.start()

    async def on_stop_async(self) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """The core periodic scan."""
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                await self._run_scan()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Discovery scan failed: %s", exc, exc_info=True)
                # Back off on repeated failures to avoid log flood
                await asyncio.sleep(min(self.interval * 2, 300))

    async def _run_scan(self) -> None:
        """Execute one full capability sweep."""
        self._scan_count += 1
        start = time.monotonic()

        body = self._get_body_schema()
        if body is None:
            logger.debug("BodySchema not available yet -- skipping scan.")
            return

        discoveries: List[str] = []
        losses: List[str] = []

        # Run the individual checks (CPU-bound work offloaded to executor)
        loop = asyncio.get_running_loop()
        exe_disc, exe_lost = await loop.run_in_executor(
            None, self._scan_executables, body
        )
        discoveries.extend(exe_disc)
        losses.extend(exe_lost)

        net_disc, net_lost = await loop.run_in_executor(
            None, self._scan_network_interfaces, body
        )
        discoveries.extend(net_disc)
        losses.extend(net_lost)

        sensor_disc, sensor_lost = await loop.run_in_executor(
            None, self._scan_sensors, body
        )
        discoveries.extend(sensor_disc)
        losses.extend(sensor_lost)

        # System resource refresh
        await loop.run_in_executor(None, self._refresh_system_resources, body)

        elapsed_ms = (time.monotonic() - start) * 1000

        if discoveries or losses:
            self._emit_to_neural_stream(discoveries, losses)

        if self._scan_count % 10 == 0 or discoveries or losses:
            logger.info(
                "Capability scan #%d complete in %.0fms -- %d new, %d lost.",
                self._scan_count, elapsed_ms, len(discoveries), len(losses),
            )

    # ------------------------------------------------------------------
    # Executable scanning
    # ------------------------------------------------------------------

    def _scan_executables(self, body: Any) -> tuple:
        """Check $PATH for tracked executables; return (discoveries, losses)."""
        from core.somatic.body_schema import Limb, LimbType

        discoveries: List[str] = []
        losses: List[str] = []
        current: Set[str] = set()

        for exe_name, description in self.TRACKED_EXECUTABLES.items():
            path = shutil.which(exe_name)
            limb_name = f"tool_{exe_name}"
            if path:
                current.add(exe_name)
                if exe_name not in self._known_executables:
                    # Newly discovered
                    body.add_limb(Limb(
                        name=limb_name,
                        limb_type=LimbType.ACTUATOR,
                        description=description,
                        available=True,
                        source=path,
                    ))
                    discoveries.append(f"executable:{exe_name} at {path}")
                else:
                    # Still present -- ensure marked available
                    body.update_limb(limb_name, available=True)
            else:
                if exe_name in self._known_executables:
                    # Was there, now gone
                    body.update_limb(limb_name, available=False, health=0.0)
                    losses.append(f"executable:{exe_name}")

        self._known_executables = current
        return discoveries, losses

    # ------------------------------------------------------------------
    # Network interface scanning
    # ------------------------------------------------------------------

    def _scan_network_interfaces(self, body: Any) -> tuple:
        """Detect new or lost network interfaces."""
        from core.somatic.body_schema import Limb, LimbType

        discoveries: List[str] = []
        losses: List[str] = []

        try:
            addrs = psutil.net_if_addrs()
            current_ifaces: Set[str] = set(addrs.keys())
        except Exception:
            return discoveries, losses

        # Filter out loopback
        current_ifaces.discard("lo")
        current_ifaces.discard("lo0")

        new_ifaces = current_ifaces - self._known_interfaces
        lost_ifaces = self._known_interfaces - current_ifaces

        for iface in new_ifaces:
            limb_name = f"net_{iface}"
            addr_info = addrs.get(iface, [])
            ip_addrs = [
                a.address for a in addr_info
                if a.family.name in ("AF_INET", "AF_INET6")
                and not a.address.startswith("fe80")
            ]
            body.add_limb(Limb(
                name=limb_name,
                limb_type=LimbType.SENSOR,
                description=f"Network interface: {iface}",
                available=True,
                source="psutil",
                metadata={"addresses": ip_addrs},
            ))
            discoveries.append(f"network:{iface} ({', '.join(ip_addrs)})")

        for iface in lost_ifaces:
            limb_name = f"net_{iface}"
            body.update_limb(limb_name, available=False, health=0.0)
            losses.append(f"network:{iface}")

        # Also update the aggregate "network" limb
        body.update_limb(
            "network",
            available=len(current_ifaces) > 0,
            metadata_patch={"interface_count": len(current_ifaces)},
        )

        self._known_interfaces = current_ifaces
        return discoveries, losses

    # ------------------------------------------------------------------
    # Sensor / library scanning
    # ------------------------------------------------------------------

    def _scan_sensors(self, body: Any) -> tuple:
        """Check whether sensor libraries are importable."""
        from core.somatic.body_schema import Limb, LimbType
        import importlib

        discoveries: List[str] = []
        losses: List[str] = []
        current: Set[str] = set()

        # Map library names to the limb names used by BodySchema
        lib_to_limb = {
            "cv2": "camera",
            "sounddevice": "microphone",
            "mss": "screen_capture",
            "pyttsx3": "speech_output",
            "speech_recognition": "speech_recognition",
        }

        for lib_name, description in self.TRACKED_SENSORS.items():
            try:
                importlib.import_module(lib_name)
                importable = True
            except ImportError:
                importable = False

            limb_name = lib_to_limb.get(lib_name, lib_name)

            if importable:
                current.add(lib_name)
                if lib_name not in self._known_sensors:
                    body.add_limb(Limb(
                        name=limb_name,
                        limb_type=LimbType.SENSOR,
                        description=description,
                        available=True,
                        source=lib_name,
                    ))
                    discoveries.append(f"sensor:{lib_name}")
                else:
                    body.update_limb(limb_name, available=True)
            else:
                if lib_name in self._known_sensors:
                    body.update_limb(limb_name, available=False)
                    losses.append(f"sensor:{lib_name}")

        self._known_sensors = current
        return discoveries, losses

    # ------------------------------------------------------------------
    # System resource refresh
    # ------------------------------------------------------------------

    def _refresh_system_resources(self, body: Any) -> None:
        """Update the system-resource limbs with fresh data."""
        try:
            battery = psutil.sensors_battery()
            body.update_limb(
                "battery_sensor",
                available=battery is not None,
                metadata_patch={
                    "percent": battery.percent if battery else None,
                    "plugged": battery.power_plugged if battery else None,
                },
            )
        except Exception:
            pass

        try:
            disk = psutil.disk_usage("/")
            body.update_limb(
                "disk_sensor",
                metadata_patch={
                    "total_gb": round(disk.total / (1024 ** 3), 1),
                    "free_gb": round(disk.free / (1024 ** 3), 1),
                    "percent_used": disk.percent,
                },
            )
            # Mark unhealthy if disk is critically full
            if disk.percent > 95:
                body.update_limb("disk_sensor", health=0.2)
            elif disk.percent > 90:
                body.update_limb("disk_sensor", health=0.5)
            else:
                body.update_limb("disk_sensor", health=1.0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Neural stream emission
    # ------------------------------------------------------------------

    def _emit_to_neural_stream(
        self, discoveries: List[str], losses: List[str]
    ) -> None:
        """Push capability changes to the neural feed for cognitive awareness."""
        try:
            from core.neural_feed import get_feed
            feed = get_feed()

            if discoveries:
                feed.push(
                    content=f"New capabilities detected: {', '.join(discoveries)}",
                    title="CAPABILITY_DISCOVERED",
                    category="SOMATIC",
                )
            if losses:
                feed.push(
                    content=f"Capabilities lost: {', '.join(losses)}",
                    title="CAPABILITY_LOST",
                    category="SOMATIC",
                )
        except Exception as exc:
            logger.debug("Neural feed emission failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot_current_state(self) -> None:
        """Capture the initial environment state so the first delta is clean."""
        for exe_name in self.TRACKED_EXECUTABLES:
            if shutil.which(exe_name):
                self._known_executables.add(exe_name)

        try:
            addrs = psutil.net_if_addrs()
            self._known_interfaces = set(addrs.keys()) - {"lo", "lo0"}
        except Exception:
            pass

        for lib_name in self.TRACKED_SENSORS:
            try:
                __import__(lib_name)
                self._known_sensors.add(lib_name)
            except ImportError:
                pass

    @staticmethod
    def _get_body_schema():
        """Resolve the BodySchema from the container, or None."""
        return ServiceContainer.get("body_schema", default=None)

    def get_status(self) -> Dict[str, Any]:
        """Return daemon status for health checks."""
        return {
            "running": self._running,
            "scan_count": self._scan_count,
            "interval_s": self.interval,
            "known_executables": len(self._known_executables),
            "known_interfaces": len(self._known_interfaces),
            "known_sensors": len(self._known_sensors),
        }


# ---------------------------------------------------------------------------
# Module-level accessor and ServiceContainer wiring
# ---------------------------------------------------------------------------

_daemon: Optional[CapabilityDiscoveryDaemon] = None
_init_lock = threading.Lock()


def get_capability_discovery(interval: float = 60.0) -> CapabilityDiscoveryDaemon:
    """Get or create the global CapabilityDiscoveryDaemon singleton.

    Also registers the instance in ServiceContainer as
    ``capability_discovery`` if not already present.
    """
    global _daemon
    if _daemon is not None:
        return _daemon

    with _init_lock:
        if _daemon is not None:
            return _daemon
        _daemon = CapabilityDiscoveryDaemon(interval=interval)
        ServiceContainer.register_instance("capability_discovery", _daemon)
    return _daemon
