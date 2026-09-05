"""Enforcement backends — the hands the immune system reaches for.

The immune system decides; this is what actually carries out a defensive action on the host.
Each backend is registered as a mitigation handler (so the decision layer stays clean and
auditable) and is strictly defensive: block an origin, quarantine a file, kill a runaway
process Aura owns, throttle a flood. Anything needing elevated privileges (a real pf firewall
rule, system-wide process control) is attempted only when those privileges exist and fails
open to an effective app-layer equivalent otherwise — Aura never pretends she enforced
something she couldn't.

It also wires the real sensors: a psutil-backed resource monitor that feeds the
exhaustion/flood detectors, and an ``arp -a`` scanner that gives the network sentinel the
actual device list for her own network. No scanning or action against machines that aren't
hers.
"""
from __future__ import annotations

import logging
import ipaddress
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Security.Enforcement")
_ENFORCEMENT_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class AppLayerFirewall:
    """An in-process blocklist Aura's own servers/clients consult — effective without root."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._blocked: set[str] = set()
        self._blocked_at: dict[str, float] = {}

    def block(self, origin: str, *, now: float | None = None) -> None:
        if not origin or origin in {"unknown", "local"}:
            return
        with self._lock:
            self._blocked.add(origin)
            self._blocked_at[origin] = time.time() if now is None else now
        logger.warning("🛡️ [Firewall] app-layer block on %s", origin)
        self._try_pf_block(origin)

    def is_blocked(self, origin: str) -> bool:
        with self._lock:
            return origin in self._blocked

    def unblock(self, origin: str) -> None:
        with self._lock:
            self._blocked.discard(origin)
            self._blocked_at.pop(origin, None)

    def blocked(self) -> list[str]:
        with self._lock:
            return sorted(self._blocked)

    @staticmethod
    def _try_pf_block(origin: str) -> None:
        """Best-effort kernel firewall rule via pf — only if we have the privileges. Fail-open."""
        if os.geteuid() != 0:  # type: ignore[attr-defined]
            return  # no root → app-layer block already applied; don't pretend
        if not re.match(r"^[0-9a-fA-F:.]+$", origin):  # only IP-ish origins to pfctl
            return
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.subprocess_gateway import get_subprocess_gateway
            with local_internal_governed_scope("security.enforcement.pf_block", domain="tool_execution"):
                get_subprocess_gateway().run(
                    ["pfctl", "-t", "aura_block", "-T", "add", origin],
                    read_only=False, timeout=3.0, source="security.enforcement",
                    accelerator_capability="none",
                )
        except _ENFORCEMENT_ERRORS as exc:
            logger.debug("pf block unavailable for %s: %s", origin, exc)


class ProcessGuard:
    """Kills a runaway/hostile process — own-user processes only, never arbitrary system pids."""

    @staticmethod
    def terminate(pid: int) -> bool:
        try:
            from core.runtime import resource_psutil as psutil
        except ImportError as exc:
            logger.debug("Process terminate unavailable for %s: %s", pid, exc)
            return False

        process_errors = (psutil.Error, *_ENFORCEMENT_ERRORS)
        try:
            p = psutil.Process(int(pid))
            # Only ever act on processes owned by the same user as Aura.
            if p.uids().real != os.getuid():  # type: ignore[attr-defined]
                logger.warning("Refusing to kill pid %s — not owned by Aura's user", pid)
                return False
            p.terminate()
            try:
                p.wait(timeout=2.0)
            except psutil.TimeoutExpired:
                p.kill()
            return True
        except process_errors as exc:
            logger.debug("Process terminate failed for %s: %s", pid, exc)
            return False


class Quarantine:
    """Moves a suspect file out of harm's way into an isolated, restorable store."""

    def __init__(self, quarantine_dir: Path | None = None) -> None:
        if quarantine_dir is None:
            try:
                from core.config import config
                quarantine_dir = Path(config.paths.home_dir) / "data" / "quarantine"
            except (ImportError, AttributeError, RuntimeError):
                quarantine_dir = state_root() / "data" / "quarantine"
        self._dir = Path(quarantine_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def isolate(self, path: str) -> str | None:
        try:
            src = Path(path)
            if not src.exists():
                return None
            dest = self._dir / f"q-{int(time.time())}-{src.name}"
            shutil.move(str(src), str(dest))
            try:
                dest.chmod(0o400)  # read-only, defang
            except OSError:
                pass
            logger.warning("🔒 [Quarantine] isolated %s → %s", path, dest)
            return str(dest)
        except (OSError, ValueError) as exc:
            logger.debug("Quarantine failed for %s: %s", path, exc)
            return None

def _data_volume_percent(psutil_module: object) -> float:
    """Percent used of the volume holding Aura's state, not of "/".

    Delegates rather than re-deriving: this monitor having its own idea of
    where "the disk" is, is the defect that put ten subsystems on a read-only
    system volume. The psutil module stays in the signature for the callers
    that inject a fake one; it is only consulted if the canonical reading is
    unavailable.
    """
    try:
        from core.runtime.disk_budget import state_volume_percent

        return float(state_volume_percent())
    except (ImportError, OSError, ValueError):
        pass
    try:
        return float(psutil_module.disk_usage("/").percent)
    except (OSError, ValueError, AttributeError):
        return 0.0



class ResourceMonitor:
    """psutil-backed sampling that feeds the exhaustion detector when the host is under strain."""

    def __init__(self, *, cpu_high: float = 92.0, mem_high: float = 90.0, disk_high: float = 95.0) -> None:
        self._cpu_high = cpu_high
        self._mem_high = mem_high
        self._disk_high = disk_high

    def sample(self) -> dict[str, float]:
        try:
            from core.runtime import resource_psutil as psutil
        except ImportError as exc:
            logger.debug("Resource sample unavailable: %s", exc)
            return {}

        process_errors = (psutil.Error, *_ENFORCEMENT_ERRORS)
        try:
            return {
                "cpu": float(psutil.cpu_percent(interval=0.0)),
                "mem": float(psutil.virtual_memory().percent),
                # The volume Aura WRITES to, named explicitly.
                #
                # This asked for "/" and got the right answer by luck: on macOS
                # "/" is a sealed read-only volume sharing an APFS container
                # with /System/Volumes/Data, and what it reports depends on
                # which of the two the reading resolves to. Raw psutil returns
                # 2.2% for "/" here while the data volume is at 72% — a monitor
                # reading that would stay quiet however full the data got.
                #
                # The 2026-08-13 "disk at 99%" alarms were CORRECT; the disk
                # really was full. This does not fix a false reading. It stops
                # the alarm depending on which mount "/" happens to resolve to,
                # and points it at the volume state_root() actually lives on.
                "disk": float(_data_volume_percent(psutil)),
                "procs": float(len(psutil.pids())),
            }
        except process_errors as exc:
            logger.debug("Resource sample failed: %s", exc)
            return {}

    def check_and_report(self) -> dict[str, Any] | None:
        s = self.sample()
        if not s:
            return None
        breaches = []
        if s.get("cpu", 0) >= self._cpu_high:
            breaches.append(("cpu", s["cpu"]))
        if s.get("mem", 0) >= self._mem_high:
            breaches.append(("mem", s["mem"]))
        if s.get("disk", 0) >= self._disk_high:
            breaches.append(("disk", s["disk"]))
        if not breaches:
            return None
        worst = max(breaches, key=lambda kv: kv[1])
        try:
            from core.security.immune_system import ThreatClass, get_immune_system
            get_immune_system().assess(
                "resource_monitor", f"resource strain: {worst[0]} at {worst[1]:.0f}%",
                severity=min(0.9, worst[1] / 100.0), origin="host",
                targeted_vuln="resource_exhaustion", vector="compute",
                threat_class=ThreatClass.RESOURCE_EXHAUSTION, evidence=s,
            )
        except _ENFORCEMENT_ERRORS as exc:
            logger.debug("Resource monitor immune report failed: %s", exc)
        return {"breaches": dict(breaches), "sample": s}


# ── network scanner (own environment) ───────────────────────────────────────

_ARP_LINE = re.compile(
    r"^(?P<host>[^\s(]+)?\s*\((?P<ip>[0-9a-fA-F:.]+)\)\s+at\s+"
    r"(?P<mac>[0-9a-fA-F:]+)\s+on\s+(?P<interface>\S+)"
)


def _normalize_mac(value: str) -> str:
    parts = str(value or "").strip().lower().replace("-", ":").split(":")
    if len(parts) != 6:
        return ""
    try:
        return ":".join(f"{int(part, 16):02x}" for part in parts)
    except ValueError:
        return ""


def _local_interface_macs() -> set[str]:
    try:
        from core.runtime import resource_psutil as psutil

        addresses = psutil.net_if_addrs()
    except (ImportError, AttributeError, OSError, RuntimeError):
        return set()
    found: set[str] = set()
    for items in addresses.values():
        for item in items:
            normalized = _normalize_mac(getattr(item, "address", ""))
            if normalized:
                found.add(normalized)
    return found


def _valid_arp_device(ip_text: str, mac: str, local_macs: set[str]) -> bool:
    if not mac or mac in local_macs or mac in {"00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"}:
        return False
    try:
        if int(mac.split(":", 1)[0], 16) & 1:
            return False
        address = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return not (
        address.is_multicast
        or address.is_unspecified
        or address.is_loopback
        or str(address) == "255.255.255.255"
    )


def arp_scan() -> list[Any]:
    """Enumerate the local network from the ARP table (unprivileged, own network only)."""
    devices: list[Any] = []
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway
        from core.security.network_sentinel import Device
        proc = get_subprocess_gateway().run(
            ["/usr/sbin/arp", "-a"], read_only=True, timeout=4.0, source="security.enforcement.arp",
            accelerator_capability="none",
        )
        local_macs = _local_interface_macs()
        for line in (proc.stdout or "").splitlines():
            m = _ARP_LINE.match(line.strip())
            if not m or "incomplete" in line.lower():
                continue
            mac = _normalize_mac(m.group("mac"))
            ip = m.group("ip")
            if not _valid_arp_device(ip, mac, local_macs):
                continue
            name = str(m.group("host") or "").strip()
            devices.append(
                Device(
                    fingerprint=mac,
                    name=ip if name in {"", "?"} else name,
                    kind="network_host",
                    ip=ip,
                    mac=mac,
                    interface=m.group("interface"),
                    arp_state="reachable",
                    scanner_source="arp",
                    observation_confidence=0.95,
                )
            )
    except subprocess.TimeoutExpired as exc:
        logger.debug("ARP scan timed out after %.1fs", float(exc.timeout or 4.0))
    except _ENFORCEMENT_ERRORS as exc:
        logger.debug("ARP scan failed: %s", exc)
    return devices


# ── installation: wire the backends into the seams ──────────────────────────

_firewall: AppLayerFirewall | None = None
_quarantine: Quarantine | None = None
_installed = False
_install_lock = threading.Lock()


def get_firewall() -> AppLayerFirewall:
    global _firewall
    if _firewall is None:
        _firewall = AppLayerFirewall()
    return _firewall


def install_default_enforcement() -> dict[str, Any]:
    """Register the real enforcement backends into the immune system + network sentinel."""
    global _installed, _quarantine
    with _install_lock:
        if _installed:
            return {"installed": True, "already": True}
        fw = get_firewall()
        _quarantine = Quarantine()

        from core.security.immune_system import ThreatEvent, get_immune_system
        immune = get_immune_system()

        def _block(ev: ThreatEvent) -> str | None:
            fw.block(ev.origin)
            return f"unblock-{ev.origin}"

        def _quarantine_handler(ev: ThreatEvent) -> str | None:
            path = ev.evidence.get("path") if isinstance(ev.evidence, dict) else None
            if path:
                return _quarantine.isolate(str(path))
            return None

        def _rate_limit(ev: ThreatEvent) -> str | None:
            fw.block(ev.origin)   # at the app layer, rate-limit a flood == block the source
            return f"unblock-{ev.origin}"

        immune.register_mitigation("isolate", _block)
        immune.register_mitigation("alert", lambda ev: None)   # alerting handled by reflex/log
        immune.register_mitigation("quarantine", _quarantine_handler)
        immune.register_mitigation("rate_limit", _rate_limit)

        try:
            from core.security.network_sentinel import get_network_sentinel
            get_network_sentinel().register_scanner(arp_scan)
        except (ImportError, AttributeError, RuntimeError):
            pass

        _installed = True
        return {"installed": True, "mitigations": sorted(immune._handlers), "scanner": "arp"}
