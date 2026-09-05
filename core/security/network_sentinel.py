"""Network environment baseline, novelty confirmation, and threat escalation."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Security.NetworkSentinel")
_SCHEMA_VERSION = 1


def _default_baseline_path() -> Path:
    try:
        from core.config import config

        return Path(config.paths.home_dir) / "data" / "security" / "network_baseline.json"
    except (ImportError, AttributeError, RuntimeError):
        return state_root() / "data" / "security" / "network_baseline.json"


@dataclass
class Device:
    fingerprint: str
    name: str = ""
    kind: str = "unknown"
    ip: str = ""
    mac: str = ""
    interface: str = ""
    arp_state: str = "reachable"
    scanner_source: str = "unknown"
    observation_confidence: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.fingerprint = str(self.fingerprint or "").strip().lower()
        self.mac = str(self.mac or self.fingerprint or "").strip().lower()
        self.ip = str(self.ip or "").strip()
        self.name = str(self.name or "").strip()
        if self.name in {"", "?"}:
            self.name = self.ip or self.fingerprint
        self.observation_confidence = max(
            0.0,
            min(1.0, float(self.observation_confidence or 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "name": self.name,
            "kind": self.kind,
            "ip": self.ip,
            "mac": self.mac,
            "interface": self.interface,
            "arp_state": self.arp_state,
            "scanner_source": self.scanner_source,
            "observation_confidence": self.observation_confidence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class DeviceVerdict:
    fingerprint: str
    known: bool
    anomalous: bool
    threat: float
    action: str
    reasons: List[str] = field(default_factory=list)
    state: str = "normal"
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "known": self.known,
            "anomalous": self.anomalous,
            "threat": round(self.threat, 3),
            "action": self.action,
            "reasons": self.reasons,
            "state": self.state,
            "evidence": dict(self.evidence),
        }


Scanner = Callable[[], List[Device]]


class NetworkSentinel:
    """Persisted baseline with repeated-observation and corroboration gates."""

    def __init__(
        self,
        *,
        settle_period_s: float = 3600.0,
        baseline_path: Path | None = None,
        confirmation_count: int = 2,
    ) -> None:
        self._lock = threading.RLock()
        self._known: Dict[str, Device] = {}
        self._novel: Dict[str, dict[str, Any]] = {}
        self._settle = max(0.0, float(settle_period_s))
        self._confirmation_count = max(2, int(confirmation_count))
        self._started_at = time.time()
        self._baseline_established_at = 0.0
        self._baseline_path = Path(baseline_path or _default_baseline_path())
        self._scanner: Optional[Scanner] = None
        self._load_baseline()

    def register_scanner(self, scanner: Scanner) -> None:
        self._scanner = scanner

    def _load_baseline(self) -> None:
        if not self._baseline_path.exists():
            return
        try:
            payload = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or int(payload.get("schema_version") or 0) != _SCHEMA_VERSION
            ):
                raise ValueError("unsupported network baseline schema")
            devices = payload.get("devices")
            if not isinstance(devices, list):
                raise ValueError("network baseline devices must be a list")
            for raw in devices:
                if not isinstance(raw, dict):
                    continue
                device = Device(**raw)
                if device.fingerprint:
                    self._known[device.fingerprint] = device
            novel = payload.get("novel_observations")
            if isinstance(novel, dict):
                for fingerprint, raw in novel.items():
                    if not isinstance(raw, dict):
                        continue
                    self._novel[str(fingerprint)] = {
                        "count": max(0, int(raw.get("count") or 0)),
                        "first_seen": max(0.0, float(raw.get("first_seen") or 0.0)),
                        "last_seen": max(0.0, float(raw.get("last_seen") or 0.0)),
                    }
            established = float(payload.get("established_at") or 0.0)
            self._baseline_established_at = established if self._known else 0.0
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.info(
                "Network baseline unavailable after validation failure: %s",
                type(exc).__name__,
            )
            self._known.clear()
            self._novel.clear()
            self._baseline_established_at = 0.0

    def _persist_baseline(self) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "saved_at": time.time(),
            "established_at": self._baseline_established_at,
            "devices": [
                device.to_dict()
                for device in sorted(
                    self._known.values(),
                    key=lambda item: item.fingerprint,
                )
            ],
            "novel_observations": dict(self._novel),
        }
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with local_internal_governed_scope(
                "security.network_sentinel.baseline",
                domain="file_write",
                receipt_prefix="network-baseline",
            ):
                get_file_write_gateway().write_text(
                    self._baseline_path,
                    json.dumps(payload, indent=2, sort_keys=True),
                    source="core.security.network_sentinel.baseline",
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "Network baseline persistence failed; in-memory evidence retained: %s",
                exc,
            )

    def learn_baseline(self, devices: List[Device]) -> None:
        changed = False
        now = time.time()
        with self._lock:
            for device in devices:
                if not isinstance(device, Device) or not device.fingerprint:
                    continue
                self._known[device.fingerprint] = device
                self._novel.pop(device.fingerprint, None)
                changed = True
            if changed and not self._baseline_established_at:
                self._baseline_established_at = now
            if changed:
                self._persist_baseline()

    def _observe_novel(self, device: Device, now: float) -> dict[str, Any]:
        state = self._novel.setdefault(
            device.fingerprint,
            {"count": 0, "first_seen": now, "last_seen": now},
        )
        state["count"] = int(state.get("count") or 0) + 1
        state["last_seen"] = now
        return state

    def observe(
        self,
        device: Device,
        *,
        now: Optional[float] = None,
        corroborating_signals: Sequence[str] = (),
    ) -> DeviceVerdict:
        now = time.time() if now is None else float(now)
        signals = tuple(
            sorted({str(item).strip() for item in corroborating_signals if str(item).strip()})
        )
        with self._lock:
            known = self._known.get(device.fingerprint)
            settling = self._settle > 0 and (now - self._started_at) < self._settle
            if known is not None:
                known.last_seen = now
                return DeviceVerdict(
                    device.fingerprint,
                    True,
                    False,
                    0.0,
                    "normal",
                    ["known device"],
                    state="known",
                    evidence=device.to_dict(),
                )

            observation = self._observe_novel(device, now)
            count = int(observation["count"])
            evidence = {
                **device.to_dict(),
                "observation_count": count,
                "corroborating_signals": list(signals),
                "baseline_established_at": self._baseline_established_at,
            }

            if settling:
                if count >= self._confirmation_count:
                    self._known[device.fingerprint] = device
                    self._novel.pop(device.fingerprint, None)
                    if not self._baseline_established_at:
                        self._baseline_established_at = now
                    self._persist_baseline()
                    return DeviceVerdict(
                        device.fingerprint,
                        False,
                        False,
                        0.0,
                        "observe",
                        ["repeated device learned during baseline settle-in"],
                        state="baseline_learned",
                        evidence=evidence,
                    )
                self._persist_baseline()
                return DeviceVerdict(
                    device.fingerprint,
                    False,
                    False,
                    0.0,
                    "observe",
                    ["awaiting repeated observation during baseline settle-in"],
                    state="baseline_observation",
                    evidence=evidence,
                )

            if not self._baseline_established_at or not self._known:
                if count <= self._confirmation_count:
                    self._persist_baseline()
                return DeviceVerdict(
                    device.fingerprint,
                    False,
                    False,
                    0.0,
                    "observe",
                    ["baseline unavailable; novelty cannot be classified as intrusion"],
                    state="baseline_unavailable",
                    evidence=evidence,
                )

            if count < self._confirmation_count:
                self._persist_baseline()
                return DeviceVerdict(
                    device.fingerprint,
                    False,
                    False,
                    0.1,
                    "observe",
                    ["first novel observation; awaiting confirmation"],
                    state="novel_observation",
                    evidence=evidence,
                )

            if not signals:
                if count == self._confirmation_count:
                    self._persist_baseline()
                return DeviceVerdict(
                    device.fingerprint,
                    False,
                    True,
                    0.25,
                    "investigate",
                    ["repeated novel device without independent threat evidence"],
                    state="confirmed_novel_device",
                    evidence=evidence,
                )

            reasons = [
                "repeated novel device",
                "independent security evidence: " + ", ".join(signals),
            ]
            threat = 0.65
            self._flag_immune(device, threat, reasons, evidence=evidence)
            return DeviceVerdict(
                device.fingerprint,
                False,
                True,
                threat,
                "alert",
                reasons,
                state="corroborated_intrusion",
                evidence=evidence,
            )

    def enumerate(self) -> List[Device]:
        if self._scanner is None:
            return []
        try:
            return list(self._scanner() or [])
        except (RuntimeError, OSError, ValueError) as exc:
            logger.debug("Network enumerate failed: %s", exc)
            return []

    def sweep(self) -> List[DeviceVerdict]:
        return [self.observe(device) for device in self.enumerate()]

    def _flag_immune(
        self,
        device: Device,
        threat: float,
        reasons: List[str],
        *,
        evidence: dict[str, Any],
    ) -> None:
        try:
            from core.security.immune_system import ThreatClass, get_immune_system

            get_immune_system().assess(
                "network_sentinel",
                f"corroborated device {device.name or device.fingerprint}: "
                + "; ".join(reasons),
                severity=threat,
                origin=device.ip or device.fingerprint,
                targeted_vuln="network_perimeter",
                vector="network",
                threat_class=ThreatClass.INTRUSION,
                evidence=evidence,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def recovery_plan(self) -> Dict[str, Any]:
        plan: Dict[str, Any] = {"restore_points": [], "recoverable": False}
        try:
            from core.security.deletion_guard import get_deletion_guard

            dg = get_deletion_guard().status()
            plan["restore_points"].append(
                {"source": "deletion_guard", "versions": dg.get("versions_kept", 0)}
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            from core.security.emergency_protocol import get_emergency_protocol

            ep = get_emergency_protocol().get_status()
            plan["restore_points"].append(
                {
                    "source": "emergency_snapshot",
                    "snapshot_taken": ep.get("snapshot_taken", False),
                }
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        plan["recoverable"] = any(
            item.get("versions", 0) or item.get("snapshot_taken")
            for item in plan["restore_points"]
        )
        return plan

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "known_devices": len(self._known),
                "novel_observations": len(self._novel),
                "scanner_registered": self._scanner is not None,
                "settled": (time.time() - self._started_at) >= self._settle,
                "baseline_state": (
                    "established" if self._baseline_established_at and self._known
                    else "settling" if (time.time() - self._started_at) < self._settle
                    else "unavailable"
                ),
                "baseline_established_at": self._baseline_established_at,
                "baseline_path": str(self._baseline_path),
            }


_sentinel: Optional[NetworkSentinel] = None
_lock = threading.Lock()


def get_network_sentinel() -> NetworkSentinel:
    global _sentinel
    if _sentinel is None:
        with _lock:
            if _sentinel is None:
                _sentinel = NetworkSentinel()
    return _sentinel
