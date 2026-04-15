"""core/somatic/body_schema.py

Body Schema -- Aura's real-time awareness of her own capabilities.

Crossing the Rubicon (Technological Autonomy):
    This module is the somatic foundation of Aura's self-model.  Just as a
    biological organism maintains an internal body schema that tracks which
    limbs exist, which are injured, and which are the best choice for a
    given motor task, this module maintains a live map of every capability
    Aura can exercise -- skills, sensors, actuators, and system resources.

    The body_schema is the single source of truth that downstream planners,
    the capability engine, and the volition stack consult before deciding
    *how* to act.  It is kept fresh by the CapabilityDiscoveryDaemon and by
    inline success/failure reports from the skill execution pipeline.

Design:
    - On init, performs a full-body discovery sweep.
    - Exposes a flat dict of ``Limb`` dataclasses keyed by canonical name.
    - Thread-safe; guarded by a single ``threading.Lock`` (no async needed
      for reads; the daemon does async writes via ``update_limb``).
    - Registered in ServiceContainer as ``body_schema``.
"""

import importlib
import logging
import os
import platform
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from core.base_module import AuraBaseModule
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Somatic.BodySchema")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class LimbType(str, Enum):
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    COGNITIVE = "cognitive"


@dataclass
class Limb:
    """A single capability (skill, sensor, or system resource)."""

    name: str
    limb_type: LimbType
    description: str = ""
    available: bool = True
    health: float = 1.0  # 0.0 = dead, 1.0 = perfect
    last_used: float = 0.0  # monotonic timestamp
    latency_estimate_ms: float = 0.0  # rolling average
    cost_cpu: float = 0.0  # estimated CPU fraction 0-1
    cost_memory_mb: float = 0.0  # estimated MB
    error_count: int = 0
    success_count: int = 0
    source: str = ""  # e.g. "skills/web_search.py" or "psutil"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- health helpers ---------------------------------------------------

    def record_success(self, latency_ms: float) -> None:
        self.success_count += 1
        self.last_used = time.monotonic()
        # Exponential moving average (alpha=0.3)
        alpha = 0.3
        self.latency_estimate_ms = (
            alpha * latency_ms + (1 - alpha) * self.latency_estimate_ms
        )
        self._recompute_health()

    def record_failure(self, error: str) -> None:
        self.error_count += 1
        self.last_used = time.monotonic()
        self.metadata["last_error"] = error
        self.metadata["last_error_time"] = time.time()
        self._recompute_health()

    def _recompute_health(self) -> None:
        total = self.success_count + self.error_count
        if total == 0:
            self.health = 1.0
            return
        success_ratio = self.success_count / total
        # Recent errors weigh more: decay health faster if last 5 had errors
        recent_penalty = min(self.error_count, 5) * 0.05
        self.health = max(0.0, min(1.0, success_ratio - recent_penalty))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["limb_type"] = self.limb_type.value
        return d


# ---------------------------------------------------------------------------
# Body Schema
# ---------------------------------------------------------------------------

class BodySchema(AuraBaseModule):
    """Maintains Aura's live body map of all available capabilities.

    Registered as ``body_schema`` in ServiceContainer.
    """

    def __init__(self) -> None:
        super().__init__("BodySchema")
        self._limbs: Dict[str, Limb] = {}
        self._lock = threading.Lock()
        self._aura_root = Path(__file__).resolve().parents[2]  # ~/Desktop/aura

        # Run initial discovery synchronously at construction time so the map
        # is populated before any downstream code asks for it.
        self._discover_all()
        logger.info(
            "Body Schema online -- %d limbs discovered.", len(self._limbs)
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_all(self) -> None:
        """Full body scan: skills, senses, system capabilities."""
        self._discover_skills()
        self._discover_senses()
        self._discover_system_capabilities()
        self._discover_external_tools()

    def _discover_skills(self) -> None:
        """Scan core/skills/ for registered skill modules."""
        skills_dir = self._aura_root / "core" / "skills"
        if not skills_dir.is_dir():
            logger.warning("Skills directory not found: %s", skills_dir)
            return

        for py_file in sorted(skills_dir.glob("*.py")):
            if py_file.name.startswith("_") or py_file.name == "base_skill.py":
                continue

            skill_name = py_file.stem
            description = ""
            metabolic_cost = 1

            # Attempt lightweight introspection without full import to avoid
            # heavy side effects at boot.  Fall back gracefully.
            try:
                module_path = f"core.skills.{skill_name}"
                mod = importlib.import_module(module_path)
                # Walk module for BaseSkill subclasses
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name, None)
                    if (
                        isinstance(obj, type)
                        and hasattr(obj, "name")
                        and hasattr(obj, "description")
                        and attr_name != "BaseSkill"
                    ):
                        skill_name = getattr(obj, "name", skill_name)
                        description = getattr(obj, "description", "")
                        metabolic_cost = getattr(obj, "metabolic_cost", 1)
                        break
            except Exception as exc:
                logger.debug(
                    "Lightweight introspection failed for %s: %s", skill_name, exc
                )

            limb_type = LimbType.ACTUATOR
            # Heuristics: senses/perception skills are sensors
            sensor_keywords = (
                "listen", "vision", "observe", "perception", "proprioception",
                "screen", "toggle_senses", "visual_context",
            )
            if any(kw in skill_name for kw in sensor_keywords):
                limb_type = LimbType.SENSOR

            cognitive_keywords = (
                "plan", "dream", "curiosity", "belief", "cognitive",
                "self_improvement", "self_evolution", "train",
            )
            if any(kw in skill_name for kw in cognitive_keywords):
                limb_type = LimbType.COGNITIVE

            cost_map = {0: 0.01, 1: 0.05, 2: 0.15, 3: 0.35}
            self._register_limb(Limb(
                name=skill_name,
                limb_type=limb_type,
                description=description or f"Skill: {skill_name}",
                available=True,
                source=str(py_file.relative_to(self._aura_root)),
                cost_cpu=cost_map.get(metabolic_cost, 0.1),
            ))

    def _discover_senses(self) -> None:
        """Probe hardware sensors via existing modules."""
        # Camera (cv2 / mss)
        try:
            import cv2  # noqa: F401
            self._register_limb(Limb(
                name="camera",
                limb_type=LimbType.SENSOR,
                description="Camera capture via OpenCV",
                available=True,
                source="cv2",
                cost_cpu=0.10,
                cost_memory_mb=50.0,
            ))
        except ImportError:
            self._register_limb(Limb(
                name="camera",
                limb_type=LimbType.SENSOR,
                description="Camera capture (unavailable -- cv2 missing)",
                available=False,
                source="cv2",
            ))

        # Microphone
        try:
            import sounddevice  # noqa: F401
            self._register_limb(Limb(
                name="microphone",
                limb_type=LimbType.SENSOR,
                description="Microphone input via sounddevice",
                available=True,
                source="sounddevice",
                cost_cpu=0.05,
                cost_memory_mb=20.0,
            ))
        except ImportError:
            self._register_limb(Limb(
                name="microphone",
                limb_type=LimbType.SENSOR,
                description="Microphone input (unavailable -- sounddevice missing)",
                available=False,
                source="sounddevice",
            ))

        # Screen capture
        try:
            import mss  # noqa: F401
            self._register_limb(Limb(
                name="screen_capture",
                limb_type=LimbType.SENSOR,
                description="Screen capture via mss",
                available=True,
                source="mss",
                cost_cpu=0.08,
                cost_memory_mb=30.0,
            ))
        except ImportError:
            self._register_limb(Limb(
                name="screen_capture",
                limb_type=LimbType.SENSOR,
                description="Screen capture (unavailable -- mss missing)",
                available=False,
                source="mss",
            ))

        # TTS / Speech output
        try:
            import pyttsx3  # noqa: F401
            tts_available = True
        except ImportError:
            tts_available = False
        self._register_limb(Limb(
            name="speech_output",
            limb_type=LimbType.ACTUATOR,
            description="Text-to-speech output",
            available=tts_available,
            source="pyttsx3",
            cost_cpu=0.05,
        ))

    def _discover_system_capabilities(self) -> None:
        """Detect OS-level capabilities via psutil and platform."""
        # File I/O -- always available
        self._register_limb(Limb(
            name="file_io",
            limb_type=LimbType.ACTUATOR,
            description="Local filesystem read/write",
            available=True,
            source="os",
            cost_cpu=0.01,
        ))

        # Network access
        net_available = False
        try:
            interfaces = psutil.net_if_addrs()
            net_available = len(interfaces) > 0
        except Exception:
            pass
        self._register_limb(Limb(
            name="network",
            limb_type=LimbType.ACTUATOR,
            description="Network connectivity",
            available=net_available,
            source="psutil",
            metadata={"interface_count": len(psutil.net_if_addrs())},
        ))

        # Process management
        self._register_limb(Limb(
            name="process_management",
            limb_type=LimbType.ACTUATOR,
            description="OS process spawn/monitor via psutil",
            available=True,
            source="psutil",
            cost_cpu=0.02,
        ))

        # Battery awareness
        battery = psutil.sensors_battery()
        self._register_limb(Limb(
            name="battery_sensor",
            limb_type=LimbType.SENSOR,
            description="Battery level and AC status",
            available=battery is not None,
            source="psutil",
            metadata={
                "percent": battery.percent if battery else None,
                "plugged": battery.power_plugged if battery else None,
            },
        ))

        # Disk space awareness
        disk = psutil.disk_usage("/")
        self._register_limb(Limb(
            name="disk_sensor",
            limb_type=LimbType.SENSOR,
            description="Disk usage monitoring",
            available=True,
            source="psutil",
            metadata={
                "total_gb": round(disk.total / (1024 ** 3), 1),
                "free_gb": round(disk.free / (1024 ** 3), 1),
            },
        ))

    def _discover_external_tools(self) -> None:
        """Check for key CLI tools on $PATH."""
        tools = {
            "git": ("git", LimbType.ACTUATOR, "Git version control"),
            "docker": ("docker", LimbType.ACTUATOR, "Docker container runtime"),
            "python3": ("python3", LimbType.ACTUATOR, "Python 3 interpreter"),
            "node": ("node", LimbType.ACTUATOR, "Node.js runtime"),
            "curl": ("curl", LimbType.ACTUATOR, "HTTP client (curl)"),
            "ssh": ("ssh", LimbType.ACTUATOR, "SSH remote access"),
            "ffmpeg": ("ffmpeg", LimbType.ACTUATOR, "FFmpeg media processing"),
            "nmap": ("nmap", LimbType.SENSOR, "Network scanner (nmap)"),
        }
        for name, (binary, limb_type, desc) in tools.items():
            path = shutil.which(binary)
            self._register_limb(Limb(
                name=f"tool_{name}",
                limb_type=limb_type,
                description=desc,
                available=path is not None,
                source=path or "",
            ))

    # ------------------------------------------------------------------
    # Registration & Mutation (thread-safe)
    # ------------------------------------------------------------------

    def _register_limb(self, limb: Limb) -> None:
        with self._lock:
            self._limbs[limb.name] = limb

    def update_limb(
        self,
        name: str,
        *,
        available: Optional[bool] = None,
        health: Optional[float] = None,
        metadata_patch: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Patch an existing limb's state (used by CapabilityDiscoveryDaemon)."""
        with self._lock:
            limb = self._limbs.get(name)
            if limb is None:
                logger.debug("update_limb: unknown limb '%s' -- ignoring.", name)
                return
            if available is not None:
                limb.available = available
            if health is not None:
                limb.health = max(0.0, min(1.0, health))
            if metadata_patch:
                limb.metadata.update(metadata_patch)

    def add_limb(self, limb: Limb) -> None:
        """Register a brand-new capability discovered at runtime."""
        with self._lock:
            if limb.name in self._limbs:
                logger.debug("add_limb: '%s' already exists -- updating.", limb.name)
            self._limbs[limb.name] = limb
        logger.info("New limb registered: %s (%s)", limb.name, limb.limb_type.value)

    def remove_limb(self, name: str) -> None:
        """Mark a capability as permanently lost."""
        with self._lock:
            limb = self._limbs.pop(name, None)
        if limb:
            logger.warning("Limb removed: %s", name)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_body_map(self) -> Dict[str, Dict[str, Any]]:
        """Return the full body map as a serialisable dict."""
        with self._lock:
            return {name: limb.to_dict() for name, limb in self._limbs.items()}

    def is_limb_available(self, name: str) -> bool:
        """Check whether a named capability is currently usable."""
        with self._lock:
            limb = self._limbs.get(name)
            return limb is not None and limb.available and limb.health > 0.1

    def get_best_limb_for(self, task_description: str) -> Optional[str]:
        """Heuristic selection of the best capability for a described task.

        Scores each available limb by keyword overlap with the task
        description, weighted by health and inverse latency.  Returns the
        name of the top-scoring limb, or ``None`` if nothing matches.
        """
        task_lower = task_description.lower()
        task_tokens = set(task_lower.split())

        best_name: Optional[str] = None
        best_score: float = -1.0

        with self._lock:
            for name, limb in self._limbs.items():
                if not limb.available or limb.health < 0.15:
                    continue

                # Keyword overlap score
                limb_tokens = set(limb.name.lower().replace("_", " ").split())
                limb_tokens.update(limb.description.lower().split())
                overlap = len(task_tokens & limb_tokens)
                if overlap == 0:
                    # Check substring match as a fallback
                    if not any(tok in limb.name.lower() or tok in limb.description.lower() for tok in task_tokens):
                        continue
                    overlap = 0.5

                # Weighted score: overlap * health * inverse-latency-penalty
                latency_penalty = 1.0 / (1.0 + limb.latency_estimate_ms / 1000.0)
                score = overlap * limb.health * latency_penalty

                if score > best_score:
                    best_score = score
                    best_name = name

        return best_name

    def get_available_limbs(self, limb_type: Optional[LimbType] = None) -> List[str]:
        """Return names of all available limbs, optionally filtered by type."""
        with self._lock:
            return [
                name
                for name, limb in self._limbs.items()
                if limb.available
                and limb.health > 0.1
                and (limb_type is None or limb.limb_type == limb_type)
            ]

    def get_limb(self, name: str) -> Optional[Dict[str, Any]]:
        """Return a single limb's state as a dict, or None."""
        with self._lock:
            limb = self._limbs.get(name)
            return limb.to_dict() if limb else None

    # ------------------------------------------------------------------
    # Skill execution feedback hooks
    # ------------------------------------------------------------------

    def report_limb_success(self, name: str, latency_ms: float) -> None:
        """Called after a successful skill/tool invocation."""
        with self._lock:
            limb = self._limbs.get(name)
            if limb:
                limb.record_success(latency_ms)

    def report_limb_failure(self, name: str, error: str) -> None:
        """Called after a failed skill/tool invocation."""
        with self._lock:
            limb = self._limbs.get(name)
            if limb:
                limb.record_failure(error)
                if limb.health < 0.15:
                    logger.warning(
                        "Limb '%s' health critical (%.2f) -- may be unusable.",
                        name, limb.health,
                    )

    # ------------------------------------------------------------------
    # Summary / Health
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Quick summary for cognitive introspection."""
        with self._lock:
            total = len(self._limbs)
            available = sum(1 for l in self._limbs.values() if l.available)
            healthy = sum(1 for l in self._limbs.values() if l.health > 0.5)
            by_type = {}
            for l in self._limbs.values():
                by_type.setdefault(l.limb_type.value, 0)
                by_type[l.limb_type.value] += 1

        return {
            "total_limbs": total,
            "available": available,
            "healthy": healthy,
            "degraded": total - healthy,
            "by_type": by_type,
        }


# ---------------------------------------------------------------------------
# Module-level accessor and ServiceContainer wiring
# ---------------------------------------------------------------------------

_body_schema: Optional[BodySchema] = None
_init_lock = threading.Lock()


def get_body_schema() -> BodySchema:
    """Get or create the global BodySchema singleton.

    Also registers the instance in ServiceContainer as ``body_schema`` if
    it is not already present.
    """
    global _body_schema
    if _body_schema is not None:
        return _body_schema

    with _init_lock:
        if _body_schema is not None:
            return _body_schema
        _body_schema = BodySchema()
        ServiceContainer.register_instance("body_schema", _body_schema)
    return _body_schema
