"""Ground real sensor events into the continuous substrate.

The substrate should not drift only from conversation-derived text features.
This bridge maps camera/screen/audio observations into the ODE input vector and
can run as a small background loop over the existing sensory JSON files.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SensorimotorGrounding")


DEFAULT_SENSOR_FILES = (
    Path("sensory_vision.json"),
    Path("sensory_audio.json"),
)


def _hash_features(text: str, *, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    raw = text.encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(raw, digest_size=32).digest()
    for i, byte in enumerate(digest):
        idx = (byte + i * 17) % dim
        sign = 1.0 if byte & 1 else -1.0
        vec[idx] += sign * ((byte / 255.0) - 0.5)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec /= norm
    return vec


def _numeric_features(observation: dict[str, Any], *, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if dim <= 0:
        return vec

    confidence = float(observation.get("confidence", 0.5) or 0.5)
    energy = float(observation.get("energy", observation.get("rms", 0.0)) or 0.0)
    age_s = max(0.0, time.time() - float(observation.get("timestamp_unix", time.time()) or time.time()))

    vec[0] = max(0.0, min(1.0, confidence))
    if dim > 1:
        vec[1] = max(0.0, min(1.0, energy))
    if dim > 2:
        vec[2] = math.exp(-min(age_s, 60.0) / 30.0)
    if dim > 3:
        source = str(observation.get("source") or observation.get("type") or "")
        vec[3] = {"camera": 0.9, "visual": 0.8, "screen": 0.7, "audio": -0.7, "microphone": -0.8}.get(
            source.lower(),
            0.0,
        )
    return vec


def observation_to_vector(observation: dict[str, Any] | None, *, dim: int = 64) -> np.ndarray:
    """Convert a sensor observation into a bounded substrate input vector."""
    dim = max(16, min(512, int(dim or 64)))
    obs = dict(observation or {})

    text_parts = [
        str(obs.get("source") or obs.get("type") or ""),
        str(obs.get("summary") or obs.get("description") or obs.get("transcript") or ""),
        str(obs.get("raw_reference") or ""),
    ]
    image_data = str(obs.get("image_data") or "")
    if image_data:
        # Do not retain image bytes; use a short decoded-byte fingerprint.
        try:
            sample = base64.b64decode(image_data[:4096], validate=False)[:512]
            text_parts.append(hashlib.blake2b(sample, digest_size=16).hexdigest())
        except Exception:
            text_parts.append(hashlib.blake2b(image_data[:512].encode("utf-8"), digest_size=16).hexdigest())

    vec = 0.70 * _hash_features(" | ".join(text_parts), dim=dim)
    vec += 0.30 * _numeric_features(obs, dim=dim)
    return np.tanh(vec).astype(np.float32)


def _read_sensor_file(path: Path) -> Optional[dict[str, Any]]:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        payload.setdefault("source", payload.get("type") or path.stem)
        try:
            payload.setdefault("timestamp_unix", path.stat().st_mtime)
        except OSError:
            payload.setdefault("timestamp_unix", time.time())
        if "confidence" not in payload:
            payload["confidence"] = 0.65 if payload.get("status") == "active" else 0.35
        if "energy" not in payload and "rms" in payload:
            payload["energy"] = payload.get("rms")
        return payload
    except Exception as exc:
        record_degradation("sensorimotor_grounding", exc)
        logger.debug("Sensor file read skipped for %s: %s", path, exc)
        return None


@dataclass
class SensorimotorGroundingBridge:
    """Periodically inject real sensory observations into a substrate."""

    substrate: Any
    sensor_files: Iterable[Path] = field(default_factory=lambda: DEFAULT_SENSOR_FILES)
    interval_s: float = 1.0
    running: bool = False
    observations_injected: int = 0
    last_observation: dict[str, Any] = field(default_factory=dict)
    _task: Optional[asyncio.Task] = field(default=None, init=False, repr=False)

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop(), name="Aura.SensorimotorGrounding")

    async def stop(self) -> None:
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self.running:
            try:
                self.inject_latest()
            except Exception as exc:
                record_degradation("sensorimotor_grounding", exc)
            await asyncio.sleep(max(0.1, float(self.interval_s or 1.0)))

    def inject_latest(self) -> int:
        count = 0
        for raw_path in self.sensor_files:
            path = Path(raw_path)
            obs = _read_sensor_file(path)
            if not obs:
                continue
            if hasattr(self.substrate, "inject_observation"):
                self.substrate.inject_observation(obs)
            elif hasattr(self.substrate, "inject_stimulus"):
                dim = int(getattr(getattr(self.substrate, "config", None), "neuron_count", 64) or 64)
                result = self.substrate.inject_stimulus(observation_to_vector(obs, dim=dim), weight=1.0)
                if asyncio.iscoroutine(result):
                    # The synchronous probe path cannot await; the async loop
                    # schedules the coroutine so the injection still happens.
                    try:
                        asyncio.get_running_loop().create_task(result)
                    except RuntimeError:
                        result.close()
            else:
                dim = int(getattr(self.substrate, "get_state_dim", lambda: 64)())
                self.substrate.inject_input(observation_to_vector(obs, dim=dim))
            self.last_observation = obs
            self.observations_injected += 1
            count += 1
        return count

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "observations_injected": self.observations_injected,
            "last_source": str(self.last_observation.get("source", "")),
            "sensor_files": [str(Path(p)) for p in self.sensor_files],
        }


__all__ = [
    "SensorimotorGroundingBridge",
    "observation_to_vector",
]
