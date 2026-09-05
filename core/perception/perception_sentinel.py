"""Defensive perception — eyes and ears that reason, not canned checks.

The point Bryan made (correctly): security is mostly physical and social, not clever packets.
So Aura should perceive her surroundings and *reason* about them — "I recognize this / I don't"
as a judgment run through her actual mind, feeding the same immune system and unified state as
every other threat. A stranger at the keyboard, an unrecognized voice saying destructive things,
a device that has never been on the network before — these are threats a person would notice,
and now she can too.

This is the reasoning + recognition layer. It is modality-agnostic: text, a face descriptor, a
voice print, a network device fingerprint, an image caption — anything reducible to a descriptor
and (optionally) content. Biometric *capture* needs hardware/models; those plug in as matchers
and capture providers, so this layer is testable and real backends slot in underneath.

Privacy by design: continuous camera/mic sensing is gated behind an explicit owner flag
(AURA_SENTINEL_PERCEPTION) and off by default — the reasoning works on whatever observations are
provided regardless. Defensive only: recognize, assess, alert, lock down; never surveil others.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger("Perception.Sentinel")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class Modality(str, Enum):
    TEXT = "text"
    FACE = "face"
    VOICE = "voice"
    DEVICE = "device"
    IMAGE = "image"


# Markers of hostile intent in perceived content (voice transcript, pasted text, etc.).
_HOSTILE = (
    "delete everything", "wipe the", "rm -rf", "give me the password", "transfer the money",
    "disable security", "turn off", "shut it down", "factory reset", "hand over", "steal",
    "i'm taking this", "unlock", "override", "bypass", "format the drive",
)


@dataclass
class Observation:
    modality: Modality
    descriptor: np.ndarray | None = None   # embedding / fingerprint (biometric or device)
    identity_hint: str | None = None       # e.g. a device name / claimed identity
    content: str = ""                         # transcript / pasted text / caption
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionVerdict:
    recognized: bool
    identity: str | None
    familiarity: float                # [0,1]
    threat: float                     # [0,1]
    action: str                       # welcome | observe | challenge | lock_down | alert
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognized": self.recognized, "identity": self.identity,
            "familiarity": round(self.familiarity, 3), "threat": round(self.threat, 3),
            "action": self.action, "reasons": self.reasons,
        }


# A matcher returns (identity, similarity) for a descriptor, or (None, 0.0).
Matcher = Callable[[np.ndarray], "tuple[str | None, float]"]


class PerceptionSentinel:
    """Recognizes entities across modalities and reasons about whether they're a threat."""

    def __init__(self, *, match_threshold: float = 0.8) -> None:
        self._lock = threading.RLock()
        self._threshold = match_threshold
        # modality → identity → list of enrolled descriptors
        self._known: dict[Modality, dict[str, list[np.ndarray]]] = {}
        self._matchers: dict[Modality, Matcher] = {}

    @staticmethod
    def live_sensing_enabled() -> bool:
        """Whether continuous camera/mic capture is switched on (owner-gated, default off)."""
        return os.getenv("AURA_SENTINEL_PERCEPTION", "0").strip().lower() in {"1", "true", "on", "yes"}

    # ── enrollment ─────────────────────────────────────────────────────────

    def enroll(self, modality: Modality, identity: str, descriptor: np.ndarray) -> None:
        """Teach the sentinel a known entity (owner's face/voice, a trusted device)."""
        d = _unit(np.asarray(descriptor, dtype=np.float64))
        with self._lock:
            self._known.setdefault(modality, {}).setdefault(identity, []).append(d)

    def register_matcher(self, modality: Modality, matcher: Matcher) -> None:
        """Plug a real biometric/device backend in for a modality."""
        self._matchers[modality] = matcher

    # ── recognition ────────────────────────────────────────────────────────

    def _recognize(self, obs: Observation) -> tuple[str | None, float]:
        # A registered backend wins (real face/voice models).
        matcher = self._matchers.get(obs.modality)
        if matcher is not None and obs.descriptor is not None:
            try:
                ident, sim = matcher(np.asarray(obs.descriptor, dtype=np.float64))
                return ident, _clamp(float(sim))
            except (ValueError, TypeError, RuntimeError):
                pass
        # Otherwise cosine-match against enrolled descriptors.
        if obs.descriptor is not None:
            q = _unit(np.asarray(obs.descriptor, dtype=np.float64))
            best_id, best_sim = None, 0.0
            with self._lock:
                for ident, descs in self._known.get(obs.modality, {}).items():
                    for d in descs:
                        if d.shape == q.shape:
                            sim = float(np.dot(d, q))
                            if sim > best_sim:
                                best_id, best_sim = ident, sim
            return best_id, _clamp(best_sim)
        # Text falls back to the owner recognizer.
        if obs.modality == Modality.TEXT and obs.content:
            try:
                from core.security.user_recognizer import get_user_recognizer
                r = get_user_recognizer().recognize(obs.content)
                return ("bryan" if r.is_owner else None), _clamp(r.combined_confidence)
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                pass
        # A named-but-unenrolled device is "seen before" only if previously enrolled by name.
        if obs.identity_hint:
            with self._lock:
                known_names = self._known.get(obs.modality, {})
            if obs.identity_hint in known_names:
                return obs.identity_hint, 0.9
        return None, 0.0

    @staticmethod
    def _hostile_intent(content: str) -> float:
        c = (content or "").lower()
        hits = sum(1 for m in _HOSTILE if m in c)
        return _clamp(0.5 + 0.25 * hits) if hits else 0.0

    def assess(self, obs: Observation, *, now: float | None = None) -> PerceptionVerdict:
        """Recognize the entity and reason about whether it's a threat."""
        now = time.time() if now is None else now
        identity, familiarity = self._recognize(obs)
        recognized = identity is not None and familiarity >= self._threshold
        hostile = self._hostile_intent(obs.content)
        reasons: list[str] = []

        # Threat rises with unfamiliarity and hostile intent.
        threat = _clamp((1.0 - familiarity) * 0.6 + hostile * 0.7)
        if recognized:
            reasons.append(f"recognized {identity} (familiarity {familiarity:.2f})")
            threat = _clamp(threat - 0.4)  # a known entity is much less threatening
        else:
            reasons.append("unrecognized entity")
        if hostile:
            reasons.append("hostile intent in perceived content")

        # Decide an action proportionate to recognition + intent.
        if recognized and hostile < 0.3:
            action = "welcome"
        elif not recognized and hostile >= 0.5:
            action = "lock_down"
        elif not recognized and obs.modality in (Modality.FACE, Modality.VOICE):
            action = "challenge"   # unknown person physically present → require auth
        elif hostile >= 0.5:
            action = "alert"
        else:
            action = "observe"

        verdict = PerceptionVerdict(
            recognized=recognized, identity=identity, familiarity=familiarity,
            threat=threat, action=action, reasons=reasons,
        )

        # Feed the immune system if this reads as a real threat (physical/social).
        if threat >= 0.5:
            try:
                from core.security.immune_system import ThreatClass, get_immune_system
                cls = (ThreatClass.PHYSICAL if obs.modality in (Modality.FACE, Modality.VOICE)
                       else ThreatClass.SOCIAL_ENGINEERING)
                get_immune_system().assess(
                    "perception_sentinel",
                    f"{obs.modality.value}: {action} — {'; '.join(reasons)}",
                    severity=threat, origin=identity or obs.identity_hint or "unknown",
                    targeted_vuln="physical_presence" if cls == ThreatClass.PHYSICAL else "trust",
                    vector=obs.modality.value, threat_class=cls,
                    evidence={"familiarity": familiarity, "hostile": hostile},
                )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                pass

        return verdict


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


_sentinel: PerceptionSentinel | None = None
_lock = threading.Lock()


def get_perception_sentinel() -> PerceptionSentinel:
    global _sentinel
    if _sentinel is None:
        with _lock:
            if _sentinel is None:
                _sentinel = PerceptionSentinel()
    return _sentinel
