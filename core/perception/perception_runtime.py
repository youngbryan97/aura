"""PerceptionRuntime contract surface.

This module ships the contracts the rest of Aura calls into. Real
hardware drivers (camera, microphone, screen, subtitle reader) are
attached at runtime via ``register_sensor`` and remain optional.

Capability access is gated by a ``CapabilityToken`` issued through
Unified Will. The runtime never starts a sensor without a valid token,
and the token records the audit receipt that authorized the access.
"""
from __future__ import annotations


import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("Aura.PerceptionRuntime")


CAPABILITY_KINDS = (
    "camera",
    "microphone",
    "screen",
    "subtitle",
    "terminal",
    "browser",
)


@dataclass(frozen=True)
class CapabilityToken:
    capability: str
    scope: str
    granted_by: str
    receipt_id: str
    expires_at: float

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) >= self.expires_at


@dataclass
class SceneEvent:
    """One observation produced by a sensor or scene segmenter."""

    timestamp: float
    source: str  # camera / microphone / subtitle / screen
    summary: str
    confidence: float
    energy: float = 0.0  # 0..1, used by silence policy
    raw_reference: Optional[str] = None
    storage_policy: str = "session"


@dataclass
class MovieSessionMemory:
    """Multi-hour movie-watching context.

    The audit demands explicit clocks, character tracking, plot events,
    user reactions, and a privacy-aware retention policy. This class
    keeps it minimal but fully testable.
    """

    title: str
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    scenes: List[SceneEvent] = field(default_factory=list)
    characters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    user_reactions: List[Dict[str, Any]] = field(default_factory=list)
    aura_comments: List[Dict[str, Any]] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    privacy_mode: bool = False

    # Clocks ---------------------------------------------------------------

    def add_scene(self, event: SceneEvent) -> None:
        if self.privacy_mode:
            event = SceneEvent(
                timestamp=event.timestamp,
                source=event.source,
                summary="<redacted: privacy mode>",
                confidence=event.confidence,
                energy=event.energy,
                raw_reference=None,
                storage_policy="session_only",
            )
        self.scenes.append(event)

    def track_character(self, name: str, **attrs: Any) -> None:
        char = self.characters.setdefault(name, {})
        char.update(attrs)
        char["last_seen"] = time.time()

    def record_user_reaction(self, kind: str, text: str = "", at: Optional[float] = None) -> None:
        self.user_reactions.append({"kind": kind, "text": text, "at": at or time.time()})

    def record_aura_comment(self, text: str, suppressed: bool = False, reason: Optional[str] = None) -> None:
        self.aura_comments.append(
            {"text": text, "at": time.time(), "suppressed": suppressed, "reason": reason}
        )

    def close_session(self) -> None:
        self.finished_at = time.time()


@dataclass
class SilencePolicy:
    """When may Aura speak during a movie / focus session?"""

    silence_min_seconds: float = 12.0
    high_energy_threshold: float = 0.6
    backchannel_cooldown_s: float = 25.0

    def should_speak(self, *, scene_energy: float, since_user_speech_s: float, last_aura_comment_age_s: float) -> bool:
        if scene_energy >= self.high_energy_threshold:
            return False
        if since_user_speech_s < self.silence_min_seconds:
            return False
        if last_aura_comment_age_s < self.backchannel_cooldown_s:
            return False
        return True


@dataclass
class SharedAttentionState:
    user_focus_estimate: str = "unknown"
    aura_focus: str = "unknown"
    confidence: float = 0.0
    recent_joint_events: List[str] = field(default_factory=list)

    def update_focus(self, *, user: Optional[str] = None, aura: Optional[str] = None, confidence: Optional[float] = None) -> None:
        if user is not None:
            self.user_focus_estimate = user
        if aura is not None:
            self.aura_focus = aura
        if confidence is not None:
            self.confidence = max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# PerceptionRuntime
# ---------------------------------------------------------------------------


SensorCallable = Callable[[CapabilityToken], Awaitable[None]]


class CapabilityDenied(RuntimeError):
    pass  # no-op: intentional


class PerceptionRuntime:
    """Single owner of perception capabilities + sessions.

    Hardware drivers must be registered via ``register_sensor``. The
    runtime will call the sensor callable only after a capability token
    is issued through ``request_capability`` (which itself routes through
    Unified Will via ``governance_decide``).
    """

    def __init__(
        self,
        *,
        governance_decide: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self._governance = governance_decide
        self._clock = clock
        self._sensors: Dict[str, SensorCallable] = {}
        self._tokens: Dict[str, CapabilityToken] = {}
        self.movie_session: Optional[MovieSessionMemory] = None
        self.silence_policy = SilencePolicy()
        self.shared_attention = SharedAttentionState()

    # --- Sensor registration ----------------------------------------------

    def register_sensor(self, capability: str, sensor: SensorCallable) -> None:
        if capability not in CAPABILITY_KINDS:
            raise ValueError(f"unknown capability '{capability}'")
        self._sensors[capability] = sensor

    # --- Capability gating -------------------------------------------------

    async def request_capability(
        self,
        capability: str,
        *,
        scope: str = "default",
        ttl_s: float = 3600.0,
    ) -> CapabilityToken:
        if capability not in CAPABILITY_KINDS:
            raise CapabilityDenied(f"unknown capability '{capability}'")
        decision = await self._invoke_governance(capability, scope)
        approved, receipt = self._extract_decision(decision)
        if not approved:
            raise CapabilityDenied(f"governance denied capability '{capability}' (scope={scope})")
        token = CapabilityToken(
            capability=capability,
            scope=scope,
            granted_by="UnifiedWill",
            receipt_id=receipt or "rcpt-pending",
            expires_at=self._clock() + ttl_s,
        )
        self._tokens[capability] = token
        return token

    async def _invoke_governance(self, capability: str, scope: str) -> Any:
        if self._governance is None:
            # Fail closed when no governance is wired.
            raise CapabilityDenied(
                f"no governance authority configured; refusing capability '{capability}'"
            )
        decision = self._governance(
            domain="perception",
            action=f"grant_capability_{capability}",
            cause="perception_runtime",
            context={"scope": scope},
        )
        if asyncio.iscoroutine(decision):
            decision = await decision
        return decision

    @staticmethod
    def _extract_decision(decision: Any):
        if decision is None:
            return False, None
        if isinstance(decision, dict):
            return bool(decision.get("approved")), decision.get("receipt_id")
        approved = getattr(decision, "is_approved", None)
        if callable(approved):
            return bool(approved()), getattr(decision, "receipt_id", None)
        return bool(decision), None

    def has_token(self, capability: str) -> bool:
        token = self._tokens.get(capability)
        return token is not None and not token.is_expired(self._clock())

    # --- Sessions ----------------------------------------------------------

    def open_movie_session(self, *, title: str, privacy_mode: bool = False) -> MovieSessionMemory:
        self.movie_session = MovieSessionMemory(title=title, privacy_mode=privacy_mode)
        return self.movie_session

    def close_movie_session(self) -> Optional[MovieSessionMemory]:
        if self.movie_session is None:
            return None
        self.movie_session.close_session()
        session = self.movie_session
        self.movie_session = None
        return session

    # --- Sensor execution --------------------------------------------------

    async def start_sensor(self, capability: str) -> None:
        sensor = self._sensors.get(capability)
        if sensor is None:
            raise CapabilityDenied(f"no sensor registered for '{capability}'")
        if not self.has_token(capability):
            raise CapabilityDenied(
                f"capability '{capability}' has no valid token; call request_capability first"
            )
        await sensor(self._tokens[capability])
