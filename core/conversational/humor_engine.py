"""Calibrated exact-agent humor outcome learning for Aura."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer, ServiceLifetime
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.social.relational_memory import (
    RelationalMemoryAuthority,
    get_relational_memory_authority,
)

logger = logging.getLogger("Aura.Humor")

HUMOR_TYPES = (
    "sarcasm",
    "dry_wit",
    "absurdist",
    "callback",
    "observational",
    "self_deprecating",
    "hyperbole",
    "pun",
    "wordplay",
    "dark",
    "surreal",
    "deadpan",
)

_DEFAULT_DATA_PATH = state_root() / "data" / "humor_profiles.json"
_SNAPSHOT_NAMESPACE = "humor_profile:v1"
_SNAPSHOT_KIND = "style_preference"
_FEEDBACK_WINDOW_SECONDS = 30 * 60
_MAX_ATTEMPTS_PER_USER = 100

_LANDED_PATTERNS = (
    re.compile(r"\b(?:lol|lmao|lmfao|rofl)\b", re.IGNORECASE),
    re.compile(r"\b(?:haha|hahaha|hehe|ahaha)\b", re.IGNORECASE),
    re.compile(r"[😂🤣]+"),
    re.compile(
        r"\b(?:that(?:'s| is) funny|hilarious|you got me|cracked me up|"
        r"that landed|peak comedy)\b",
        re.IGNORECASE,
    ),
)
_MISSED_PATTERNS = (
    re.compile(
        r"\b(?:not (?:funny|hilarious)|please don't joke|stop joking|that was cringe)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:i don't get the joke|what was the joke|explain the joke)\b", re.IGNORECASE),
)
_OVERT_HUMOR_RESPONSE = re.compile(
    r"(?:[😂🤣]|\b(?:just kidding|i(?:'m| am) kidding|kidding aside|"
    r"i(?:'m| am) teasing|teasing you|pun intended|bad pun|couldn't resist)\b)",
    re.IGNORECASE,
)

_TOPIC_CATEGORIES = {
    "technology": ("ai", "automation", "software", "computer", "model"),
    "work": ("work", "job", "career", "company", "meeting"),
    "learning": ("learn", "study", "school", "research", "explain"),
    "creativity": ("create", "design", "art", "write", "imagine"),
    "relationships": ("relationship", "friend", "family", "partner", "team"),
    "daily_life": ("food", "coffee", "sleep", "home", "weather"),
}
_REGISTER_VALUES = frozenset(
    {
        "casual",
        "direct",
        "formal",
        "neutral",
        "playful",
        "serious",
        "supportive",
        "technical",
    }
)


def _bounded_float(value: Any, default: float, *, low: float, high: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(result):
        return default
    return min(high, max(low, result))


def _bounded_int(value: Any, default: int, *, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(high, max(low, result))


def _bounded_text(value: Any, *, limit: int, default: str = "") -> str:
    normalized = " ".join(str(value or "").strip().split())
    return normalized[:limit] or default


def _digest_or_empty(value: Any) -> str:
    digest = str(value or "").strip().lower()
    return digest if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest) else ""


def _normalize_register(value: Any) -> str:
    normalized = _bounded_text(value, limit=40, default="casual").lower()
    return normalized if normalized in _REGISTER_VALUES else "neutral"


def _normalize_topic_category(value: Any) -> str:
    normalized = _bounded_text(value, limit=120, default="general").lower()
    if normalized in _TOPIC_CATEGORIES or normalized == "general":
        return normalized
    return _topic_category(normalized)


@dataclass
class HumorAttempt:
    """Privacy-safe delivered-attempt evidence awaiting one reaction."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    delivered_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    humor_type: str = "observational"
    response_digest: str = ""
    response_chars: int = 0
    topic_category: str = "general"
    context_register: str = "casual"
    delivery_receipt_id: str = ""
    landed: bool | None = None
    reaction_digest: str = ""
    reaction_at: float | None = None
    classification_confidence: float = 0.0
    evidence_source: str = "delivered_response"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HumorAttempt:
        data = dict(payload or {})
        delivered_at = _bounded_float(
            data.get("delivered_at", data.get("timestamp")),
            0.0,
            low=0.0,
            high=10**12,
        )
        expires_at = _bounded_float(
            data.get("expires_at"),
            delivered_at + _FEEDBACK_WINDOW_SECONDS,
            low=delivered_at,
            high=10**12,
        )
        raw_content = str(data.pop("content_snippet", "") or "")
        response_digest = _digest_or_empty(data.get("response_digest"))
        if raw_content and not response_digest:
            response_digest = hashlib.sha256(
                raw_content.encode("utf-8", errors="replace")
            ).hexdigest()
        raw_reaction = str(data.pop("user_reaction", "") or "")
        reaction_digest = _digest_or_empty(data.get("reaction_digest"))
        if raw_reaction and not reaction_digest:
            reaction_digest = hashlib.sha256(
                raw_reaction.encode("utf-8", errors="replace")
            ).hexdigest()
        raw_topic = str(data.pop("topic", "") or "")
        topic_category = _normalize_topic_category(
            data.get("topic_category") or raw_topic
        )
        reaction_at_raw = data.get("reaction_at")
        reaction_at = (
            _bounded_float(
                reaction_at_raw,
                delivered_at,
                low=delivered_at,
                high=10**12,
            )
            if reaction_at_raw is not None
            else None
        )
        receipt_id = _bounded_text(data.get("delivery_receipt_id"), limit=200)
        raw_landed = data.get("landed")
        landed = (
            raw_landed
            if isinstance(raw_landed, bool) and receipt_id and reaction_at is not None
            else None
        )
        humor_type = _bounded_text(
            data.get("humor_type"),
            limit=40,
            default="observational",
        ).lower()
        return cls(
            id=_bounded_text(data.get("id"), limit=64, default=uuid.uuid4().hex[:16]),
            delivered_at=delivered_at,
            expires_at=expires_at,
            humor_type=humor_type if humor_type in HUMOR_TYPES else "observational",
            response_digest=response_digest,
            response_chars=_bounded_int(
                data.get("response_chars", len(raw_content)),
                len(raw_content),
                low=0,
                high=1_000_000,
            ),
            topic_category=topic_category,
            context_register=_normalize_register(data.get("context_register")),
            delivery_receipt_id=receipt_id,
            landed=landed,
            reaction_digest=reaction_digest,
            reaction_at=reaction_at,
            classification_confidence=_bounded_float(
                data.get("classification_confidence"),
                0.0,
                low=0.0,
                high=1.0,
            ),
            evidence_source=_bounded_text(
                data.get("evidence_source"),
                limit=80,
                default="legacy_unverified",
            ),
        )


@dataclass
class HumorProfile:
    """Bayesian aggregate over scored delivered attempts for one exact agent."""

    user_id: str
    total_attempts: int = 0
    scored_attempts: int = 0
    total_landed: int = 0
    total_missed: int = 0
    landing_rate: float = 0.5
    confidence: float = 0.0
    type_scores: dict[str, float] = field(default_factory=dict)
    type_sample_counts: dict[str, int] = field(default_factory=dict)
    best_topics: list[str] = field(default_factory=list)
    worst_topics: list[str] = field(default_factory=list)
    banter_streak_record: int = 0
    sarcasm_ceiling: float = 0.2
    irony_layers_max: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HumorProfile:
        data = dict(payload or {})
        raw_scores = data.get("type_scores") or {}
        raw_counts = data.get("type_sample_counts") or {}
        scores = {
            kind: _bounded_float(raw_scores.get(kind), 0.5, low=0.0, high=1.0)
            for kind in HUMOR_TYPES
            if isinstance(raw_scores, dict) and kind in raw_scores
        }
        counts = {
            kind: _bounded_int(raw_counts.get(kind), 0, low=0, high=_MAX_ATTEMPTS_PER_USER)
            for kind in HUMOR_TYPES
            if isinstance(raw_counts, dict) and kind in raw_counts
        }
        valid_topics = set(_TOPIC_CATEGORIES) | {"general"}

        def _topics(key: str) -> list[str]:
            values = data.get(key) or []
            if not isinstance(values, list):
                return []
            return sorted(
                {
                    normalized
                    for value in values[:20]
                    if (normalized := _normalize_topic_category(value)) in valid_topics
                }
            )[:10]

        return cls(
            user_id=_bounded_text(data.get("user_id"), limit=160, default="local_user"),
            total_attempts=_bounded_int(
                data.get("total_attempts"),
                0,
                low=0,
                high=_MAX_ATTEMPTS_PER_USER,
            ),
            scored_attempts=_bounded_int(
                data.get("scored_attempts"),
                0,
                low=0,
                high=_MAX_ATTEMPTS_PER_USER,
            ),
            total_landed=_bounded_int(
                data.get("total_landed"),
                0,
                low=0,
                high=_MAX_ATTEMPTS_PER_USER,
            ),
            total_missed=_bounded_int(
                data.get("total_missed"),
                0,
                low=0,
                high=_MAX_ATTEMPTS_PER_USER,
            ),
            landing_rate=_bounded_float(
                data.get("landing_rate"), 0.5, low=0.0, high=1.0
            ),
            confidence=_bounded_float(
                data.get("confidence"), 0.0, low=0.0, high=0.95
            ),
            type_scores=scores,
            type_sample_counts=counts,
            best_topics=_topics("best_topics"),
            worst_topics=_topics("worst_topics"),
            banter_streak_record=_bounded_int(
                data.get("banter_streak_record"),
                0,
                low=0,
                high=_MAX_ATTEMPTS_PER_USER,
            ),
            sarcasm_ceiling=_bounded_float(
                data.get("sarcasm_ceiling"), 0.2, low=0.0, high=0.75
            ),
            irony_layers_max=_bounded_int(
                data.get("irony_layers_max"), 1, low=1, high=2
            ),
        )


@dataclass
class BanterState:
    """Ephemeral per-agent banter state; never a global relationship signal."""

    active: bool = False
    streak: int = 0
    escalation_level: float = 0.0
    last_humor_type: str = ""
    momentum: float = 0.0
    should_escalate: bool = False
    should_land: bool = False
    max_safe_escalation: float = 0.2
    _last_volley_time: float = field(default_factory=time.time)

    def public_copy(self) -> BanterState:
        return BanterState(**asdict(self))


@dataclass(frozen=True)
class PendingHumorAttempt:
    attempt_id: str
    expires_at: float


class HumorEngine:
    """Owns delivered-attempt pairing and projects authority-backed humor guidance."""

    BANTER_ENTRY_STREAK = 2
    BANTER_LANDING_THRESHOLD = 4
    BANTER_MOMENTUM_DECAY = 0.15
    BANTER_EXIT_PAUSE = 45.0

    def __init__(
        self,
        data_path: Path | None = None,
        *,
        authority: RelationalMemoryAuthority | None = None,
        now_fn: Any = time.time,
    ) -> None:
        if data_path is None:
            try:
                from core.config import config

                data_path = config.paths.data_dir / "humor_profiles.json"
            except (ImportError, AttributeError, RuntimeError):
                data_path = _DEFAULT_DATA_PATH
        self._legacy_path = Path(data_path)
        self._authority = authority or get_relational_memory_authority()
        self._now = now_fn
        self._attempts: dict[str, list[HumorAttempt]] = {}
        self._profiles: dict[str, HumorProfile] = {}
        self._banter_states: dict[str, BanterState] = {}
        self._pending: dict[str, PendingHumorAttempt] = {}
        self._lock = threading.RLock()
        migrated = self._authority.quarantine_legacy_snapshot_file(
            self._legacy_path,
            namespace=_SNAPSHOT_NAMESPACE,
            kind=_SNAPSHOT_KIND,
        )
        logger.info(
            "HumorEngine online (authority-backed, %d legacy profiles quarantined)",
            migrated,
        )

    def _clear_user(self, user_id: str) -> None:
        with self._lock:
            self._attempts.pop(user_id, None)
            self._profiles.pop(user_id, None)
            self._banter_states.pop(user_id, None)
            self._pending.pop(user_id, None)

    def _load_user(self, user_id: str, *, purpose: str) -> bool:
        with self._lock:
            if not self._authority.allows(user_id, _SNAPSHOT_KIND, purpose):
                self._clear_user(user_id)
                return False
            payload = self._authority.load_snapshot(
                user_id,
                namespace=_SNAPSHOT_NAMESPACE,
                kind=_SNAPSHOT_KIND,
                purpose=purpose,
            )
            if payload is not None:
                raw_attempts = payload.get("attempts") or []
                raw_profile = payload.get("profile") or {}
                if isinstance(raw_attempts, list):
                    self._attempts[user_id] = [
                        HumorAttempt.from_dict(item)
                        for item in raw_attempts[-_MAX_ATTEMPTS_PER_USER:]
                        if isinstance(item, dict)
                    ]
                if isinstance(raw_profile, dict):
                    profile = HumorProfile.from_dict(raw_profile)
                    profile.user_id = user_id
                    self._profiles[user_id] = profile
            self._attempts.setdefault(user_id, [])
            self._profiles.setdefault(user_id, HumorProfile(user_id=user_id))
            self._banter_states.setdefault(user_id, BanterState())
            if payload is not None:
                # The aggregate is a projection, never an independent truth.
                # Rebuild it from validated, receipt-paired attempts on every load.
                self._recompute_profile(user_id)
            return True

    def _persist_user(self, user_id: str) -> None:
        with self._lock:
            if not self._authority.allows(user_id, _SNAPSHOT_KIND, "recall"):
                return
            attempts = [
                attempt.to_dict()
                for attempt in self._attempts.get(user_id, [])[-_MAX_ATTEMPTS_PER_USER:]
            ]
            profile = self._profiles.get(user_id, HumorProfile(user_id=user_id))
            payload = {"attempts": attempts, "profile": profile.to_dict()}
            try:
                self._authority.upsert_snapshot(
                    user_id,
                    namespace=_SNAPSHOT_NAMESPACE,
                    kind=_SNAPSHOT_KIND,
                    payload=payload,
                    confidence=profile.confidence,
                    provenance="humor.delivered_outcome_pairing",
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                record_degradation("humor_engine", exc)
                logger.error("Humor authority save failed: %s", exc)

    def observe_delivered_response(
        self,
        user_id: str,
        response_text: str,
        *,
        metadata: dict[str, Any] | None = None,
        delivered_at: float | None = None,
        delivery_receipt_id: str = "",
    ) -> HumorAttempt | None:
        """Open feedback only after the caller confirms output delivery succeeded."""
        with self._lock:
            if not self._load_user(user_id, purpose="recall"):
                return None
            descriptor = self._classify_delivered_humor(response_text, metadata or {})
            if descriptor is None:
                self._pending.pop(user_id, None)
                return None
            humor_type, topic, register, evidence_source = descriptor
            at = float(self._now()) if delivered_at is None else float(delivered_at)
            response = str(response_text or "")
            receipt_id = _bounded_text(delivery_receipt_id, limit=200)
            if not receipt_id:
                return None
            return self.record_attempt(
                user_id,
                humor_type,
                response,
                topic,
                register,
                delivered_at=at,
                delivery_receipt_id=receipt_id,
                evidence_source=evidence_source,
            )

    def record_attempt(
        self,
        user_id: str,
        humor_type: str,
        content: str,
        topic: str,
        register: str = "casual",
        *,
        delivered_at: float | None = None,
        delivery_receipt_id: str,
        evidence_source: str = "explicit_response_metadata",
    ) -> HumorAttempt | None:
        receipt_id = _bounded_text(delivery_receipt_id, limit=200)
        if not receipt_id:
            raise ValueError("humor attempt requires successful-delivery receipt evidence")
        with self._lock:
            if not self._load_user(user_id, purpose="recall"):
                return None
            at = float(self._now()) if delivered_at is None else float(delivered_at)
            response = str(content or "")
            attempt = HumorAttempt(
                delivered_at=at,
                expires_at=at + _FEEDBACK_WINDOW_SECONDS,
                humor_type=humor_type if humor_type in HUMOR_TYPES else "observational",
                response_digest=hashlib.sha256(
                    response.encode("utf-8", errors="replace")
                ).hexdigest(),
                response_chars=min(1_000_000, len(response)),
                topic_category=_topic_category(topic),
                context_register=_normalize_register(register),
                delivery_receipt_id=receipt_id,
                evidence_source=_bounded_text(
                    evidence_source,
                    limit=80,
                    default="delivered_response",
                ),
            )
            attempts = self._attempts.setdefault(user_id, [])
            attempts.append(attempt)
            self._attempts[user_id] = attempts[-_MAX_ATTEMPTS_PER_USER:]
            self._pending[user_id] = PendingHumorAttempt(
                attempt_id=attempt.id,
                expires_at=attempt.expires_at,
            )
            self._persist_user(user_id)
            return attempt

    def record_reaction(
        self,
        user_id: str,
        user_message: str,
        timestamp: float = 0.0,
    ) -> bool | None:
        at = float(timestamp or self._now())
        landed, classification_confidence = self._detect_humor_landing(user_message)
        with self._lock:
            if not self._load_user(user_id, purpose="recall"):
                return None
            pending = self._pending.get(user_id)
            if pending is None or at > pending.expires_at:
                self._pending.pop(user_id, None)
                return None
            attempt = next(
                (
                    item
                    for item in reversed(self._attempts.get(user_id, []))
                    if item.id == pending.attempt_id
                ),
                None,
            )
            if attempt is None or attempt.landed is not None or attempt.reaction_at is not None:
                self._pending.pop(user_id, None)
                return None
            if at <= attempt.delivered_at:
                return None
            self._pending.pop(user_id, None)
            attempt.landed = landed
            attempt.reaction_digest = hashlib.sha256(
                str(user_message or "").encode("utf-8", errors="replace")
            ).hexdigest()
            attempt.reaction_at = at
            attempt.classification_confidence = classification_confidence
            state = self._banter_states.setdefault(user_id, BanterState())
            if landed is True:
                state.streak += 1
                state._last_volley_time = at
                state.momentum = min(1.0, state.momentum + 0.25)
                state.last_humor_type = attempt.humor_type
            elif landed is False:
                state.streak = 0
                state.momentum = max(0.0, state.momentum - 0.4)
            self._recompute_profile(user_id)
            self._persist_user(user_id)
            return landed

    @staticmethod
    def _detect_humor_landing(user_message: str) -> tuple[bool | None, float]:
        message = str(user_message or "").strip()
        if not message:
            return None, 0.0
        if any(pattern.search(message) for pattern in _MISSED_PATTERNS):
            return False, 0.9
        if any(pattern.search(message) for pattern in _LANDED_PATTERNS):
            return True, 0.9
        return None, 0.0

    @staticmethod
    def _classify_delivered_humor(
        response_text: str,
        metadata: dict[str, Any],
    ) -> tuple[str, str, str, str] | None:
        candidates: list[dict[str, Any]] = []
        for key in ("humor_attempt", "response_metadata", "response_modifiers"):
            nested = metadata.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
        for candidate in candidates:
            if (
                candidate.get("provenance") != "aura.response_generation"
                or candidate.get("response_humor_attempt") is not True
            ):
                continue
            explicit_type = str(candidate.get("humor_type") or "").strip().lower()
            explicitly_active = candidate.get("humor_frame_active") is True
            if explicit_type in HUMOR_TYPES or explicitly_active:
                return (
                    explicit_type if explicit_type in HUMOR_TYPES else "observational",
                    str(candidate.get("topic") or "general"),
                    str(candidate.get("register") or "casual"),
                    "explicit_response_metadata",
                )
        if not _OVERT_HUMOR_RESPONSE.search(str(response_text or "")):
            return None
        lowered = response_text.lower()
        humor_type = "pun" if "pun" in lowered else "observational"
        return humor_type, str(metadata.get("topic") or "general"), "casual", "overt_response_marker"

    def _recompute_profile(self, user_id: str) -> None:
        with self._lock:
            attempts = list(self._attempts.get(user_id, []))
            state = self._banter_states.setdefault(user_id, BanterState())
        scored = [attempt for attempt in attempts if attempt.landed is not None]
        landed_count = sum(attempt.landed is True for attempt in scored)
        missed_count = sum(attempt.landed is False for attempt in scored)
        type_totals: dict[str, int] = {}
        type_landed: dict[str, int] = {}
        topic_totals: dict[str, int] = {}
        topic_landed: dict[str, int] = {}
        for attempt in scored:
            type_totals[attempt.humor_type] = type_totals.get(attempt.humor_type, 0) + 1
            topic_totals[attempt.topic_category] = topic_totals.get(attempt.topic_category, 0) + 1
            if attempt.landed is True:
                type_landed[attempt.humor_type] = type_landed.get(attempt.humor_type, 0) + 1
                topic_landed[attempt.topic_category] = topic_landed.get(attempt.topic_category, 0) + 1
        type_scores = {
            humor_type: (type_landed.get(humor_type, 0) + 1) / (count + 2)
            for humor_type, count in type_totals.items()
        }
        best_topics = [
            topic
            for topic, count in topic_totals.items()
            if count >= 3 and (topic_landed.get(topic, 0) + 1) / (count + 2) >= 0.7
        ]
        worst_topics = [
            topic
            for topic, count in topic_totals.items()
            if count >= 3 and (topic_landed.get(topic, 0) + 1) / (count + 2) <= 0.3
        ]
        sarcasm_types = ("sarcasm", "dark", "dry_wit")
        sarcasm_total = sum(type_totals.get(kind, 0) for kind in sarcasm_types)
        sarcasm_landed = sum(type_landed.get(kind, 0) for kind in sarcasm_types)
        sarcasm_ceiling = 0.2
        if sarcasm_total >= 3:
            posterior = (sarcasm_landed + 1) / (sarcasm_total + 2)
            sarcasm_ceiling = min(0.75, 0.15 + posterior * 0.6)
        irony_types = ("absurdist", "surreal", "sarcasm")
        irony_total = sum(type_totals.get(kind, 0) for kind in irony_types)
        irony_landed = sum(type_landed.get(kind, 0) for kind in irony_types)
        irony_layers = 1
        if irony_total >= 5 and (irony_landed + 1) / (irony_total + 2) >= 0.7:
            irony_layers = 2
        previous = self._profiles.get(user_id, HumorProfile(user_id=user_id))
        profile = HumorProfile(
            user_id=user_id,
            total_attempts=len(attempts),
            scored_attempts=len(scored),
            total_landed=landed_count,
            total_missed=missed_count,
            landing_rate=(landed_count + 1) / (len(scored) + 2),
            confidence=min(0.95, len(scored) / 10.0),
            type_scores=type_scores,
            type_sample_counts=type_totals,
            best_topics=sorted(best_topics),
            worst_topics=sorted(worst_topics),
            banter_streak_record=max(previous.banter_streak_record, state.streak),
            sarcasm_ceiling=sarcasm_ceiling,
            irony_layers_max=irony_layers,
        )
        state.max_safe_escalation = sarcasm_ceiling
        with self._lock:
            self._profiles[user_id] = profile

    def update_banter_state(
        self,
        user_message: str,
        dynamics_state: Any = None,
        *,
        user_id: str = "",
    ) -> None:
        del user_message
        if not user_id or not self._load_user(user_id, purpose="recall"):
            return
        now = float(self._now())
        with self._lock:
            state = self._banter_states.setdefault(user_id, BanterState())
            elapsed = now - state._last_volley_time
            state.momentum = max(
                0.0,
                state.momentum - self.BANTER_MOMENTUM_DECAY * max(0.0, elapsed),
            )
            if elapsed > self.BANTER_EXIT_PAUSE:
                state.active = False
                state.streak = 0
                state.escalation_level = 0.0
                state.momentum = 0.0
            if dynamics_state is not None:
                frame = str(getattr(dynamics_state, "partner_frame", "neutral"))
                register = str(getattr(dynamics_state, "register", "casual"))
                humor_active = bool(getattr(dynamics_state, "humor_frame_active", False))
                escalation_invited = bool(getattr(dynamics_state, "escalation_invited", False))
                if frame in {"vulnerable", "serious", "anxious"}:
                    state.active = False
                    state.streak = 0
                    state.escalation_level = 0.0
                    state.momentum = 0.0
                elif (
                    humor_active
                    and register == "playful"
                    and state.streak >= self.BANTER_ENTRY_STREAK
                ):
                    state.active = True
                    if escalation_invited:
                        state.escalation_level = min(1.0, state.escalation_level + 0.15)
            state.should_escalate = (
                state.active
                and state.momentum > 0.5
                and state.escalation_level < state.max_safe_escalation - 0.1
            )
            state.should_land = (
                state.active
                and (
                    state.streak >= self.BANTER_LANDING_THRESHOLD
                    or state.escalation_level >= state.max_safe_escalation - 0.05
                )
            )

    def get_humor_guidance(self, user_id: str) -> str:
        if not self._load_user(user_id, purpose="prompt"):
            return ""
        with self._lock:
            profile = self._profiles.get(user_id, HumorProfile(user_id=user_id))
        lines = [
            "## HUMOR OUTCOME EVIDENCE",
            "- Use only paired delivered-attempt outcomes. Never infer taste from generic praise, brevity, or unrelated reactions.",
        ]
        if profile.scored_attempts < 3:
            lines.extend(
                [
                    f"- Evidence is sparse ({profile.scored_attempts} scored attempts; confidence={profile.confidence:.2f}).",
                    "- Keep humor low-risk, optional, and easy to ignore; do not escalate or use edgy material.",
                ]
            )
        else:
            lines.append(
                f"- Bayesian landing estimate: {profile.landing_rate:.2f} across "
                f"{profile.scored_attempts} scored attempts (confidence={profile.confidence:.2f})."
            )
            supported = [
                (kind, score, profile.type_sample_counts.get(kind, 0))
                for kind, score in profile.type_scores.items()
                if profile.type_sample_counts.get(kind, 0) >= 3
            ]
            positive = sorted(
                (item for item in supported if item[1] >= 0.65),
                key=lambda item: item[1],
                reverse=True,
            )
            negative = sorted(item for item in supported if item[1] <= 0.35)
            if positive:
                lines.append(
                    "- Better-supported forms: "
                    + ", ".join(
                        f"{kind} ({score:.2f}, n={count})"
                        for kind, score, count in positive[:3]
                    )
                )
            if negative:
                lines.append(
                    "- Poorly supported forms: "
                    + ", ".join(
                        f"{kind} ({score:.2f}, n={count})"
                        for kind, score, count in negative[:3]
                    )
                )
            lines.append(
                f"- Conservative sarcasm ceiling: {profile.sarcasm_ceiling:.2f}."
            )
        directive = self.get_banter_directive(user_id)
        if directive:
            lines.append(f"- {directive}")
        return "\n".join(lines)

    def get_banter_directive(self, user_id: str | None = None) -> str:
        if not user_id:
            return ""
        if not self._authority.allows(user_id, _SNAPSHOT_KIND, "prompt"):
            if not self._authority.allows(user_id, _SNAPSHOT_KIND, "recall"):
                self._clear_user(user_id)
            return ""
        with self._lock:
            state = self._banter_states.get(user_id)
            if state is None or not state.active:
                return ""
            if state.should_land:
                callback = self._get_callback_suggestion(user_id)
                return (
                    f"Land the bit with the consented callback {callback!r}."
                    if callback
                    else "Land the bit cleanly; do not keep escalating."
                )
            if state.should_escalate:
                return "A slight escalation is supported by the current paired streak; stay reversible."
            return "Banter is active; keep it brief and let the user redirect immediately."

    def _get_callback_suggestion(self, user_id: str) -> str:
        try:
            from core.memory.shared_ground import get_shared_ground

            entries = get_shared_ground().get_top_entries(3, agent_id=user_id)
            jokes = [entry for entry in entries if {"joke", "humor"} & set(entry.tags)]
            selected = jokes[0] if jokes else entries[0] if entries else None
            return selected.reference if selected is not None else ""
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("humor_engine.callback", exc)
            return ""

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            known_agents = set(self._profiles) | set(self._attempts) | set(self._banter_states)
        for user_id in known_agents:
            if not self._authority.allows(user_id, _SNAPSHOT_KIND, "recall"):
                self._clear_user(user_id)
        with self._lock:
            return {
                "profiles": len(self._profiles),
                "total_attempts": sum(len(items) for items in self._attempts.values()),
                "pending_feedback_windows": len(self._pending),
                "active_banter_agents": sum(state.active for state in self._banter_states.values()),
                "canonical_owner": "relational_memory",
            }

    def get_profile(self, user_id: str) -> HumorProfile | None:
        if not self._load_user(user_id, purpose="recall"):
            return None
        with self._lock:
            profile = self._profiles.get(user_id)
            return HumorProfile.from_dict(profile.to_dict()) if profile is not None else None

    def get_banter_state(self, user_id: str | None = None) -> BanterState:
        if not user_id:
            return BanterState()
        if not self._authority.allows(user_id, _SNAPSHOT_KIND, "recall"):
            self._clear_user(user_id)
            return BanterState()
        with self._lock:
            return self._banter_states.get(user_id, BanterState()).public_copy()


def _topic_category(value: str) -> str:
    normalized = " ".join(str(value or "").strip().lower().split())
    for category, markers in _TOPIC_CATEGORIES.items():
        if any(
            re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", normalized)
            for marker in markers
        ):
            return category
    return "general"


def register_humor_engine() -> None:
    try:
        ServiceContainer.register(
            "humor_engine",
            factory=HumorEngine,
            lifetime=ServiceLifetime.SINGLETON,
        )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("humor_engine.registration", exc)


_instance: HumorEngine | None = None
_instance_lock = threading.Lock()


def get_humor_engine() -> HumorEngine:
    global _instance
    if _instance is None:
        try:
            _instance = ServiceContainer.get("humor_engine", default=None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("humor_engine.container_lookup", exc)
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = HumorEngine()
                try:
                    ServiceContainer.register_instance("humor_engine", _instance)
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation("humor_engine.registration", exc)
    return _instance
