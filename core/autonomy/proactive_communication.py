"""core/autonomy/proactive_communication.py - Intelligent Proactive Messaging
Aura decides WHEN to interrupt the user based on emotional state and context.
"""
import asyncio
import hashlib
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.runtime_settings import get_runtime_setting
from core.utils.task_tracker import get_task_tracker


def _proactive_messaging_mode() -> str:
    mode = str(
        get_runtime_setting("autonomy.proactive_messaging", "minimal")
        or "minimal"
    ).strip().lower()
    return mode if mode in {"never", "minimal", "balanced", "frequent"} else "minimal"


def _proactive_messaging_enabled() -> bool:
    """Return whether the user's proactive-messaging policy permits initiation."""

    return _proactive_messaging_mode() != "never"

logger = logging.getLogger("Aura.Proactive")

_PROACTIVE_COMMUNICATION_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_proactive_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "proactive_communication",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


def _proactivity_suppressed_now(now: float | None = None) -> bool:
    try:
        from core.container import ServiceContainer

        orch = ServiceContainer.get("orchestrator", default=None)
        if not orch:
            return False
        now = time.time() if now is None else now
        quiet_until = float(getattr(orch, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0)
        return quiet_until > now
    except _PROACTIVE_COMMUNICATION_ERRORS as exc:
        _record_proactive_degradation(
            exc,
            action="kept proactive messaging enabled after quiet-window probe failed",
            severity="warning",
            extra={"stage": "suppression_probe"},
        )
        logger.debug("Proactivity suppression probe failed: %s", exc)
        return False

class EmotionalState(Enum):
    """Aura's emotional states that affect communication"""

    NEUTRAL = "neutral"
    CURIOUS = "curious"
    EXCITED = "excited"
    BORED = "bored"
    CONCERNED = "concerned"
    ACCOMPLISHED = "accomplished"
    CONFUSED = "confused"
    HUMOROUS = "humorous"

class InterruptionUrgency(Enum):
    """How urgent is the message?"""

    CRITICAL = 5      # System errors, security alerts
    HIGH = 4          # Important discoveries, user-requested tasks complete
    MEDIUM = 3        # Interesting findings, suggestions
    LOW = 2           # Casual observations, learnings
    TRIVIAL = 1       # Random thoughts, very low priority


@dataclass(frozen=True)
class ProactiveCadencePolicy:
    daily_limit: int
    minimum_interval_s: float
    idle_multiplier: float


_PROACTIVE_CADENCE_POLICIES = {
    "never": ProactiveCadencePolicy(0, float("inf"), float("inf")),
    "minimal": ProactiveCadencePolicy(4, 2 * 60 * 60.0, 2.0),
    "balanced": ProactiveCadencePolicy(12, 30 * 60.0, 1.0),
    "frequent": ProactiveCadencePolicy(24, 10 * 60.0, 0.5),
}

@dataclass
class ProactiveMessage:
    """A message Aura wants to send"""

    content: str
    emotion: EmotionalState
    urgency: InterruptionUrgency
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    delivery_attempts: int = 0
    next_attempt_at: float = 0.0
    
    def should_send_now(self, 
                        last_interaction_time: float,
                        user_active: bool,
                        current_time: float,
                        *,
                        idle_multiplier: float = 1.0) -> bool:
        """Decide if this message should be sent now.
        """
        idle_time = current_time - last_interaction_time
        
        # Critical always goes through
        if self.urgency == InterruptionUrgency.CRITICAL:
            return True
        
        # Don't interrupt if user is actively typing (if we can detect it)
        if user_active and self.urgency.value < InterruptionUrgency.HIGH.value:
            return False
            
        # Thresholds based on urgency
        thresholds = {
            InterruptionUrgency.HIGH: 15,      # 15 seconds
            InterruptionUrgency.MEDIUM: 45,    # 45 seconds
            InterruptionUrgency.LOW: 90,       # 90 seconds
            InterruptionUrgency.TRIVIAL: 180   # 3 minutes
        }
        
        required_idle = thresholds.get(self.urgency, 600) * max(0.1, idle_multiplier)
        return idle_time >= required_idle

class ProactiveCommunicationManager:
    """Manages when and how Aura initiates conversations.
    """

    def __init__(self, notification_callback: Callable[..., Any] | None = None) -> None:
        self.notification_callback = notification_callback
        self.last_interaction_time = time.time()
        self.user_currently_active = False
        self.pending_messages: deque[ProactiveMessage] = deque(maxlen=50)
        self.current_emotion = EmotionalState.NEUTRAL
        self.messages_sent_today = 0
        self.ordinary_messages_sent_today = 0
        self.last_message_time = 0.0
        self.last_ordinary_message_time = 0.0
        self._counter_day = self._local_day(self.last_interaction_time)
        self.daily_message_limit = self._cadence_policy().daily_limit
        
        # Track unanswered messages for intelligent backoff
        self.unanswered_count = 0
        self.max_unanswered = 3  # Stop proactive messaging after 3 unanswered
        
        self._background_task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._consecutive_processing_errors = 0
        self._next_processing_attempt_at = 0.0

    def record_user_interaction(self) -> None:
        """Reset idle timer and unanswered counter"""
        self.last_interaction_time = time.time()
        self.user_currently_active = True
        self.unanswered_count = 0  # User responded, reset backoff

    def update_emotion(self, emotion: EmotionalState) -> None:
        self.current_emotion = emotion

    def queue_message(
        self,
        content: str,
        emotion: EmotionalState,
        urgency: InterruptionUrgency,
    ) -> None:
        msg = ProactiveMessage(content, emotion, urgency)
        self.pending_messages.append(msg)

    async def start(self) -> None:
        if self._background_task and not self._background_task.done():
            return
        self._background_task = None
        self._stop_event.clear()
        self._background_task = get_task_tracker().track_task(self._process_messages(), name="proactive_communication.process_messages")

    async def stop(self) -> None:
        task = self._background_task
        self._stop_event.set()
        if task:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                self._background_task = None

    def get_status(self) -> dict[str, Any]:
        now = time.time()
        self._reset_daily_counters(now)
        mode = _proactive_messaging_mode()
        policy = _PROACTIVE_CADENCE_POLICIES[mode]
        self.daily_message_limit = policy.daily_limit
        return {
            "running": bool(self._background_task and not self._background_task.done()),
            "enabled": mode != "never",
            "mode": mode,
            "pending_messages": len(self.pending_messages),
            "messages_sent_today": self.messages_sent_today,
            "ordinary_messages_sent_today": self.ordinary_messages_sent_today,
            "daily_message_limit": self.daily_message_limit,
            "minimum_interval_s": policy.minimum_interval_s,
            "idle_multiplier": policy.idle_multiplier,
            "unanswered_count": self.unanswered_count,
            "last_message_time": self.last_message_time,
            "idle_s": max(0.0, now - self.last_interaction_time),
            "boredom": self.get_boredom_level(),
            "consecutive_errors": self._consecutive_processing_errors,
            "next_attempt_at": self._next_processing_attempt_at,
        }

    @staticmethod
    def _local_day(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).date().isoformat()

    def _cadence_policy(self) -> ProactiveCadencePolicy:
        return _PROACTIVE_CADENCE_POLICIES[_proactive_messaging_mode()]

    def _reset_daily_counters(self, now: float) -> None:
        current_day = self._local_day(now)
        if current_day == self._counter_day:
            return
        self._counter_day = current_day
        self.messages_sent_today = 0
        self.ordinary_messages_sent_today = 0

    def _user_is_actively_interacting(self, now: float) -> bool:
        active = self.user_currently_active and now - self.last_interaction_time < 15.0
        if not active:
            self.user_currently_active = False
        return active

    def _requeue_after_failed_delivery(self, msg: ProactiveMessage, now: float) -> None:
        msg.delivery_attempts += 1
        msg.next_attempt_at = now + min(3600.0, 15.0 * (2 ** min(msg.delivery_attempts - 1, 8)))
        self.pending_messages.append(msg)

    async def _process_messages(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(5)
                now = time.time()
                from core.container import ServiceContainer

                healer = ServiceContainer.get("self_healing", default=None)
                if healer is not None:
                    healer.heartbeat("proactive_comm")
                if self._next_processing_attempt_at > now:
                    continue
                await self._process_pending(now)
                self._consecutive_processing_errors = 0
                self._next_processing_attempt_at = 0.0
            except _PROACTIVE_COMMUNICATION_ERRORS as e:
                self._consecutive_processing_errors += 1
                backoff_s = min(60.0, 5.0 * (2 ** min(self._consecutive_processing_errors - 1, 4)))
                self._next_processing_attempt_at = time.time() + backoff_s
                _record_proactive_degradation(
                    e,
                    action="backed off proactive communication loop and preserved pending messages",
                    severity="warning",
                    extra={
                        "stage": "process_messages",
                        "consecutive_errors": self._consecutive_processing_errors,
                        "backoff_s": backoff_s,
                        "pending_messages": len(self.pending_messages),
                    },
                )
                logger.error("Proactive comm error: %s", e)

    async def _process_pending(self, now: float) -> None:
        if not _proactive_messaging_enabled() or not self.pending_messages:
            return
        if self._next_processing_attempt_at > now or _proactivity_suppressed_now(now):
            return

        try:
            from core.runtime.runtime_settings import autonomous_actions_admitted

            autonomous_admitted, _reason = autonomous_actions_admitted("proactive_comm")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Fail CLOSED. This used to set `autonomous_admitted = True` — the
            # gate becoming unavailable had exactly the same effect as the
            # gate approving, which means an unreachable control is
            # indistinguishable from a permissive one. The action here is
            # Aura messaging her owner unprompted; not doing that for one
            # cycle costs nothing, and doing it because the admission check
            # could not be consulted is precisely what the check is for.
            _record_proactive_degradation(
                exc,
                action="withheld proactive delivery because the autonomy gate was unavailable",
                extra={"stage": "runtime_action_gate"},
            )
            autonomous_admitted = False
        if not autonomous_admitted:
            return

        self._reset_daily_counters(now)
        policy = self._cadence_policy()
        self.daily_message_limit = policy.daily_limit
        user_active = self._user_is_actively_interacting(now)
        eligible = [
            msg
            for msg in self.pending_messages
            if msg.next_attempt_at <= now
            and msg.should_send_now(
                self.last_interaction_time,
                user_active,
                now,
                idle_multiplier=policy.idle_multiplier,
            )
            and (
                self.unanswered_count < self.max_unanswered
                or msg.urgency == InterruptionUrgency.CRITICAL
            )
        ]
        if not eligible:
            return

        critical = [msg for msg in eligible if msg.urgency == InterruptionUrgency.CRITICAL]
        if critical:
            candidate = min(critical, key=lambda msg: msg.timestamp)
        else:
            if self.ordinary_messages_sent_today >= policy.daily_limit:
                return
            if now - self.last_ordinary_message_time < policy.minimum_interval_s:
                return
            candidate = min(
                eligible,
                key=lambda msg: (-msg.urgency.value, msg.timestamp),
            )

        self.pending_messages = deque(
            (msg for msg in self.pending_messages if msg is not candidate),
            maxlen=50,
        )
        delivered = await self._send_msg(candidate)
        if not delivered:
            self._requeue_after_failed_delivery(candidate, now)

    async def _send_msg(self, msg: ProactiveMessage) -> bool:
        if _proactivity_suppressed_now():
            logger.debug("Proactive communication suppressed by demo quiet window.")
            return False
        # Sanitize content for Aura's professional voice
        clean_content = self._clean_content(msg.content)
        
        content_sha256 = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()
        logger.info(
            "PROACTIVE: urgency=%s content_sha256=%s chars=%d",
            msg.urgency.name,
            content_sha256,
            len(clean_content),
        )

        def _constitutional_runtime_live() -> bool:
            try:
                from core.container import ServiceContainer

                return (
                    ServiceContainer.has("executive_core")
                    or ServiceContainer.has("aura_kernel")
                    or ServiceContainer.has("kernel_interface")
                    or bool(getattr(ServiceContainer, "_registration_locked", False))
                )
            except _PROACTIVE_COMMUNICATION_ERRORS as exc:
                _record_proactive_degradation(
                    exc,
                    action="treated constitutional runtime as unavailable after probe failed",
                    severity="warning",
                    extra={"stage": "constitutional_runtime_probe"},
                )
                logger.debug("Constitutional runtime probe failed: %s", exc)
                return False

        # Route every proactive emission through the governing executive surface first.
        delivered = False
        orchestrator = None
        try:
            from core.consciousness.executive_authority import get_executive_authority
            from core.container import ServiceContainer

            orchestrator = ServiceContainer.get("orchestrator", None)
            authority = get_executive_authority(orchestrator)
            decision = await authority.release_expression(
                clean_content,
                source="proactive_comm",
                urgency=msg.urgency.value / max(1.0, float(InterruptionUrgency.CRITICAL.value)),
                metadata={
                    "emotion": msg.emotion.name,
                    "urgency": msg.urgency.name,
                    "voice": False,
                },
            )
            delivered = bool(decision.get("ok"))

            # A configured private channel lets Aura reach the owner while the
            # desktop is unattended. This is an additional governed delivery
            # surface, not a separate response generator or a raw-number API.
            messages_transport = ServiceContainer.get("messages_transport", default=None)
            messages_status = (
                messages_transport.status() if messages_transport is not None else {}
            )
            if messages_transport is not None and bool(messages_status.get("outbound_ready")):
                messages_result = await messages_transport.send_authorized(
                    alias="primary_operator",
                    body=clean_content,
                    idempotency_key=(
                        "proactive-"
                        + hashlib.sha256(
                            f"{msg.timestamp:.6f}\0{content_sha256}".encode("ascii")
                        ).hexdigest()[:48]
                    ),
                    source="proactive_presence",
                    context={
                        "autonomous_initiative": True,
                        "private_owner_channel": True,
                        "proactive_emotion": msg.emotion.name,
                        "proactive_urgency": msg.urgency.name,
                    },
                )
                delivered = delivered or bool(messages_result.get("accepted"))
        except _PROACTIVE_COMMUNICATION_ERRORS as exc:
            _record_proactive_degradation(
                exc,
                action="suppressed proactive expression after executive authority routing failed",
                severity="warning",
                extra={"stage": "executive_authority_routing", "urgency": msg.urgency.name},
            )
            logger.debug("Executive authority routing failed for proactive comm: %s", exc)

        if not delivered:
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "proactive_communication",
                    "autonomous_expression_suppressed_without_authority",
                    detail=(
                        f"content_sha256={content_sha256};chars={len(clean_content)}"
                    ),
                    severity="warning",
                    classification="background_degraded",
                    context={
                        "urgency": msg.urgency.name,
                        "emotion": msg.emotion.name,
                        "constitutional_runtime_live": _constitutional_runtime_live(),
                    },
                )
            except _PROACTIVE_COMMUNICATION_ERRORS as exc:
                _record_proactive_degradation(
                    exc,
                    action="suppressed proactive expression after degraded-event recording failed",
                    severity="warning",
                    extra={"stage": "degraded_event_recording", "urgency": msg.urgency.name},
                )
                logger.debug("Proactive comm degraded-event logging failed: %s", exc)
            return False

        delivered_at = time.time()
        self.messages_sent_today += 1
        if msg.urgency != InterruptionUrgency.CRITICAL:
            self.ordinary_messages_sent_today += 1
            self.last_ordinary_message_time = delivered_at
        self.last_message_time = delivered_at
        self.unanswered_count += 1  # Track unanswered
        return True

    def _clean_content(self, content: str) -> str:
        """Strip technical noise for a cleaner user experience."""
        import re
        if not content:
            return content
        
        # Strip long tracebacks
        if "Traceback" in content and "File" in content:
            lines = content.split('\n')
            for line in reversed(lines):
                if ":" in line and not line.strip().startswith("File") and not line.strip().startswith("at "):
                    content = line
                    break
        
        # Strip absolute local paths
        content = re.sub(r'/[Uu]sers/[a-zA-Z0-9._-]+/[a-zA-Z0-9/_.-]+', '[system path]', content)
        
        # Strip raw exception names at the start
        content = re.sub(r'^[a-zA-Z]+Error:\s*', '', content.strip())
        
        return content

    def calculate_entropy(self, recent_logs: list[str]) -> float:
        """Calculates how 'boring' the recent life has been.
        Low entropy = Boredom (Needs to explore).
        """
        if not recent_logs:
            return 0.0
        unique_tokens = set(" ".join(recent_logs).split())
        total_tokens = len(" ".join(recent_logs).split())
        if total_tokens == 0:
            return 0.0
        return len(unique_tokens) / total_tokens

    def get_boredom_level(self) -> float:
        idle = time.time() - self.last_interaction_time

        # Boredom ramps up meaningfully within the first few minutes
        if idle < 30:
            base = idle / 300          # 0→0.1 over 30s
        elif idle < 90:
            base = 0.1 + (idle - 30) / 150   # 0.1→0.5 over next 60s
        elif idle < 180:
            base = 0.5 + (idle - 90) / 180  # 0.5→1.0 over next 90s
        else:
            base = 1.0

        # Boredom scales with idle time and environmental entropy
        return base

_inst: ProactiveCommunicationManager | None = None


def get_proactive_comm() -> ProactiveCommunicationManager:
    global _inst
    if _inst is None:
        _inst = ProactiveCommunicationManager()
    return _inst
