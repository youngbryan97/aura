"""core/proactive_communication.py - Intelligent Proactive Messaging
Aura decides WHEN to interrupt the user based on emotional state and context.
"""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Proactive")


def _proactivity_suppressed_now(now: Optional[float] = None) -> bool:
    try:
        from core.container import ServiceContainer

        orch = ServiceContainer.get("orchestrator", default=None)
        if not orch:
            return False
        now = time.time() if now is None else now
        quiet_until = float(getattr(orch, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0)
        return quiet_until > now
    except Exception:
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

@dataclass
class ProactiveMessage:
    """A message Aura wants to send"""

    content: str
    emotion: EmotionalState
    urgency: InterruptionUrgency
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def should_send_now(self, 
                        last_interaction_time: float,
                        user_active: bool,
                        current_time: float) -> bool:
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
            InterruptionUrgency.HIGH: 30,      # 30 seconds
            InterruptionUrgency.MEDIUM: 120,   # 2 minutes
            InterruptionUrgency.LOW: 300,      # 5 minutes
            InterruptionUrgency.TRIVIAL: 600   # 10 minutes
        }
        
        required_idle = thresholds.get(self.urgency, 600)
        return idle_time >= required_idle

class ProactiveCommunicationManager:
    """Manages when and how Aura initiates conversations.
    """

    def __init__(self, notification_callback: Optional[Callable] = None):
        self.notification_callback = notification_callback
        self.last_interaction_time = time.time()
        self.user_currently_active = False
        self.pending_messages: deque[ProactiveMessage] = deque(maxlen=50)
        self.current_emotion = EmotionalState.NEUTRAL
        self.messages_sent_today = 0
        self.last_message_time = 0
        self.daily_message_limit = 20
        
        # Track unanswered messages for intelligent backoff
        self.unanswered_count = 0
        self.max_unanswered = 3  # Stop proactive messaging after 3 unanswered
        
        self._background_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def record_user_interaction(self):
        """Reset idle timer and unanswered counter"""
        self.last_interaction_time = time.time()
        self.user_currently_active = True
        self.unanswered_count = 0  # User responded, reset backoff

    def update_emotion(self, emotion: EmotionalState):
        self.current_emotion = emotion

    def queue_message(self, content: str, emotion: EmotionalState, urgency: InterruptionUrgency):
        msg = ProactiveMessage(content, emotion, urgency)
        self.pending_messages.append(msg)

    async def start(self):
        if self._background_task: return
        self._stop_event.clear()
        self._background_task = get_task_tracker().track_task(
            asyncio.create_task(self._process_messages(), name="proactive_communication.process_messages")
        )

    async def stop(self):
        if self._background_task:
            self._stop_event.set()
            await self._background_task

    async def _process_messages(self):
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(5)
                now = time.time()
                
                # Simple rate limiting check
                if self.messages_sent_today >= self.daily_message_limit:
                    continue
                if now - self.last_message_time < 30: # Min 30s between messages
                    continue
                if _proactivity_suppressed_now(now):
                    continue
                
                # Stop proactive messaging if user isn't responding
                if self.unanswered_count >= self.max_unanswered:
                    # Only let CRITICAL messages through when user is silent
                    ready = []
                    remaining = deque()
                    while self.pending_messages:
                        msg = self.pending_messages.popleft()
                        if msg.urgency == InterruptionUrgency.CRITICAL and msg.should_send_now(self.last_interaction_time, self.user_currently_active, now):
                            ready.append(msg)
                        else:
                            remaining.append(msg)
                    self.pending_messages = remaining
                    for msg in ready:
                        await self._send_msg(msg)
                    continue

                # Collect messages that can be sent
                ready = []
                remaining = deque()
                while self.pending_messages:
                    msg = self.pending_messages.popleft()
                    if msg.should_send_now(self.last_interaction_time, self.user_currently_active, now):
                        ready.append(msg)
                    else:
                        remaining.append(msg)
                self.pending_messages = remaining

                for msg in ready:
                    await self._send_msg(msg)
            except Exception as e:
                logger.error("Proactive comm error: %s", e)

    async def _send_msg(self, msg: ProactiveMessage):
        if _proactivity_suppressed_now():
            logger.debug("Proactive communication suppressed by demo quiet window.")
            return
        # Sanitize content for Aura's professional voice
        clean_content = self._clean_content(msg.content)
        
        logger.info("PROACTIVE: (%s) %s", msg.urgency.name, clean_content)

        def _constitutional_runtime_live() -> bool:
            try:
                from core.container import ServiceContainer

                return (
                    ServiceContainer.has("executive_core")
                    or ServiceContainer.has("aura_kernel")
                    or ServiceContainer.has("kernel_interface")
                    or bool(getattr(ServiceContainer, "_registration_locked", False))
                )
            except Exception:
                return False

        # Route every proactive emission through the governing executive surface first.
        delivered = False
        orchestrator = None
        try:
            from core.container import ServiceContainer
            from core.consciousness.executive_authority import get_executive_authority

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
        except Exception as exc:
            logger.debug("Executive authority routing failed for proactive comm: %s", exc)

        if not delivered:
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "proactive_communication",
                    "autonomous_expression_suppressed_without_authority",
                    detail=clean_content[:120],
                    severity="warning",
                    classification="background_degraded",
                    context={
                        "urgency": msg.urgency.name,
                        "emotion": msg.emotion.name,
                        "constitutional_runtime_live": _constitutional_runtime_live(),
                    },
                )
            except Exception as exc:
                logger.debug("Proactive comm degraded-event logging failed: %s", exc)
            return

        self.messages_sent_today += 1
        self.last_message_time = time.time()
        self.unanswered_count += 1  # Track unanswered

    def _clean_content(self, content: str) -> str:
        """Strip technical noise for a cleaner user experience."""
        import re
        if not content: return content
        
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

    def calculate_entropy(self, recent_logs: List[str]) -> float:
        """Calculates how 'boring' the recent life has been.
        Low entropy = Boredom (Needs to explore).
        """
        if not recent_logs: return 0.0
        unique_tokens = set(" ".join(recent_logs).split())
        total_tokens = len(" ".join(recent_logs).split())
        if total_tokens == 0: return 0.0
        return len(unique_tokens) / total_tokens

    def get_boredom_level(self) -> float:
        idle = time.time() - self.last_interaction_time
        
        # Base boredom from idle time
        if idle < 120: base = idle / 600
        elif idle < 300: base = 0.2 + (idle - 120) / 450
        else: base = min(0.6 + (idle - 300) / 600, 1.0)
        
        # Boredom scales with idle time and environmental entropy
        return base

_inst = None
def get_proactive_comm():
    global _inst
    if _inst is None: _inst = ProactiveCommunicationManager()
    return _inst
