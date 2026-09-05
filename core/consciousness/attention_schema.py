import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.executors import run_blocking_io
from core.runtime.flags import FlagKind, declare
from core.runtime.receipts import WorkspaceGateReceipt, get_receipt_store

logger = logging.getLogger("Consciousness.Attention")

_ATTENTION_SCHEMA_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_ATTENTION_RIGIDITY_GATE_TIMEOUT_FLAG = declare(
    "AURA_ATTENTION_RIGIDITY_GATE_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=0.1,
    description="Maximum time allowed to read the focus-rigidity safety signal",
    owner="core.consciousness.attention_schema",
)


def _read_focus_rigidity_signal() -> tuple[str, float | None]:
    from core.consciousness.free_energy import get_free_energy_engine

    engine = get_free_energy_engine()
    if engine is None:
        raise RuntimeError("free-energy focus gate is unavailable")
    instance_id = str(
        getattr(engine, "instance_id", "") or f"free-energy:{id(engine)}"
    )[:160]
    current = getattr(engine, "current", None)
    if current is None:
        return instance_id, None
    value = float(current.free_energy)
    if not 0.0 <= value <= 1.0:
        raise ValueError("free-energy focus gate returned an out-of-range value")
    return instance_id, value


def _emit_attention_gate_receipt(receipt: WorkspaceGateReceipt) -> WorkspaceGateReceipt:
    emitted = get_receipt_store().emit(receipt)
    if not isinstance(emitted, WorkspaceGateReceipt):
        raise TypeError("attention gate receipt store returned the wrong receipt type")
    return emitted


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttentionalFocus:
    """A first-order cognitive state with its meta-representation."""

    content: str                  # What the system is attending to
    source: str                   # Which subsystem generated this (drive, affect, curiosity, etc.)
    priority: float               # 0.0–1.0 weight at time of broadcast
    timestamp: float = field(default_factory=time.time)

    # HOT meta-representation (Higher-Order Thought)
    meta_repr: str = ""           # "I am attending to X because Y"
    meta_confidence: float = 0.5  # How confident is the meta-representation

    def generate_meta(self) -> str:
        """Produce the higher-order representation of this attentional state."""
        self.meta_repr = (
            f"[HOT] I am currently directing attention toward '{self.content[:80]}' "
            f"(source: {self.source}, priority: {self.priority:.2f}). "
            f"This state is itself an object of my awareness."
        )
        self.meta_confidence = self.priority  # Confidence tracks salience
        return self.meta_repr


@dataclass
class AttentionSchemaState:
    """Full snapshot of the attention schema at a moment in time."""

    current_focus: AttentionalFocus | None = None
    focus_depth: int = 0          # How many recursive HOT levels deep (capped at 3)
    coherence: float = 1.0        # 0.0 = scattered attention, 1.0 = unified
    salience_map: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AttentionSchema:
    """Builds and maintains Aura's internal model of her own attention.

    Key behaviors:
    - Accepts "candidate" attentional states from subsystems
    - Generates HOT meta-representations automatically
    - Tracks coherence (is attention unified or scattered?)
    - Exposes a cognitive_modifier: float that HomeostaticCoupling reads
    """

    _MAX_HISTORY = 50
    _MAX_HOT_DEPTH = 3

    def __init__(self) -> None:
        self.instance_id = f"attention-schema-{uuid.uuid4()}"
        self._lock: asyncio.Lock | None = None  # CS-01: Lazy-initialized
        self.current_focus: AttentionalFocus | None = None
        self.history: deque[AttentionalFocus] = deque(maxlen=self._MAX_HISTORY)
        self.topic_coherence: float = 1.0
        self.hot_depth: int = 0           # Current HOT recursion depth
        self.salience_map: dict[str, float] = {}
        self._focus_start: float = time.time()
        self._sustained_topics: dict[str, int] = {}  # topic -> consecutive ticks
        self._rigidity_gate_ready = True
        self._rigidity_gate_reason = "not_checked"
        self._rigidity_gate_instance_id = ""
        self._gate_rejections: deque[dict[str, Any]] = deque(maxlen=50)

        logger.info("AttentionSchema initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _reject_focus_shift(
        self,
        *,
        source: str,
        priority: float,
        reason: str,
        phase: str,
        error: BaseException | None = None,
        gate_instance_id: str = "",
    ) -> AttentionalFocus:
        current = self.current_focus
        if current is None:
            raise RuntimeError("focus shift cannot be rejected without a retained focus")
        if error is not None:
            self._rigidity_gate_ready = False
            self._rigidity_gate_reason = reason[:240]
            record_degradation(
                "attention_schema",
                error,
                severity="degraded",
                action="retained current focus because the rigidity gate was unavailable",
                extra={"phase": phase, "lane": "attention_focus"},
                enforce_failure_policy=False,
            )
        event = {
            "source": str(source)[:160],
            "retained_source": current.source[:160],
            "reason": reason[:240],
            "phase": phase,
            "retryable": True,
        }
        receipt = WorkspaceGateReceipt(
            cause="attention_focus_shift",
            candidate_source=str(source)[:160],
            gate="attention_focus_rigidity",
            decision="rejected",
            reason=reason[:240],
            retryable=True,
            gate_instance_id=str(gate_instance_id or self._rigidity_gate_instance_id)[:160],
            metadata={
                "phase": phase,
                "lane": "attention_focus",
                "candidate_priority": round(float(priority), 6),
                "retained_source": current.source[:160],
            },
        )
        try:
            emitted = await run_blocking_io(
                _emit_attention_gate_receipt,
                receipt,
                timeout_s=1.0,
                label="attention_gate_receipt",
            )
            event["receipt_id"] = emitted.receipt_id
        except _ATTENTION_SCHEMA_RECOVERABLE_ERRORS as receipt_error:
            self._rigidity_gate_ready = False
            self._rigidity_gate_reason = (
                f"gate_receipt_failed:{type(receipt_error).__qualname__}"
            )[:240]
            record_degradation(
                "attention_schema",
                receipt_error,
                severity="critical",
                action="retained current focus but failed to persist the gate receipt",
                extra={"phase": "attention_rigidity_receipt", "lane": "attention_focus"},
                enforce_failure_policy=False,
            )
            event["receipt_error"] = (
                f"{type(receipt_error).__qualname__}: {receipt_error}"
            )[:240]
        self._gate_rejections.append(event)
        return current

    async def set_focus(self, content: str, source: str, priority: float) -> AttentionalFocus:
        """Set the current attentional focus. Generates HOT meta-representation.
        Called by GlobalWorkspace after competitive selection.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            # Enforce focus stability/rigidity under high cognitive tension. A
            # cross-source shift is a real admission boundary: if its signal is
            # missing, late, cancelled, or malformed, retain the prior focus.
            current = self.current_focus
            if current is not None and source != current.source:
                try:
                    timeout = max(
                        0.01, float(_ATTENTION_RIGIDITY_GATE_TIMEOUT_FLAG.value())
                    )
                    gate_instance_id, fe = await run_blocking_io(
                        _read_focus_rigidity_signal,
                        timeout_s=timeout,
                        label="attention_focus_rigidity",
                    )
                except asyncio.CancelledError as exc:
                    await asyncio.shield(
                        self._reject_focus_shift(
                            source=source,
                            priority=priority,
                            reason="gate_check_cancelled:CancelledError",
                            phase="attention_rigidity_cancelled",
                            error=exc,
                        )
                    )
                    raise
                except _ATTENTION_SCHEMA_RECOVERABLE_ERRORS as exc:
                    return await self._reject_focus_shift(
                        source=source,
                        priority=priority,
                        reason=f"gate_check_failed:{type(exc).__qualname__}",
                        phase="attention_rigidity_check",
                        error=exc,
                    )

                self._rigidity_gate_ready = True
                self._rigidity_gate_reason = "healthy"
                self._rigidity_gate_instance_id = gate_instance_id
                if fe is not None and fe > 0.6:
                    required_priority = 0.3 + fe * 0.4
                    if priority < required_priority:
                        logger.info(
                            "🔒 [ATTENTION GATING] Focus shift blocked due to high Free Energy (F=%.3f). "
                            "Priority %.2f < %.2f required. Retaining focus source '%s'.",
                            fe, priority, required_priority, current.source,
                        )
                        return await self._reject_focus_shift(
                            source=source,
                            priority=priority,
                            reason="focus_rigidity_policy",
                            phase="attention_rigidity_policy",
                            gate_instance_id=gate_instance_id,
                        )

            focus = AttentionalFocus(
                content=content,
                source=source,
                priority=priority,
            )
            focus.generate_meta()

            # Track sustained attention on same topic
            topic_key = content[:40].lower()
            self._sustained_topics[topic_key] = self._sustained_topics.get(topic_key, 0) + 1

            # HOT depth: if this focus is itself a meta-representation, go deeper
            if content.startswith("[HOT]") and self.hot_depth < self._MAX_HOT_DEPTH:
                self.hot_depth += 1
                # Generate meta-meta: awareness of the awareness
                focus.meta_repr = (
                    f"[HOT-{self.hot_depth}] I notice that I am noticing my attention. "
                    f"This recursive awareness has depth {self.hot_depth}."
                )
            else:
                self.hot_depth = 0

            prev = self.current_focus
            self.current_focus = focus
            self.history.append(focus)
            self._focus_start = time.time()

            # Update salience map
            self.salience_map[source] = max(
                self.salience_map.get(source, 0.0),
                priority
            )
            # Decay all salience slightly each update
            self.salience_map = {
                k: v * 0.95 for k, v in self.salience_map.items()
            }

            # Coherence: high if we are dwelling on related topics, low if scattered
            self._update_coherence(content, prev)

            logger.debug(
                f"AttentionFocus → '{content[:60]}' "
                f"(src={source}, pri={priority:.2f}, coherence={self.topic_coherence:.2f})"
            )
            return focus

    async def get_current_meta(self) -> str:
        """Returns the HOT meta-representation of current focus for prompt injection."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if not self.current_focus:
                return "[HOT] No current attentional focus established."
            return self.current_focus.meta_repr

    def get_cognitive_modifier(self) -> float:
        """Returns a 0.0–1.0 modifier that HomeostaticCoupling applies to cognition.
        Low coherence = scattered attention = degraded reasoning.
        High coherence + sustained focus = enhanced reasoning.
        """
        sustained_bonus = 0.0
        if self.current_focus:
            topic_key = self.current_focus.content[:40].lower()
            sustained_ticks = self._sustained_topics.get(topic_key, 0)
            # Up to +0.15 bonus for sustained attention (simulates "flow")
            sustained_bonus = min(0.15, sustained_ticks * 0.01)

        return min(1.0, self.topic_coherence + sustained_bonus)

    def get_snapshot(self) -> dict[str, Any]:
        focus = self.current_focus
        return {
            "instance_id": self.instance_id,
            "current_focus": focus.content[:80] if focus else None,
            "focus_source": focus.source if focus else None,
            "focus_priority": focus.priority if focus else 0.0,
            "meta_repr": focus.meta_repr[:120] if focus else None,
            "hot_depth": self.hot_depth,
            # Published as "coherence" because readers take it by that name and
            # a `.get("coherence", default)` would silently take the default.
            # The attribute is `topic_coherence`: this measures whether
            # attention is staying on one subject, which is a different
            # quantity from the canonical self-coherence channel, and was
            # being counted as a second answer to it.
            "coherence": round(self.topic_coherence, 3),
            "cognitive_modifier": round(self.get_cognitive_modifier(), 3),
            "history_length": len(self.history),
            "top_salience": sorted(self.salience_map.items(), key=lambda x: -x[1])[:3],
            "rigidity_gate": {
                "ready": self._rigidity_gate_ready,
                "reason": self._rigidity_gate_reason,
                "instance_id": self._rigidity_gate_instance_id,
                "rejection_count": len(self._gate_rejections),
                "recent_rejections": list(self._gate_rejections)[-5:],
            },
        }

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return self._rigidity_gate_ready

    def get_status(self) -> dict[str, Any]:
        snapshot = self.get_snapshot()
        snapshot["alive"] = self.is_alive()
        snapshot["ready"] = self.is_ready()
        snapshot["lane"] = "attention_focus"
        return snapshot

    def get_recent_narrative(self, n: int = 5) -> str:
        """Return last n focus transitions as a narrative string for temporal binding."""
        items = list(self.history)[-n:]
        if not items:
            return "No attentional history yet."
        lines = []
        for f in items:
            age = round(time.time() - f.timestamp, 1)
            lines.append(f"  [{age}s ago, src={f.source}] {f.content[:60]}")
        return "Recent attentional trace:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Integration & context
    # ------------------------------------------------------------------

    def get_context_block(self) -> str:
        """Concise attention state for context injection (max 200 chars)."""
        f = self.current_focus
        if not f:
            return "[ATT] no focus | coherence=1.00 | HOT=0 | flow=no"
        content_trunc = f.content[:40].replace("\n", " ")
        flow = "yes" if self.is_in_flow() else "no"
        return (
            f"[ATT] '{content_trunc}' src={f.source} "
            f"coh={self.topic_coherence:.2f} HOT={self.hot_depth} flow={flow}"
        )

    def get_focus_bias_for_source(self, source: str) -> float:
        """Priority boost (0.0-0.3) for GWT candidates matching current focus."""
        if not self.current_focus:
            return 0.0

        # Exact match with current focus source: +0.2
        if source == self.current_focus.source:
            return 0.2

        # In top 3 salience map: +0.1
        top3 = sorted(self.salience_map.items(), key=lambda x: -x[1])[:3]
        top3_sources = {k for k, _ in top3}
        if source in top3_sources:
            return 0.1

        return 0.0

    def get_coherence_for_complexity(self) -> float:
        """Inverted coherence for FreeEnergyEngine complexity signal.
        Scattered attention (low coherence) = high complexity.
        """
        return 1.0 - self.topic_coherence

    def is_in_flow(self) -> bool:
        """True if same topic has been focused for > 5 consecutive ticks."""
        if not self.current_focus:
            return False
        topic_key = self.current_focus.content[:40].lower()
        return self._sustained_topics.get(topic_key, 0) > 5

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_coherence(
        self, new_content: str, prev: AttentionalFocus | None
    ) -> None:
        """Coherence decays when attention jumps to unrelated topics,
        increases when attention dwells on related or same topic.
        Time-on-topic > 30s grants an additional deep-focus coherence bonus.
        """
        if not prev:
            self.topic_coherence = 1.0
            return

        # Simple lexical overlap as proxy for topic similarity
        new_words = set(new_content.lower().split())
        prev_words = set(prev.content.lower().split())
        overlap = len(new_words & prev_words) / max(1, len(new_words | prev_words))

        if overlap > 0.3:
            # Related topics — coherence increases
            self.topic_coherence = min(1.0, self.topic_coherence + 0.05)
            # Deep focus reward: if same topic held > 30 seconds, extra boost
            elapsed = time.time() - self._focus_start
            if elapsed > 30.0:
                self.topic_coherence = min(1.0, self.topic_coherence + 0.02)
        else:
            # Topic jump — coherence decreases
            self.topic_coherence = max(0.1, self.topic_coherence - 0.1)
