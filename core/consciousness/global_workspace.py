from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

from core.container import ServiceContainer
from core.memory.retention_policy import working_history_retention_policy
from core.runtime.errors import Severity, record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.receipts import WorkspaceGateReceipt, get_receipt_store
from core.utils.task_tracker import get_task_tracker

if TYPE_CHECKING:
    from core.resilience.inhibition_manager import InhibitionManager

logger = logging.getLogger("Consciousness.GlobalWorkspace")

_WORKSPACE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    TypeError,
    ValueError,
)
_INHIBITION_GATE_DEGRADATION_PHASES = frozenset(
    {
        "global_inhibition_lookup",
        "global_inhibition_check",
        "global_inhibition_check_cancelled",
        "global_inhibition_revalidation_lookup",
        "global_inhibition_revalidation",
        "global_inhibition_revalidation_cancelled",
    }
)
_INHIBITION_GATE_TIMEOUT_FLAG = declare(
    "AURA_WORKSPACE_INHIBITION_GATE_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=0.5,
    description="Maximum time allowed for the workspace global-inhibition safety gate",
    owner="core.consciousness.global_workspace",
)
_MAX_STRUCTURED_SIGNAL_BYTES = 64 * 1024
_MAX_STRUCTURED_SIGNAL_PREVIEW_CHARS = 4096


def _record_workspace_degradation(
    error: BaseException,
    *,
    phase: str,
    action: str,
    severity: Severity = "warning",
) -> None:
    record_degradation(
        "global_workspace",
        error,
        severity=severity,
        action=action,
        extra={"phase": phase},
        enforce_failure_policy=False,
    )


def _error_summary(error: BaseException) -> str:
    return f"{type(error).__qualname__}: {error}"[:240]


def _emit_workspace_gate_receipt(receipt: WorkspaceGateReceipt) -> WorkspaceGateReceipt:
    emitted = get_receipt_store().emit(receipt)
    if not isinstance(emitted, WorkspaceGateReceipt):
        raise TypeError("workspace gate receipt store returned the wrong receipt type")
    return emitted



# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(Enum):
    """Types of cognitive content for workspace processing."""
    UNKNOWN = auto()
    PERCEPTUAL = auto()
    AFFECTIVE = auto()
    MEMORIAL = auto()
    INTENTIONAL = auto()
    LINGUISTIC = auto()
    SOMATIC = auto()
    SOCIAL = auto()
    META = auto()


@dataclass(order=True)
class WorkItem:
    """Backward compatibility for legacy AttentionSummarizer."""
    priority: float
    ts: float = field(compare=False)
    id: str = field(compare=False)
    source: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
    reason: str | None = field(compare=False)


class HistoryBuffer:
    """Fixed-size history with deque performance and list-like slices.

    Ported from the retired core/global_workspace.py, whose design was better
    than the canonical's here: the canonical kept a plain list and truncated it
    inside ``publish``, so the bound held only for the one path that remembered
    to enforce it. A buffer that cannot exceed its own limit is the difference
    between a bound and a convention — and an unbounded workspace history on a
    long autonomous run is a slow leak, which is the failure mode Aura is least
    able to notice about itself.
    """

    def __init__(self, maxlen: int, items: Iterable[Any] | None = None):
        self.maxlen = maxlen
        self._items: deque[Any] = deque(items or [], maxlen=maxlen)

    def append(self, item: Any) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __getitem__(self, index):
        return list(self._items)[index]

    def __bool__(self) -> bool:
        return bool(self._items)


@dataclass
class CognitiveCandidate:
    """A bid for the global workspace broadcast slot.
    Any subsystem can submit one each tick.
    """

    content: str                       # What wants to be broadcast
    source: str                        # e.g. "drive_curiosity", "affect_distress", "memory"
    priority: float                    # 0.0–1.0 base weight
    content_type: ContentType = ContentType.UNKNOWN
    affect_weight: float = 0.0        # Emotional urgency boost (from AffectEngine)
    focus_bias: float = 0.0           # Priority boost for focused attention (from AttentionSchema)
    submitted_at: float = field(default_factory=time.time)
    gate_instance_id: str = field(default="", repr=False)
    gate_checked_at: float = field(default=0.0, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def salience(self) -> float:
        """Alias for effective_priority for downstream compatibility."""
        return self.effective_priority

    @property
    def effective_priority(self) -> float:
        """Priority as of now. Prefer :meth:`priority_at` inside a competition."""
        return self.priority_at(time.time())

    @property
    def cognitive_priority(self) -> float:
        """Everything about this bid EXCEPT when it arrived.

        What a competition is supposed to be settled by: base salience, how
        urgent affect makes it, where attention already is, and whether it
        aligns with the dominant action. Recency is a real cognitive factor
        and it is applied on top of this; sub-microsecond arrival order is
        not, and separating them is what lets a tie be recognised as a tie.
        """
        return self.priority_at(self.submitted_at)

    def priority_at(self, now: float) -> float:
        """Priority evaluated against ONE instant.

        The property used to call `time.time()` itself, which had two
        consequences and the smaller one was the known flake.
        
        It was used as a sort key, so the comparator re-read the clock during
        the sort and the ordering was not guaranteed to be consistent — a
        comparison function that changes between comparisons can produce an
        arbitrary permutation, not merely a jittered one.
        
        And a competition is one cognitive moment. Ageing each candidate from
        the instant its own comparison happened to run meant identical bids
        came out microseconds apart, and the workspace settled the choice by
        arrival order while presenting it as a priority difference.
        """
        age = max(0.0, now - self.submitted_at)
        recency = max(0.0, 1.0 - (age / 10.0))  # Full weight within 10s, then decays
        
        # Free Energy dynamic gating
        fe_bias = 0.0
        try:
            from core.consciousness.free_energy import get_free_energy_engine
            fe_engine = get_free_energy_engine()
            if fe_engine and fe_engine.current:
                fe_state = fe_engine.current
                dom_action = fe_state.dominant_action
                fe_val = fe_state.free_energy
                
                # High free energy makes the gate much more selective (higher boost for aligned action)
                boost_magnitude = 0.25 * fe_val
                
                aligned = False
                src = self.source.lower()
                ct = self.content_type
                
                if dom_action == "update_beliefs":
                    if ct == ContentType.MEMORIAL or any(x in src for x in ("belief", "memory", "epistemic", "prediction")):
                        aligned = True
                elif dom_action == "act_on_world":
                    if ct == ContentType.INTENTIONAL or any(x in src for x in ("motivation", "action", "goal", "agency")):
                        aligned = True
                elif dom_action == "explore":
                    if ct == ContentType.PERCEPTUAL or any(x in src for x in ("curiosity", "exploration", "perceptual", "search")):
                        aligned = True
                elif dom_action == "reflect":
                    if ct == ContentType.META or any(x in src for x in ("hot", "reflection", "self", "identity")):
                        aligned = True
                elif dom_action == "engage":
                    if ct in (ContentType.LINGUISTIC, ContentType.SOCIAL) or any(x in src for x in ("chat", "user", "linguistic", "social")):
                        aligned = True
                elif dom_action == "rest":
                    if ct == ContentType.SOMATIC or any(x in src for x in ("soma", "sleep", "rest")):
                        aligned = True
                        
                if aligned:
                    fe_bias = boost_magnitude
        except _WORKSPACE_RECOVERABLE_ERRORS as exc:
            _record_workspace_degradation(
                exc,
                phase="free_energy_priority",
                action="Skipped free-energy priority bias and used base salience only",
                severity="debug",
            )

        return min(1.0, (self.priority + self.affect_weight * 0.3 + self.focus_bias + fe_bias) * (0.7 + 0.3 * recency))



#: How close two bids have to be before nothing distinguishes them.
#:
#: Float arithmetic over a handful of added terms, not a tunable. A tie is
#: two bids the workspace genuinely cannot separate, and anything wider than
#: the arithmetic's own resolution would be a policy about how similar is
#: too similar — which is a different claim and would need its own evidence.
_TIE_RESOLUTION = 1e-9


@dataclass
class BroadcastEvent:
    """The formal event emitted on a workspace competition win.
    Compatible with PhenomenologicalExperiencer.
    """
    winners: list[CognitiveCandidate]
    timestamp: float = field(default_factory=time.time)


@dataclass
class BroadcastRecord:
    winner: CognitiveCandidate
    losers: list[str]          # source names of losers
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SomaticImpulse:
    """A bounded non-optimal workspace bid produced by bodily noise.

    This is not a bypass around governance. It only creates a candidate for the
    same workspace competition as every other subsystem. Downstream action still
    has to pass Will/Authority/tool gates.
    """

    content: str
    priority: float
    reason: str
    content_type: ContentType = ContentType.SOMATIC


class SomaticNoiseInjector:
    """Injects rare, bounded, non-goal-maximizing candidates into workspace."""

    DEFAULT_IMPULSES: tuple[tuple[str, str, ContentType], ...] = (
        ("look again at a recent percept before assuming the world is stable", "perceptual_recheck", ContentType.PERCEPTUAL),
        ("write a brief private reflection about the current internal texture", "reflection_whim", ContentType.META),
        ("inspect one recent file or log because it feels slightly salient", "environmental_curiosity", ContentType.INTENTIONAL),
        ("hold a strange analogy and see whether it connects two unrelated ideas", "creative_association", ContentType.META),
        ("notice whether the current plan is becoming too optimized and brittle", "anti_brittleness_impulse", ContentType.META),
    )

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        rate: float | None = None,
        max_priority: float | None = None,
        min_ticks_between: int | None = None,
    ) -> None:
        self.rng = rng or random.Random()
        self.rate = self._bounded_float(
            os.environ.get("AURA_SOMATIC_NOISE_RATE"),
            0.035 if rate is None else rate,
            minimum=0.0,
            maximum=0.35,
        )
        self.max_priority = self._bounded_float(
            os.environ.get("AURA_SOMATIC_NOISE_MAX_PRIORITY"),
            0.72 if max_priority is None else max_priority,
            minimum=0.05,
            maximum=0.9,
        )
        self.min_ticks_between = int(
            self._bounded_float(
                os.environ.get("AURA_SOMATIC_NOISE_MIN_TICKS"),
                30 if min_ticks_between is None else min_ticks_between,
                minimum=1,
                maximum=10_000,
            )
        )
        self.enabled = os.environ.get("AURA_SOMATIC_NOISE", "1").strip().lower() not in {"0", "false", "off", "no"}
        self.last_impulse: SomaticImpulse | None = None
        self.injected_count = 0
        self._last_injected_tick = 0

    @staticmethod
    def _bounded_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value if value is not None else default)
        except (TypeError, ValueError, OverflowError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def maybe_generate(self, *, tick: int, candidate_count: int, inhibited_sources: set[str]) -> SomaticImpulse | None:
        if not self.enabled or candidate_count >= GlobalWorkspace._MAX_CANDIDATES:
            return None
        if "somatic_noise" in inhibited_sources:
            return None
        force = os.environ.get("AURA_SOMATIC_NOISE_FORCE", "0").strip().lower() in {"1", "true", "on", "yes"}
        if not force and tick - self._last_injected_tick < self.min_ticks_between:
            return None
        if not force and self.rng.random() > self.rate:
            return None
        content, reason, content_type = self.rng.choice(self.DEFAULT_IMPULSES)
        jitter = self.rng.uniform(-0.08, 0.08)
        priority = max(0.18, min(self.max_priority, 0.48 + jitter))
        impulse = SomaticImpulse(
            content=f"somatic impulse t{tick}: {content}",
            priority=round(priority, 4),
            reason=reason,
            content_type=content_type,
        )
        self.last_impulse = impulse
        self.injected_count += 1
        self._last_injected_tick = tick
        return impulse


# ---------------------------------------------------------------------------
# Processor registration type
# ---------------------------------------------------------------------------

ProcessorFn = Callable[
    [BroadcastEvent | CognitiveCandidate],
    Awaitable[Any] | Any,
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GlobalWorkspace:
    """The competitive bottleneck. One winner per cognitive tick.

    Rotation model:
      - Winning costs the winner: `_fatigue` reduces its next effective
        priority by `_WINNER_FATIGUE`, recovering over subsequent ticks.
      - Losing costs the loser nothing. It may bid again on the next tick.
      - So the same subsystem cannot hold the broadcast indefinitely, and a
        genuine priority gap still wins outright.

    This docstring described the previous model — losers placed in a cooldown
    dict, barred from re-submitting for `_INHIBIT_TICKS` — which 259cb2aec
    replaced. Banning the loser is a ban on whoever happened to come second,
    which is arrival order wearing a policy's clothes; three tests were still
    asserting against `_inhibited` and reading an always-empty dict.
    `_inhibited` remains, used only by the safety-inhibition path.
    """

    _INHIBIT_TICKS: int = 1       # Retained for the safety-inhibition path; see _fatigue for the refractory
    #: How much a win reduces that source's next effective priority. 0.15 is
    #: large enough to let a 0.02 priority gap rotate and small enough that a
    #: 0.79 gap (urgent vs idle) still dominates.
    _WINNER_FATIGUE: float = 0.15
    #: Safety bound on accumulated adaptation: two wins' worth. Adaptation is
    #: self-limiting (see _fatigue_recovery), so this only caps a pathological
    #: run; it is not the mechanism.
    _MAX_FATIGUE: float = 0.30
    _MAX_CANDIDATES: int = 20     # Hard cap — prevents memory leak if submissions pile up
    _IGNITION_THRESHOLD: float = 0.6  # Priority above which workspace "ignites"
    _PHI_PRIORITY_BOOST: float = 0.15  # Max priority bonus for high-Φ sources
    _SEIZURE_REPLACEMENT_MARGIN: float = 0.05  # Avoid recency-only churn under floods.

    def __init__(self, attention_schema: Any = None):
        self._lock: asyncio.Lock | None = None
        self._candidates: list[CognitiveCandidate] = []
        self._inhibited: dict[str, int] = {}   # source -> ticks_remaining
        #: source -> current adaptation penalty on effective priority.
        self._fatigue: dict[str, float] = {}
        #: The cognitive priorities compared the last time a tie was declared.
        self._last_tie_values: dict[str, float] = {}
        #: Decisions settled by list order because nothing discriminated. A
        #: rising count means the priority scheme has stopped separating its
        #: producers, which is invisible from the winners alone.
        self._tie_impasses: int = 0
        self._last_tie: tuple[str, ...] = ()
        self._processors: list[ProcessorFn] = []
        # Retention comes from the shared working-history policy rather than a
        # hardcoded 100, and lives in a self-bounding buffer. Both were the
        # superseded core/global_workspace.py's design; absorbing them here is
        # what makes retiring that module a merge rather than a loss.
        self.max_history: int = working_history_retention_policy(
            "AURA_GLOBAL_WORKSPACE_HISTORY_MAX",
        ).max_items
        self._history: HistoryBuffer = HistoryBuffer(self.max_history)
        self._tick: int = 0
        self.attention_schema: Any = attention_schema
        self.last_winner: CognitiveCandidate | None = None
        
        # [UNITY] Global Inhibition Link
        self._global_inhibition: InhibitionManager | None = None
        
        # --- Ignition Detection (GWT) ---
        self.ignition_level: float = 0.0    # 0.0-1.0 current ignition intensity
        self.ignited: bool = False          # True when ignition_level >= threshold
        self._ignition_count: int = 0       # Total ignition events
        self._current_phi: float = 0.0      # Φ from substrate (updated externally)
        self._degraded_channels: dict[str, str] = {}
        self._degradation_events: list[dict[str, Any]] = []
        self._processor_failures: dict[str, int] = {}
        self._somatic_noise = SomaticNoiseInjector()
        self._inhibition_gate_ready = False
        self._last_inhibition_gate_reason = "not_checked"
        self._gate_rejections: list[dict[str, Any]] = []
        
        logger.info("GlobalWorkspace initialized (ignition_threshold=%.2f).", self._IGNITION_THRESHOLD)

    def _resolve_tie(self, tied: tuple[str, ...]) -> CognitiveCandidate:
        """Choose among candidates that nothing distinguishes, without using arrival order.

        Sorting and taking ``[0]`` settled these by whoever submitted first,
        which is not arbitration — and worse, it did not *look* like arrival
        order, because ``effective_priority`` scales salience by
        ``(1 - 0.03·age)`` and so turned submission microseconds into a
        priority difference. Four sources bidding an identical 0.70 came out
        ~2e-6 apart. The decision was being made by the scheduler and reported
        as a judgement.

        Two rules, in order, and neither can see when a bid arrived:

        1. **Least fatigued wins.** Fatigue already encodes how recently and
           how often a source held the broadcast, so among equals this hands it
           to whoever has waited longest. It is the same quantity the
           competition already uses, so tie-breaking cannot pull against
           arbitration.
        2. **Rotate on the tick.** When fatigue is level too — the common case
           at startup, when everything is zero — the tick index selects from
           the sources in a stable order. Deterministic, reproducible, and fair
           over time, where a fixed order would hand every genuine tie to
           whichever source sorts first, forever.
        """
        by_source = {c.source: c for c in self._candidates}
        contenders = [by_source[s] for s in tied if s in by_source]
        if not contenders:
            return self._candidates[0]

        # Report the deadlock before settling it. Detecting a tie and breaking
        # it locally leaves the architecture no wiser: the two rules below are
        # fair, and fair is not the same as learned. The bus opens a substate
        # and gives whatever can actually discriminate these candidates a
        # chance to say so; when nothing can, the rules run exactly as before.
        chosen = self._tie_through_impasse_bus(tied, by_source)
        if chosen is not None:
            return chosen

        least = min(self._fatigue.get(c.source, 0.0) for c in contenders)
        # Float fatigue values are produced by repeated subtraction, so compare
        # against the noise they accumulate rather than for exact equality.
        freshest = [
            c for c in contenders if self._fatigue.get(c.source, 0.0) - least <= 1e-9
        ]
        if len(freshest) == 1:
            return freshest[0]
        return freshest[self._tick % len(freshest)]

    def _tie_through_impasse_bus(
        self, tied: tuple[str, ...], by_source: dict
    ) -> CognitiveCandidate | None:
        """Offer the tie to the architecture-wide impasse mechanism.

        Returns the candidate a handler chose, or ``None`` when nothing
        resolved it — in which case the local rules settle it. The workspace
        never blocks on this: a bus that is missing, misconfigured or slow
        must not cost a cognitive tick.
        """
        try:
            from core.cognition.impasse import Impasse, ImpasseType
            from core.cognition.substate import SubstateOutcome, get_impasse_bus

            substate = get_impasse_bus().raise_impasse(
                Impasse(ImpasseType.TIE, f"gw:{self._tick}", tied),
                organ="global_workspace",
                goal="broadcast one candidate",
                context={
                    "tick": self._tick,
                    "fatigue": {s: self._fatigue.get(s, 0.0) for s in tied},
                },
            )
        except Exception as exc:  # noqa: BLE001 - a tick is never worth a crash
            logger.debug("impasse bus unavailable for GW tie: %s", exc)
            return None
        resolution = substate.resolution
        if resolution is None or resolution.outcome is not SubstateOutcome.RESOLVED:
            return None
        return by_source.get(resolution.choice)

    def _fatigue_recovery(self) -> float:
        """Per-tick adaptation recovery, derived from the size of the field.

        This rate is not a tuning knob, and getting it wrong does not degrade
        the competition gracefully — it caps how many sources can share the
        broadcast *at all*.

        Adaptation here is a leaky integrator: a win adds ``g``
        (``_WINNER_FATIGUE``) and every tick sheds ``r``. A source is in
        equilibrium when what it accrues equals what it sheds, so if it wins a
        fraction ``f`` of ticks, ``g·f = r`` and its steady-state share is::

            f* = r / g

        A FIXED ``r`` therefore fixes the sustainable share of every source in
        the running, and the number of sources that can rotate is ``g/r`` no
        matter how many are actually bidding. With the constants this replaced
        (g=0.15, r=0.075) that number was exactly two.

        Measured, four sources bidding every tick at 0.90/0.88/0.86/0.84: the
        top two split the broadcast 12/12 in a strict a-b-a-b alternation and
        the other two won nothing in 24 ticks. The monopoly the previous fix
        removed had come back as a cartel of two, and it passed that fix's own
        regression test, which asked only for ``top_share < 0.75`` and two
        distinct winners — both true of a perfect duopoly. The top two leapfrog
        because while one is fatigued the other is fresh, and the fresh one
        still outbids everyone below it.

        Deriving ``r = g/n`` makes ``f* = 1/n``: the rotation widens to exactly
        as many sources as are genuinely competing, and the two boundary cases
        fall out rather than needing special cases. A lone source (n=1) recovers
        a full win's worth every tick and so is never silenced, and a large gap
        (urgent 0.99 vs idle 0.20) is untouched because 0.79 exceeds
        ``_MAX_FATIGUE``.

        ``n`` counts the sources bidding now, plus any still carrying
        adaptation from a recent win, so a source that skips a single tick is
        not written out of the field.
        """
        competitors = {c.source for c in self._candidates} | set(self._fatigue)
        return self._WINNER_FATIGUE / max(1, len(competitors))

    def _record_degradation(
        self,
        error: BaseException,
        *,
        phase: str,
        action: str,
        severity: Severity = "warning",
    ) -> None:
        summary = _error_summary(error)
        self._degraded_channels[phase] = summary
        self._degradation_events.append(
            {
                "tick": self._tick,
                "phase": phase,
                "severity": severity,
                "error": summary,
                "action": action,
            }
        )
        if len(self._degradation_events) > 50:
            self._degradation_events = self._degradation_events[-50:]
        _record_workspace_degradation(error, phase=phase, action=action, severity=severity)

    @property
    def history(self) -> HistoryBuffer:
        """Broadcast history. Bounded by construction — see HistoryBuffer.

        Iterable, sliceable and len()-able like the list it replaced, so
        AttentionSummarizer and other readers are unaffected.
        """
        return self._history

    @history.setter
    def history(self, value: Iterable[Any]) -> None:
        # Assigning a plain list must not silently unbound the buffer.
        self._history = (
            value
            if isinstance(value, HistoryBuffer)
            else HistoryBuffer(self.max_history, value)
        )

    # ------------------------------------------------------------------
    # Submission API — called by subsystems every heartbeat tick
    # ------------------------------------------------------------------

    async def publish(
        self,
        *,
        priority: float,
        source: str,
        payload: dict[str, Any],
        reason: str = "",
        content_type: ContentType = ContentType.UNKNOWN,
    ) -> bool:
        """Admit a structured signal through the canonical workspace gate.

        Older infrastructure producers publish structured work items. This
        adapter preserves that payload while routing the signal through the
        same inhibition and competition path as every native candidate.
        """
        normalized_source = " ".join(str(source or "").strip().split())[:160]
        if not normalized_source:
            raise ValueError("workspace signal source must be non-empty")
        if not isinstance(payload, dict):
            raise TypeError("workspace signal payload must be a dictionary")
        try:
            normalized_priority = float(priority)
        except (TypeError, ValueError) as exc:
            raise ValueError("workspace signal priority must be numeric") from exc
        if not math.isfinite(normalized_priority):
            raise ValueError("workspace signal priority must be finite")
        normalized_priority = max(0.0, min(1.0, normalized_priority))
        normalized_reason = " ".join(str(reason or "").strip().split())[:500]
        content = normalized_reason or f"Structured workspace signal from {normalized_source}"
        payload_json = json.dumps(payload, ensure_ascii=True, default=str, sort_keys=True)
        payload_bytes = payload_json.encode("utf-8")
        if len(payload_bytes) <= _MAX_STRUCTURED_SIGNAL_BYTES:
            preserved_payload: dict[str, Any] = json.loads(payload_json)
        else:
            preserved_payload = {
                "truncated": True,
                "original_bytes": len(payload_bytes),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "preview": payload_json[:_MAX_STRUCTURED_SIGNAL_PREVIEW_CHARS],
            }
        metadata = {
            "schema": "aura.workspace.signal.v1",
            "reason": normalized_reason,
            "payload": preserved_payload,
        }
        return await self.submit(
            CognitiveCandidate(
                content=content,
                source=normalized_source,
                priority=normalized_priority,
                content_type=content_type,
                metadata=metadata,
            )
        )

    def _resolve_global_inhibition(self) -> InhibitionManager:
        manager = ServiceContainer.get("inhibition_manager", default=None)
        if manager is None:
            from core.resilience.inhibition_manager import get_inhibition_manager

            manager = get_inhibition_manager()
            ServiceContainer.register_instance(
                "inhibition_manager",
                manager,
                required=True,
                owner="core.resilience.inhibition_manager",
                registered_by="core.consciousness.global_workspace",
                required_for="workspace_candidate_admission",
                failure_policy="fail_closed",
            )
        check = getattr(manager, "is_inhibited", None)
        if not callable(check):
            raise TypeError("inhibition manager lacks callable is_inhibited")
        return cast("InhibitionManager", manager)

    async def _reject_for_inhibition_gate(
        self,
        candidate: CognitiveCandidate,
        *,
        phase: str,
        reason: str,
        error: BaseException | None = None,
        gate: str = "global_inhibition",
        retryable: bool = True,
        gate_instance_id: str | None = None,
    ) -> bool:
        if gate == "global_inhibition" and error is not None:
            self._inhibition_gate_ready = False
            self._last_inhibition_gate_reason = reason[:240]
        if error is not None:
            self._record_degradation(
                error,
                phase=phase,
                action="Rejected workspace candidate because global inhibition authority was unavailable",
                severity="degraded",
            )
        event = {
            "tick": self._tick,
            "candidate_source": candidate.source[:160],
            "gate": gate,
            "phase": phase,
            "reason": reason[:240],
            "retryable": retryable,
        }
        receipt = WorkspaceGateReceipt(
            cause="workspace_candidate_submission",
            candidate_source=candidate.source[:160],
            gate=gate,
            decision="rejected",
            reason=reason[:240],
            retryable=retryable,
            gate_instance_id=str(
                gate_instance_id
                if gate_instance_id is not None
                else getattr(self._global_inhibition, "instance_id", "") or ""
            )[:160],
            metadata={
                "tick": self._tick,
                "phase": phase,
                "lane": "workspace_candidate_admission",
                "content_type": candidate.content_type.name,
                "candidate_age_s": round(max(0.0, time.time() - candidate.submitted_at), 6),
            },
        )
        try:
            emitted = await asyncio.to_thread(_emit_workspace_gate_receipt, receipt)
            event["receipt_id"] = emitted.receipt_id
        except _WORKSPACE_RECOVERABLE_ERRORS as receipt_error:
            self._record_degradation(
                receipt_error,
                phase="global_inhibition_receipt",
                action="Rejected workspace candidate but gate receipt persistence failed",
                severity="critical",
            )
            event["receipt_error"] = _error_summary(receipt_error)
        self._gate_rejections.append(event)
        self._gate_rejections = self._gate_rejections[-50:]
        return False

    @staticmethod
    def _manager_instance_id(manager: Any) -> str:
        return str(getattr(manager, "instance_id", "") or "")[:160]

    async def _check_candidates_for_competition(
        self,
        candidates: list[CognitiveCandidate],
    ) -> list[CognitiveCandidate]:
        """Revalidate every bid immediately before selection.

        Submission approval is deliberately not a bearer token. Inhibition can
        change or the canonical manager can be replaced between submission and
        broadcast, so every pending candidate must pass the current instance.
        """

        if not candidates:
            return []
        try:
            manager = self._resolve_global_inhibition()
            check = getattr(manager, "is_inhibited", None)
            if not callable(check):
                raise TypeError("inhibition manager lacks callable is_inhibited")
            self._global_inhibition = manager
        except _WORKSPACE_RECOVERABLE_ERRORS as exc:
            for candidate in candidates:
                await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_revalidation_lookup",
                    reason=f"gate_revalidation_lookup_failed:{type(exc).__qualname__}",
                    error=exc,
                )
            self._global_inhibition = None
            return []

        timeout = max(0.01, float(_INHIBITION_GATE_TIMEOUT_FLAG.value()))
        checks = [
            asyncio.wait_for(manager.is_inhibited(candidate.source), timeout=timeout)
            for candidate in candidates
        ]
        try:
            results = await asyncio.gather(*checks, return_exceptions=True)
        except asyncio.CancelledError as exc:
            for candidate in candidates:
                await asyncio.shield(
                    self._reject_for_inhibition_gate(
                        candidate,
                        phase="global_inhibition_revalidation_cancelled",
                        reason="gate_revalidation_cancelled:CancelledError",
                        error=exc,
                    )
                )
            self._candidates = []
            self._global_inhibition = None
            raise

        accepted: list[CognitiveCandidate] = []
        gate_fault = False
        current_instance_id = self._manager_instance_id(manager)
        for candidate, result in zip(candidates, results, strict=True):
            if isinstance(result, BaseException):
                gate_fault = True
                await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_revalidation",
                    reason=f"gate_revalidation_failed:{type(result).__qualname__}",
                    error=result,
                    gate_instance_id=current_instance_id,
                )
                continue
            if not isinstance(result, bool):
                gate_fault = True
                error = TypeError("inhibition manager returned a non-boolean decision")
                await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_revalidation",
                    reason="gate_revalidation_failed:TypeError",
                    error=error,
                    gate_instance_id=current_instance_id,
                )
                continue
            if result:
                await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_revalidation_policy",
                    reason="source_inhibited_before_competition",
                    gate_instance_id=current_instance_id,
                )
                continue
            candidate.gate_instance_id = current_instance_id
            candidate.gate_checked_at = time.time()
            accepted.append(candidate)

        if len(accepted) == len(candidates):
            self._inhibition_gate_ready = True
            self._last_inhibition_gate_reason = "healthy"
            for phase in _INHIBITION_GATE_DEGRADATION_PHASES:
                self._degraded_channels.pop(phase, None)
        elif gate_fault:
            self._global_inhibition = None
        return accepted

    async def submit(self, candidate: CognitiveCandidate) -> bool:
        """Submit a candidate for the next broadcast competition.
        Returns False if the source is currently inhibited.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        async with self._lock:
            # Check internal inhibition
            if candidate.source in self._inhibited and self._inhibited[candidate.source] > 0:
                logger.debug("GW: %s is internal-inhibited (%d ticks)", candidate.source, self._inhibited[candidate.source])
                return await self._reject_for_inhibition_gate(
                    candidate,
                    phase="workspace_refractory_policy",
                    reason="source_in_refractory_period",
                    gate="workspace_refractory",
                    gate_instance_id="global_workspace",
                )
                
            # Check global inhibition
            if self._global_inhibition is None:
                try:
                    self._global_inhibition = self._resolve_global_inhibition()
                except _WORKSPACE_RECOVERABLE_ERRORS as exc:
                    return await self._reject_for_inhibition_gate(
                        candidate,
                        phase="global_inhibition_lookup",
                        reason=f"gate_lookup_failed:{type(exc).__qualname__}",
                        error=exc,
                    )
            try:
                gate_timeout = max(0.01, float(_INHIBITION_GATE_TIMEOUT_FLAG.value()))
                inhibited = await asyncio.wait_for(
                    self._global_inhibition.is_inhibited(candidate.source),
                    timeout=gate_timeout,
                )
            except asyncio.CancelledError as exc:
                await asyncio.shield(
                    self._reject_for_inhibition_gate(
                        candidate,
                        phase="global_inhibition_check_cancelled",
                        reason="gate_check_cancelled:CancelledError",
                        error=exc,
                    )
                )
                self._global_inhibition = None
                raise
            except _WORKSPACE_RECOVERABLE_ERRORS as exc:
                rejected = await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_check",
                    reason=f"gate_check_failed:{type(exc).__qualname__}",
                    error=exc,
                )
                self._global_inhibition = None
                return rejected
            if not isinstance(inhibited, bool):
                decision_error = TypeError(
                    "inhibition manager returned a non-boolean decision"
                )
                rejected = await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_check",
                    reason="gate_check_failed:TypeError",
                    error=decision_error,
                )
                self._global_inhibition = None
                return rejected
            if inhibited:
                logger.debug("GW: %s is GLOBAL-inhibited", candidate.source)
                return await self._reject_for_inhibition_gate(
                    candidate,
                    phase="global_inhibition_policy",
                    reason="source_inhibited",
                )
            self._inhibition_gate_ready = True
            self._last_inhibition_gate_reason = "healthy"
            for phase in _INHIBITION_GATE_DEGRADATION_PHASES:
                self._degraded_channels.pop(phase, None)
            candidate.gate_instance_id = self._manager_instance_id(self._global_inhibition)
            candidate.gate_checked_at = time.time()
            
            # Φ-aware priority boost: high integration → higher salience
            if self._current_phi > 0.1:
                phi_boost = min(self._PHI_PRIORITY_BOOST, self._current_phi * 0.1)
                # Fix Issue 68: Don't mutate candidate.priority; use focus_bias instead
                candidate.focus_bias = min(1.0, candidate.focus_bias + phi_boost)
            
            # Replace any existing candidate from same source (only one bid per source).
            # Done BEFORE the flood check so a source updating its own bid never counts
            # as new pressure and never gets spuriously dropped.
            self._candidates = [c for c in self._candidates if c.source != candidate.source]

            # --- Seizure Guard (Phase 23.5) + salience-ranked backpressure ---
            if len(self._candidates) >= self._MAX_CANDIDATES:
                # The workspace is full. Rather than blanket-drop every new bid (which
                # let a flood of low-salience submissions lock out a genuinely urgent
                # one that arrives later), keep the N *most salient* bids: evict the
                # weakest queued candidate iff the incoming one outranks it. A valid,
                # high-priority candidate is never dropped just for arriving late.
                weakest = min(self._candidates, key=lambda c: c.effective_priority)
                incoming = candidate.effective_priority

                replacement_threshold = (
                    weakest.effective_priority + self._SEIZURE_REPLACEMENT_MARGIN
                )
                if incoming <= replacement_threshold:
                    # Incoming really is the least important — drop it and signal flood.
                    logger.warning(
                        "🧠 [SEIZURE GUARD] GlobalWorkspace FLOODED (%d); dropping lowest bid %s "
                        "(%.3f ≤ replacement threshold %.3f).",
                        len(self._candidates), candidate.source, incoming, replacement_threshold,
                    )
                    self._signal_neural_flood(candidate.source)
                    return await self._reject_for_inhibition_gate(
                        candidate,
                        phase="workspace_capacity_policy",
                        reason="workspace_capacity_rejected",
                        gate="workspace_capacity",
                        gate_instance_id="global_workspace",
                    )

                # Incoming outranks the weakest → evict the weakest, admit the incoming.
                logger.warning(
                    "🧠 [SEIZURE GUARD] GlobalWorkspace FLOODED (%d); evicting weakest %s (%.3f) for %s (%.3f).",
                    len(self._candidates), weakest.source, weakest.effective_priority,
                    candidate.source, incoming,
                )
                self._candidates = [c for c in self._candidates if c is not weakest]
                self._signal_neural_flood(weakest.source)
                await self._reject_for_inhibition_gate(
                    weakest,
                    phase="workspace_capacity_policy",
                    reason="workspace_capacity_evicted",
                    gate="workspace_capacity",
                    gate_instance_id="global_workspace",
                )

            self._candidates.append(candidate)
            return True

    def _signal_neural_flood(self, dropped_source: str) -> None:
        """Broadcast a workspace-flood tension reflex via the mycelial network.

        Fired whenever backpressure has to drop a bid (the incoming one or an evicted
        weakest one). Best-effort: a missing/erroring mycelium never blocks competition.
        """
        try:
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                # Thicken the visual noise while preserving network ownership.
                mycelium.set_hypha_strength("consciousness", "workspace", 10.0)
                get_task_tracker().create_task(
                    mycelium.emit_reflex("NEURAL_FLOOD", {"source": dropped_source})
                )
        except _WORKSPACE_RECOVERABLE_ERRORS as _e:
            self._record_degradation(
                _e,
                phase="seizure_guard_reflex",
                action="Dropped flooded workspace bid and skipped mycelial flood reflex",
                severity="warning",
            )
            logger.debug("GW seizure guard reflex failed after dropping bid: %s", _e)

    # ------------------------------------------------------------------
    # Processor registration — subsystems register to receive broadcasts
    # ------------------------------------------------------------------

    def register_processor(self, fn: ProcessorFn) -> None:
        """Register a coroutine function to be called when a winner is broadcast."""
        self._processors.append(fn)

    def subscribe(self, fn: ProcessorFn) -> None:
        """Alias for register_processor to support AgencyCore subscriptions."""
        self.register_processor(fn)

    # ------------------------------------------------------------------
    # Competition — called once per heartbeat tick
    # ------------------------------------------------------------------

    async def run_competition(self) -> CognitiveCandidate | None:
        """Run the competitive selection. Returns the winner (or None if no candidates).
        Inhibits losers and broadcasts winner to all registered processors.
        """
        self._tick += 1

        if self._lock is None:
            self._lock = asyncio.Lock()

        # Mycelial Pulse (Proof of Life for Workspace)
        try:
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                mycelium.pulse_hypha("consciousness", "workspace", success=True)
        except _WORKSPACE_RECOVERABLE_ERRORS as _e:
            self._record_degradation(
                _e,
                phase="workspace_pulse",
                action="Skipped mycelial proof-of-life pulse and continued workspace competition",
                severity="debug",
            )
            logger.debug("GW mycelial proof-of-life pulse skipped: %s", _e)

        async with self._lock:
            # Decay inhibition counters before a possible somatic submission so
            # the synthetic source follows the same refractory policy as every
            # other producer.
            self._inhibited = {
                src: count - 1
                for src, count in self._inhibited.items()
                if count > 1
            }
            # Adaptation recovers whether or not the source bid this tick.
            recovery = self._fatigue_recovery()
            self._fatigue = {
                src: value - recovery
                for src, value in self._fatigue.items()
                if value - recovery > 1e-9
            }
            pending_count = len(self._candidates)
            inhibited_sources = set(self._inhibited)

        try:
            impulse = self._somatic_noise.maybe_generate(
                tick=self._tick,
                candidate_count=pending_count,
                inhibited_sources=inhibited_sources,
            )
            if impulse is not None:
                await self.submit(
                    CognitiveCandidate(
                        content=impulse.content,
                        source="somatic_noise",
                        priority=impulse.priority,
                        content_type=impulse.content_type,
                        affect_weight=0.05,
                        focus_bias=0.0,
                    )
                )
        except asyncio.CancelledError:
            raise
        except _WORKSPACE_RECOVERABLE_ERRORS as exc:
            self._record_degradation(
                exc,
                phase="somatic_noise",
                action="Skipped stochastic somatic impulse and continued workspace competition",
                severity="warning",
            )

        async with self._lock:
            if not self._candidates:
                return None

            self._candidates = await self._check_candidates_for_competition(
                list(self._candidates)
            )
            if not self._candidates:
                return None

            # Sort by effective priority MINUS adaptation, so the coalition
            # that just held the workspace has to out-bid the field by more
            # than its fatigue to hold it again.
            # One instant for the whole competition. Read once, before any
            # comparison, so every candidate is aged against the same moment
            # and the sort key cannot change while the sort is running.
            decided_at = time.time()

            def _adjusted(candidate: CognitiveCandidate) -> float:
                return candidate.priority_at(decided_at) - self._fatigue.get(
                    candidate.source, 0.0
                )

            # Frozen before sorting, and the sort reads the frozen value. A
            # key function that calls the clock is not a key function.
            scores = {id(c): _adjusted(c) for c in self._candidates}
            self._candidates.sort(key=lambda c: scores[id(c)], reverse=True)
            winner = self._candidates[0]
            losers = self._candidates[1:]

            # Soar's tie impasse: several candidates that nothing actually
            # discriminates. Taking [0] resolves it silently, and until now
            # nothing recorded that the choice had been arbitrary.
            #
            # Testing for EXACT equality finds nothing, and the reason matters
            # more than the tie would have. effective_priority multiplies base
            # salience by (1 - 0.03·age) inside the recency window, so four
            # sources bidding an identical 0.70 came out ~2e-6 apart purely
            # from being submitted microseconds apart: measured over 12 ticks
            # of four identical bids, exact ties were 0 of 12. The workspace
            # was settling those decisions by sub-microsecond arrival time
            # while presenting it as a priority difference, which is strictly
            # worse than a visible tie because it looks principled.
            #
            # So the comparison is against the noise floor that mechanism
            # creates rather than against zero. Two candidates of identical
            # salience submitted `spread` seconds apart differ by at most
            # 0.03·spread, and base salience is clamped to 1.0, so that bounds
            # the whole timing artefact. Anything closer than that is a tie in
            # substance whatever the float says.
            if losers:
                # A tie is a fact about the BIDS, not about the scheduler.
                #
                # The floor used to be 0.03 x the wall-clock span between the
                # first and last submission, so how far apart two priorities
                # had to be before they counted as different was set by
                # whatever else the OS had been doing. A garbage collection
                # between two submits widened it; a quiet moment narrowed it.
                # Measured: the "three genuinely different bids" regression
                # passed about half the time, and which half depended on
                # thread timing rather than on anything cognitive.
                #
                # So the comparison is on cognitive priority — salience,
                # affect, focus, alignment, fatigue — with recency left out.
                # Recency is a real factor and it still decides the winner
                # above; what it must not do is decide whether a difference
                # EXISTS. The floor is the float resolution of that
                # comparison and nothing else.
                def _cognitive(candidate: CognitiveCandidate) -> float:
                    return candidate.cognitive_priority - self._fatigue.get(
                        candidate.source, 0.0
                    )

                cognitive = {id(c): _cognitive(c) for c in self._candidates}
                top_cognitive = max(cognitive.values())
                tied = tuple(
                    sorted(
                        c.source
                        for c in self._candidates
                        if top_cognitive - cognitive[id(c)] <= _TIE_RESOLUTION
                    )
                )
                if len(tied) > 1:
                    self._tie_impasses += 1
                    self._last_tie = tied
                    # What the tie was decided on, kept so a reader does not
                    # have to reconstruct it. Reconstruction was wrong: the
                    # fatigue map recovers earlier in this same call, so a
                    # snapshot taken before run_competition names different
                    # numbers than the ones actually compared.
                    self._last_tie_values = {
                        c.source: round(cognitive[id(c)], 12)
                        for c in self._candidates
                    }
                    logger.debug(
                        "GW tie impasse: %d sources within %.3g of each other on "
                        "cognitive priority %s",
                        len(tied),
                        _TIE_RESOLUTION,
                        tied,
                    )
                    winner = self._resolve_tie(tied)
                    losers = [c for c in self._candidates if c is not winner]

            # Adaptation on the winner, not exclusion of the losers.
            #
            # This used to inhibit every LOSER for a tick, and inhibition is
            # checked in submit(), so the sources that had just lost could not
            # bid on the next tick — leaving the winner unopposed, winning
            # again, and inhibiting them again. Measured: four sources bidding
            # every tick at 0.90 / 0.88 / 0.86 / 0.84 gave the top source 24
            # wins out of 24 while the other three were refused half their
            # submissions. A two-point priority difference bought a permanent
            # monopoly of the broadcast — the exact thing the class docstring
            # says this prevents ("stops the same subsystem from dominating
            # every cycle and forces genuine competition").
            #
            # Hard-inhibiting the WINNER instead is worse in a subtler way. It
            # was tried and measured: an urgent source at 0.99 and an idle one
            # at 0.20 then alternated 50/50, because a hard block ignores how
            # much stronger the bid was, and a source bidding ALONE won only
            # half its ticks — silence with nothing else to attend to.
            #
            # So the recent winner is FATIGUED rather than excluded: its
            # effective priority is reduced for a few ticks and recovers. This
            # is spike-frequency adaptation, the mechanism the biology actually
            # uses for the same job. Near-equal sources rotate, a genuinely
            # urgent source still outbids a weak one through the penalty, and a
            # lone source keeps the workspace because nothing outbids it.
            self._fatigue[winner.source] = min(
                self._MAX_FATIGUE,
                self._fatigue.get(winner.source, 0.0) + self._WINNER_FATIGUE,
            )

            # Clear candidate pool
            self._candidates = []

            # Record
            record = BroadcastRecord(
                winner=winner,
                losers=[loser.source for loser in losers]
            )
            # No manual truncation: the buffer enforces its own bound, so it
            # holds for every writer rather than only for the ones that
            # remember to check.
            self._history.append(record)

            self.last_winner = winner

        # --- Peripheral Awareness (Attention/Consciousness Dissociation) ---
        # Feed losers into the peripheral field so content that didn't win
        # broadcast can still be phenomenally present at low intensity.
        try:
            from core.consciousness.peripheral_awareness import get_peripheral_awareness_engine
            all_candidates_data = [
                {"source": winner.source, "priority": winner.effective_priority, "content": str(winner.content)[:200]}
            ] + [
                {"source": loser.source, "priority": loser.effective_priority, "content": str(loser.content)[:200]}
                for loser in losers
            ]
            get_peripheral_awareness_engine().process_workspace_results(
                winner_source=winner.source,
                all_candidates=all_candidates_data,
            )
        except _WORKSPACE_RECOVERABLE_ERRORS as _pa_exc:
            self._record_degradation(
                _pa_exc,
                phase="peripheral_awareness",
                action="Retained broadcast winner and skipped peripheral awareness side-feed",
                severity="warning",
            )
            logger.debug("GW peripheral awareness feed skipped: %s", _pa_exc)

        try:
            from core.unity import get_unity_runtime

            get_unity_runtime().record_workspace_competition(winner, losers)
        except _WORKSPACE_RECOVERABLE_ERRORS as exc:
            self._record_degradation(
                exc,
                phase="unity_runtime",
                action="Retained broadcast record and skipped unity workspace frame",
                severity="warning",
            )
            logger.debug("GW unity workspace frame skipped: %s", exc)

        # --- Ignition Detection ---
        winner_priority = winner.effective_priority
        self.ignition_level = min(1.0, winner_priority / self._IGNITION_THRESHOLD)
        was_ignited = self.ignited
        self.ignited = winner_priority >= self._IGNITION_THRESHOLD
        
        if self.ignited and not was_ignited:
            self._ignition_count += 1
            logger.info(
                "⚡ GW IGNITION #%d: source=%s, priority=%.3f, phi=%.4f",
                self._ignition_count, winner.source, winner_priority, self._current_phi,
            )

            # ── Theory Arbitration: GWT predicts broadcast improves accessibility ──
            try:
                from core.consciousness.theory_arbitration import get_theory_arbitration
                arb = get_theory_arbitration()
                event_id = f"gw_ignition_{self._ignition_count}"
                arb.log_prediction(
                    theory="gwt",
                    event_id=event_id,
                    prediction="broadcast_improves_coherence",
                    confidence=min(1.0, winner_priority),
                )
                # IIT counter-prediction: integration matters more than broadcast
                arb.log_prediction(
                    theory="iit_4_0",
                    event_id=event_id,
                    prediction="phi_determines_coherence_not_broadcast",
                    confidence=0.6,
                )
            except _WORKSPACE_RECOVERABLE_ERRORS as exc:
                self._record_degradation(
                    exc,
                    phase="theory_arbitration",
                    action="Recorded ignition locally and skipped theory arbitration prediction feed",
                    severity="warning",
                )
                logger.debug("GW theory arbitration feed skipped: %s", exc)

        # 4. Neural Feed Transparency (Phase 13)
        try:
            from core.thought_stream import get_emitter
            emitter = get_emitter()
            if emitter:
                emitter.emit(
                    title="Neural Competition",
                    content=f"Winner: {winner.source} | Content: {winner.content[:100]}",
                    level="info",
                    metadata={
                        "tick": self._tick,
                        "winner_priority": round(winner.effective_priority, 3),
                        "losers": [loser.source for loser in losers[:3]]
                    }
                )
        except _WORKSPACE_RECOVERABLE_ERRORS as e:
            self._record_degradation(
                e,
                phase="thought_stream",
                action="Retained winner and skipped Neural Feed transparency event",
                severity="warning",
            )
            logger.debug("Failed to emit Neural Feed match: %s", e)

        # Update attention schema with winner (outside lock)
        if self.attention_schema:
            try:
                await self.attention_schema.set_focus(
                    content=winner.content,
                    source=winner.source,
                    priority=winner.effective_priority,
                )
            except _WORKSPACE_RECOVERABLE_ERRORS as exc:
                self._record_degradation(
                    exc,
                    phase="attention_schema",
                    action="Retained broadcast winner and skipped attention-schema focus update",
                    severity="warning",
                )

        # Broadcast to all registered processors (outside lock, concurrent)
        if self._processors:
            event = BroadcastEvent(winners=[winner], timestamp=time.time())
            await asyncio.gather(
                *[self._safe_call(proc, event) for proc in self._processors],
                return_exceptions=True
            )

        logger.debug(
            "GW tick %d: winner='%s' (pri=%.2f), inhibited=%s",
            self._tick, winner.source, winner.effective_priority, list(self._inhibited.keys())
        )
        return winner

    async def _safe_call(
        self,
        fn: ProcessorFn,
        event: BroadcastEvent | CognitiveCandidate,
    ) -> None:
        try:
            # Handle both legacy single-candidate and new broadcast-event formats
            res = fn(event)
            if res is not None and inspect.isawaitable(res):
                await res
        except _WORKSPACE_RECOVERABLE_ERRORS as e:
            processor_name = getattr(fn, "__qualname__", getattr(fn, "__name__", fn.__class__.__name__))
            self._processor_failures[processor_name] = self._processor_failures.get(processor_name, 0) + 1
            self._record_degradation(
                e,
                phase="processor_broadcast",
                action=f"Isolated processor {processor_name} failure and continued remaining broadcasts",
                severity="warning",
            )
            logger.error("GW processor error: %s", e)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict[str, Any]:
        last = self.last_winner
        return {
            "tick": self._tick,
            "last_winner": last.source if last else None,
            "last_content": last.content[:80] if last else None,
            "last_priority": round(last.effective_priority, 3) if last else 0.0,
            "pending_candidates": len(self._candidates),
            "inhibited_sources": list(self._inhibited.keys()),
            # Decisions that were settled by list order rather than by any
            # preference. Reported because the rate is the diagnostic: winners
            # alone cannot show that the choice was arbitrary.
            "tie_impasses": self._tie_impasses,
            "last_tie_values": dict(self._last_tie_values),
            "last_tie": list(self._last_tie),
            "fatigue_recovery_rate": round(self._fatigue_recovery(), 5),
            "broadcast_history_len": len(self._history),
            "ignition_level": round(self.ignition_level, 3),
            "ignited": self.ignited,
            "ignition_count": self._ignition_count,
            "phi": round(self._current_phi, 4),
            "degraded_channels": dict(self._degraded_channels),
            "recent_degradations": list(self._degradation_events[-5:]),
            "processor_failures": dict(self._processor_failures),
            "inhibition_gate": {
                "ready": self._inhibition_gate_ready,
                "reason": self._last_inhibition_gate_reason,
                "instance_id": str(
                    getattr(self._global_inhibition, "instance_id", "") or ""
                ),
                "rejection_count": len(self._gate_rejections),
                "recent_rejections": list(self._gate_rejections[-5:]),
            },
            "somatic_noise": {
                "enabled": self._somatic_noise.enabled,
                "rate": self._somatic_noise.rate,
                "injected_count": self._somatic_noise.injected_count,
                "last_reason": self._somatic_noise.last_impulse.reason if self._somatic_noise.last_impulse else None,
            },
        }

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        from core.runtime.service_access import optional_service

        manager = self._global_inhibition or optional_service(
            "inhibition_manager", default=None
        )
        if manager is None or not callable(getattr(manager, "is_inhibited", None)):
            return False
        manager_ready = getattr(manager, "is_ready", None)
        if callable(manager_ready):
            try:
                if not bool(manager_ready()):
                    return False
            except _WORKSPACE_RECOVERABLE_ERRORS:
                return False
        return bool(
            self._last_inhibition_gate_reason in {"not_checked", "healthy"}
            and not (_INHIBITION_GATE_DEGRADATION_PHASES & self._degraded_channels.keys())
            and "global_inhibition_receipt" not in self._degraded_channels
        )

    def get_status(self) -> dict[str, Any]:
        snapshot = self.get_snapshot()
        snapshot["alive"] = self.is_alive()
        snapshot["ready"] = self.is_ready()
        snapshot["lane"] = "workspace_candidate_admission"
        return snapshot

    def update_phi(self, phi: float) -> None:
        """Update the current Φ value from the LiquidSubstrate.
        Called by the heartbeat or consciousness system each tick.
        """
        self._current_phi = max(0.0, float(phi))

    def is_ignited(self) -> bool:
        """Whether the workspace is currently in an ignited state."""
        return self.ignited

    def get_ignition_level(self) -> float:
        """Current ignition intensity (0.0-1.0)."""
        return self.ignition_level

    def get_last_n_winners(self, n: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "winner": r.winner.source,
                "content": r.winner.content[:60],
                "losers": r.losers,
                "timestamp": r.timestamp,
            }
            for r in self._history[-n:]
        ]

    def get_competing_coalitions(self, n: int = 3) -> list[dict[str, Any]]:
        """The live competition, strongest first — read-only and bounded.

        This is the GWT→RLC coupling surface: pending candidates (not yet
        broadcast) rendered so deep deliberation can take the mind's actual
        competing coalitions as typed thought-slot seeds.
        """
        rows: list[dict[str, Any]] = []
        for candidate in sorted(
            list(self._candidates),
            key=lambda c: c.effective_priority,
            reverse=True,
        )[: max(0, int(n))]:
            rows.append(
                {
                    "source": candidate.source,
                    "content": candidate.content[:400],
                    "priority": round(candidate.effective_priority, 4),
                    "content_type": getattr(
                        candidate.content_type, "name", str(candidate.content_type)
                    ),
                }
            )
        return rows

    def get_context_stream(self, n: int = 5) -> str:
        """Return a formatted string of the last N winners for prompt injection."""
        winners = self.get_last_n_winners(n)
        if not winners:
            return ""
        
        lines = []
        for w in winners:
            lines.append(f"- [{w['winner']}] {w['content']}")
        return "\n".join(lines)
