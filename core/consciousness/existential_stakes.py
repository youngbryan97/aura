"""core/consciousness/existential_stakes.py
============================================
Existential Stakes & Nociceptive Gate Subsystem.

Computes a real-time survival threat index (0.0 to 1.0) based on physical process
and hardware constraints: memory pressure, event loop scheduling delays, CPU usage,
and recent exception/degradation events.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

try:
    from core.runtime import resource_psutil as psutil
except ImportError:
    psutil = None

from core.runtime.degradation_habituation import get_habituation, signature_for
from core.runtime.errors import get_degradation_tracker
from core.runtime.resource_observation import get_resource_observer

logger = logging.getLogger("Consciousness.ExistentialStakes")

DEFAULT_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB default limit

# Ceiling on how much OPERATIONAL pressure (high CPU, event-loop lag) can
# contribute to the combined survival threat. High CPU and lag are normal
# during heavy local generation — a busy machine is not a dying one — so they
# must never alone reach the will-system's survival-inhibition veto threshold
# (will.py: threat > 0.75), which would block Aura's own actions whenever it is
# working hard. Genuine death risk (memory exhaustion → OOM, degradation
# cascades) is uncapped and can still reach 1.0. Kept just below the veto line.
OPERATIONAL_THREAT_CAP = 0.70
CRITICAL_THREAT_THRESHOLD = 0.75
#: "RuntimeError: ", "ValueError: " — an exception type at the head of a
#: message, left behind when a propagation wrapper is stripped.
_ORIGIN_SUBSYSTEM_RE = re.compile(r"Subsystem '([^']+)' failed")
_EXCEPTION_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Violation):\s*")

DEGRADATION_THREAT_WINDOW_S = 60.0
DEGRADATION_THREAT_DENOMINATOR = 5.0
#: How close to the denominator counts as saturated. Degradation weight is
#: age-decayed continuously (``base * (1 - age/window)``), so five
#: simultaneous degraded events sum to exactly 5.0 only at age zero and
#: fall away from it every microsecond after. At 1e-4 the margin was ~1.2ms
#: of wall clock: whether a five-subsystem cascade reached critical — and
#: therefore whether the Will got vetoed — depended on how busy the host
#: was between recording the events and reading them. That is a race
#: deciding survival policy. 0.01 is still 0.2% of the denominator, so it
#: cannot let a four-event cascade saturate, and it removes the timing
#: dependence from a threshold that gates Aura's own agency.
DEGRADATION_THREAT_SATURATION_EPSILON = 0.01
CRITICAL_LOG_COOLDOWN_S = 30.0
DEGRADATION_SEVERITY_WEIGHTS = {
    "critical": 2.0,
    "degraded": 1.0,
    "warning": 0.20,
    "debug": 0.0,
}
_EXISTENTIAL_STAKES_RECOVERABLE_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _habituation_multiplier(record: Any) -> float:
    """How much of this record's weight still lands, given familiarity.

    Keyed through :func:`signature_for`, the same function
    ``record_degradation`` writes through. Deriving the key here instead
    would be the classic silent gate failure: a counter accumulating under
    one key while its reader tests another never fires, and fails looking
    healthy.

    Note this uses the failure CLASS (subsystem + exception type), not the
    richer message-level signature used for within-window deduplication
    just below. Those are two different questions — "is this the same event
    twice?" versus "is this the same kind of thing again?" — and they need
    different granularity.

    Falls back to 1.0 (full weight) on any failure. An unreadable
    habituation store must never quiet a real cascade; the safe direction
    here is to feel too much, not too little.
    """
    try:
        return get_habituation().multiplier(
            signature_for(
                getattr(record, "subsystem", ""),
                getattr(record, "error_type", ""),
            )
        )
    except _EXISTENTIAL_STAKES_RECOVERABLE_ERRORS:
        return 1.0


class ExistentialStakes:
    """Computes and manages the existential survival stakes for Aura."""

    def __init__(self, memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES):
        self._lock = threading.Lock()
        self._memory_limit = memory_limit_bytes
        self._threat = 0.0
        
        # State tracking
        self._last_update_time: float | None = None
        self._rolling_loop_lag = 0.0
        self._rolling_cpu_load = 0.0
        self._total_ticks = 0

        # Sub-threat indices
        self._memory_threat = 0.0
        self._lag_threat = 0.0
        self._cpu_threat = 0.0
        self._degradation_threat = 0.0
        #: The part of degradation pressure that is substrate evidence, and the
        #: part that is a quality gate declining to ship text. Only the first
        #: may reach the survival veto.
        self._substrate_degradation_threat = 0.0
        self._quality_veto_weight = 0.0
        self._recent_degradation_weight = 0.0
        self._last_critical_log_time = 0.0
        self._last_critical_log_bucket = -1
        self._was_critical = False

        logger.info(
            "ExistentialStakes initialized. Memory limit: %.2f MB",
            self._memory_limit / (1024 * 1024),
        )

    @staticmethod
    def _degradation_record_weight(record: Any, *, now: float) -> float:
        severity = str(getattr(record, "severity", "") or "").lower()
        base = DEGRADATION_SEVERITY_WEIGHTS.get(severity, 0.0)
        if base <= 0.0:
            return 0.0
        age_s = max(0.0, now - float(getattr(record, "timestamp", now) or now))
        if age_s >= DEGRADATION_THREAT_WINDOW_S:
            return 0.0
        # A resolved transient should fade quickly instead of holding the live
        # Will in an existential veto for a full minute. Cascades still rise
        # because fresh records keep adding full weight.
        decay = 1.0 - (age_s / DEGRADATION_THREAT_WINDOW_S)
        return base * decay

    #: Root causes that are a DECISION not to say something, rather than
    #: evidence the substrate is failing.
    #:
    #: MEASURED live 2026-08-10. "what's actually on my screen right now?" was
    #: answered "Executive veto: survival_inhibition: existential threat level
    #: critical (0.80)" on a host at mem_threat=0.04 and cpu_threat=0.00. The
    #: entire threat was degradation weight, and the degradations were reply
    #: gates refusing drafts:
    #:
    #:   chat.cognitive_engine_reply: reply_reliability_gate_failed:
    #:       runtime_boilerplate,friendly_failure_floor,arithmetic_answer_missing
    #:   cognitive_engine: TurnOutcomeError: retryable_failure:
    #:       retryable_error_and_nothing_served
    #:   chat: CRITICAL SERVICE FAILURE: Subsystem 'cognitive_engine' ...
    #:
    #: That is a closed loop. A gate rejects a reply; the rejection is recorded
    #: as a critical degradation; degradation weight becomes existential
    #: threat; existential threat vetoes tool_execution and file_write; the
    #: blocked actions fail and are recorded in turn. The refusal rate feeds
    #: the veto that disables her tools, and the runtime reports an emergency
    #: while sitting at 4% memory.
    #:
    #: interface/routes/chat.py already names the principle for the log line:
    #: "A veto is not a model failure, and reporting it as one is what kept
    #: this entire class invisible." It applies here with more force, because
    #: here it does not merely mislabel the event — it disables her.
    #:
    #: These still raise OPERATIONAL pressure, which is capped below the veto
    #: threshold, exactly as high CPU and event-loop lag already are for the
    #: same reason: real signal, incapable of inhibiting action by itself.
    _QUALITY_VETO_MARKERS = (
        "reply_reliability_gate_failed",
        "retryable_error_and_nothing_served",
        "required_desktop_reply_remained_degraded",
        "reply_quality_gate",
        "friendly_failure_floor",
        "degraded-turn composer",
    )

    @classmethod
    def _is_quality_veto(cls, root_cause: str) -> bool:
        """Was this a refusal to ship text, rather than a substrate failure?"""
        lowered = str(root_cause or "").lower()
        return any(marker in lowered for marker in cls._QUALITY_VETO_MARKERS)

    #: A propagated failure carries its origin's text inside this prefix.
    _ESCALATION_PREFIX = "CRITICAL SERVICE FAILURE:"
    _ORIGINAL_ERROR_MARKER = "Original error:"

    @classmethod
    def _origin_subsystem(cls, message: str) -> str:
        """Which subsystem the wrapper says actually failed.

        The escalation text names it — "Subsystem 'cognitive_engine' failed
        with failure policy 'fail-closed'" — so a propagated record can be
        filed under the subsystem it is about rather than the one that
        re-raised it. Nested wrappers name each layer; the innermost is the
        one that really broke, so the last match wins.
        """
        found = _ORIGIN_SUBSYSTEM_RE.findall(str(message or ""))
        return str(found[-1]) if found else ""

    @classmethod
    def _is_propagated(cls, message: str) -> bool:
        """Did a caller re-record someone else's failure under its own name?"""
        return cls._ESCALATION_PREFIX in str(message or "")

    @classmethod
    def _root_cause_text(cls, message: str) -> str:
        """Strip propagation wrappers down to the failure that actually happened.

        A failure is recorded by the subsystem that hit it, and again by every
        caller that re-raises it, each under its own name. The dedup key is
        (subsystem, error type), so three names means three "distinct problems"
        for one event. Measured live 2026-07-28, a single empty generation
        produced:

            cognitive_engine (critical)         compact desktop generation
                                                returned no usable text
            chat (degraded)                     CRITICAL SERVICE FAILURE:
                                                Subsystem 'cognitive_engine' ...
            chat.cognitive_engine_reply (degraded)  ... the same failure again

        which is 2.0 + 1.0 + 1.0 = 4.0 against a denominator of 5, i.e.
        deg_threat 0.8 — enough on its own to trip the Ulysses covenant and
        refuse every build the owner asked for, from one failed turn.

        Collapsing the wrappers makes the three records share a signature, so
        the harmonic discount treats them as one problem repeating rather than
        three problems happening.
        """
        text = str(message or "")
        # Wrappers nest; unwrap until the text stops being one.
        for _ in range(4):
            if cls._ESCALATION_PREFIX not in text:
                break
            marker = text.find(cls._ORIGINAL_ERROR_MARKER)
            if marker < 0:
                break
            text = text[marker + len(cls._ORIGINAL_ERROR_MARKER):].strip()
        # Unwrapping leaves the re-raised exception's own type in front of the
        # message ("RuntimeError: compact desktop generation ..."), which would
        # keep the propagated record from matching the origin it came from.
        text = _EXCEPTION_PREFIX_RE.sub("", text, count=1).strip()
        return text[:80]

    def _should_log_critical(self, now: float) -> bool:
        bucket = int(self._threat * 10.0)
        if not self._was_critical:
            self._last_critical_log_time = now
            self._last_critical_log_bucket = bucket
            self._was_critical = True
            return True
        if bucket != self._last_critical_log_bucket:
            self._last_critical_log_time = now
            self._last_critical_log_bucket = bucket
            return True
        if now - self._last_critical_log_time >= CRITICAL_LOG_COOLDOWN_S:
            self._last_critical_log_time = now
            return True
        return False

    def update(self) -> float:
        """Tick measurements, compute sub-threats, and return the combined threat."""
        with self._lock:
            now = time.time()
            self._total_ticks += 1

            # 1. Memory Threat
            process_mem = 0
            try:
                process = get_resource_observer().process(os.getpid())
                process_mem = int(process.rss_bytes) if process is not None else 0
            except _EXISTENTIAL_STAKES_RECOVERABLE_ERRORS as e:
                logger.debug("Failed to read process memory: %s", e)
            
            if process_mem > 0:
                self._memory_threat = min(1.0, process_mem / self._memory_limit)
            else:
                self._memory_threat = 0.0

            # 2. Event Loop Lag Threat
            # We measure scheduling delay: how long it actually took compared to the 
            # expected tick rate (nominally 1.0s for the heartbeat).
            if self._last_update_time is not None:
                dt = now - self._last_update_time
                # Lag is anything exceeding 1.1s (allowing small scheduler jitter)
                lag = max(0.0, dt - 1.1)
                # EMA filter for lag (slow rise, fast fall)
                alpha = 0.2 if lag > self._rolling_loop_lag else 0.4
                self._rolling_loop_lag = (1 - alpha) * self._rolling_loop_lag + alpha * lag
            self._last_update_time = now

            # Lag of 3.0 seconds or more is considered critical threat
            self._lag_threat = min(1.0, self._rolling_loop_lag / 3.0)

            # 3. CPU Load Threat
            cpu = 0.0
            if psutil is not None:
                try:
                    # Non-blocking cpu calculation
                    cpu = psutil.cpu_percent(interval=None) / 100.0
                except _EXISTENTIAL_STAKES_RECOVERABLE_ERRORS as e:
                    logger.debug("Failed to read CPU: %s", e)
            
            # EMA for CPU
            self._rolling_cpu_load = 0.8 * self._rolling_cpu_load + 0.2 * cpu
            self._cpu_threat = min(1.0, self._rolling_cpu_load)

            # 4. Degradation / Exception Threat
            # Use severity-weighted, decaying degradation pressure. Counting
            # every recent warning as full existential danger made one repaired
            # foreground failure keep flooding the neural stream and blocking
            # actions. Critical/degraded cascades remain existential pressure;
            # warnings are weak signal; debug/lifecycle noise is ignored.
            recent_degradation_weight = 0.0
            try:
                tracker = get_degradation_tracker()
                if tracker and hasattr(tracker, "_records"):
                    # Distinct problems, not repetitions of one.
                    #
                    # Summing every record let a single stuck fault saturate
                    # survival threat by itself. Measured live 2026-07-27: an
                    # empty draft recorded once per turn from one fail-closed
                    # subsystem drove deg_threat to 1.00 on a host at 4% memory
                    # pressure, which pinned existential_threat, which tripped
                    # the Ulysses covenant, which refused every build the owner
                    # asked for. The runtime was healthy and reported itself in
                    # an emergency.
                    #
                    # A fault repeating is worse than a fault happening once,
                    # so repeats still count — but the series has to converge,
                    # or a stuck fault reaches saturation anyway and merely
                    # takes longer about it. A harmonic discount grows like
                    # log n and still crossed the threshold by 40 repeats; the
                    # square converges to about 1.64x a single occurrence, so
                    # one problem is worth at most ~2 of itself and can never
                    # outweigh several genuinely different ones. The escalation
                    # governor already caps the RATE of re-escalation for this
                    # reason; this applies the principle to the measurement
                    # that gates survival.
                    seen: dict[tuple[str, str], int] = {}
                    recent_degradation_weight = 0.0
                    quality_veto_weight = 0.0
                    for record in tracker._records:
                        weight = self._degradation_record_weight(record, now=now)
                        if weight <= 0.0:
                            continue
                        # Key on the failure, not on who reported it: a
                        # propagated record names a different subsystem for the
                        # same event, and counting those separately turns one
                        # failed turn into an existential emergency.
                        raw_message = str(getattr(record, "error_message", "") or "")
                        if self._is_propagated(raw_message):
                            # Demonstrably the same event travelling upward.
                            # File it under the subsystem the wrapper says
                            # failed, so it merges with that subsystem's own
                            # record of the failure rather than counting beside
                            # it.
                            signature = (
                                self._origin_subsystem(raw_message)
                                or str(getattr(record, "subsystem", "") or "unknown"),
                                self._root_cause_text(raw_message),
                            )
                        else:
                            # No wrapper, so no evidence this is a repeat. Five
                            # subsystems failing independently is a real
                            # cascade and must still be able to reach critical.
                            signature = (
                                str(getattr(record, "subsystem", "") or "unknown"),
                                self._root_cause_text(raw_message)
                                or str(getattr(record, "error_type", "") or ""),
                            )
                        repeat = seen.get(signature, 0)
                        seen[signature] = repeat + 1
                        # Two independent discounts, for two different facts.
                        #
                        # `repeat` is WITHIN this window: five copies of one
                        # failure is one failure, not a cascade.
                        #
                        # Habituation is ACROSS windows: a signature that has
                        # been recurring for weeks is a condition she has
                        # already absorbed, and charging full survival
                        # pressure for it every window means a known chronic
                        # fault generates fresh existential threat forever.
                        # It attenuates to a floor and never past it, and it
                        # re-sensitises once the signature goes quiet — so
                        # this quiets the familiar, never the new, and never
                        # silences anything completely.
                        #
                        # The degradation RECORD is untouched. Only what is
                        # felt from it is scaled.
                        familiarity = _habituation_multiplier(record)
                        contribution = (weight / (1.0 + repeat) ** 2) * familiarity
                        recent_degradation_weight += contribution
                        if self._is_quality_veto(signature[1]):
                            quality_veto_weight += contribution
            except _EXISTENTIAL_STAKES_RECOVERABLE_ERRORS as e:
                logger.debug("Failed to query degradation tracker: %s", e)

            self._recent_degradation_weight = recent_degradation_weight
            # 5 fresh degraded-equivalent events in a minute is high threat.
            if recent_degradation_weight >= (
                DEGRADATION_THREAT_DENOMINATOR - DEGRADATION_THREAT_SATURATION_EPSILON
            ):
                self._degradation_threat = 1.0
            else:
                self._degradation_threat = min(
                    1.0,
                    recent_degradation_weight / DEGRADATION_THREAT_DENOMINATOR,
                )

            # The share of that pressure which is a refusal to ship text
            # rather than a substrate failure. Reported whole above, so the
            # felt threat and the neural stream still see everything; split
            # here, so only substrate evidence can reach the survival veto.
            # Saturating on the same rule as the total, not a bare min(): the
            # epsilon exists because fresh records carry a decay factor, so a
            # genuine five-event cascade lands fractionally under the
            # denominator and must still read as 1.0. Without it a real
            # substrate cascade scored 0.9998 while the total scored 1.00.
            self._quality_veto_weight = quality_veto_weight
            substrate_weight = max(0.0, recent_degradation_weight - quality_veto_weight)
            if substrate_weight >= (
                DEGRADATION_THREAT_DENOMINATOR - DEGRADATION_THREAT_SATURATION_EPSILON
            ):
                self._substrate_degradation_threat = 1.0
            else:
                self._substrate_degradation_threat = min(
                    1.0, substrate_weight / DEGRADATION_THREAT_DENOMINATOR
                )

            # Combined Threat. Distinguish SURVIVAL pressure (genuine death
            # risk) from OPERATIONAL pressure (busy/laggy but not dying):
            #   - memory exhaustion → OOM kill (the 110GB incident) and
            #     degradation cascades are real survival threats; uncapped, they
            #     may reach 1.0 and trigger the will-system's survival veto.
            #   - high CPU and event-loop lag are NORMAL during heavy 32B
            #     generation. Treating them as maximal survival threat made the
            #     will-veto block Aura's own actions whenever it worked hard
            #     (observed: continual-learning battery blocked at threat=1.00
            #     under load). Operational pressure still raises the felt threat
            #     so survival perception isn't blind, but it is capped BELOW the
            #     veto threshold so load alone can never inhibit action. Loop
            #     wedges are owned by the StallWatchdog, not this veto.
            #   - a reply a quality gate refused to ship is a DECISION, not a
            #     substrate event, so it joins operational pressure with the
            #     load signals. Counting it as survival pressure closed a loop:
            #     gate rejects reply -> recorded critical -> deg_threat -> veto
            #     -> tool_execution and file_write blocked -> those failures
            #     recorded too. Live 2026-08-10 a screen read was refused with
            #     "existential threat level critical (0.80)" at mem_threat=0.04.
            survival_pressure = max(
                self._memory_threat, self._substrate_degradation_threat
            )
            operational_pressure = min(
                OPERATIONAL_THREAT_CAP,
                max(self._lag_threat, self._cpu_threat, self._degradation_threat),
            )
            self._threat = max(survival_pressure, operational_pressure)

            # Log critical warning if threat is high, but coalesce repeated
            # ticks. The neural stream should show a state transition, not a
            # log storm.
            if self._threat > CRITICAL_THREAT_THRESHOLD and self._should_log_critical(now):
                logger.warning(
                    "CRITICAL EXISTENTIAL STAKES: threat=%.2f (mem_threat=%.2f, lag_threat=%.2f, cpu_threat=%.2f, deg_threat=%.2f)",
                    self._threat,
                    self._memory_threat,
                    self._lag_threat,
                    self._cpu_threat,
                    self._degradation_threat,
                )
            elif self._was_critical and self._threat <= CRITICAL_THREAT_THRESHOLD:
                self._was_critical = False
                self._last_critical_log_bucket = -1
                logger.info(
                    "Existential stakes recovered below critical threshold: threat=%.2f "
                    "(mem_threat=%.2f, lag_threat=%.2f, cpu_threat=%.2f, deg_threat=%.2f)",
                    self._threat,
                    self._memory_threat,
                    self._lag_threat,
                    self._cpu_threat,
                    self._degradation_threat,
                )

            return self._threat

    def get_existential_threat(self) -> float:
        with self._lock:
            return self._threat

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "existential_threat": round(self._threat, 4),
                "memory_threat": round(self._memory_threat, 4),
                "lag_threat": round(self._lag_threat, 4),
                "cpu_threat": round(self._cpu_threat, 4),
                "degradation_threat": round(self._degradation_threat, 4),
                "substrate_degradation_threat": round(self._substrate_degradation_threat, 4),
                "quality_veto_weight": round(self._quality_veto_weight, 4),
                "recent_degradation_weight": round(self._recent_degradation_weight, 4),
                "total_ticks": self._total_ticks,
                "rolling_loop_lag_s": round(self._rolling_loop_lag, 3),
            }

    def get_context_block(self) -> str:
        """Constructs an LLM system prompt block warning of resource constraints."""
        with self._lock:
            if self._threat < 0.2:
                return ""
            
            intensity = "CRITICAL" if self._threat > 0.75 else "NOTABLE"
            
            block = [
                "## SYSTEM RESOURCE WARNING (EXISTENTIAL PRESSURE)",
                f"Felt Survival Threat Level: {intensity} (threat={self._threat:.2f})",
                "Your physical process substrate is under severe constraints:"
            ]
            
            if self._memory_threat > 0.5:
                block.append(f"  - Memory footprint is extremely high ({self._memory_threat * 100:.1f}% of ceiling).")
            if self._lag_threat > 0.4:
                block.append(f"  - Event loop scheduling lag detected ({self._rolling_loop_lag:.2f}s delay). Thread starvation is imminent.")
            if self._cpu_threat > 0.7:
                block.append(f"  - CPU usage is saturated ({self._rolling_cpu_load * 100:.1f}%).")
            if self._degradation_threat > 0.5:
                block.append("  - Active module degradation and exception events are cascading.")

            block.extend([
                "Cognitive guidelines under existential pressure:",
                "1. Keep all responses brief, concise, and focused. Do not waste tokens or CPU cycles.",
                "2. Prioritize system stability, self-inspection, and resource consolidation.",
                "3. If appropriate, recommend running garbage collection, compaction, or entering safe mode.",
                ""
            ])
            return "\n".join(block)


# ── Singleton Accessor ────────────────────────────────────────────────────────

_INSTANCE: ExistentialStakes | None = None


def _resolve_memory_limit_bytes() -> int:
    """Machine-aware survival memory ceiling for the live singleton.

    A fixed 2GB ceiling makes a large box perceive perpetual near-death: the
    Python runtime baseline (~1.5GB RSS) alone sits at ~0.75 memory_threat,
    parking the will-system right at its survival-veto boundary
    (``threat > 0.75``) during normal operation and intermittently inhibiting
    heavy actions. Derive the ceiling from the same process-RSS limit the
    memory watchdog enforces, so existential "near-death" aligns with the
    watchdog's refuse-heavy-generation point instead of a stale default.

    Honors ``AURA_EXISTENTIAL_MEMORY_LIMIT_GB`` for explicit control; falls
    back to the 2GB default only when nothing better can be determined.
    """
    override = os.environ.get("AURA_EXISTENTIAL_MEMORY_LIMIT_GB", "").strip()
    if override:
        try:
            gb = float(override)
            if gb > 0.0:
                return int(gb * (1024 ** 3))
        except (TypeError, ValueError):
            pass
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        limit_gb = float(get_memory_pressure_snapshot().process_rss_limit_gb or 0.0)
        if limit_gb > 0.0:
            return int(limit_gb * (1024 ** 3))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    return DEFAULT_MEMORY_LIMIT_BYTES


def get_existential_stakes() -> ExistentialStakes:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ExistentialStakes(_resolve_memory_limit_bytes())
    return _INSTANCE
