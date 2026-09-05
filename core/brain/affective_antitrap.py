"""core/brain/affective_antitrap.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Anti-trap governor for affect-modulated sampling: breaks the
digital-depression feedback loop.

The trap (external review, July 3): an error cascade spikes distress
telemetry (cortisol, error pressure, low coherence) → the latent bridge
clamps sampling temperature to its deterministic floor for safety → at
floor temperature the model loses the semantic variance needed to invent
the fix → errors persist → distress persists → the clamp never lifts.
The system wedges itself in a hyper-deterministic depression.

Three mechanisms, all bounded and observable:

1. TRAP DETECTION — a ring of recent observations. Trap = temperature
   pinned at/near floor across the whole window, spanning enough wall time
   to be a condition rather than a burst, AND distress not improving across
   it. All three required: pinned temperature while distress RECOVERS is
   the clamp working; distress without pinning is ordinary stress; eight
   calls in the same millisecond are one moment, not a sustained state.
2. GOVERNED ESCAPE — when trapped, raise the temperature floor to an
   exploration band for a bounded number of computations (annealing
   escape), record an AFFECT-TRAP fault occurrence carrying the escape id
   and the observed window, then re-measure: the escape's efficacy is the
   mean distress across the escape against the mean before it, and it is
   written into a completion record carrying the same escape id. A cooldown
   prevents oscillation.
3. LANE DECOUPLING — self-repair/ideation lanes get a guaranteed
   exploration floor at ALL times: safety clamps may govern speech, but
   the lanes that must invent a way out are never starved of variance.
   (Deterministic final code EMISSION stays at temp 0 by design — the
   floor protects idea search, not token-exact codegen.)

CP126 review found eleven defects in the first implementation, most of
which are the same mistake wearing different clothes: **the module wrote
down what it intended rather than what happened.**

  * A repair-lane call returned before the escape counter was touched, so
    an escape could stay open forever while repair work streamed through.
  * The ring recorded the REQUESTED temperature with no lane, so repair
    calls that actually ran at 0.50 filled the trap window with 0.15 and
    could open an escape for speech on evidence that never existed.
  * "Distress" substituted optimistic constants for missing axes, so a
    disconnected substrate read as a calm one.
  * Escape efficacy was one endpoint sample: any positive delta was
    "recovered", including noise.
  * The fault records held a prose summary of the thresholds instead of the
    observations, and the completion record had no way to name which escape
    it belonged to.
  * A failed fault write was logged at debug while the escape counters went
    on reporting a recorded escape.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.AffectiveAntitrap")


@dataclass(frozen=True)
class AntitrapCalibration:
    """The numbers, with a version and a stated basis.

    They used to be bare module constants and ``status()`` exposed two of
    them. Anything that decides when to override a safety clamp has to say
    where its thresholds came from and which set is running, or the next
    person tuning it cannot tell a calibrated value from a guess.
    """

    version: str = "2026-08-antitrap-1"

    #: Observations examined for the trap signature.
    window: int = 8
    #: "Pinned" = within this of the hard floor. Sized to the bridge's own
    #: rounding, not to a desired hit rate.
    floor_epsilon: float = 0.05
    #: The latent bridge's deterministic floor; mirrors compute_inference_params.
    temp_hard_floor: float = 0.15
    #: Exploration band floor during an escape.
    escape_temp_floor: float = 0.45
    #: Computations the escape floor persists for.
    escape_span: int = 6
    #: Minimum seconds between escapes.
    cooldown_s: float = 600.0
    #: Repair/ideation lanes never sample below this.
    lane_exploration_floor: float = 0.50
    #: A window has to cover this much wall time to be a sustained state.
    #: Eight calls inside one turn are one moment.
    min_window_span_s: float = 20.0
    #: Distress must fall by at least this much, on the mean, for an escape
    #: to be called recovered. Below it the delta is not distinguishable
    #: from the ordinary drift of the score.
    min_efficacy_delta: float = 0.05
    #: How much the improvement test tolerates before calling it "not
    #: improving" — the same drift band, on the other side.
    improvement_tolerance: float = 0.02

    basis: str = (
        "temp_hard_floor and floor_epsilon mirror compute_inference_params' "
        "own clamp; window/escape_span/cooldown are the original July-3 "
        "anti-trap sizing; min_window_span_s and min_efficacy_delta were "
        "added by CP126 review and are NOT empirically calibrated — they are "
        "the smallest values that make the burst and noise cases impossible."
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "window": self.window,
            "floor_epsilon": self.floor_epsilon,
            "temp_hard_floor": self.temp_hard_floor,
            "escape_temp_floor": self.escape_temp_floor,
            "escape_span": self.escape_span,
            "cooldown_s": self.cooldown_s,
            "lane_exploration_floor": self.lane_exploration_floor,
            "min_window_span_s": self.min_window_span_s,
            "min_efficacy_delta": self.min_efficacy_delta,
            "improvement_tolerance": self.improvement_tolerance,
            "basis": self.basis,
        }


DEFAULT_CALIBRATION = AntitrapCalibration()

# Module-level names kept for the callers and tests that import them. They
# are views on the calibration, not a second source of truth.
WINDOW = DEFAULT_CALIBRATION.window
FLOOR_EPSILON = DEFAULT_CALIBRATION.floor_epsilon
TEMP_HARD_FLOOR = DEFAULT_CALIBRATION.temp_hard_floor
ESCAPE_TEMP_FLOOR = DEFAULT_CALIBRATION.escape_temp_floor
ESCAPE_SPAN = DEFAULT_CALIBRATION.escape_span
COOLDOWN_S = DEFAULT_CALIBRATION.cooldown_s
LANE_EXPLORATION_FLOOR = DEFAULT_CALIBRATION.lane_exploration_floor


class Lane(StrEnum):
    """Which kind of work is asking for sampling parameters.

    The lane used to be an arbitrary string, and membership in a public set
    was the only thing standing between a caller and a 0.50 temperature
    floor. A closed set is not authentication — binding the lane to the
    call path is the latent bridge's job, and ``resolve_lane`` is the seam
    where that check would land — but an unrecognised label can no longer
    buy exploration variance by spelling itself "repair".
    """

    SPEECH = "speech"
    REPAIR = "repair"
    IDEATION = "ideation"
    DISCOVERY = "discovery"
    SELF_REPAIR = "self_repair"
    CODE_EMISSION = "code_emission"


#: Lanes whose purpose is inventing a way out — never variance-starved.
EXPLORATION_LANES = frozenset(
    {Lane.REPAIR, Lane.IDEATION, Lane.DISCOVERY, Lane.SELF_REPAIR}
)

#: Axes the distress score is built from, and what each is worth.
_DISTRESS_WEIGHTS: tuple[tuple[str, float, float], ...] = (
    # (key, weight, value assumed when the axis is absent)
    ("cortisol", 0.35, 0.3),
    ("active_error_pressure", 0.30, 0.0),
    ("frustration", 0.20, 0.0),
    ("organismal_coherence", 0.15, 1.0),
)


def resolve_lane(value: Any) -> tuple[Lane, bool]:
    """Map a caller's label to a known lane, and say whether it was one."""
    if isinstance(value, Lane):
        return value, True
    try:
        return Lane(str(value).strip().lower()), True
    except ValueError:
        return Lane.SPEECH, False


def _finite(value: Any) -> float | None:
    """A float, or None for anything that is not a usable number.

    NaN is the case that mattered: it fails every comparison, so it slid
    through both floor tests and pin detection as though it had been
    checked.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


@dataclass(frozen=True)
class DistressReading:
    """A distress score that says how much of the substrate it actually saw.

    The score alone could not distinguish a calm organism from a
    disconnected one: absent axes were replaced with optimistic constants
    and the result was returned as an ordinary number.
    """

    score: float
    axes_present: tuple[str, ...]
    axes_missing: tuple[str, ...]
    axes_invalid: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.axes_missing and not self.axes_invalid

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "axes_present": list(self.axes_present),
            "axes_missing": list(self.axes_missing),
            "axes_invalid": list(self.axes_invalid),
            "complete": self.complete,
        }


def distress_reading(substrate: Any) -> DistressReading:
    """Collapse the distress-relevant axes to one score, and report coverage."""
    present: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    raw = 0.0
    source = substrate if isinstance(substrate, dict) else {}
    if not isinstance(substrate, dict):
        invalid.append("<substrate is not a mapping>")

    for key, weight, absent_value in _DISTRESS_WEIGHTS:
        if key not in source:
            missing.append(key)
            value = absent_value
        else:
            number = _finite(source.get(key))
            if number is None:
                invalid.append(key)
                value = absent_value
            else:
                present.append(key)
                value = number
        contribution = (1.0 - value) if key == "organismal_coherence" else value
        raw += weight * contribution

    return DistressReading(
        score=max(0.0, min(1.0, raw)),
        axes_present=tuple(present),
        axes_missing=tuple(missing),
        axes_invalid=tuple(invalid),
    )


def distress_score(substrate: Any) -> float:
    """The score alone, for callers that only need the number."""
    return distress_reading(substrate).score


@dataclass
class _Observation:
    at: float
    lane: Lane
    #: What the caller asked for.
    requested_temperature: float
    #: What this module returned — the value the model actually sampled at.
    effective_temperature: float
    distress: float
    complete: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": round(self.at, 3),
            "lane": str(self.lane),
            "requested_temperature": round(self.requested_temperature, 4),
            "effective_temperature": round(self.effective_temperature, 4),
            "distress": round(self.distress, 4),
            "complete": self.complete,
        }


@dataclass
class _PendingFault:
    """A fault occurrence to emit AFTER the lock is released."""

    details: str
    recovered: bool


class AffectiveTrapGuard:
    """Observes every sampling-parameter computation; breaks closed loops.

    Thread-safe. ``observe_and_adjust`` is O(window) on a tiny deque —
    negligible on the inference-parameter path — and it no longer calls the
    fault registry while holding the lock, so a slow or reentrant registry
    can neither serialize every parameter computation nor deadlock it.
    """

    def __init__(
        self,
        calibration: AntitrapCalibration | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._calibration = calibration or DEFAULT_CALIBRATION
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._ring: deque[_Observation] = deque(maxlen=self._calibration.window)
        self._escape_remaining = 0
        self._last_escape_at = -self._calibration.cooldown_s
        self._escape_id = ""
        self._escape_entry_distress = 0.0
        self._escape_baseline = 0.0
        self._escape_samples: list[float] = []
        self._escapes_fired = 0
        self._fault_records_written = 0
        self._fault_record_failures = 0
        self._readings_rejected = 0
        self._unknown_lane_requests = 0

    # ── the seam: called by the latent bridge per computation ────────

    def observe_and_adjust(
        self,
        temperature: float,
        substrate: dict[str, float],
        *,
        lane: str = "speech",
    ) -> tuple[float, str]:
        """Record the observation; return (adjusted_temperature, note).

        note is empty when nothing changed — the latent bridge appends it
        to its rationale so every intervention is visible in telemetry.
        """
        cal = self._calibration
        resolved_lane, recognised = resolve_lane(lane)
        reading = distress_reading(substrate)
        requested = _finite(temperature)
        now = float(self._clock())

        notes: list[str] = []
        if not recognised:
            notes.append(f"antitrap: unknown lane {lane!r} treated as speech")
        if requested is None:
            # A non-finite temperature fails every comparison, so it used to
            # pass both floor tests and pin detection untouched.
            requested = cal.temp_hard_floor
            notes.append(
                f"antitrap: non-finite temperature replaced with "
                f"{cal.temp_hard_floor:.2f}"
            )

        pending: list[_PendingFault] = []
        with self._lock:
            if not recognised:
                self._unknown_lane_requests += 1
            if not reading.complete:
                self._readings_rejected += 1

            effective = requested

            # The escape advances on EVERY observed computation. It used to
            # be decremented after the exploration-lane early return, so a
            # stream of repair work kept an escape open indefinitely and its
            # completion — with the efficacy measurement — never ran.
            escape_active = self._escape_remaining > 0
            if escape_active:
                self._escape_samples.append(reading.score)
                self._escape_remaining -= 1
                if effective < cal.escape_temp_floor:
                    effective = cal.escape_temp_floor
                    notes.append(
                        f"antitrap: escape floor {cal.escape_temp_floor:.2f} "
                        f"({self._escape_remaining} computations remain)"
                    )
                if self._escape_remaining == 0:
                    pending.append(self._complete_escape_locked(reading.score))

            if resolved_lane in EXPLORATION_LANES and effective < cal.lane_exploration_floor:
                notes.append(
                    f"antitrap: lane '{resolved_lane}' exploration floor "
                    f"{cal.lane_exploration_floor:.2f} (was {requested:.2f})"
                )
                effective = cal.lane_exploration_floor

            if (
                not escape_active
                and self._is_trapped_locked(now)
                and (now - self._last_escape_at) >= cal.cooldown_s
            ):
                pending.append(self._begin_escape_locked(now, reading.score))
                if effective < cal.escape_temp_floor:
                    effective = cal.escape_temp_floor
                notes.append(
                    "antitrap: TRAP DETECTED — temperature pinned at floor "
                    f"with non-improving distress ({reading.score:.2f}); escape "
                    f"band opened for {cal.escape_span} computations"
                )

            self._ring.append(
                _Observation(
                    at=now,
                    lane=resolved_lane,
                    requested_temperature=requested,
                    effective_temperature=effective,
                    distress=reading.score,
                    complete=reading.complete,
                )
            )

        # Outside the lock on purpose: the registry is another subsystem.
        for record in pending:
            self._record_fault(details=record.details, recovered=record.recovered)

        return effective, "; ".join(notes)

    # ── trap signature ────────────────────────────────────────────────

    def _trap_window_locked(self) -> list[_Observation]:
        """The observations that may testify about a speech-lane trap.

        Exploration lanes are excluded because they are never clamped — a
        repair call that ran at 0.50 used to enter the ring as the 0.15 it
        asked for, so repair traffic could manufacture the pinned-floor
        signature for speech. Incomplete readings are excluded because a
        disconnected substrate is not evidence of a calm one.
        """
        return [
            o
            for o in self._ring
            if o.lane not in EXPLORATION_LANES and o.complete
        ]

    def _window_pinned_locked(self, observations: list[_Observation]) -> bool:
        """Is the model actually sampling at the floor right now?

        Judged on the EFFECTIVE temperature — what was returned and sampled
        at — not on what the caller asked for. During an escape the model
        samples at the escape floor, so escape-era observations are not
        evidence of a pinned model, which is what stopped an escape from
        immediately re-arming on its own output.
        """
        cal = self._calibration
        return bool(observations) and all(
            o.effective_temperature <= cal.temp_hard_floor + cal.floor_epsilon
            for o in observations
        )

    def _is_trapped_locked(self, now: float) -> bool:
        cal = self._calibration
        observations = self._trap_window_locked()
        if len(observations) < cal.window:
            return False
        if not self._window_pinned_locked(observations):
            return False
        # A burst of eight calls inside one turn is one moment. The window
        # has to cover real time before it describes a sustained condition.
        span = observations[-1].at - observations[0].at
        if span < cal.min_window_span_s:
            return False
        half = len(observations) // 2
        early = sum(o.distress for o in observations[:half]) / half
        late = sum(o.distress for o in observations[half:]) / (len(observations) - half)
        # Distress not improving: the clamp is no longer buying recovery.
        return late >= early - cal.improvement_tolerance

    # ── escape lifecycle ──────────────────────────────────────────────

    def _begin_escape_locked(self, now: float, entry_distress: float) -> _PendingFault:
        cal = self._calibration
        observations = self._trap_window_locked()
        baseline = (
            sum(o.distress for o in observations) / len(observations)
            if observations
            else entry_distress
        )

        self._escape_id = uuid.uuid4().hex[:12]
        self._escape_remaining = cal.escape_span
        self._last_escape_at = now
        self._escape_entry_distress = entry_distress
        self._escape_baseline = baseline
        self._escape_samples = []
        self._escapes_fired += 1

        logger.warning(
            "AFFECTIVE TRAP %s: temperature pinned at floor with non-improving "
            "distress (%.2f, baseline %.2f) — opening exploration escape band "
            "for %d computations",
            self._escape_id, entry_distress, baseline, cal.escape_span,
        )
        # The record carries the observations themselves. It used to carry a
        # sentence restating the thresholds, which cannot be replayed and
        # cannot be checked.
        window_payload = [o.as_dict() for o in observations]
        return _PendingFault(
            details=(
                f"escape {self._escape_id} opened; calibration "
                f"{cal.version}; entry_distress {entry_distress:.3f}; "
                f"baseline {baseline:.3f}; window {window_payload}"
            ),
            recovered=False,
        )

    def _complete_escape_locked(self, exit_distress: float) -> _PendingFault:
        cal = self._calibration
        samples = list(self._escape_samples)
        mean_during = sum(samples) / len(samples) if samples else exit_distress
        delta = self._escape_baseline - mean_during
        recovered = delta >= cal.min_efficacy_delta

        logger.info(
            "AFFECTIVE TRAP %s escape complete: baseline %.3f → mean %.3f over "
            "%d samples (delta %+.3f, recovered=%s)",
            self._escape_id, self._escape_baseline, mean_during, len(samples),
            delta, recovered,
        )
        details = (
            f"escape {self._escape_id} complete: baseline "
            f"{self._escape_baseline:.3f} -> mean {mean_during:.3f} over "
            f"{len(samples)} samples (delta {delta:+.3f}, threshold "
            f"{cal.min_efficacy_delta:.3f}); exit_distress {exit_distress:.3f}. "
            "Uncontrolled: this compares the escape window against the window "
            "before it, and does not prove the temperature caused the change."
        )
        self._escape_samples = []
        return _PendingFault(details=details, recovered=recovered)

    def _record_fault(self, *, details: str, recovered: bool) -> None:
        """Write the occurrence, and count it when the write does not happen.

        The counters used to advance whether or not this succeeded, and the
        failure was a debug line — so ``status`` could report a fired and
        completed escape whose promised audit record did not exist.
        """
        try:
            from core.resilience.fault_taxonomy import get_fault_registry

            get_fault_registry().record_fault(
                "AFFECT-TRAP", subsystem="latent_bridge.antitrap",
                details=details, recovered=recovered,
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            with self._lock:
                self._fault_record_failures += 1
            record_degradation(
                "affective_antitrap",
                exc,
                action="AFFECT-TRAP occurrence was not recorded",
            )
            return
        with self._lock:
            self._fault_records_written += 1

    def status(self) -> dict[str, Any]:
        with self._lock:
            window = self._trap_window_locked()
            return {
                "window_filled": len(self._ring),
                "trap_window_filled": len(window),
                "trap_window_pinned": self._window_pinned_locked(window),
                "escape_active": self._escape_remaining > 0,
                "escape_remaining": self._escape_remaining,
                "escape_id": self._escape_id,
                "escapes_fired": self._escapes_fired,
                "fault_records_written": self._fault_records_written,
                "fault_record_failures": self._fault_record_failures,
                "audit_complete": self._fault_record_failures == 0,
                "readings_rejected": self._readings_rejected,
                "unknown_lane_requests": self._unknown_lane_requests,
                "cooldown_s": self._calibration.cooldown_s,
                "lane_floor": self._calibration.lane_exploration_floor,
                "calibration": self._calibration.as_dict(),
            }


_guard: AffectiveTrapGuard | None = None
_guard_lock = threading.Lock()


def get_affective_trap_guard() -> AffectiveTrapGuard:
    global _guard
    if _guard is None:
        with _guard_lock:
            if _guard is None:
                _guard = AffectiveTrapGuard()
    return _guard


__all__ = [
    "COOLDOWN_S",
    "DEFAULT_CALIBRATION",
    "ESCAPE_SPAN",
    "ESCAPE_TEMP_FLOOR",
    "EXPLORATION_LANES",
    "FLOOR_EPSILON",
    "LANE_EXPLORATION_FLOOR",
    "TEMP_HARD_FLOOR",
    "WINDOW",
    "AffectiveTrapGuard",
    "AntitrapCalibration",
    "DistressReading",
    "Lane",
    "distress_reading",
    "distress_score",
    "get_affective_trap_guard",
    "resolve_lane",
]
