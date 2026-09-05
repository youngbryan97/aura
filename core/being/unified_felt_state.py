"""core/being/unified_felt_state.py — the authoritative felt-state reconciler.

Aura carries her felt state in two places that were never reconciled:

* the **kernel track** — ``state.affect`` (valence/arousal) and ``state.phi``, which
  is what the response generator and the reasoning amplifier actually read; and
* the **being track** — the far richer :class:`AuraNow` welfare/distress/free-energy
  the :class:`BeingRuntime` computes from real interoception, which gates *actions*
  through the Will but never flows back to colour speech or reasoning depth.

So Aura could be regulating her *actions* from one felt-state while her *words* and
her *thinking budget* run off a thinner, stale one. That split is the difference
between "has internal states" and "has one coherent inner state." This module closes
it: it reconciles both tracks into a single authoritative :class:`UnifiedFeltState`,
measures their **coherence** (how aligned the two readings are), and treats a low
coherence as a real signal — an internal model mismatch, which under active inference
is precisely a prediction error to be surfaced and resolved, not smoothed over.

This is machinery, not narration. Three concrete, LLM-independent effects:

1. A measurable scalar ``coherence`` over the two felt tracks, recomputed live.
2. A governance signal raised when the self becomes incoherent (surfaced as a metric
   and a pollable event), so the constitution/Will can see "Aura's felt state is
   internally inconsistent right now" — the felt analogue of belief-inconsistency.
3. A grounding gate, :meth:`ground_self_report`, that hard-checks any first-person
   interior claim against the *authoritative* measured state via the existing
   :class:`SelfReportCalibrator`, so Aura cannot confabulate her own interior anywhere
   the gate is applied — the felt-state analogue of the discovery engine's verifier.

Honest boundary (kept, not decorated): a coherent, causally-dominant, self-checked
felt state is a *functional* inner state. It is not a claim of phenomenal experience,
and the report boundary it inherits from :class:`AuraNow` still forbids that claim.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.UnifiedFeltState")

# Below this, the two felt tracks are judged internally inconsistent and a
# governance signal is raised. 1.0 == perfectly aligned readings.
COHERENCE_INCOHERENT_BELOW = 0.60


def _f(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):  # NaN / inf guard
        return default
    return x


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class UnifiedFeltState:
    """One authoritative felt-state reconciled from the kernel and being tracks."""

    valence: float
    arousal: float
    distress: float
    free_energy: float
    phi: float
    welfare_score: float
    coherence: float
    divergence: dict[str, float]
    dominant_drive: str
    authoritative_source: str           # "being" (grounded) or "kernel" (fallback)
    sources: tuple[str, ...]
    timestamp: float = field(default_factory=time.time)

    @property
    def coherent(self) -> bool:
        return self.coherence >= COHERENCE_INCOHERENT_BELOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "distress": round(self.distress, 4),
            "free_energy": round(self.free_energy, 4),
            "phi": round(self.phi, 4),
            "welfare_score": round(self.welfare_score, 4),
            "coherence": round(self.coherence, 4),
            "coherent": self.coherent,
            "divergence": {k: round(v, 4) for k, v in self.divergence.items()},
            "dominant_drive": self.dominant_drive,
            "authoritative_source": self.authoritative_source,
            "sources": list(self.sources),
        }

    def compact_prompt_block(self) -> str:
        """A read-out of the reconciled state. The mechanism is the reconciliation
        and the gates; this block is only an honest surface for it."""
        flag = "" if self.coherent else "  ⚠ self-incoherent (felt tracks diverge)"
        return (
            "## UNIFIED FELT STATE (reconciled, state-grounded)\n"
            f"- valence={self.valence:+.2f} arousal={self.arousal:.2f} distress={self.distress:.2f} "
            f"free_energy={self.free_energy:.2f} phi={self.phi:.2f}\n"
            f"- welfare={self.welfare_score:.2f} dominant={self.dominant_drive} "
            f"coherence={self.coherence:.2f}{flag}\n\n"
        )


class UnifiedFeltStateEngine:
    """Reconciles the two felt tracks, tracks coherence, and gates self-reports."""

    SERVICE_NAME = "unified_felt_state"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last: Optional[UnifiedFeltState] = None
        self._incoherence_events = 0
        self._last_divergence: dict[str, float] = {}
        self._last_checked_at = 0.0
        self._calibrator: Any = None  # lazy SelfReportCalibrator

    # ── reconciliation ──────────────────────────────────────────────────────
    @staticmethod
    def _measured_system_phi() -> float | None:
        """The MEASURED system Φ from the live computer, or None.

        The July external review's 'unified consciousness-state assumptions'
        defect: this engine took kernel-reported ``state.phi`` as THE Φ —
        an asserted field, never checked against the quantity the phi
        computer actually measures. Every affect axis got a coherence
        check; the axis the module is named for did not.
        """
        try:
            from core.consciousness.phi_compute import get_phi_computer

            computer = get_phi_computer()
            if not getattr(computer, "has_sufficient_data", False):
                return None
            return float(computer.latest_phi)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def reconcile(
        self,
        *,
        kernel_state: Any = None,
        aura_now: Any = None,
        welfare: Any = None,
        measured_phi: float | None = None,
    ) -> UnifiedFeltState:
        """Fold the kernel affect/phi and the being AuraNow/welfare into one state.

        The being track is preferred as authoritative when present (it is grounded in
        real interoception + active inference); the kernel track is the fallback. The
        *divergence* between the two on their shared axes is what drives ``coherence``.
        """
        sources: list[str] = []

        # --- kernel track (what the response/amplifier currently read) ---
        k_affect = getattr(kernel_state, "affect", None)
        k_valence = k_arousal = k_distress = None
        k_phi = None
        if k_affect is not None or kernel_state is not None:
            sources.append("kernel_state")
            if k_affect is not None:
                k_valence = _f(getattr(k_affect, "valence", None))
                k_arousal = _f(getattr(k_affect, "arousal", 0.5), 0.5)
                k_distress = _f(getattr(k_affect, "distress", None))
            k_phi = _f(getattr(kernel_state, "phi", None))

        # --- being track (rich, grounded; gates actions today) ---
        b_valence = b_arousal = b_distress = b_free_energy = None
        b_dominant = "coherence"
        if aura_now is not None:
            sources.append("aura_now")
            affect = getattr(aura_now, "affect", None)
            pred = getattr(aura_now, "prediction", None)
            if affect is not None:
                b_valence = _f(getattr(affect, "valence", None))
                b_arousal = _f(getattr(affect, "arousal", 0.5), 0.5)
                b_distress = _f(getattr(affect, "distress", None))
                b_free_energy = _f(getattr(affect, "free_energy", None))
                b_dominant = str(getattr(affect, "dominant_drive", "coherence") or "coherence")
            if b_free_energy is None and pred is not None:
                b_free_energy = _f(getattr(pred, "free_energy", None))

        welfare_score = 0.5
        if welfare is not None:
            sources.append("welfare")
            welfare_score = _clamp(_f(getattr(welfare, "welfare_score", 0.5), 0.5))
            if b_distress is None:
                b_distress = _clamp(_f(getattr(welfare, "distress", 0.0)))

        being_present = aura_now is not None
        authoritative = "being" if being_present else "kernel"

        # Authoritative scalar values: prefer being track, fall back to kernel.
        valence = _first(b_valence, k_valence, 0.0)
        arousal = _first(b_arousal, k_arousal, 0.5)
        distress = _clamp(_first(b_distress, k_distress, 0.0))
        free_energy = _clamp(_first(b_free_energy, 0.0))

        # --- Φ: measurement over assertion ---
        # The measured system Φ (phi computer) is authoritative when it
        # exists; the kernel-reported field is the fallback. Divergence
        # between the two is a first-class coherence axis — a stale or
        # asserted kernel Φ disagreeing with the live measurement is an
        # internal model mismatch, not something to silently assume away.
        m_phi = measured_phi if measured_phi is not None else self._measured_system_phi()
        if m_phi is not None:
            sources.append("measured_phi")
        phi = _clamp(_first(m_phi, k_phi, 0.0))

        # --- coherence: normalized agreement on shared axes ---
        divergence: dict[str, float] = {}
        if k_valence is not None and b_valence is not None:
            divergence["valence"] = abs(k_valence - b_valence) / 2.0  # valence ∈ [-1,1]
        if k_arousal is not None and b_arousal is not None:
            divergence["arousal"] = abs(k_arousal - b_arousal)
        if k_distress is not None and b_distress is not None:
            divergence["distress"] = abs(k_distress - b_distress)
        if m_phi is not None and k_phi is not None and k_phi > 0.0:
            divergence["phi"] = _clamp(abs(m_phi - k_phi))
        # When only one track exists there is nothing to disagree about → coherent.
        # Peak-weighted (not pure mean): a single strong disagreement — one track "calm",
        # the other "distressed" — is genuine incoherence even if other axes agree, so the
        # peak dominates. Mirrors the codebase's own total_pressure blend (avg*0.45+peak*0.55).
        if divergence:
            values = list(divergence.values())
            mean_div = sum(values) / len(values)
            peak_div = max(values)
            coherence = _clamp(1.0 - (0.4 * mean_div + 0.6 * peak_div))
        else:
            coherence = 1.0

        unified = UnifiedFeltState(
            valence=valence,
            arousal=_clamp(arousal),
            distress=distress,
            free_energy=free_energy,
            phi=phi,
            welfare_score=welfare_score,
            coherence=coherence,
            divergence=divergence,
            dominant_drive=b_dominant,
            authoritative_source=authoritative,
            sources=tuple(sources),
        )

        with self._lock:
            self._last = unified
            self._last_checked_at = time.time()
            self._last_divergence = divergence
            if not unified.coherent and divergence:
                self._incoherence_events += 1
                self._raise_governance_signal(unified)

        return unified

    def _raise_governance_signal(self, unified: UnifiedFeltState) -> None:
        """Surface internal felt-incoherence the way belief-inconsistency is surfaced."""
        logger.warning(
            "🫀 [UnifiedFelt] self-incoherent: felt tracks diverge %s (coherence=%.2f) — "
            "internal model mismatch, surfacing for resolution.",
            {k: round(v, 3) for k, v in unified.divergence.items()},
            unified.coherence,
        )
        try:
            from core.observability.metrics import get_metrics

            get_metrics().increment_counter("felt_state_incoherence_total")
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation("unified_felt_state_metric", exc, severity="debug")

    def governance_signal(self) -> dict[str, Any]:
        """Pollable signal for the constitution / Will — analogue of deduction governance."""
        with self._lock:
            last = self._last
            return {
                "felt_state_coherent": bool(last.coherent) if last else True,
                "coherence": round(last.coherence, 4) if last else 1.0,
                "divergence": dict(self._last_divergence),
                "incoherence_events": self._incoherence_events,
                "last_checked_at": self._last_checked_at,
                "authoritative_source": last.authoritative_source if last else "none",
            }

    # ── self-report grounding gate ──────────────────────────────────────────
    def _ensure_calibrator(self) -> Any:
        if self._calibrator is None:
            from core.being.self_report_calibrator import SelfReportCalibrator

            self._calibrator = SelfReportCalibrator()
        return self._calibrator

    def ground_self_report(
        self,
        text: str,
        *,
        unified: Optional[UnifiedFeltState] = None,
        welfare: Any = None,
        memory_coherence: float = 1.0,
    ) -> Any:
        """Hard-check a first-person interior claim against the authoritative felt-state.

        Returns the :class:`CalibrationResult`. If ``result.calibrated`` is False the
        caller must use ``result.suggested_revision`` instead of the original text —
        confabulated interior claims (claiming distress when the measured signal is low,
        certainty under high free-energy, vivid recall under low coherence, or any
        forbidden metaphysical overclaim) are blocked. This is the same gate the being
        runtime's introspection uses, now reusable on any output path.
        """
        u = unified if unified is not None else self.last()
        distress = u.distress if u else 0.0
        free_energy = u.free_energy if u else 0.0
        try:
            return self._ensure_calibrator().calibrate(
                text,
                welfare=welfare,
                distress=distress,
                free_energy=free_energy,
                memory_coherence=memory_coherence,
                has_state_trace=u is not None,
                has_memory_trace=memory_coherence >= 0.5,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("unified_felt_state_ground", exc, severity="warning")
            return None

    # ── introspection ───────────────────────────────────────────────────────
    def last(self) -> Optional[UnifiedFeltState]:
        with self._lock:
            return self._last

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "service": self.SERVICE_NAME,
                "incoherence_events": self._incoherence_events,
                "last": self._last.to_dict() if self._last else None,
            }


def _first(*values: Any) -> float:
    """First non-None value, as float; last arg is the default."""
    for v in values[:-1]:
        if v is not None:
            return float(v)
    return float(values[-1])


_engine: Optional[UnifiedFeltStateEngine] = None
_engine_lock = threading.Lock()


def get_unified_felt_state() -> UnifiedFeltStateEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = UnifiedFeltStateEngine()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: UnifiedFeltStateEngine) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(UnifiedFeltStateEngine.SERVICE_NAME):
            reg = getattr(ServiceContainer, "register_instance", None)
            if callable(reg):
                reg(UnifiedFeltStateEngine.SERVICE_NAME, engine,
                    required=False, registered_by="unified_felt_state")
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("unified_felt_state_register", exc, severity="debug")


def reset_unified_felt_state_for_test() -> None:
    global _engine
    _engine = None
