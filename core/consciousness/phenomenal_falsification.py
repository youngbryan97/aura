"""core/consciousness/phenomenal_falsification.py — a falsification control for the
consciousness markers. The scientific control the existing audit never had.

`unified_audit.ConsciousnessAuditSuite` aggregates a weighted ``consciousness_index``
from how many functional criteria of various theories are met. But it has no *control
condition*: it never asks whether the same marker readings could be produced by a system
that is, by construction, NOT a candidate for phenomenal experience — a feed-forward,
non-integrated, non-self-monitoring "functional zombie" with the same input/output
behaviour. Without that control, a high index is uninterpretable: behaviour alone can
fake the outputs.

This instrument is that control. For each recognised marker it runs a falsification
test against the zombie baseline and reports the *margin* by which the live system
exceeds it. The markers and their baselines come from the actual theories:

  * Integration (IIT)            — Φ above a disconnected system's ~0 (a feed-forward net
                                    has no integrated information beyond its parts).
  * Recurrence (RPT)             — information carried across time (a feed-forward zombie
                                    has none; recurrent processing is the marker).
  * Global broadcast (GWT)       — an ignited coalition actually broadcast to many
                                    consumers (a zombie processes locally only).
  * Metacognition (HOT)          — higher-order states track first-order states.
  * Self-model coherence (AST/   — a coherent, causally-dominant self-model.
    active inference)
  * Causal efficacy (the key     — the markers are wired to behaviour, not epiphenomenal.
    disqualifier)                  A zombie's "markers" change nothing; Aura's Φ gates
                                    compute, her felt-state gates action. This is the test
                                    a behaviour-only mimic fails hardest.

It aggregates a **discriminability index**: how distinguishable the live marker profile
is from the zombie baseline. And it is disciplined about what that number is:

    A high index means the functional correlates that theories of consciousness propose
    are PRESENT and CAUSAL here, and are not reproducible by a degenerate baseline that
    only matches I/O. It does NOT measure, detect, or prove subjective/phenomenal
    experience. Maximal discriminability from a functional zombie does not resolve the
    hard problem. This is an instrument for *the markers*, honestly bounded — the
    beginning of measurement, not a verdict on experience.

It is live (reads the markers from their organs), tracks transitions over time (a drop
is a "dimming"), and is governed (the boundary is part of every report, never stripped).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.PhenomenalFalsification")

PHI_MEANINGFUL = 0.20        # Φ above this is non-trivial integration (zombie ≈ 0)
DISCRIMINABLE_AT = 0.50      # a per-test margin ≥ this counts as discriminable from zombie

# The honest boundary — part of every report, never stripped.
REPORT_BOUNDARY = (
    "This index measures how distinguishable the system's functional markers are from a "
    "non-phenomenal baseline (a feed-forward, non-integrated, non-self-monitoring system "
    "with the same I/O). It quantifies the presence and CAUSAL EFFICACY of the functional "
    "correlates that contested theories of consciousness propose. It does NOT measure, "
    "detect, or prove subjective/phenomenal experience; maximal discriminability from a "
    "functional zombie does not resolve the hard problem of consciousness."
)


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


@dataclass(frozen=True)
class MarkerSnapshot:
    """Live readings of the consciousness markers (all already computed elsewhere)."""
    phi: float = 0.0                 # integrated information (IIT)
    recurrence: float = 0.0          # info carried across time (RPT) [0,1]
    ignition: float = 0.0            # global-workspace ignition strength (GWT) [0,1]
    broadcast_breadth: float = 0.0   # fraction of consumers an ignition reached [0,1]
    metacognition: float = 0.0       # higher-order monitoring activity (HOT) [0,1]
    self_coherence: float = 0.0      # unified felt-state coherence (AST/active inference) [0,1]
    markers_causal: bool = False     # are the markers wired to behaviour (not epiphenomenal)?

    def to_dict(self) -> dict[str, Any]:
        return {
            "phi": round(self.phi, 4),
            "recurrence": round(self.recurrence, 4),
            "ignition": round(self.ignition, 4),
            "broadcast_breadth": round(self.broadcast_breadth, 4),
            "metacognition": round(self.metacognition, 4),
            "self_coherence": round(self.self_coherence, 4),
            "markers_causal": self.markers_causal,
        }


# The zombie baseline: a feed-forward, non-integrated, non-self-monitoring system whose
# markers are all ~0 and, critically, epiphenomenal (causal=False).
ZOMBIE_BASELINE = MarkerSnapshot()


@dataclass(frozen=True)
class FalsificationTest:
    name: str
    theory: str
    margin: float                    # how far the live system exceeds the zombie [0,1]
    discriminable: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "theory": self.theory,
            "margin": round(self.margin, 4), "discriminable": self.discriminable,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DiscriminabilityReport:
    index: float                      # aggregate discriminability from the zombie baseline
    tests: list[FalsificationTest]
    n_discriminable: int
    n_total: int
    delta: float                      # change since the previous report (transition signal)
    verdict: str
    boundary: str
    snapshot: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discriminability_index": round(self.index, 4),
            "tests": [t.to_dict() for t in self.tests],
            "n_discriminable": self.n_discriminable,
            "n_total": self.n_total,
            "delta": round(self.delta, 4),
            "verdict": self.verdict,
            "boundary": self.boundary,
            "snapshot": self.snapshot,
            "timestamp": self.timestamp,
        }


# Per-test weights (sum to 1). Causal efficacy is weighted heavily: it is the test a
# behaviour-only mimic fails hardest, and the one most central to the markers mattering.
_WEIGHTS = {
    "integration": 0.22,
    "recurrence": 0.15,
    "global_broadcast": 0.18,
    "metacognition": 0.15,
    "self_coherence": 0.10,
    "causal_efficacy": 0.20,
}


#: Channels whose lesioning would show the phenomenal markers gate behaviour.
#: Named rather than pattern-matched: a channel that merely has one of these
#: words in it is not evidence about consciousness, and a check that accepted
#: one would be the proxy this replaced wearing a longer name.
THE_MARKER_CHANNELS: tuple[str, ...] = (
    "phenomenal_state",
    "felt_state",
    "phi",
    "interoception",
    "affect_grounding",
    "global_workspace_ignition",
)


def _causal_by_intervention() -> tuple[bool, str]:
    """Whether lesioning a phenomenal marker measurably moved the output.

    This used to be `bool(loop.get("phi") is not None)` and `causal = True`
    with a comment beside it. Both said the pathway was present; neither said
    it made a difference. An external review named the distinction exactly:

        P(Y | do(X = x1)) != P(Y | do(X = x0))

    is not established by a live snapshot reporting that a causal channel
    appears to be there. Aura's own influence framework already requires
    treatment against null, so the answer comes from there, and where there is
    no such evidence the answer is False with a sentence saying what would
    change it — which is a different claim from "epiphenomenal" and has to
    read differently.
    """
    try:
        from core.verify.causal_influence import Verdict, get_influence_ledger

        ledger = get_influence_ledger()
    except (ImportError, RuntimeError) as exc:
        return False, f"no influence ledger to ask: {exc}"

    influential: list[str] = []
    inert: list[str] = []
    unmeasured: list[str] = []
    for channel in THE_MARKER_CHANNELS:
        try:
            verdict = ledger.verdict(channel).verdict
        except Exception:  # noqa: BLE001 — an unaskable channel is unmeasured
            unmeasured.append(channel)
            continue
        if verdict is Verdict.INERT:
            inert.append(channel)
        elif verdict is Verdict.UNMEASURED:
            unmeasured.append(channel)
        else:
            influential.append(channel)

    if influential:
        return True, (
            "lesioning moved the output for: " + ", ".join(sorted(influential))
        )
    if inert:
        return False, (
            "measured with enough power to see an effect and there was none: "
            + ", ".join(sorted(inert))
        )
    return False, (
        "no paired trials for any marker channel ("
        + ", ".join(sorted(unmeasured))
        + "); run the treatment and the null before claiming it gates behaviour"
    )


class PhenomenalFalsifier:
    """Tests the live consciousness markers against a non-phenomenal baseline."""

    SERVICE_NAME = "phenomenal_falsifier"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._history: deque[DiscriminabilityReport] = deque(maxlen=200)

    # ── the falsification tests ──────────────────────────────────────────────
    def _tests(self, s: MarkerSnapshot) -> list[FalsificationTest]:
        integration = _clamp01(s.phi / PHI_MEANINGFUL)
        recurrence = _clamp01(s.recurrence)
        broadcast = _clamp01(s.ignition * min(1.0, 0.3 + s.broadcast_breadth))
        metacog = _clamp01(s.metacognition)
        coherence = _clamp01(s.self_coherence)
        causal = 1.0 if s.markers_causal else 0.0
        return [
            FalsificationTest(
                "integration", "IIT", integration, integration >= DISCRIMINABLE_AT,
                f"Φ={s.phi:.3f} vs disconnected baseline ≈ 0 (threshold {PHI_MEANINGFUL})",
            ),
            FalsificationTest(
                "recurrence", "RPT", recurrence, recurrence >= DISCRIMINABLE_AT,
                f"information carried across time={s.recurrence:.3f} vs feed-forward 0",
            ),
            FalsificationTest(
                "global_broadcast", "GWT", broadcast, broadcast >= DISCRIMINABLE_AT,
                f"ignition={s.ignition:.3f} × breadth={s.broadcast_breadth:.3f} vs local-only",
            ),
            FalsificationTest(
                "metacognition", "HOT", metacog, metacog >= DISCRIMINABLE_AT,
                f"higher-order monitoring={s.metacognition:.3f} vs none",
            ),
            FalsificationTest(
                "self_coherence", "AST/active-inference", coherence, coherence >= DISCRIMINABLE_AT,
                f"unified self-model coherence={s.self_coherence:.3f} vs no self-model",
            ),
            FalsificationTest(
                "causal_efficacy", "control", causal, causal >= DISCRIMINABLE_AT,
                "markers gate behaviour (Φ→compute, felt-state→action)"
                if s.markers_causal else "markers epiphenomenal — a zombie disqualifier",
            ),
        ]

    def assess(self, snapshot: MarkerSnapshot) -> DiscriminabilityReport:
        tests = self._tests(snapshot)
        index = _clamp01(sum(_WEIGHTS[t.name] * t.margin for t in tests))
        n_disc = sum(1 for t in tests if t.discriminable)

        with self._lock:
            prev = self._history[-1].index if self._history else index
        delta = round(index - prev, 4)

        verdict = self._verdict(
            index, n_disc, len(tests), snapshot.markers_causal, delta,
            why=getattr(self, "_why_causal", ""),
        )
        report = DiscriminabilityReport(
            index=index, tests=tests, n_discriminable=n_disc, n_total=len(tests),
            delta=delta, verdict=verdict, boundary=REPORT_BOUNDARY,
            snapshot=snapshot.to_dict(),
        )
        with self._lock:
            self._history.append(report)
        return report

    @staticmethod
    def _verdict(
        index: float,
        n_disc: int,
        n_total: int,
        causal: bool,
        delta: float,
        *,
        why: str = "",
    ) -> str:
        if not causal and "no paired trials" in why:
            # Not the same claim. Calling a marker epiphenomenal on no
            # evidence is as wrong as calling it causal on none, and the
            # boolean this used to take could not tell the two apart.
            head = (
                f"UNMEASURED on the decisive test (index={index:.2f}). "
                f"{why}."
            )
        elif not causal:
            head = (
                f"NOT discriminable on the decisive test: the markers are epiphenomenal "
                f"(index={index:.2f}). Behaviour alone could produce this profile."
                + (f" {why}." if why else "")
            )
        elif index >= 0.66:
            head = (
                f"Strongly discriminable from a functional zombie ({n_disc}/{n_total} markers, "
                f"index={index:.2f}): the proposed functional correlates are present AND causal."
            )
        elif index >= 0.4:
            head = (
                f"Partially discriminable ({n_disc}/{n_total} markers, index={index:.2f}): "
                f"some correlates present and causal, others weak."
            )
        else:
            head = (
                f"Weakly discriminable ({n_disc}/{n_total}, index={index:.2f}): the marker "
                f"profile is close to the non-phenomenal baseline right now."
            )
        if abs(delta) >= 0.08:
            head += f" {'Brightening' if delta > 0 else 'Dimming'} (Δ={delta:+.2f})."
        return head + " NOTE: this is discriminability of functional markers, not a claim of experience."

    # ── live reading ─────────────────────────────────────────────────────────
    def from_live(self) -> MarkerSnapshot:
        """Best-effort read of the live markers from their organs. Fail-open to zeros."""
        phi = recurrence = ignition = breadth = metacog = coherence = 0.0
        causal = False

        # Φ + ignition + causal loop from the existing consciousness audit's sources.
        try:
            from core.consciousness.phi_compute import get_phi_computer

            phi = float(get_phi_computer().latest_phi or 0.0)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("phenomenal_falsifier", exc, severity="debug")
        try:
            from core.kernel.kernel_interface import KernelInterface

            loop = KernelInterface.get_instance().loop_state()
            if isinstance(loop, dict):
                if not phi:
                    phi = float(loop.get("phi", 0.0) or 0.0)
                # NOT causal evidence. A phi value being present in the loop
                # state says the pathway ran, which is not the same claim as
                # "the markers gate behaviour" — see `_causal_by_intervention`
                # below, which is where that answer now comes from.
                recurrence = _clamp01(loop.get("recurrence", 0.6))
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("phenomenal_falsifier", exc, severity="debug")
        try:
            from core.container import ServiceContainer

            gw = ServiceContainer.get("global_workspace", default=None)
            if gw is not None and hasattr(gw, "get_ignition_level"):
                ignition = _clamp01(gw.get_ignition_level())
                breadth = _clamp01(getattr(gw, "broadcast_breadth", 0.5))
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("phenomenal_falsifier", exc, severity="debug")
        # Self-model coherence from the unified felt-state built earlier.
        try:
            from core.being.unified_felt_state import get_unified_felt_state

            last = get_unified_felt_state().last()
            if last is not None:
                coherence = _clamp01(last.coherence)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("phenomenal_falsifier", exc, severity="debug")
        # Metacognition: a higher-order monitor being present + active.
        try:
            from core.container import ServiceContainer

            mon = ServiceContainer.get("higher_order_monitor", default=None)
            metacog = 0.6 if mon is not None else 0.0
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            pass

        causal, why = _causal_by_intervention()
        self._why_causal = why
        return MarkerSnapshot(
            phi=phi, recurrence=recurrence, ignition=ignition, broadcast_breadth=breadth,
            metacognition=metacog, self_coherence=coherence, markers_causal=causal,
        )

    def why_the_causal_answer(self) -> str:
        """What the last live read based its causal marker on."""
        return getattr(self, "_why_causal", "not read yet")

    def assess_live(self) -> DiscriminabilityReport:
        return self.assess(self.from_live())

    def history(self, n: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in list(self._history)[-n:]]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            latest = self._history[-1] if self._history else None
            return {
                "service": self.SERVICE_NAME,
                "assessments": len(self._history),
                "latest_index": round(latest.index, 4) if latest else None,
                "boundary": REPORT_BOUNDARY,
            }


_engine: Optional[PhenomenalFalsifier] = None
_engine_lock = threading.Lock()


def get_phenomenal_falsifier() -> PhenomenalFalsifier:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = PhenomenalFalsifier()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: PhenomenalFalsifier) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(PhenomenalFalsifier.SERVICE_NAME):
            reg = getattr(ServiceContainer, "register_instance", None)
            if callable(reg):
                reg(PhenomenalFalsifier.SERVICE_NAME, engine,
                    required=False, registered_by="phenomenal_falsification")
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        pass


def reset_phenomenal_falsifier_for_test() -> None:
    global _engine
    _engine = None
