"""A model of Aura's own faculties, and of what "better" means for each.

Aura could already improve. What she could not do was decide *what* to
improve. ``RecursiveSelfImprovementLoop.record_signal(...)`` waits to be told
where the problem is, and ``core/consciousness/metacognition.py`` judges one
episode of reasoning at a time. Between them sits the thing that was missing:
a standing model of the cognitive stack — which faculties exist, what a better
version of each would look like, how far each currently is from that, and
which one is actually holding the rest back.

Four ideas carry the design.

**Improvement is a declared contract, not an opinion.** A faculty declares
metrics, and a metric declares its unit, its direction, the floor it must not
fall below, the target that counts as good enough, and the ceiling that
represents its potential. "Better memory" stops being a sentiment and becomes
recall@k against a stated ceiling. Nothing here can improve a faculty that has
not said what improvement means.

**Unmeasured is never "fine".** A probe that cannot run returns ``None``, and
that reading is ``measured=False`` with a reason. Unmeasured readings are
excluded from scores rather than defaulted, because a faculty that silently
scores well because nothing measured it is precisely the failure this codebase
keeps finding. A faculty with no measurable metric at all is not healthy — it
is *invisible*, and :meth:`CognitiveSelfModel.blind_spots` reports it as a gap
in self-knowledge, which is itself a first-class improvement target.

**Holistic means leverage, not just deficit.** The faculty with the worst
score is not automatically the one to fix. Faculties gate one another —
degraded memory caps what attention can do, which caps temporal reasoning —
so priority is headroom weighted by how much of the rest of the stack a
faculty limits, computed over the transitive closure of the ``gates`` graph.
That is what makes this a model of the stack rather than a list of numbers.

**It closes the loop causally.** :func:`emit_improvement_signals` pushes the
binding constraint into the existing RSI loop as a signal it can plan against,
so the improvement machinery keeps its single owner and simply stops being
blind. Aura generates her own targets; she does not wait to be handed one.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock

Direction = Literal["higher_is_better", "lower_is_better"]

#: A probe returns the current value of a metric, or None when it genuinely
#: cannot be measured right now. None is a first-class answer, never 0.0.
Probe = Callable[[], float | None]

_RECOVERABLE = (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)

#: Bound on the gates-graph walk, so a cyclic declaration cannot spin.
MAX_GATE_DEPTH = 8

#: Re-entrancy guard. Probes are arbitrary callables reading live subsystems,
#: and a subsystem may itself consult the self-model — directly, or through
#: any number of intermediate calls. Without this, one such probe turns an
#: assessment into unbounded recursion.
#:
#: The guard is thread-local rather than global so concurrent assessments on
#: different threads stay independent; what it forbids is an assessment
#: re-entering ITSELF on the same thread.
_assessing = threading.local()

#: The same faculty must not be re-proposed on every tick. Improvement takes
#: time to show up in a metric, so a target repeated before its effect could
#: land is a livelock: the same goal chosen forever, each time believing it is
#: new. Suppression is per faculty and expires.
DEFAULT_REPROPOSE_COOLDOWN_S = 900.0


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class ImprovementMetric:
    """What "better" means for one aspect of one faculty.

    ``floor``, ``target`` and ``ceiling`` are what turn a raw number into a
    judgement. ``ceiling`` is the potential — the value beyond which further
    effort is wasted — so headroom is measured against what is achievable
    rather than against perfection.
    """

    metric_id: str
    unit: str
    direction: Direction
    probe: Probe
    floor: float
    target: float
    ceiling: float
    weight: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.direction == "higher_is_better":
            ordered = self.floor <= self.target <= self.ceiling
        else:
            ordered = self.floor >= self.target >= self.ceiling
        if not ordered:
            raise ValueError(
                f"{self.metric_id}: floor/target/ceiling are not ordered for "
                f"{self.direction} (got {self.floor}/{self.target}/{self.ceiling})"
            )
        if self.floor == self.ceiling:
            raise ValueError(f"{self.metric_id}: floor and ceiling are equal; no scale")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"{self.metric_id}: weight must be finite and non-negative")

    def normalize(self, value: float) -> float:
        """Map a raw value onto 0..1, where 1.0 means "at potential"."""
        span = self.ceiling - self.floor
        fraction = (value - self.floor) / span
        return max(0.0, min(1.0, fraction))

    def headroom(self, value: float) -> float:
        """How much of this metric's potential is still unrealised (0..1)."""
        return 1.0 - self.normalize(value)

    def meets_target(self, value: float) -> bool:
        if self.direction == "higher_is_better":
            return value >= self.target
        return value <= self.target

    def read(self) -> MetricReading:
        """Measure. Never raises — a broken probe is an unmeasured metric.

        A probe that re-enters the self-model is refused rather than followed.
        The result is an ordinary unmeasured reading, so a recursive probe
        degrades that one metric instead of hanging the assessment — and the
        reason says so, which makes the loop findable rather than silent.
        """
        # Set around the PROBE CALL, not around the assessment: a probe that
        # reaches back into the self-model lands here again with the flag
        # already set and is refused. Setting it in assess() instead would
        # refuse every probe on the first pass.
        if getattr(_assessing, "active", False):
            return MetricReading(
                metric_id=self.metric_id,
                value=None,
                measured=False,
                reason="probe re-entered the self-model; refused to recurse",
            )
        _assessing.active = True
        try:
            raw = self.probe()
        except _RECOVERABLE as exc:
            return MetricReading(
                metric_id=self.metric_id,
                value=None,
                measured=False,
                reason=f"probe raised {type(exc).__name__}: {exc}"[:200],
            )
        finally:
            # Always cleared, including on the exception path above — a probe
            # that raises must not leave the guard latched and silently turn
            # every later metric into "re-entered".
            _assessing.active = False
        if raw is None:
            return MetricReading(
                metric_id=self.metric_id,
                value=None,
                measured=False,
                reason="probe reported the metric is not currently measurable",
            )
        value = _finite(raw)
        if value is None:
            return MetricReading(
                metric_id=self.metric_id,
                value=None,
                measured=False,
                reason=f"probe returned a non-finite value ({raw!r})",
            )
        return MetricReading(
            metric_id=self.metric_id,
            value=value,
            measured=True,
            reason="",
            normalized=self.normalize(value),
            headroom=self.headroom(value),
            meets_target=self.meets_target(value),
            weight=self.weight,
            unit=self.unit,
        )


@dataclass(frozen=True)
class MetricReading:
    """One measurement, carrying whether it is a measurement at all."""

    metric_id: str
    value: float | None
    measured: bool
    reason: str = ""
    normalized: float | None = None
    headroom: float | None = None
    meets_target: bool | None = None
    weight: float = 1.0
    unit: str = ""
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "value": self.value,
            "measured": self.measured,
            "reason": self.reason,
            "normalized": self.normalized,
            "headroom": self.headroom,
            "meets_target": self.meets_target,
            "unit": self.unit,
            "at": self.at,
        }


@dataclass(frozen=True)
class Faculty:
    """One capacity of the cognitive stack.

    ``gates`` names the faculties this one limits. Memory gating attention is
    the claim that however well attention is allocated, poor recall caps what
    it can be allocated over. That graph is what makes the assessment holistic
    instead of a ranked list of complaints.
    """

    faculty_id: str
    description: str
    owner: str
    metrics: tuple[ImprovementMetric, ...] = ()
    gates: tuple[str, ...] = ()

    def assess(self) -> FacultyAssessment:
        readings = tuple(metric.read() for metric in self.metrics)
        measured = [r for r in readings if r.measured]
        total_weight = sum(r.weight for r in measured)
        if measured and total_weight > 0:
            headroom = sum(
                (r.headroom or 0.0) * r.weight for r in measured
            ) / total_weight
            attainment = sum(
                (r.normalized or 0.0) * r.weight for r in measured
            ) / total_weight
        else:
            headroom = None
            attainment = None
        return FacultyAssessment(
            faculty_id=self.faculty_id,
            readings=readings,
            headroom=headroom,
            attainment=attainment,
            measured_metrics=len(measured),
            declared_metrics=len(readings),
        )


@dataclass(frozen=True)
class FacultyAssessment:
    """What is currently known about one faculty — including "nothing"."""

    faculty_id: str
    readings: tuple[MetricReading, ...]
    headroom: float | None
    attainment: float | None
    measured_metrics: int
    declared_metrics: int
    leverage: float = 1.0
    priority: float | None = None

    @property
    def measurable(self) -> bool:
        """Whether anything at all could be measured about this faculty."""
        return self.measured_metrics > 0

    @property
    def coverage(self) -> float:
        """Fraction of declared metrics that actually produced a reading."""
        if not self.declared_metrics:
            return 0.0
        return self.measured_metrics / self.declared_metrics

    def unmet(self) -> tuple[MetricReading, ...]:
        return tuple(r for r in self.readings if r.measured and r.meets_target is False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "faculty_id": self.faculty_id,
            "measurable": self.measurable,
            "coverage": round(self.coverage, 4),
            "headroom": self.headroom,
            "attainment": self.attainment,
            "leverage": round(self.leverage, 4),
            "priority": self.priority,
            "measured_metrics": self.measured_metrics,
            "declared_metrics": self.declared_metrics,
            "unmet_metrics": [r.metric_id for r in self.unmet()],
            "readings": [r.as_dict() for r in self.readings],
        }


@dataclass(frozen=True)
class CognitiveSelfModel:
    """A snapshot of what Aura currently knows about her own faculties."""

    assessments: tuple[FacultyAssessment, ...]
    binding_constraint: str | None
    at: float = field(default_factory=time.time)

    def by_id(self, faculty_id: str) -> FacultyAssessment | None:
        for assessment in self.assessments:
            if assessment.faculty_id == faculty_id:
                return assessment
        return None

    def blind_spots(self) -> tuple[str, ...]:
        """Faculties nothing could measure.

        Not a health report — a report on the limits of the health report. A
        faculty here is one Aura cannot currently form any opinion about, and
        that is an improvement target in its own right: the fix is to build
        the probe, not to tune the faculty.
        """
        return tuple(a.faculty_id for a in self.assessments if not a.measurable)

    def self_knowledge_coverage(self) -> float:
        """Fraction of all declared metrics that produced a reading."""
        declared = sum(a.declared_metrics for a in self.assessments)
        measured = sum(a.measured_metrics for a in self.assessments)
        return (measured / declared) if declared else 0.0

    def ranked(self) -> tuple[FacultyAssessment, ...]:
        return tuple(
            sorted(
                (a for a in self.assessments if a.priority is not None),
                key=lambda a: a.priority or 0.0,
                reverse=True,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.cognitive_self_model.v1",
            "at": self.at,
            "binding_constraint": self.binding_constraint,
            "self_knowledge_coverage": round(self.self_knowledge_coverage(), 4),
            "blind_spots": list(self.blind_spots()),
            "faculties": [a.as_dict() for a in self.assessments],
        }


class FacultyRegistry:
    """The declared faculties and their improvement contracts."""

    def __init__(self) -> None:
        self._lock = checked_lock("metacognition.faculty_registry", rank=LockRank.REGISTRY, reentrant=True)
        self._faculties: dict[str, Faculty] = {}

    def declare(self, faculty: Faculty) -> Faculty:
        if not isinstance(faculty, Faculty) or not faculty.faculty_id:
            raise ValueError("a faculty needs an id")
        metric_ids = [m.metric_id for m in faculty.metrics]
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError(f"{faculty.faculty_id}: duplicate metric ids")
        with self._lock:
            self._faculties[faculty.faculty_id] = faculty
        return faculty

    def get(self, faculty_id: str) -> Faculty | None:
        with self._lock:
            return self._faculties.get(faculty_id)

    def all(self) -> tuple[Faculty, ...]:
        with self._lock:
            return tuple(self._faculties.values())

    def clear(self) -> None:
        with self._lock:
            self._faculties.clear()

    # -- leverage ---------------------------------------------------------
    def leverage(self, faculty_id: str) -> float:
        """How much of the stack this faculty limits.

        1.0 plus the count of faculties reachable through ``gates``. Transitive
        because the constraint compounds: if memory gates attention and
        attention gates temporal reasoning, memory limits both. Bounded and
        cycle-safe, since a declaration is data and may be wrong.
        """
        seen: set[str] = set()
        frontier = [(faculty_id, 0)]
        while frontier:
            current, depth = frontier.pop()
            if depth >= MAX_GATE_DEPTH:
                continue
            faculty = self.get(current)
            if faculty is None:
                continue
            for gated in faculty.gates:
                if gated in seen or gated == faculty_id:
                    continue
                seen.add(gated)
                frontier.append((gated, depth + 1))
        return 1.0 + float(len(seen))

    # -- assessment -------------------------------------------------------
    def assess(self) -> CognitiveSelfModel:
        """Measure every faculty and rank them by what would help most."""
        assessments: list[FacultyAssessment] = []
        for faculty in self.all():
            base = faculty.assess()
            leverage = self.leverage(faculty.faculty_id)
            # Priority exists only where there is evidence. An unmeasured
            # faculty gets None, not zero — zero would sort it last and read
            # as "nothing to do here", which is the opposite of the truth.
            if base.headroom is None:
                priority = None
            else:
                priority = base.headroom * leverage * base.coverage
            assessments.append(
                FacultyAssessment(
                    faculty_id=base.faculty_id,
                    readings=base.readings,
                    headroom=base.headroom,
                    attainment=base.attainment,
                    measured_metrics=base.measured_metrics,
                    declared_metrics=base.declared_metrics,
                    leverage=leverage,
                    priority=priority,
                )
            )
        ranked = sorted(
            (a for a in assessments if a.priority is not None),
            key=lambda a: a.priority or 0.0,
            reverse=True,
        )
        binding = ranked[0].faculty_id if ranked and (ranked[0].priority or 0) > 0 else None
        return CognitiveSelfModel(
            assessments=tuple(assessments), binding_constraint=binding
        )


_registry_lock = checked_lock("metacognition.faculty_registry_singleton", rank=LockRank.REGISTRY, reentrant=True)
_registry: FacultyRegistry | None = None


def get_faculty_registry() -> FacultyRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = FacultyRegistry()
        return _registry


# ---------------------------------------------------------------------------
# Closing the loop
# ---------------------------------------------------------------------------


def emit_improvement_signals(
    loop: Any,
    model: CognitiveSelfModel | None = None,
    *,
    max_signals: int = 3,
) -> list[dict[str, Any]]:
    """Feed the self-model's conclusions into the RSI loop as signals.

    This is what makes the model causal rather than a report. The loop's
    planner already knows how to act on a signal; it simply had no way to
    discover one on its own, so every improvement had to be handed to it. Now
    the binding constraint arrives as a signal like any other, and the
    improvement machinery keeps its single owner.

    Blind spots are emitted too, as ``kind="self_knowledge_gap"``. A faculty
    nothing can measure is a real deficiency — the actionable fix is to build
    the probe — and leaving it silent is how a stack acquires areas no one
    ever looks at.
    """
    assessment_model = model if model is not None else get_faculty_registry().assess()
    emitted: list[dict[str, Any]] = []
    recorder = getattr(loop, "record_signal", None)
    if not callable(recorder):
        return emitted

    for assessment in assessment_model.ranked()[:max_signals]:
        if not assessment.headroom or assessment.headroom <= 0:
            continue
        try:
            recorder(
                source=f"faculty_model:{assessment.faculty_id}",
                kind="faculty_headroom",
                severity=max(0.0, min(1.0, assessment.headroom)),
                metric=assessment.faculty_id,
                delta=assessment.headroom,
                evidence={
                    "faculty": assessment.faculty_id,
                    "headroom": assessment.headroom,
                    "attainment": assessment.attainment,
                    "leverage": assessment.leverage,
                    "coverage": assessment.coverage,
                    "unmet_metrics": [r.metric_id for r in assessment.unmet()],
                    "binding_constraint": (
                        assessment.faculty_id == assessment_model.binding_constraint
                    ),
                },
            )
            emitted.append({"faculty": assessment.faculty_id, "kind": "faculty_headroom"})
        except _RECOVERABLE as exc:
            record_degradation(
                "faculty_model",
                exc,
                action="could not emit a faculty improvement signal",
                enforce_failure_policy=False,
            )

    for faculty_id in assessment_model.blind_spots():
        try:
            recorder(
                source=f"faculty_model:{faculty_id}",
                kind="self_knowledge_gap",
                severity=0.5,
                metric=faculty_id,
                delta=0.0,
                evidence={
                    "faculty": faculty_id,
                    "reason": "no declared metric could be measured",
                    "actionable": "build a probe for this faculty",
                },
            )
            emitted.append({"faculty": faculty_id, "kind": "self_knowledge_gap"})
        except _RECOVERABLE as exc:
            record_degradation(
                "faculty_model",
                exc,
                action="could not emit a self-knowledge-gap signal",
                enforce_failure_policy=False,
            )
    return emitted


_proposed_lock = checked_lock("metacognition.proposal_history", rank=LockRank.REGISTRY, reentrant=True)
_last_proposed: dict[str, float] = {}


def _recently_proposed(faculty_id: str, cooldown_s: float) -> bool:
    """Whether this faculty was proposed too recently to propose again.

    Improvement does not land in a metric instantly, so a target re-chosen
    before its effect could appear is a livelock: the same goal selected
    forever, each time believing it is new. This is the same shape as the
    immune lane re-issuing one remedy 247 times.
    """
    if cooldown_s <= 0:
        return False
    with _proposed_lock:
        last = _last_proposed.get(faculty_id)
    return last is not None and (time.time() - last) < cooldown_s


def _mark_proposed(faculty_id: str) -> None:
    with _proposed_lock:
        _last_proposed[faculty_id] = time.time()
        if len(_last_proposed) > 256:
            oldest = sorted(_last_proposed.items(), key=lambda kv: kv[1])[:128]
            for key, _ in oldest:
                _last_proposed.pop(key, None)


def clear_proposal_history() -> None:
    """Forget the cooldown state. For tests and deliberate re-planning."""
    with _proposed_lock:
        _last_proposed.clear()


def improvement_goal(
    model: CognitiveSelfModel | None = None,
    *,
    cooldown_s: float | None = None,
) -> dict[str, Any] | None:
    """The goal Aura would set herself, given what she knows about herself.

    Her competence drive previously produced one fixed sentence — "run a
    self-diagnosis and fix anything broken" — every time it fired, regardless
    of what was actually wrong. That is improvement as a reflex. This returns a
    goal that names the faculty, the metric, and the measured gap, so the
    deliberation that follows has REASONS and the choice can be reviewed
    against evidence rather than taken on faith.

    Returns None when there is nothing grounded to want. A drive with nothing
    to say should say nothing; inventing a target would be the same
    manufactured confidence this codebase keeps removing.
    """
    if cooldown_s is None:
        cooldown_s = DEFAULT_REPROPOSE_COOLDOWN_S
    if model is not None:
        self_model = model
    else:
        try:
            from core.metacognition.default_faculties import ensure_default_faculties

            self_model = ensure_default_faculties().assess()
        except _RECOVERABLE:
            self_model = get_faculty_registry().assess()

    ranked = [
        a for a in self_model.ranked()
        if (a.priority or 0.0) > 0.0 and not _recently_proposed(a.faculty_id, cooldown_s)
    ]
    if ranked:
        top = ranked[0]
        _mark_proposed(top.faculty_id)
        unmet = top.unmet()
        if unmet:
            worst = min(unmet, key=lambda r: r.normalized if r.normalized is not None else 1.0)
            detail = (
                f"{worst.metric_id} is at {worst.value:.3g}{worst.unit}"
                if worst.value is not None
                else worst.metric_id
            )
        else:
            detail = f"{top.measured_metrics} metric(s) below potential"
        return {
            "objective": (
                f"Improve my {top.faculty_id.replace('_', ' ')}: {detail}, with "
                f"{top.headroom:.0%} of its potential unrealised. It limits "
                f"{top.leverage:.0f} part(s) of my cognition."
            ),
            "origin": "intrinsic_competence_faculty_model",
            "complexity": 0.6,
            "faculty": top.faculty_id,
            "evidence": {
                "headroom": top.headroom,
                "attainment": top.attainment,
                "leverage": top.leverage,
                "coverage": top.coverage,
                "unmet_metrics": [r.metric_id for r in unmet],
                "binding_constraint": self_model.binding_constraint,
            },
        }

    blind = [f for f in self_model.blind_spots() if not _recently_proposed(f, cooldown_s)]
    if blind:
        _mark_proposed(blind[0])
        # Not being able to tell how good a faculty is IS the deficiency, and
        # the honest goal is to build the instrument rather than to tune
        # something unmeasured and call it better.
        return {
            "objective": (
                f"Build a way to measure my {blind[0].replace('_', ' ')} — I "
                "currently have no probe for it, so I cannot tell whether it is "
                "improving or degrading."
            ),
            "origin": "intrinsic_competence_self_knowledge",
            "complexity": 0.7,
            "faculty": blind[0],
            "evidence": {
                "reason": "no declared metric could be measured",
                "blind_spots": list(blind),
                "self_knowledge_coverage": self_model.self_knowledge_coverage(),
            },
        }
    return None


__all__ = [
    "CognitiveSelfModel",
    "Direction",
    "Faculty",
    "FacultyAssessment",
    "FacultyRegistry",
    "ImprovementMetric",
    "MetricReading",
    "Probe",
    "emit_improvement_signals",
    "clear_proposal_history",
    "improvement_goal",
    "get_faculty_registry",
]
