"""The faculties Aura actually has, and the probes that can currently see them.

This is where the model stops being a mechanism and starts being a picture of
a specific mind. Every probe here reads a real runtime surface, and every
faculty without one says so rather than being quietly omitted — an undeclared
faculty is invisible, and invisible is worse than known-unmeasured.

The honest state today: attention allocation and operational integrity read
live counters; memory reads its real recall hit rate, which is None until
something has actually been recalled — unknown, not zero. Temporal reasoning
and multi-step reasoning have no probe at all.

Those last two are not gaps in this file, they are true statements about
Aura, and the model exists to surface them as targets rather than hide them.
Wiring a convenient proxy would be worse than leaving them blind:
reasoning_solved_cache.stats() was available and deliberately not used,
because it measures cache reuse rather than whether reasoning was correct,
and it reports 0.0 for zero attempts — a fabricated score in the exact shape
this model is built to reject.
"""

from __future__ import annotations

from core.metacognition.faculty_model import (
    Faculty,
    FacultyRegistry,
    ImprovementMetric,
    get_faculty_registry,
)
from core.runtime.lockdep import LockRank, checked_lock

_declared_lock = checked_lock("metacognition.default_faculties", rank=LockRank.REGISTRY, reentrant=True)
_declared = False


# ── probes ────────────────────────────────────────────────────────────────
# Each returns a number, or None when it genuinely cannot measure. None is a
# real answer; a fabricated default would be the exact failure this campaign
# exists to remove.


def _loop_blocking_holds() -> float | None:
    """How often a lock froze the event loop. Attention that cannot be paid.

    A blocking hold on the loop is attention allocation failing in its most
    literal form: for that window the runtime could attend to nothing.
    """
    try:
        from core.runtime.lockdep import lockdep_report

        report = lockdep_report()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return None
    splats = report.get("splats")
    if not isinstance(splats, (list, tuple)):
        return None
    return float(
        sum(1 for s in splats if "loop_blocking" in str(getattr(s, "kind", s) or ""))
    )


def _open_degradations() -> float | None:
    """Recorded degradations — capability the runtime knows it has lost."""
    try:
        from core.runtime.errors import get_degradation_tracker

        # count() is per-subsystem; the process-wide total lives in status().
        status = get_degradation_tracker().status()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return None
    if not isinstance(status, dict):
        return None
    total = status.get("total_degradations")
    try:
        return float(total)
    except (TypeError, ValueError):
        return None


def _recall_hit_rate() -> float | None:
    """How often a recall attempt actually returned something usable.

    The real memory metric: not "is the machinery present" but "does asking
    my memory a question get an answer". ``RecallTelemetry`` already reports
    None when nothing has been attempted, which is exactly the right answer —
    a hit rate over zero attempts is not zero, it is unknown, and this model
    is built to keep that distinction.
    """
    try:
        from core.memory.recall_telemetry import get_recall_telemetry

        snapshot = get_recall_telemetry().snapshot()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return None
    if not isinstance(snapshot, dict):
        return None
    # Prefer the recent window; fall back to lifetime when the window is cold.
    for scope in ("window", "lifetime"):
        section = snapshot.get(scope)
        if isinstance(section, dict) and section.get("hit_rate") is not None:
            try:
                return float(section["hit_rate"])
            except (TypeError, ValueError):
                return None
    return None


def _dense_retrieval_available() -> float | None:
    """Whether recall runs on dense embeddings or has fallen back to lexical.

    A coarse probe, and deliberately labelled as one: it answers "is the good
    path available", not "how good is recall". Recall@k against a fixed probe
    set is the metric this should become.
    """
    try:
        from core.memory import rag

        if getattr(rag, "_EMBED_ENGINE_FAILED", False):
            return 0.0
        return 1.0 if getattr(rag, "_EMBED_ENGINE", None) is not None else 0.0
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        return None


def _unmeasured(reason: str):
    """A probe for a faculty nothing can currently see.

    Declaring the faculty with an honest non-probe is the point: it makes the
    blind spot countable and gives the self-model something to want.
    """

    def _probe() -> float | None:
        return None

    _probe.__doc__ = reason
    return _probe


# ── declarations ──────────────────────────────────────────────────────────


def declare_default_faculties(registry: FacultyRegistry | None = None) -> FacultyRegistry:
    """Declare Aura's faculties into ``registry`` (idempotent)."""
    target = registry if registry is not None else get_faculty_registry()

    target.declare(
        Faculty(
            faculty_id="memory",
            description="Storing and recalling what happened and what is true.",
            owner="core.memory",
            gates=("attention_allocation", "temporal_reasoning", "reasoning"),
            metrics=(
                ImprovementMetric(
                    metric_id="recall_hit_rate",
                    unit="",
                    direction="higher_is_better",
                    probe=_recall_hit_rate,
                    floor=0.0,
                    target=0.75,
                    ceiling=1.0,
                    weight=3.0,
                    description=(
                        "Fraction of recall attempts that returned something "
                        "usable — measured from live retrievals, not a benchmark."
                    ),
                ),
                ImprovementMetric(
                    metric_id="dense_retrieval_available",
                    unit="",
                    direction="higher_is_better",
                    probe=_dense_retrieval_available,
                    floor=0.0,
                    target=1.0,
                    ceiling=1.0,
                    weight=1.0,
                    description=(
                        "Whether the good retrieval path is even up. A capacity "
                        "check, weighted below the quality metric above."
                    ),
                ),
            ),
        )
    )

    target.declare(
        Faculty(
            faculty_id="attention_allocation",
            description="Where cognitive effort goes, and whether the loop can spend it.",
            owner="core.runtime",
            gates=("temporal_reasoning", "reasoning"),
            metrics=(
                ImprovementMetric(
                    metric_id="loop_blocking_holds",
                    unit=" holds",
                    direction="lower_is_better",
                    probe=_loop_blocking_holds,
                    floor=20.0,
                    target=0.0,
                    ceiling=0.0,
                    description="Windows in which the event loop could attend to nothing.",
                ),
            ),
        )
    )

    target.declare(
        Faculty(
            faculty_id="operational_integrity",
            description="Capability the runtime knows it has lost.",
            owner="core.runtime",
            gates=("memory", "attention_allocation", "reasoning"),
            metrics=(
                ImprovementMetric(
                    metric_id="open_degradations",
                    unit=" degradations",
                    direction="lower_is_better",
                    probe=_open_degradations,
                    floor=50.0,
                    target=0.0,
                    ceiling=0.0,
                ),
            ),
        )
    )

    # Declared WITHOUT a working probe on purpose. These are the faculties the
    # user named that Aura genuinely cannot see yet; leaving them out would
    # make the self-model look complete when it is not.
    target.declare(
        Faculty(
            faculty_id="temporal_reasoning",
            description="Ordering events, estimating durations, reasoning about when.",
            owner="core.cognition",
            metrics=(
                ImprovementMetric(
                    metric_id="event_order_accuracy",
                    unit="",
                    direction="higher_is_better",
                    probe=_unmeasured(
                        "no temporal benchmark is wired in; needs a task set of "
                        "event-ordering and duration questions scored at runtime"
                    ),
                    floor=0.0,
                    target=0.9,
                    ceiling=1.0,
                ),
            ),
        )
    )

    target.declare(
        Faculty(
            faculty_id="reasoning",
            description="Multi-step inference and its calibration.",
            owner="core.cognition",
            metrics=(
                ImprovementMetric(
                    metric_id="verifier_pass_rate",
                    unit="",
                    direction="higher_is_better",
                    probe=_unmeasured(
                        "no live verifier stream is aggregated into a rate yet. "
                        "reasoning_solved_cache.stats() is deliberately NOT used: "
                        "it measures cache reuse, not whether reasoning was "
                        "correct, and it reports 0.0 for zero attempts"
                    ),
                    floor=0.0,
                    target=0.85,
                    ceiling=1.0,
                ),
            ),
        )
    )
    return target


def _declare_owned_faculties(registry: FacultyRegistry) -> None:
    """Let a package declare its own faculty, without this module knowing it.

    Engineering design is measurable — whether its formulas still reproduce
    their published answers, what share of its results carried their working
    — and the probes for that belong beside the code they measure, not here.
    The import is local so that a runtime without the package still has a
    self-model, and a failure to declare is recorded rather than raised: a
    faculty that cannot be declared is a blind spot, and a blind spot must
    not take metacognition down with it.
    """
    from core.runtime.errors import record_degradation

    try:
        from core.engineering.faculty import declare_engineering_faculty

        declare_engineering_faculty(registry)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "metacognition.faculties", exc,
            action="engineering design could not declare itself as a faculty",
        )


def ensure_default_faculties() -> FacultyRegistry:
    """Declare the defaults once, on first use of the self-model."""
    global _declared
    registry = get_faculty_registry()
    with _declared_lock:
        if _declared:
            return registry
        _declared = True
    declared = declare_default_faculties(registry)
    _declare_owned_faculties(declared)
    return declared


def reset_default_faculties_for_test() -> None:
    global _declared
    with _declared_lock:
        _declared = False


__all__ = [
    "declare_default_faculties",
    "ensure_default_faculties",
    "reset_default_faculties_for_test",
]
