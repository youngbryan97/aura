"""core/pipeline/pass_manager.py — the cognitive pass manager.

Clean-room adoption of LLVM's new PassManager, its AnalysisManager, its
PassInstrumentation hooks, and `-opt-bisect-limit`.

Three ideas, each of which Aura's cognitive pipeline needs and did not
have:

**1. Analyses are cached and invalidated precisely.** In LLVM a pass does
not recompute dominator trees; it asks the AnalysisManager, which either
returns the cached result or recomputes it because some earlier pass said
it did not preserve it. Aura's phases recompute the same derived facts —
current affect, retrieved evidence, salience ranking — several times per
tick, and worse, sometimes use a *stale* one because nothing tracks who
invalidated what. A pass returns ``PreservedAnalyses``; that return value
is the invalidation contract.

**2. Instrumentation is a first-class hook, not print statements.** Every
pass execution is announced before and after. That single seam gives
timing histograms, trace slices, structural verification after each pass,
and —

**3. `-opt-bisect-limit`, which is the reason this file exists.** When a
compiler miscompiles, you do not read 200 passes. You bisect: run the
first N passes and stop, binary-search N until the output goes bad, and
the pass at the boundary is the culprit. Aura has exactly this problem in
a worse form — a turn comes out wrong and the pipeline has ~30 phases, any
of which could have caused it, and the failure is not deterministic enough
to reason about statically. With bisect the question "which phase ruined
this answer" becomes ~5 runs instead of an afternoon of guessing.

Instrumentation is a process-wide singleton so the existing kernel phase
loop can consult it without adopting the whole PassManager: a pipeline
that is already written and load-bearing gets bisect and timing by asking
``should_run()``, and new pipelines get the full machinery.
"""

from __future__ import annotations

from core.runtime.sheddable import CacheHolder

import contextvars
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

logger = logging.getLogger("Aura.PassManager")

Unit = TypeVar("Unit")

#: Pass numbering for the turn currently being served. A one-element list so
#: the counter can be bumped without rebinding the ContextVar on every pass.
#: ``None`` until :meth:`PassInstrumentation.begin_run` is called on this
#: task, where numbering falls back to the process-wide counter.
_RUN_ORDINAL: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "aura_pass_run_ordinal", default=None
)


# ══════════════════════════════════════════════════════════════════════
# Preservation contract
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PreservedAnalyses:
    """What a pass promises it did not invalidate.

    The default for a pass that says nothing is :meth:`none` — pessimistic
    on purpose. A pass that forgets to declare preservation costs
    recomputation; a pass that wrongly *claims* preservation costs
    correctness, and correctness bugs from stale analyses are the hardest
    kind to find.
    """

    _all: bool = False
    _preserved: frozenset[str] = frozenset()
    _abandoned: frozenset[str] = frozenset()

    @classmethod
    def all(cls) -> PreservedAnalyses:
        """The pass changed nothing any analysis depends on."""
        return cls(_all=True)

    @classmethod
    def none(cls) -> PreservedAnalyses:
        """The pass may have changed anything."""
        return cls(_all=False)

    def preserve(self, *names: str) -> PreservedAnalyses:
        return PreservedAnalyses(
            _all=self._all,
            _preserved=self._preserved | frozenset(names),
            _abandoned=self._abandoned - frozenset(names),
        )

    def abandon(self, *names: str) -> PreservedAnalyses:
        """Explicitly invalidate, even under :meth:`all`."""
        return PreservedAnalyses(
            _all=self._all,
            _preserved=self._preserved - frozenset(names),
            _abandoned=self._abandoned | frozenset(names),
        )

    def preserved(self, name: str) -> bool:
        if name in self._abandoned:
            return False
        return self._all or name in self._preserved

    def intersect(self, other: PreservedAnalyses) -> PreservedAnalyses:
        """Running two passes preserves only what both preserved."""
        return PreservedAnalyses(
            _all=self._all and other._all,
            _preserved=frozenset(
                n for n in (self._preserved | other._preserved) if self.preserved(n) and other.preserved(n)
            ),
            _abandoned=self._abandoned | other._abandoned,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "all": self._all,
            "preserved": sorted(self._preserved),
            "abandoned": sorted(self._abandoned),
        }


# ══════════════════════════════════════════════════════════════════════
# Analyses
# ══════════════════════════════════════════════════════════════════════

class Analysis(Protocol[Unit]):
    """A derived fact computed from the unit and cached until invalidated."""

    name: str

    def run(self, unit: Unit) -> Any: ...


@dataclass
class _CacheEntry:
    value: Any
    computed_at: float
    computed_in_s: float
    hits: int = 0


class AnalysisManager(CacheHolder, Generic[Unit]):
    """Caches analysis results and invalidates them on the preservation contract."""

    def __init__(self) -> None:
        self._analyses: dict[str, Analysis[Unit]] = {}
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self.computations = 0
        self.hits = 0
        self.invalidations = 0

    #: Analyses already computed for the current unit. The preservation
    #: contract invalidates them routinely, so dropping them early is the
    #: same event the manager already handles.
    sheddable_caches = ("_cache",)
    oom_rationale = "cached pass analyses, recomputed on the next request"

    def register(self, analysis: Analysis[Unit]) -> None:
        name = getattr(analysis, "name", "") or type(analysis).__name__
        with self._lock:
            self._analyses[name] = analysis

    def register_fn(self, name: str, fn: Callable[[Unit], Any]) -> None:
        """Register a plain function as an analysis."""

        class _FnAnalysis:
            def __init__(self) -> None:
                self.name = name

            def run(self, unit: Unit) -> Any:
                return fn(unit)

        self.register(_FnAnalysis())

    def get(self, name: str, unit: Unit) -> Any:
        with self._lock:
            entry = self._cache.get(name)
            if entry is not None:
                entry.hits += 1
                self.hits += 1
                return entry.value
            analysis = self._analyses.get(name)
        if analysis is None:
            raise KeyError(
                f"no analysis named {name!r} is registered; a pass asked for a "
                "derived fact nothing computes"
            )
        started = time.perf_counter()
        value = analysis.run(unit)
        elapsed = time.perf_counter() - started
        with self._lock:
            self._cache[name] = _CacheEntry(
                value=value, computed_at=time.time(), computed_in_s=elapsed
            )
            self.computations += 1
        get_instrumentation().after_analysis(name, elapsed)
        return value

    def cached(self, name: str) -> bool:
        with self._lock:
            return name in self._cache

    def invalidate(self, preserved: PreservedAnalyses) -> list[str]:
        """Drop every cached analysis the pass did not promise to preserve."""
        with self._lock:
            dropped = [name for name in self._cache if not preserved.preserved(name)]
            for name in dropped:
                del self._cache[name]
            self.invalidations += len(dropped)
        return dropped

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "registered": sorted(self._analyses),
                "cached": sorted(self._cache),
                "computations": self.computations,
                "hits": self.hits,
                "invalidations": self.invalidations,
                "hit_rate": (
                    self.hits / (self.hits + self.computations)
                    if (self.hits + self.computations)
                    else 0.0
                ),
            }


# ══════════════════════════════════════════════════════════════════════
# Instrumentation
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PassRecord:
    name: str
    ordinal: int
    duration_s: float
    skipped: bool
    reason: str = ""
    preserved: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ordinal": self.ordinal,
            "duration_s": round(self.duration_s, 6),
            "skipped": self.skipped,
            "reason": self.reason,
            "preserved": self.preserved,
            "error": self.error,
        }


class PassInstrumentation:
    """The one seam every pipeline announces itself through.

    ``before_pass`` returning False skips the pass — that is how bisect and
    any future pass-level policy (a budget guard, a lesion controller, an
    ablation experiment) get their leverage without any pipeline knowing
    they exist.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ordinal = 0
        self._bisect_limit: int | None = _declared_bisect_limit()
        self._trace = _declared_trace()
        self._records: list[PassRecord] = []
        self._before_hooks: list[Callable[[str, int], bool]] = []
        self._after_hooks: list[Callable[[PassRecord], None]] = []
        self._analysis_hooks: list[Callable[[str, float], None]] = []
        self._max_records = 4096
        self.skips = 0

    # ── configuration ─────────────────────────────────────────────────
    def set_bisect_limit(self, limit: int | None) -> None:
        """Run passes with ordinal <= limit; skip the rest.

        ``0`` runs nothing, ``None`` runs everything. Binary-search the
        limit until the output changes and the pass at the boundary is the
        one that did it.
        """
        with self._lock:
            self._bisect_limit = limit
        logger.info("🔬 pass bisect limit set to %s", "off" if limit is None else limit)

    def bisect_limit(self) -> int | None:
        return self._bisect_limit

    def set_trace(self, enabled: bool) -> None:
        self._trace = bool(enabled)

    def add_before_hook(self, hook: Callable[[str, int], bool]) -> None:
        """Return False from the hook to skip the pass."""
        with self._lock:
            self._before_hooks.append(hook)

    def add_after_hook(self, hook: Callable[[PassRecord], None]) -> None:
        with self._lock:
            self._after_hooks.append(hook)

    def add_analysis_hook(self, hook: Callable[[str, float], None]) -> None:
        with self._lock:
            self._analysis_hooks.append(hook)

    # ── the hooks themselves ──────────────────────────────────────────
    def next_ordinal(self) -> int:
        """The next pass number, scoped to the current run if there is one."""
        counter = _RUN_ORDINAL.get()
        if counter is not None:
            counter[0] += 1
            return counter[0]
        with self._lock:
            self._ordinal += 1
            return self._ordinal

    def begin_run(self, label: str = "") -> None:
        """Restart pass numbering for the turn about to be served.

        Bisect compares an ordinal against a limit, so "the first N passes"
        only means anything if every turn starts counting at the same place.
        Process-monotonic numbering made ``AURA_PASS_BISECT_LIMIT=5`` mean
        "the first five passes since boot": correct on the first turn, total
        silence on every turn after it — worse than not having the knob.
        Documented behaviour is per-turn, so numbering is per-turn.

        The counter lives in a ContextVar rather than on the instrumentation,
        so it is scoped to the asyncio task serving this turn. Two turns in
        flight number their own passes instead of interleaving into one
        sequence, and no caller has to remember to close a scope.
        """
        _RUN_ORDINAL.set([0])
        if self._trace and label:
            logger.info("🔬 BISECT begin run %s", label)

    def reset_ordinals(self) -> None:
        """Restart pass numbering, whichever counter is in force.

        Both counters, deliberately. Resetting only the process-wide one
        would silently do nothing in any context where :meth:`begin_run` has
        been called — and since a ContextVar set outside a task outlives the
        call that set it, that is most contexts once a turn has been served.
        A reset that quietly stops resetting is how bisect would come to
        report the wrong boundary pass.
        """
        counter = _RUN_ORDINAL.get()
        if counter is not None:
            counter[0] = 0
        with self._lock:
            self._ordinal = 0

    def should_run(self, name: str, ordinal: int | None = None) -> tuple[bool, int, str]:
        """Consulted before every pass. Returns (run?, ordinal, reason)."""
        ordinal = self.next_ordinal() if ordinal is None else ordinal
        limit = self._bisect_limit
        if limit is not None and ordinal > limit:
            self.skips += 1
            reason = f"opt-bisect: ordinal {ordinal} > limit {limit}"
            if self._trace:
                logger.info("🔬 BISECT NOT running pass (%s) on %s", reason, name)
            return False, ordinal, reason
        with self._lock:
            hooks = list(self._before_hooks)
        for hook in hooks:
            try:
                if hook(name, ordinal) is False:
                    self.skips += 1
                    return False, ordinal, f"skipped by {getattr(hook, '__name__', 'hook')}"
            except Exception:  # noqa: BLE001 — a broken hook must not stop the pipeline
                logger.debug("pass before-hook failed", exc_info=True)
        if self._trace:
            logger.info("🔬 BISECT running pass (%d) %s", ordinal, name)
        return True, ordinal, ""

    def after_pass(self, record: PassRecord) -> None:
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                del self._records[: -self._max_records]
            hooks = list(self._after_hooks)
        for hook in hooks:
            try:
                hook(record)
            except Exception:  # noqa: BLE001
                logger.debug("pass after-hook failed", exc_info=True)

    def after_analysis(self, name: str, duration_s: float) -> None:
        with self._lock:
            hooks = list(self._analysis_hooks)
        for hook in hooks:
            try:
                hook(name, duration_s)
            except Exception:  # noqa: BLE001
                logger.debug("analysis hook failed", exc_info=True)

    # ── reporting ─────────────────────────────────────────────────────
    def records(self) -> list[PassRecord]:
        with self._lock:
            return list(self._records)

    def report(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
            limit = self._bisect_limit
            ordinal = self._ordinal
        by_name: dict[str, dict[str, Any]] = {}
        for record in records:
            entry = by_name.setdefault(
                record.name,
                {"runs": 0, "skips": 0, "total_s": 0.0, "max_s": 0.0, "errors": 0},
            )
            if record.skipped:
                entry["skips"] += 1
                continue
            entry["runs"] += 1
            entry["total_s"] += record.duration_s
            entry["max_s"] = max(entry["max_s"], record.duration_s)
            if record.error:
                entry["errors"] += 1
        for entry in by_name.values():
            entry["mean_s"] = entry["total_s"] / entry["runs"] if entry["runs"] else 0.0
        hottest = sorted(by_name.items(), key=lambda kv: -kv[1]["total_s"])[:5]
        return {
            "bisect_limit": limit,
            "ordinals_issued": ordinal,
            "skips": self.skips,
            "passes": by_name,
            "hottest": [name for name, _ in hottest],
            "recent": [r.to_dict() for r in records[-16:]],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._ordinal = 0
            self._records.clear()
            self._before_hooks.clear()
            self._after_hooks.clear()
            self._analysis_hooks.clear()
            self._bisect_limit = None
            self.skips = 0


def _declared_bisect_limit() -> int | None:
    """The bisect limit, read through the flag registry.

    Declared rather than read raw so the knob shows up in `flag_report()`
    alongside every other one — a knob nobody can enumerate is a knob
    nobody can retire. ``-1`` means off, matching "no limit".
    """
    from core.runtime.flags import FlagKind, declare

    flag = declare(
        "AURA_PASS_BISECT_LIMIT",
        kind=FlagKind.INT,
        default=-1,
        description=(
            "Run only cognitive passes with ordinal <= this value; -1 disables. "
            "Binary-search it to find which pass caused a regression."
        ),
        owner="core/pipeline/pass_manager.py",
    )
    value = int(flag.value())
    return None if value < 0 else value


def _declared_trace() -> bool:
    from core.runtime.flags import FlagKind, declare

    return bool(
        declare(
            "AURA_PASS_TRACE",
            kind=FlagKind.BOOL,
            default=False,
            description="Log every cognitive pass as it runs or is skipped.",
            owner="core/pipeline/pass_manager.py",
        ).value()
    )


_INSTRUMENTATION = PassInstrumentation()


def get_instrumentation() -> PassInstrumentation:
    return _INSTRUMENTATION


# ══════════════════════════════════════════════════════════════════════
# Passes and the manager
# ══════════════════════════════════════════════════════════════════════

class Pass(Protocol[Unit]):
    name: str

    def run(self, unit: Unit, am: AnalysisManager[Unit]) -> PreservedAnalyses: ...


@dataclass
class FunctionPass(Generic[Unit]):
    """Adapts a plain callable into a pass."""

    name: str
    fn: Callable[[Unit, AnalysisManager[Unit]], PreservedAnalyses | None]
    preserves: PreservedAnalyses = field(default_factory=PreservedAnalyses.none)

    def run(self, unit: Unit, am: AnalysisManager[Unit]) -> PreservedAnalyses:
        result = self.fn(unit, am)
        return result if isinstance(result, PreservedAnalyses) else self.preserves


class PassManager(Generic[Unit]):
    """An ordered pipeline with preservation, instrumentation, and bisect."""

    def __init__(self, name: str = "pipeline", *, verify_each: str | None = None) -> None:
        self.name = name
        #: Scope passed to the structural verifier after each pass, LLVM's
        #: `-verify-each`. None disables it.
        self.verify_each = verify_each
        self._passes: list[Pass[Unit]] = []

    def add(self, p: Pass[Unit]) -> PassManager[Unit]:
        self._passes.append(p)
        return self

    def add_fn(
        self,
        name: str,
        fn: Callable[[Unit, AnalysisManager[Unit]], PreservedAnalyses | None],
        *,
        preserves: PreservedAnalyses | None = None,
    ) -> PassManager[Unit]:
        return self.add(
            FunctionPass(name=name, fn=fn, preserves=preserves or PreservedAnalyses.none())
        )

    def pass_names(self) -> list[str]:
        return [getattr(p, "name", type(p).__name__) for p in self._passes]

    def run(self, unit: Unit, am: AnalysisManager[Unit] | None = None) -> PreservedAnalyses:
        manager = am if am is not None else AnalysisManager()
        instrumentation = get_instrumentation()
        # One run, one numbering. A bisect limit is meaningless if the second
        # run of the same pipeline starts counting where the first left off.
        instrumentation.begin_run(self.name)
        overall = PreservedAnalyses.all()

        for p in self._passes:
            name = getattr(p, "name", type(p).__name__)
            run_it, ordinal, reason = instrumentation.should_run(f"{self.name}/{name}")
            if not run_it:
                instrumentation.after_pass(
                    PassRecord(name=name, ordinal=ordinal, duration_s=0.0, skipped=True, reason=reason)
                )
                continue

            started = time.perf_counter()
            error = ""
            try:
                preserved = p.run(unit, manager)
                if not isinstance(preserved, PreservedAnalyses):
                    preserved = PreservedAnalyses.none()
            except Exception as exc:  # noqa: BLE001 — the record must exist before the raise
                error = f"{type(exc).__name__}: {exc}"
                instrumentation.after_pass(
                    PassRecord(
                        name=name,
                        ordinal=ordinal,
                        duration_s=time.perf_counter() - started,
                        skipped=False,
                        error=error,
                    )
                )
                raise

            elapsed = time.perf_counter() - started
            manager.invalidate(preserved)
            overall = overall.intersect(preserved)
            instrumentation.after_pass(
                PassRecord(
                    name=name,
                    ordinal=ordinal,
                    duration_s=elapsed,
                    skipped=False,
                    preserved=preserved.to_dict(),
                )
            )

            if self.verify_each:
                from core.verify.invariants import verify

                report = verify(self.verify_each)
                if not report.ok:
                    logger.error(
                        "🔎 VERIFIER after pass %s/%s: %s", self.name, name, report.summary()
                    )
        return overall


def install_default_instrumentation() -> dict[str, Any]:
    """Wire timing and tracing into the process-wide instrumentation.

    Idempotent: safe to call from boot and from tests.
    """
    instrumentation = get_instrumentation()
    if getattr(instrumentation, "_defaults_installed", False):
        return {"installed": False, "reason": "already installed"}

    def _record_timing(record: PassRecord) -> None:
        if record.skipped:
            return
        try:
            from core.observability.metrics import get_metrics

            get_metrics().record_duration(f"cognitive_pass.{record.name}", record.duration_s)
        except Exception:  # noqa: BLE001 — telemetry is never load-bearing
            logger.debug("pass timing metric failed", exc_info=True)

    instrumentation.add_after_hook(_record_timing)
    instrumentation._defaults_installed = True  # type: ignore[attr-defined]
    return {
        "installed": True,
        "bisect_limit": instrumentation.bisect_limit(),
    }


def bisect_pipeline(
    run: Callable[[], Any],
    is_good: Callable[[Any], bool],
    *,
    max_ordinal: int,
) -> dict[str, Any]:
    """Binary-search the pass that turns a good result bad.

    ``run`` executes the pipeline under the current bisect limit and
    returns its result; ``is_good`` judges it. Returns the boundary
    ordinal and the pass name recorded at it — the first pass whose
    inclusion makes the result bad.

    This is a debugging entry point. It resets pass numbering between
    attempts, so it is meant for a pipeline you are driving deliberately,
    not for the live tick loop while it is serving.
    """
    instrumentation = get_instrumentation()
    original = instrumentation.bisect_limit()
    high = max(0, int(max_ordinal))

    def attempt(limit: int) -> Any:
        instrumentation.set_bisect_limit(limit)
        instrumentation.reset_ordinals()
        return run()

    try:
        if is_good(attempt(high)):
            return {"found": False, "reason": "the full pipeline is already good"}
        if not is_good(attempt(0)):
            return {"found": False, "reason": "the empty pipeline is already bad"}

        # Invariant: limit 0 is good, limit `high` is bad. Find the
        # smallest limit that is bad; the pass at that ordinal is the one
        # whose inclusion flipped it.
        low = 1
        while low < high:
            mid = (low + high) // 2
            if is_good(attempt(mid)):
                low = mid + 1
            else:
                high = mid
        attempt(low)
        culprit = next(
            (
                r.name
                for r in reversed(instrumentation.records())
                if r.ordinal == low and not r.skipped
            ),
            "<unknown>",
        )
        return {"found": True, "ordinal": low, "pass": culprit}
    finally:
        instrumentation.set_bisect_limit(original)


def pass_manager_report() -> dict[str, Any]:
    return get_instrumentation().report()


def reset_pass_manager_for_test() -> None:
    _INSTRUMENTATION.reset_for_test()
    if hasattr(_INSTRUMENTATION, "_defaults_installed"):
        delattr(_INSTRUMENTATION, "_defaults_installed")


__all__ = [
    "Analysis",
    "AnalysisManager",
    "FunctionPass",
    "Pass",
    "PassInstrumentation",
    "PassManager",
    "PassRecord",
    "PreservedAnalyses",
    "bisect_pipeline",
    "get_instrumentation",
    "install_default_instrumentation",
    "pass_manager_report",
    "reset_pass_manager_for_test",
]
