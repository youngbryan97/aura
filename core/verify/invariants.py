"""core/verify/invariants.py — the structural verifier.

Clean-room adoption of LLVM's `Verifier` pass and its `-verify-each`
discipline.

LLVM does not trust its own transforms. After a pass rewrites the IR, the
verifier re-checks every structural invariant — a terminator ends every
block, PHI nodes have one entry per predecessor, types line up. The payoff
is not that the invariants are exotic; it is *where the failure surfaces*.
Without it, a pass that corrupts the IR is diagnosed twelve passes later
in the backend, and the person debugging it starts from the wrong file.
With it, the corrupting pass names itself.

Aura has the same class of invariant and, until now, the same absence of a
checker. "Every required service is registered", "every alias resolves",
"declared lock ranks are consistent with observed acquisition order",
"every load-bearing organ is OOM-immune" — these are all true right up
until a refactor makes one false, and the symptom shows up somewhere else
entirely: a boot that hangs, a lane that never admits, a green health
report over a runtime that shed its own Will.

The contract:

* An invariant is **declarative**: name, scope, severity, owner,
  description, and a check that yields violations. Registration is a
  decorator, so adding one is three lines next to the thing it protects
  rather than an edit to a central list.
* Severity means what it says. ``ERROR`` is "the runtime's structure is
  wrong and something will break because of it". ``WARNING`` is "this is
  not what we intended". ``NOTE`` is informational.
* Verification is **scoped**, so the cost of checking after every mutation
  is proportional to what the mutation could have broken —
  ``verify_after("container")`` is what makes `-verify-each` affordable.
* A check that *raises* is itself a violation. A verifier that silently
  skips a broken invariant is worse than no verifier, because it reports
  clean.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Verify")


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


@dataclass(frozen=True)
class Violation:
    """One concrete breach, naming the subject that breached it.

    ``invariant`` and ``severity`` may be left unset by the check; the
    verifier fills them from the owning spec. A check should not have to
    repeat its own name on every yield.
    """

    subject: str
    message: str
    remedy: str = ""
    invariant: str = ""
    severity: Severity | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant": self.invariant,
            "severity": str(self.severity) if self.severity else str(Severity.ERROR),
            "subject": self.subject,
            "message": self.message,
            "remedy": self.remedy,
        }

    def __str__(self) -> str:
        tail = f" — fix: {self.remedy}" if self.remedy else ""
        return f"[{self.severity}] {self.invariant} @ {self.subject}: {self.message}{tail}"


CheckFn = Callable[[], Iterable[Violation]]


@dataclass(frozen=True)
class InvariantSpec:
    name: str
    scope: str
    severity: Severity
    description: str
    owner: str
    check: CheckFn
    observational: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope,
            "severity": str(self.severity),
            "description": self.description,
            "owner": self.owner,
            "observational": self.observational,
        }


@dataclass
class VerifyReport:
    scopes: tuple[str, ...]
    checked: int
    violations: list[Violation] = field(default_factory=list)
    duration_s: float = 0.0
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing at ERROR severity was found."""
        return not self.errors

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.WARNING]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "scopes": list(self.scopes),
            "checked": self.checked,
            "duration_s": round(self.duration_s, 4),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "violations": [v.to_dict() for v in self.violations],
            "skipped": list(self.skipped),
        }

    def summary(self) -> str:
        if not self.violations:
            return f"{self.checked} invariants verified clean"
        return (
            f"{self.checked} invariants: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s) — "
            + "; ".join(str(v) for v in self.violations[:3])
            + ("…" if len(self.violations) > 3 else "")
        )


def _definition_site(check: CheckFn) -> tuple[str, str]:
    """What identifies *one definition* of an invariant.

    Not the function object. Reloading a module re-executes its body and
    produces a fresh object for the same source definition, so an identity
    test reads a reload as a name collision.

    LIVE DEFECT, 2026-08-10: a hot-reload from the desktop UI reported "377
    reloaded, 1 failed" — ``core.conversation.disposition_invariants`` could
    not be reloaded because re-importing it re-registered
    ``surface.advisory_never_destroys``. Every module that declares an
    invariant at import time was un-reloadable for the same reason, and the
    guide tells authors to put invariants next to what they protect, so the
    set could only grow.

    The duplicate check exists to stop two *different* definitions claiming
    one name. Re-running the same definition is not that.
    """
    return (
        getattr(check, "__module__", "") or "",
        getattr(check, "__qualname__", None) or getattr(check, "__name__", "") or "",
    )


class InvariantRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._specs: dict[str, InvariantSpec] = {}

    def register(self, spec: InvariantSpec) -> InvariantSpec:
        with self._lock:
            existing = self._specs.get(spec.name)
            if existing is not None:
                previous = _definition_site(existing.check)
                current = _definition_site(spec.check)
                if previous != current:
                    raise ValueError(
                        f"invariant {spec.name!r} already registered by "
                        f"{existing.owner} ({previous[0]}.{previous[1]}); "
                        f"{spec.owner} ({current[0]}.{current[1]}) cannot claim "
                        "the same name — an invariant has one definition"
                    )
            self._specs[spec.name] = spec
            # Re-registering the same definition is a hot reload. Its previous
            # result described the old function body and must not survive as
            # evidence about the replacement.
            with _LAST_RESULTS_LOCK:
                _LAST_RESULTS.pop(spec.name, None)
            return spec

    def specs(self, scopes: Iterable[str] | None = None) -> list[InvariantSpec]:
        wanted = set(scopes) if scopes else None
        with self._lock:
            values = list(self._specs.values())
        if wanted is None:
            return sorted(values, key=lambda s: (s.scope, s.name))
        return sorted(
            (s for s in values if s.scope in wanted), key=lambda s: (s.scope, s.name)
        )

    def scopes(self) -> list[str]:
        with self._lock:
            return sorted({s.scope for s in self._specs.values()})

    def clear_for_test(self) -> None:
        with self._lock:
            self._specs.clear()
        with _LAST_RESULTS_LOCK:
            _LAST_RESULTS.clear()


_REGISTRY = InvariantRegistry()


def get_registry() -> InvariantRegistry:
    return _REGISTRY


def invariant(
    name: str,
    *,
    scope: str,
    severity: Severity = Severity.ERROR,
    description: str = "",
    owner: str = "unknown",
    observational: bool = True,
) -> Callable[[CheckFn], CheckFn]:
    """Declare a structural invariant next to the thing it protects.

    The decorated function yields :class:`Violation` for each breach and
    yields nothing when the invariant holds::

        @invariant("container.no_dangling_alias", scope="container",
                   owner="core/container.py")
        def _check_aliases():
            for alias, target in aliases().items():
                if not registered(target):
                    yield Violation(..., subject=alias, message=...)
    """

    def decorate(fn: CheckFn) -> CheckFn:
        _REGISTRY.register(
            InvariantSpec(
                name=name,
                scope=scope,
                severity=severity,
                description=description or (fn.__doc__ or "").strip().split("\n")[0],
                owner=owner,
                check=fn,
                observational=observational,
            )
        )
        return fn

    return decorate


def verify(
    *scopes: str,
    fail_fast: bool = False,
    record: bool = True,
    observational_only: bool = False,
) -> VerifyReport:
    """Run invariants in the given scopes (all scopes when empty).

    ``observational_only`` is for read surfaces such as health. Active probes
    are real mechanism tests that may stage and roll back temporary state; a
    health read must never execute them. Full verification still runs every
    invariant and records the result for observational surfaces to report.
    """
    started = time.perf_counter()
    specs = _REGISTRY.specs(scopes or None)
    if observational_only:
        specs = [spec for spec in specs if spec.observational]
    violations: list[Violation] = []
    skipped: list[str] = []
    outcomes: dict[str, tuple[Violation, ...]] = {}

    for spec in specs:
        spec_violations: list[Violation] = []
        try:
            produced = list(spec.check() or ())
        except Exception as exc:  # noqa: BLE001 — a broken check IS a violation
            skipped.append(spec.name)
            failure = Violation(
                invariant=spec.name,
                severity=Severity.ERROR,
                subject=spec.owner,
                message=(
                    f"the invariant check itself failed with {type(exc).__name__}: "
                    f"{exc}. A verifier that cannot check reports clean, which is "
                    "worse than reporting a breach"
                ),
                remedy="fix the check, or remove the invariant if it is obsolete",
            )
            violations.append(failure)
            spec_violations.append(failure)
            produced = []
        for item in produced:
            # Fill in what the check left to its spec.
            normalized = Violation(
                subject=item.subject,
                message=item.message,
                remedy=item.remedy,
                invariant=item.invariant or spec.name,
                severity=item.severity or spec.severity,
            )
            violations.append(normalized)
            spec_violations.append(normalized)
        outcomes[spec.name] = tuple(spec_violations)
        if fail_fast and any(v.severity is Severity.ERROR for v in violations):
            break

    report = VerifyReport(
        scopes=tuple(scopes) if scopes else tuple(_REGISTRY.scopes()),
        checked=len(specs),
        violations=violations,
        duration_s=time.perf_counter() - started,
        skipped=skipped,
    )
    if record:
        _record(report, outcomes)
    return report


_LAST_REPORT: VerifyReport | None = None
_LAST_RESULTS_LOCK = checked_lock("core.verify.invariants.singleton")
_LAST_RESULTS: dict[str, tuple[float, tuple[Violation, ...]]] = {}


def _record(
    report: VerifyReport,
    outcomes: dict[str, tuple[Violation, ...]],
) -> None:
    global _LAST_REPORT
    _LAST_REPORT = report
    checked_at = time.time()
    with _LAST_RESULTS_LOCK:
        for name, result in outcomes.items():
            _LAST_RESULTS[name] = (checked_at, result)
    if report.ok and not report.warnings:
        return
    for violation in report.errors:
        logger.error("🔎 VERIFIER %s", violation)
    for violation in report.warnings:
        logger.warning("🔎 verifier %s", violation)
    if not report.errors:
        return
    try:
        from core.runtime.errors import record_degradation
        from core.runtime.taint import TaintFlag, taint

        taint(
            TaintFlag.ASSERTION,
            f"structural verifier found {len(report.errors)} error(s): "
            + "; ".join(v.invariant for v in report.errors[:4]),
            subsystem="verifier",
        )
        record_degradation(
            "verifier",
            AssertionError(report.summary()),
            severity="degraded",
            action="recorded structural violations; runtime continued",
            extra={"violations": [v.to_dict() for v in report.errors[:16]]},
            enforce_failure_policy=False,
        )
    except Exception:  # noqa: BLE001 — reporting must not break verification
        logger.debug("verifier degradation record failed", exc_info=True)


def last_report() -> dict[str, Any] | None:
    return _LAST_REPORT.to_dict() if _LAST_REPORT is not None else None


def latest_invariant_result(name: str) -> dict[str, Any] | None:
    """Return the last explicit proof for one invariant without re-running it."""
    with _LAST_RESULTS_LOCK:
        observed = _LAST_RESULTS.get(str(name))
    if observed is None:
        return None
    checked_at, violations = observed
    errors = [one for one in violations if one.severity is Severity.ERROR]
    return {
        "checked_at": checked_at,
        "ok": not errors,
        "violations": [one.to_dict() for one in violations],
    }


@contextmanager
def verify_after(*scopes: str, label: str = "") -> Iterator[None]:
    """`-verify-each`: re-check the affected scopes after a mutation.

    Wrap anything that changes the runtime's structure — registering a
    service, hot-swapping a module, declaring a lock rank — so the
    operation that broke the invariant is the one that names itself::

        with verify_after("container", label="register_all_services"):
            register_all_services()
    """
    yield
    report = verify(*scopes)
    if not report.ok:
        logger.error(
            "🔎 VERIFIER after %s: %s",
            label or "/".join(scopes) or "mutation",
            report.summary(),
        )


def verifier_report() -> dict[str, Any]:
    """Full standing state: what is declared, and the last result."""
    specs = _REGISTRY.specs()
    return {
        "declared": [s.to_dict() for s in specs],
        "scopes": _REGISTRY.scopes(),
        "count": len(specs),
        "last_report": last_report(),
    }


def reset_verifier_for_test() -> None:
    global _LAST_REPORT
    _LAST_REPORT = None
    with _LAST_RESULTS_LOCK:
        _LAST_RESULTS.clear()


__all__ = [
    "CheckFn",
    "InvariantRegistry",
    "InvariantSpec",
    "Severity",
    "VerifyReport",
    "Violation",
    "get_registry",
    "invariant",
    "last_report",
    "latest_invariant_result",
    "reset_verifier_for_test",
    "verifier_report",
    "verify",
    "verify_after",
]
