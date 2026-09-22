"""core/runtime/admission.py — the admission chain.

Clean-room adoption of Kubernetes admission control: an ordered chain of
*mutating* then *validating* hooks that every write to the system passes
through before it becomes real.

Aura already has a decision authority — the Unified Will — and it is the
right shape for "may this actor do this at all". What it is not is a place
to hang *cross-cutting policy about the request itself*: defaulting a
missing budget, stamping provenance, rejecting a payload that is
structurally impossible, enforcing a naming rule, capping a fan-out.
Today those live scattered across call sites, which means each one is
enforced wherever somebody remembered.

Kubernetes solved exactly this by making admission a pipeline with three
properties worth copying verbatim:

1. **Mutation happens before validation, always.** Every mutating hook
   runs first and may rewrite the request; then every validating hook sees
   the *final* object. Interleaving them is how you get a validator that
   approved something a later mutator changed.
2. **Validators may not mutate.** The chain re-checks this, because a
   validator that quietly edits its input makes the admitted object
   depend on hook ordering, and ordering is the thing nobody remembers.
3. **Failure policy is declared per hook.** ``Fail`` means a hook that
   errors denies the request — correct for anything security-relevant.
   ``Ignore`` means the request proceeds without it — correct for
   enrichment. Leaving this implicit is how a crashed policy hook becomes
   an open door.

The chain is *the* seam for policy that must apply everywhere, so adding a
rule is one registration rather than an audit of every call site.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Admission")

#: A hook that runs longer than this is treated per its failure policy.
DEFAULT_HOOK_TIMEOUT_S = 2.0


class Operation(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"


class FailurePolicy(StrEnum):
    #: A hook that errors denies the request. For anything that enforces a
    #: rule: silence must not mean consent.
    FAIL = "fail"
    #: A hook that errors is skipped. For enrichment only.
    IGNORE = "ignore"


@dataclass(frozen=True)
class AdmissionRequest:
    """What is being admitted."""

    operation: Operation
    kind: str
    name: str
    obj: Any
    principal: str = "runtime"
    context: dict[str, Any] = field(default_factory=dict)
    old_obj: Any = None

    def with_object(self, obj: Any) -> AdmissionRequest:
        return replace(self, obj=obj)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": str(self.operation),
            "kind": self.kind,
            "name": self.name,
            "principal": self.principal,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class AdmissionResponse:
    allowed: bool
    reason: str = ""
    hook: str = ""
    warnings: tuple[str, ...] = ()

    @classmethod
    def allow(cls, *warnings: str) -> AdmissionResponse:
        return cls(allowed=True, warnings=tuple(warnings))

    @classmethod
    def deny(cls, reason: str) -> AdmissionResponse:
        return cls(allowed=False, reason=reason)


@dataclass
class AdmissionVerdict:
    """The chain's outcome, including what the object became."""

    allowed: bool
    obj: Any
    reason: str = ""
    denied_by: str = ""
    warnings: list[str] = field(default_factory=list)
    mutated_by: list[str] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "denied_by": self.denied_by,
            "warnings": list(self.warnings),
            "mutated_by": list(self.mutated_by),
            "ran": list(self.ran),
            "skipped": dict(self.skipped),
            "duration_s": round(self.duration_s, 5),
        }


MutatingFn = Callable[[AdmissionRequest], Any]
ValidatingFn = Callable[[AdmissionRequest], AdmissionResponse | bool | None]


@dataclass(frozen=True)
class Hook:
    name: str
    kinds: frozenset[str]
    operations: frozenset[Operation]
    order: int
    failure_policy: FailurePolicy
    owner: str
    mutating: bool
    fn: Callable[..., Any]
    timeout_s: float = DEFAULT_HOOK_TIMEOUT_S

    def matches(self, request: AdmissionRequest) -> bool:
        if self.kinds and "*" not in self.kinds and request.kind not in self.kinds:
            return False
        return not self.operations or request.operation in self.operations

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kinds": sorted(self.kinds) or ["*"],
            "operations": sorted(str(o) for o in self.operations) or ["*"],
            "order": self.order,
            "failure_policy": str(self.failure_policy),
            "owner": self.owner,
            "mutating": self.mutating,
        }


def _fingerprint(obj: Any) -> str:
    """A cheap structural fingerprint, used to catch mutating validators."""
    try:
        return json.dumps(obj, sort_keys=True, default=repr)[:4096]
    except Exception:  # noqa: BLE001 — unserializable objects fall back to repr
        logger.debug("admission fingerprint JSON encoding failed", exc_info=True)
        return repr(obj)[:4096]


class AdmissionChain:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mutating: list[Hook] = []
        self._validating: list[Hook] = []
        self.admitted = 0
        self.denied = 0
        self._recent: list[dict[str, Any]] = []

    # ── registration ──────────────────────────────────────────────────
    def register(self, hook: Hook) -> Hook:
        with self._lock:
            bucket = self._mutating if hook.mutating else self._validating
            for existing in bucket:
                if existing.name == hook.name:
                    raise ValueError(
                        f"admission hook {hook.name!r} already registered by "
                        f"{existing.owner}"
                    )
            bucket.append(hook)
            bucket.sort(key=lambda h: (h.order, h.name))
        return hook

    def unregister(self, name: str) -> bool:
        with self._lock:
            for bucket in (self._mutating, self._validating):
                for index, hook in enumerate(bucket):
                    if hook.name == name:
                        del bucket[index]
                        return True
        return False

    def hooks(self) -> list[Hook]:
        with self._lock:
            return [*self._mutating, *self._validating]

    # ── the chain ─────────────────────────────────────────────────────
    def admit(self, request: AdmissionRequest) -> AdmissionVerdict:
        started = time.perf_counter()
        with self._lock:
            mutating = [h for h in self._mutating if h.matches(request)]
            validating = [h for h in self._validating if h.matches(request)]

        verdict = AdmissionVerdict(allowed=True, obj=request.obj)
        current = request

        # Phase 1 — mutation. Validators must see the final object.
        for hook in mutating:
            outcome, failure = self._call(hook, current)
            if failure is not None:
                if hook.failure_policy is FailurePolicy.FAIL:
                    return self._finish(
                        verdict,
                        started,
                        allowed=False,
                        reason=f"mutating hook {hook.name!r} failed: {failure}",
                        denied_by=hook.name,
                        request=request,
                    )
                verdict.skipped[hook.name] = failure
                continue
            verdict.ran.append(hook.name)
            if outcome is not None and outcome is not current.obj:
                current = current.with_object(outcome)
                verdict.mutated_by.append(hook.name)
        verdict.obj = current.obj

        # Phase 2 — validation over the final object.
        before = _fingerprint(current.obj)
        for hook in validating:
            outcome, failure = self._call(hook, current)
            if failure is not None:
                if hook.failure_policy is FailurePolicy.FAIL:
                    return self._finish(
                        verdict,
                        started,
                        allowed=False,
                        reason=f"validating hook {hook.name!r} failed: {failure}",
                        denied_by=hook.name,
                        request=request,
                    )
                verdict.skipped[hook.name] = failure
                continue
            verdict.ran.append(hook.name)

            response = _coerce_response(outcome)
            verdict.warnings.extend(response.warnings)
            if not response.allowed:
                return self._finish(
                    verdict,
                    started,
                    allowed=False,
                    reason=response.reason or f"denied by {hook.name}",
                    denied_by=hook.name,
                    request=request,
                )

            after = _fingerprint(current.obj)
            if after != before:
                # A validator that edits its input makes the admitted
                # object depend on hook ordering. Report, and treat the
                # edit as not having happened for ordering purposes.
                logger.error(
                    "🚦 admission: validating hook %r mutated the object; "
                    "validators must not mutate",
                    hook.name,
                )
                verdict.warnings.append(
                    f"validating hook {hook.name!r} mutated the request object"
                )
                from core.runtime.sanitizers import get_sanitizer_log

                get_sanitizer_log().report(
                    "admission",
                    f"validator-mutates:{hook.name}",
                    f"validating hook {hook.name!r} mutated the admission object; "
                    "the admitted result now depends on hook order",
                )
                before = after

        return self._finish(verdict, started, allowed=True, request=request)

    def _call(self, hook: Hook, request: AdmissionRequest) -> tuple[Any, str | None]:
        started = time.perf_counter()
        try:
            outcome = hook.fn(request)
        except Exception as exc:  # noqa: BLE001 — policy decides what a failure means
            logger.warning("admission hook %r raised: %s", hook.name, exc)
            return None, f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        if elapsed > hook.timeout_s:
            logger.warning(
                "admission hook %r took %.3fs (budget %.3fs)",
                hook.name,
                elapsed,
                hook.timeout_s,
            )
        return outcome, None

    def _finish(
        self,
        verdict: AdmissionVerdict,
        started: float,
        *,
        allowed: bool,
        request: AdmissionRequest,
        reason: str = "",
        denied_by: str = "",
    ) -> AdmissionVerdict:
        verdict.allowed = allowed
        verdict.reason = reason
        verdict.denied_by = denied_by
        verdict.duration_s = time.perf_counter() - started
        # Aura.Admission.DurationMs named this file as its owner and had no
        # writer, so the histogram read empty for the life of the declaration.
        # _finish is the one exit every verdict passes through, allowed or not —
        # timing only the allowed path would hide the case that matters, a slow
        # denial.
        try:
            from core.observability.histograms import record as record_histogram

            record_histogram("Aura.Admission.DurationMs", max(0.0, verdict.duration_s) * 1000.0)
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            # A silent except here is how the histogram read empty the first
            # time. Say it, quietly, rather than go back to that.
            logger.debug(
                "Aura.Admission.DurationMs not recorded (%s: %s)",
                type(exc).__name__,
                exc,
            )
        with self._lock:
            if allowed:
                self.admitted += 1
            else:
                self.denied += 1
            self._recent.append({**request.to_dict(), **verdict.to_dict()})
            if len(self._recent) > 128:
                del self._recent[:-128]
        if not allowed:
            logger.info(
                "🚦 admission denied %s/%s by %s: %s",
                request.kind,
                request.name,
                denied_by,
                reason,
            )
        return verdict

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mutating": [h.to_dict() for h in self._mutating],
                "validating": [h.to_dict() for h in self._validating],
                "admitted": self.admitted,
                "denied": self.denied,
                "recent": list(self._recent[-8:]),
            }

    def reset_for_test(self) -> None:
        with self._lock:
            self._mutating.clear()
            self._validating.clear()
            self._recent.clear()
            self.admitted = 0
            self.denied = 0


def _coerce_response(outcome: Any) -> AdmissionResponse:
    if isinstance(outcome, AdmissionResponse):
        return outcome
    if outcome is None or outcome is True:
        return AdmissionResponse.allow()
    if outcome is False:
        return AdmissionResponse.deny("hook returned False")
    if isinstance(outcome, str):
        return AdmissionResponse.deny(outcome)
    return AdmissionResponse.allow()


_CHAIN = AdmissionChain()


def get_admission_chain() -> AdmissionChain:
    return _CHAIN


def mutating(
    name: str,
    *,
    kinds: tuple[str, ...] = ("*",),
    operations: tuple[Operation, ...] = (),
    order: int = 100,
    failure_policy: FailurePolicy = FailurePolicy.FAIL,
    owner: str = "unknown",
    timeout_s: float = DEFAULT_HOOK_TIMEOUT_S,
) -> Callable[[MutatingFn], MutatingFn]:
    """Declare a mutating hook. Return the new object, or None to leave it."""

    def decorate(fn: MutatingFn) -> MutatingFn:
        _CHAIN.register(
            Hook(
                name=name,
                kinds=frozenset(kinds),
                operations=frozenset(operations),
                order=order,
                failure_policy=failure_policy,
                owner=owner,
                mutating=True,
                fn=fn,
                timeout_s=timeout_s,
            )
        )
        return fn

    return decorate


def validating(
    name: str,
    *,
    kinds: tuple[str, ...] = ("*",),
    operations: tuple[Operation, ...] = (),
    order: int = 100,
    failure_policy: FailurePolicy = FailurePolicy.FAIL,
    owner: str = "unknown",
    timeout_s: float = DEFAULT_HOOK_TIMEOUT_S,
) -> Callable[[ValidatingFn], ValidatingFn]:
    """Declare a validating hook. Return AdmissionResponse, bool, or a
    deny-reason string. Must not mutate the request object."""

    def decorate(fn: ValidatingFn) -> ValidatingFn:
        _CHAIN.register(
            Hook(
                name=name,
                kinds=frozenset(kinds),
                operations=frozenset(operations),
                order=order,
                failure_policy=failure_policy,
                owner=owner,
                mutating=False,
                fn=fn,
                timeout_s=timeout_s,
            )
        )
        return fn

    return decorate


def admit(
    kind: str,
    name: str,
    obj: Any,
    *,
    operation: Operation = Operation.CREATE,
    principal: str = "runtime",
    context: dict[str, Any] | None = None,
    old_obj: Any = None,
    copy_object: bool = False,
) -> AdmissionVerdict:
    """Run the chain. ``copy_object`` protects the caller's object from
    mutating hooks when the caller wants the original preserved."""
    payload = copy.deepcopy(obj) if copy_object else obj
    return _CHAIN.admit(
        AdmissionRequest(
            operation=operation,
            kind=kind,
            name=name,
            obj=payload,
            principal=principal,
            context=dict(context or {}),
            old_obj=old_obj,
        )
    )


def admission_report() -> dict[str, Any]:
    return _CHAIN.report()


def reset_admission_for_test() -> None:
    _CHAIN.reset_for_test()


__all__ = [
    "AdmissionChain",
    "AdmissionRequest",
    "AdmissionResponse",
    "AdmissionVerdict",
    "FailurePolicy",
    "Hook",
    "Operation",
    "admission_report",
    "admit",
    "get_admission_chain",
    "mutating",
    "reset_admission_for_test",
    "validating",
]
