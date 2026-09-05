import logging
# core/brain/execution.py
import asyncio
import hashlib
import inspect
import json
import math
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.brain.trace_logger import TraceLogger
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.Brain.Execution")

_MAX_RETRIES = 20
_SENSITIVE_META_MARKERS = ("secret", "password", "passwd", "token", "key", "credential", "auth")

#: Name shapes that are dangerous regardless of whether anyone listed them.
#: CP126 c0dd3c01: safe mode denied ONLY names present in a caller-supplied
#: ``dangerous_whitelist``, so an unknown or newly added destructive action
#: sailed through the one gate that exists.
_DANGEROUS_NAME_PATTERNS = (
    re.compile(r"\b(delete|destroy|drop|purge|wipe|erase|truncate|format)\b", re.I),
    re.compile(r"\b(rm|rmdir|unlink|shred|mkfs|dd)\b", re.I),
    re.compile(r"\b(kill|terminate|shutdown|reboot|halt)\b", re.I),
    re.compile(r"\b(deploy|publish|push|release|migrate|rollback)\b", re.I),
    re.compile(r"\b(send|email|post|tweet|transfer|pay|purchase|charge)\b", re.I),
    re.compile(r"\b(grant|revoke|chmod|chown|sudo|escalate)\b", re.I),
    re.compile(r"\b(write|overwrite|patch|modify|mutate)\b", re.I),
    re.compile(r"\b(exec|eval|spawn|subprocess|shell)\b", re.I),
)


def is_dangerous_action(action_name: str, dangerous_whitelist: set | None = None) -> bool:
    """Whether this action needs explicit danger authorization.

    Union of the caller's declared set and the structural patterns above, so a
    name nobody remembered to list still has to be authorized.

    The name is tokenized on separators and camelCase first: ``\b`` does not
    fire between "delete" and "_all", so a raw word-boundary match would have
    missed ``delete_all_records`` — exactly the unlisted-name case that
    CP126 c0dd3c01 is about.
    """
    name = str(action_name or "")
    if name in (dangerous_whitelist or set()):
        return True
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    spaced = re.sub(r"[^A-Za-z0-9]+", " ", spaced)
    return any(pattern.search(spaced) for pattern in _DANGEROUS_NAME_PATTERNS)


def _danger_authorized(
    action_name: str, allow_danger: Any, capability_token: str
) -> tuple[bool, str]:
    """Whether a dangerous action is genuinely authorized.

    CP126 575d878c: ``allow_danger`` was an unauthenticated caller boolean —
    any caller could clear the only safety gate with no principal, signed
    scope, standing-authority lease, gesture or receipt. A bare True is no
    longer sufficient; it must be accompanied by a capability token that
    validates for this action.
    """
    if not allow_danger:
        return False, "allow_danger not set"
    token = str(capability_token or "").strip()
    if not token:
        return False, "allow_danger carried no capability token"
    try:
        from core.agency.capability_token import get_token_store

        get_token_store().validate(token, domain="tool_execution", action=action_name)
    except (PermissionError, ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return False, f"capability token rejected: {exc}"
    return True, "capability token validated"


def _finite(value: Any, *, default: float, low: float, high: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return max(low, min(high, num))


def _copy_meta(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Copy caller metadata (so we never return their mutable dict by reference)
    and redact secret-bearing keys before it can enter traces/results."""
    out: dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        if isinstance(k, str) and any(m in k.lower() for m in _SENSITIVE_META_MARKERS):
            out[k] = "***redacted***"
        else:
            out[k] = v
    return out

_EXECUTION_RECOVERABLE_ERRORS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _record_execution_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        "execution",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


@dataclass
class ExecResult:
    ok: bool
    result: Any = None
    error: str | None = None
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionManager:
    """
    Responsible for executing actions safely.
    - action_fn: Callable[[str], Any] or async
    - safety_check(action_name, context) -> bool (allow/deny)
    - supports timeouts, retries, and safe-mode gating for dangerous actions
    """

    def __init__(
        self,
        trace: TraceLogger,
        safe_mode: bool = True,
        dangerous_whitelist: set | None = None,
        safety_check: Callable[[str, str], bool] | None = None,
    ):
        self.trace = trace
        self.safe_mode = safe_mode
        self.dangerous_whitelist = dangerous_whitelist or set()
        # CP126 c0dd3c01: the class contract documented a `safety_check`
        # callback that did not exist anywhere in the file.
        self.safety_check = safety_check
        #: Effect receipts by idempotency key, so a retry can prove whether the
        #: previous attempt already landed (CP126 3c626bb4).
        self._effects: dict[str, dict[str, Any]] = {}

    def _make_receipt(
        self,
        *,
        action_name: str,
        context: str,
        metadata: dict[str, Any],
        operation_id: str,
        idempotency_key: str,
        attempt: int,
        duration: float,
        result: Any,
    ) -> dict[str, Any]:
        """A state-mutation receipt for a completed action (CP126 aae07b1a)."""
        body = {
            "action": action_name,
            "principal": str(metadata.get("principal") or "unattributed"),
            "target": str(metadata.get("target") or ""),
            "authority": metadata.get("danger_authorization", "not_required"),
            "context": context[:200],
            "operation_id": operation_id,
            "idempotency_key": idempotency_key,
            "attempt": attempt,
            "duration_s": round(duration, 6),
            "result_sha256": _result_digest(result),
            "at": time.time(),
        }
        body["receipt_id"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:32]
        return body

    async def execute(
        self,
        action_name: str,
        action_fn: Callable[..., Any],
        context: str = "",
        timeout_seconds: float = 30.0,
        retries: int = 1,
        retry_delay: float = 1.0,
        allow_danger: bool = False,
        metadata: dict[str, Any] | None = None,
        success_predicate: Callable[[Any], bool] | None = None,
        **legacy_kwargs: Any,
    ) -> ExecResult:
        if "timeout" in legacy_kwargs:
            timeout_seconds = float(legacy_kwargs.pop("timeout"))
        if legacy_kwargs:
            raise TypeError(f"Unsupported execution options: {sorted(legacy_kwargs)}")
        # Secret redaction runs on the COPY that reaches traces and results,
        # but authorization and idempotency need the real values — and both
        # of those key names match the sensitive markers ("token", "key"), so
        # they must be read before the copy is redacted.
        raw_metadata = dict(metadata or {})
        supplied_token = str(raw_metadata.get("capability_token") or "")
        supplied_idempotency_key = str(raw_metadata.get("idempotency_key") or "")
        metadata = _copy_meta(metadata)
        # NaN would slip past a bare `<= 0` check; validate finiteness first.
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Execution timeout must be a positive, finite number.")
        # retries is an ATTEMPT count (min 1); cap it so a huge value cannot
        # monopolize the lane.
        try:
            retries = max(1, min(_MAX_RETRIES, int(retries)))
        except (TypeError, ValueError):
            retries = 1
        retry_delay = _finite(retry_delay, default=1.0, low=0.0, high=60.0)
        operation_id = uuid.uuid4().hex
        metadata.setdefault("operation_id", operation_id)
        # CP126 3c626bb4: an idempotency key lets a retry prove whether the
        # previous attempt already produced its effect.
        idempotency_key = supplied_idempotency_key or f"{action_name}:{operation_id}"
        # The redacted copy keeps a redacted marker; the real key stays local.
        metadata.setdefault("idempotency_key", "***redacted***" if supplied_idempotency_key else idempotency_key)
        idempotent = bool(metadata.get("idempotent", False))
        prior = self._effects.get(idempotency_key)
        if prior is not None and prior.get("state") == "succeeded":
            self.trace.log({
                "type": "execution_deduplicated", "action": action_name,
                "idempotency_key": idempotency_key, "operation_id": operation_id,
            })
            metadata["deduplicated"] = True
            return ExecResult(
                ok=True, result=prior.get("result"), duration=0.0, metadata=metadata
            )

        # safety gating — the caller's callback runs first and can deny
        # anything, then the structural danger check applies.
        if self.safety_check is not None:
            try:
                permitted = bool(self.safety_check(action_name, context))
            except Exception as exc:  # noqa: BLE001 - an unusable check denies
                _record_execution_degradation(
                    exc,
                    action="denied an action because its safety check could not run",
                    extra={"action": action_name},
                )
                permitted = False
            if not permitted:
                msg = f"Action '{action_name}' denied by safety_check"
                self.trace.log({
                    "type": "execution_denied", "action": action_name,
                    "reason": msg, "context": context[:200],
                })
                return ExecResult(ok=False, error=msg, duration=0.0, metadata=metadata)

        if self.safe_mode and is_dangerous_action(action_name, self.dangerous_whitelist):
            authorized, why = _danger_authorized(action_name, allow_danger, supplied_token)
            if not authorized:
                msg = f"Action '{action_name}' denied by safe_mode ({why})"
                self.trace.log({
                    "type": "execution_denied", "action": action_name,
                    "reason": msg, "context": context[:200],
                })
                return ExecResult(ok=False, error=msg, duration=0.0, metadata=metadata)
            metadata["danger_authorization"] = why

        # Monotonic clock: a wall-clock jump must not make a duration negative.
        start = time.monotonic()
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                res = await asyncio.wait_for(_invoke_action(action_fn), timeout=timeout_seconds)
                dur = time.monotonic() - start
                # A returned object is not automatically a success — honor an
                # explicit success predicate and treat obvious failure envelopes
                # (False/None, or a dict carrying ok=False / an error) as failures.
                if not _looks_successful(res, success_predicate):
                    last_err = _failure_reason(res)
                    self.trace.log({
                        "type": "execution_unsuccessful_result", "action": action_name,
                        "attempt": attempt, "operation_id": operation_id,
                    })
                    if attempt < retries:
                        await asyncio.sleep(retry_delay)
                        continue
                    return ExecResult(ok=False, result=res, error=last_err, duration=dur, metadata=metadata)
                # CP126 aae07b1a: trace logging is not an action receipt. This
                # binds principal, target, authority, inputs and a result hash.
                receipt = self._make_receipt(
                    action_name=action_name,
                    context=context,
                    metadata=metadata,
                    operation_id=operation_id,
                    idempotency_key=idempotency_key,
                    attempt=attempt,
                    duration=dur,
                    result=res,
                )
                self._effects[idempotency_key] = {
                    "state": "succeeded", "result": res, "receipt": receipt,
                }
                metadata["receipt"] = receipt
                self.trace.log({
                    "type": "execution", "action": action_name, "ok": True,
                    "duration": dur, "attempt": attempt, "operation_id": operation_id,
                    "receipt_id": receipt["receipt_id"],
                })
                return ExecResult(ok=True, result=res, duration=dur, metadata=metadata)
            except TimeoutError:
                # The wait_for cancels our await, but a sync callable already
                # running in a worker thread cannot be forced to stop — mark the
                # outcome uncertain so a retry is not assumed side-effect-free.
                last_err = "timeout"
                metadata["outcome"] = "uncertain_timeout"
                self._effects[idempotency_key] = {"state": "uncertain", "result": None}
                self.trace.log({
                    "type": "execution_timeout", "action": action_name, "attempt": attempt,
                    "timeout": timeout_seconds, "operation_id": operation_id,
                })
                # CP126 6e82df46 / 3c626bb4: wait_for cancels our await but
                # cannot stop a sync callable already running in a worker
                # thread. Retrying would run the effect a second time while the
                # first is still in flight, so an action that has not declared
                # itself idempotent stops here with an honest uncertain result.
                if not idempotent:
                    metadata["retry_suppressed"] = "non_idempotent_after_uncertain_timeout"
                    self.trace.log({
                        "type": "execution_retry_suppressed", "action": action_name,
                        "reason": "non_idempotent_after_uncertain_timeout",
                        "operation_id": operation_id,
                    })
                    return ExecResult(
                        ok=False, error="timeout_outcome_uncertain",
                        duration=time.monotonic() - start, metadata=metadata,
                    )
            except asyncio.CancelledError:
                self.trace.log(
                    {"type": "execution_cancelled", "action": action_name, "attempt": attempt}
                )
                raise
            except _EXECUTION_RECOVERABLE_ERRORS as e:
                _record_execution_degradation(
                    e,
                    action="returned failed execution result after action callable failed",
                    extra={"action": action_name, "attempt": attempt},
                )
                last_err = str(e)
                self.trace.log({
                    "type": "execution_exception", "action": action_name,
                    "attempt": attempt, "error": last_err, "operation_id": operation_id,
                })
            except Exception as e:  # noqa: BLE001 — unexpected faults become a receipt, never escape mid-effect
                _record_execution_degradation(
                    e,
                    action="converted an unexpected execution fault into a failed result receipt",
                    severity="degraded",
                    extra={"action": action_name, "attempt": attempt, "operation_id": operation_id},
                )
                last_err = f"unexpected:{type(e).__name__}:{e}"
                self.trace.log({
                    "type": "execution_unexpected_error", "action": action_name,
                    "attempt": attempt, "error": last_err[:300], "operation_id": operation_id,
                })
            # retry backoff
            if attempt < retries:
                await asyncio.sleep(retry_delay)
        dur = time.monotonic() - start
        return ExecResult(ok=False, error=last_err, duration=dur, metadata=metadata)


def _result_digest(result: Any) -> str:
    """A stable digest of the result, for binding into the receipt."""
    try:
        payload = json.dumps(result, sort_keys=True, default=str)
    except (TypeError, ValueError, RecursionError):
        payload = repr(result)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _looks_successful(res: Any, predicate: Callable[[Any], bool] | None) -> bool:
    """Judge whether a returned object represents success.

    With a caller ``predicate`` the caller decides. By default we reject only an
    explicit failure envelope (a dict with ok=False or an error) so a void
    (None) result from an ordinary side-effecting action stays successful.
    """
    if predicate is not None:
        try:
            return bool(predicate(res))
        except Exception as exc:  # noqa: BLE001 — a predicate fault is a failed judgment
            # False is the right answer: an unevaluable predicate has not
            # judged the result successful. But a predicate that CRASHES and a
            # predicate that returned False are different events, and only one
            # of them needs fixing.
            try:
                from core.runtime.errors import record_degradation

                record_degradation(
                    "execution_predicate",
                    exc,
                    severity="warning",
                    action="treated an unevaluable success predicate as not satisfied",
                )
            except Exception:  # noqa: BLE001
                logger.debug("Success predicate raised and was unrecorded: %s", exc)
            return False
    if isinstance(res, dict):
        if res.get("ok") is False:
            return False
        if res.get("error") and res.get("ok") is not True:
            return False
    return True


def _failure_reason(res: Any) -> str:
    if isinstance(res, dict):
        return str(res.get("error") or res.get("reason") or "unsuccessful_result")[:200]
    return "unsuccessful_result"


async def _invoke_action(action_fn: Callable[..., Any]) -> Any:
    if inspect.iscoroutinefunction(action_fn):
        return await action_fn()
    result = await asyncio.to_thread(action_fn)
    if inspect.isawaitable(result):
        return await result
    return result
