"""Base Skill — v41 Full Realization

All Aura skills inherit from this class. It provides:
  - Automatic timeout enforcement (default 30s, configurable per-skill)
  - Standardized result format (ok, error, duration_ms, skill_name)
  - Execution logging with duration and success/failure tracking
  - Graceful degradation on unhandled exceptions
  - Input validation via Pydantic models
  - Metabolic cost tagging for resource management
"""

import asyncio
import builtins
import contextvars
import functools
import inspect
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel

from core.exceptions import ContainerError
from core.runtime.errors import record_degradation

logger = logging.getLogger("Skills")

_SKILL_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TimeoutError,
    OSError,
    LookupError,
    ContainerError,
    sqlite3.Error,
    TypeError,
    ValueError,
)

#: Where a caller that has already sized THIS request puts the budget it
#: negotiated. The capability engine asks a skill's ``timeout_for`` what a
#: particular objective will cost; without somewhere to put the answer, the
#: engine held the larger number and the skill went on enforcing its flat
#: declared default, which is smaller by construction and therefore always
#: fired first. Live 2026-07-29: a research task sized at 405s was killed at
#: 180s by the class attribute and reported as "Completed 0/0 steps".
SKILL_TIMEOUT_CONTEXT_KEY = "_skill_timeout_s"


#: True while `safe_execute` is running its own `self.execute(...)` call, so
#: the guard below knows the governance check has already happened and does not
#: repeat it. A ContextVar rather than an instance flag: one skill instance is
#: shared across concurrent tasks, and an attribute would leak one task's
#: exemption into another's raw call.
_INSIDE_SAFE_EXECUTE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aura_skill_inside_safe_execute", default=False
)


def _record_skill_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    """Record degradation inside base skill execution."""
    record_degradation("base_skill", exc, severity=severity, action=action)


def _raw_execute_refusal(skill: "BaseSkill") -> dict[str, Any] | None:
    """Governance verdict for a direct `execute()` call, or None to proceed.

    `safe_execute` was documented as "the PUBLIC entry point that the skill
    router should call", and the active-runtime governance requirement lived
    only inside it. Nothing made it the only door. `DesktopPlanner`'s adapter
    called the computer-use skill's `execute()` directly, and so did
    `core/tools/computer_use.py` and two capability modules — so the claim that
    no consequential action runs outside the governed lane was false for every
    one of those paths.

    Moving the check into `execute` itself is what makes the claim structural
    rather than conventional: there is no longer a raw door to find. A call
    from inside `safe_execute` is exempt because that wrapper has already run
    exactly this check.
    """
    if _INSIDE_SAFE_EXECUTE.get():
        return None
    try:
        from core.governance_context import (
            GovernanceViolation,
            governance_runtime_active,
            require_governance,
        )

        require_governance(
            f"skill:{skill.name}",
            strict=governance_runtime_active(),
            allowed_domains=("tool_execution",),
        )
    except GovernanceViolation as exc:
        logger.warning(
            "🛡️ Ungoverned direct execute() on skill '%s' refused: %s", skill.name, exc
        )
        return {
            "ok": False,
            "skill": skill.name,
            "error": f"Ungoverned skill execution blocked: {exc}",
            "summary": "blocked: skill executed outside the governed lane",
            "duration_ms": 0,
        }
    except (ImportError, AttributeError, RuntimeError):
        # Governance is not booted. Same posture `safe_execute` takes: during
        # boot there is no runtime to be ungoverned against.
        return None
    return None


def _governed_execute(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a subclass `execute` so a direct call is governed like a routed one.

    Idempotent by marker, because hot-reload re-runs `__init_subclass__` against
    a class whose `execute` may already be wrapped, and a second layer would
    check governance twice and log twice.
    """
    if getattr(func, "__aura_governed_execute__", False):
        return func

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(self: "BaseSkill", *args: Any, **kwargs: Any) -> Any:
            refusal = _raw_execute_refusal(self)
            if refusal is not None:
                return refusal
            return await func(self, *args, **kwargs)

        wrapper: Callable[..., Any] = async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(self: "BaseSkill", *args: Any, **kwargs: Any) -> Any:
            refusal = _raw_execute_refusal(self)
            if refusal is not None:
                return refusal
            return func(self, *args, **kwargs)

        wrapper = sync_wrapper

    wrapper.__aura_governed_execute__ = True  # type: ignore[attr-defined]
    #: Kept so tests and the raw-execute ratchet can reach the unwrapped body.
    wrapper.__aura_unwrapped_execute__ = func  # type: ignore[attr-defined]
    return wrapper



def _infer_ok_flag(result: dict[str, Any]) -> bool:
    # Negative evidence is authoritative. A contradictory payload such as
    # {"ok": True, "success": False} cannot be normalized into success.
    # An *empty* error field ("" / [] / None) is the absence of an error, not
    # evidence of one — several skills populate error=stderr, which is "" on a
    # clean run (e.g. test_generator: pytest passed, empty stderr). Only a
    # truthy error/errors payload is negative evidence.
    if result.get("error") or result.get("errors"):
        return False
    if result.get("failed") is True:
        return False
    # A skill that reports its own outcome with a 'success' key (11 skills do)
    # was being marked ok=True on {"success": False} — a dishonest success
    # (e.g. local_reference returning "corpus empty" as a successful lookup).
    if result.get("success") is False:
        return False
    if str(result.get("status", "")).lower() in {"blocked", "error", "failed"}:
        return False
    if "ok" in result:
        return bool(result["ok"])
    return True


class SkillResult(BaseModel):
    """Standardized result from any skill execution."""
    ok: bool
    skill: str = ""
    summary: str = ""
    error: str | None = None
    data: dict[str, Any] | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values for cleanliness."""
        return dict(self.model_dump(exclude_none=True))


class BaseSkill(ABC):
    """Base class for all Aura skills.

    Provides automatic timeout enforcement, error handling, input validation,
    and standardized result formatting. Subclasses implement `run()` with
    their specific logic; the `safe_execute()` wrapper handles the rest.

    Class Attributes:
        name: Unique identifier for this skill.
        description: Human-readable description for LLM tool routing.
        input_model: Optional Pydantic model for input validation.
        timeout_seconds: Maximum execution time (default 30s).
        metabolic_cost: 0=Core, 1=Light, 2=Medium, 3=Heavy.
        is_core_personality: True if this skill defines personality traits.
        requires_approval: True if destructive actions need user confirmation.
    """

    name: str = "base_skill"
    description: str = "Base skill description"

    # Input validation
    input_model: type[BaseModel] | None = None

    # Execution limits
    timeout_seconds: float = 30.0

    # Metabolic Tagging (0=Core, 1=Light, 2=Medium, 3=Heavy)
    metabolic_cost: int = 1
    is_core_personality: bool = False
    requires_approval: bool = False

    # Retry safety is explicit opt-in. safe_execute retries transient failures
    # up to 3x, which is correct only for operations proven idempotent or
    # read-only. An unclassified skill must execute at most once: a skill that
    # completes an external effect and then times out reading the response
    # cannot safely be replayed.
    retry_safe: bool = False

    # Execution stats (instance-level)
    _total_executions: int = 0
    _total_failures: int = 0

    # Error types considered transient (safe to retry)
    _TRANSIENT_EXCEPTIONS: tuple[type, ...] = (
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        ConnectionRefusedError,
        OSError,  # Covers ENETUNREACH, ECONNRESET, etc.
    )

    # Error types considered permanent (do NOT retry)
    _PERMANENT_EXCEPTIONS: tuple[type, ...] = (
        FileNotFoundError,
        PermissionError,
        ValueError,
        TypeError,
        KeyError,
        builtins.NotImplementedError,
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Make every subclass's `execute` govern itself.

        Wrapping here rather than asking call sites to use `safe_execute` is
        the difference between a convention and an invariant: a skill written
        next year, or reached through a path nobody enumerated, is governed
        because it is a skill, not because its caller remembered.
        """
        super().__init_subclass__(**kwargs)
        own_execute = cls.__dict__.get("execute")
        if own_execute is None or getattr(own_execute, "__isabstractmethod__", False):
            return
        if not callable(own_execute):
            return
        cls.execute = _governed_execute(own_execute)  # type: ignore[assignment]

    def _ensure_stats_initialized(self) -> None:
        if "_total_executions" not in self.__dict__:
            self._total_executions = 0
        if "_total_failures" not in self.__dict__:
            self._total_failures = 0

    def _classify_error(self, exc: Exception) -> str:
        """Classify an exception as 'transient' or 'permanent'.

        Transient errors (timeouts, network blips) are safe to retry.
        Permanent errors (bad input, missing files) should fail immediately.
        Unknown errors default to 'transient' to give the retry loop a chance.
        """
        # Check httpx exceptions dynamically to avoid hard import dependency
        exc_module = type(exc).__module__ or ""
        exc_name = type(exc).__name__ or ""
        if "httpx" in exc_module and "Timeout" in exc_name:
            return "transient"

        if isinstance(exc, self._PERMANENT_EXCEPTIONS):
            return "permanent"
        if isinstance(exc, self._TRANSIENT_EXCEPTIONS):
            return "transient"

        # Heuristic: check the error message for transient-sounding keywords
        err_lower = str(exc).lower()
        transient_markers = ("timeout", "timed out", "connection", "network", "retry", "rate limit", "429", "503")
        if any(marker in err_lower for marker in transient_markers):
            return "transient"

        return "transient"  # Default: give the retry loop a chance

    @abstractmethod
    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the skill's core logic.

        Subclasses MUST implement this method. It should focus purely on
        the skill's business logic — error handling, timeouts, and logging
        are handled by `safe_execute()`.

        Args:
            params: Validated input (Pydantic model or dict).
            context: Execution context (agent state, user info, etc.).

        Returns:
            Result dict. Must include 'ok' (bool). May include 'summary',
            'error', 'content', or any skill-specific keys.
        """
        pass  # no-op: intentional

    def _effective_timeout_seconds(self, context: dict[str, Any] | None) -> float:
        """The budget for THIS request, falling back to the declared default.

        A caller that sized the request (capability engine, via ``timeout_for``)
        publishes its number under SKILL_TIMEOUT_CONTEXT_KEY. Only a larger
        budget is honoured: the declared value is a floor the skill author
        chose, and a caller must not quietly shorten it here — callers that
        need to constrain a skill do it with their own outer wait.
        """
        declared = float(self.timeout_seconds)
        if not isinstance(context, dict):
            return declared
        try:
            negotiated = float(context.get(SKILL_TIMEOUT_CONTEXT_KEY) or 0.0)
        except (TypeError, ValueError):
            return declared
        return max(declared, negotiated)

    async def safe_execute(
        self,
        params: Any,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute with governance, timeout, error handling, and standardized results.

        This is the PUBLIC entry point that the skill router should call.
        It wraps `execute()` with:
          - Governance verification (Will receipt check)
          - Input validation (if input_model is defined)
          - Timeout enforcement
          - Exception catching with graceful error messages
          - Duration tracking
          - Execution stats

        Args:
            params: Raw input (dict or Pydantic model).
            context: Optional execution context.

        Returns:
            Standardized result dict with 'ok', 'skill', 'summary',
            'duration_ms', and optionally 'error'.
        """
        context = context or {}
        start = time.monotonic()
        self._ensure_stats_initialized()
        self._total_executions += 1

        # ── GOVERNANCE CHECK ─────────────────────────────────────────
        # Verify this skill execution is governed (has a Will receipt
        # somewhere in the call stack). Log violations but don't block
        # during early boot or testing.
        try:
            from core.governance_context import (
                GovernanceViolation,
                governance_runtime_active,
                require_governance,
            )

            require_governance(
                f"skill:{self.name}",
                strict=governance_runtime_active(),
                allowed_domains=("tool_execution",),
            )
        except GovernanceViolation as e:
            self._total_failures += 1
            return self._error_result(
                f"Ungoverned skill execution blocked: {e}",
                time.monotonic() - start
            )
        except (ImportError, AttributeError, RuntimeError):
            pass  # governance not booted yet

        # Input validation
        if self.input_model and isinstance(params, dict):
            try:
                params = self.input_model(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_skill_degradation(e, action="returned invalid input error result to skill execution caller", severity="error")
                return self._error_result(
                    f"Invalid input: {e}",
                    time.monotonic() - start
                )

        from infrastructure.resilience import _get_or_create_breaker
        breaker = _get_or_create_breaker(self.name)
        if not hasattr(breaker, "allow_request"):
            # Fallback if breaker API changes
            pass  # no-op: intentional
        elif not breaker.allow_request():
            self._total_failures += 1
            return self._error_result(
                f"Circuit tripped: {self.name} is currently failing consistently.",
                time.monotonic() - start
            )

        # Only retry when it is safe to re-run this skill's side effects.
        # requires_approval implies a destructive/consequential action that
        # must never double-fire; retry_safe is the explicit opt-out for
        # non-destructive-but-side-effectful skills (send/post/notify).
        retryable = bool(self.retry_safe) and not bool(self.requires_approval)
        max_attempts = 3 if retryable else 1
        base_delay = 1.0
        result: Any = None
        last_err: Exception = RuntimeError("skill execution did not start")
        effective_timeout = self._effective_timeout_seconds(context)

        for attempt in range(max_attempts):
            # The governance check above covers this call, so the guard on
            # `execute` must not run it a second time. Reset in `finally`
            # rather than left set, or a skill that calls another skill's raw
            # execute would inherit this exemption.
            inside_token = _INSIDE_SAFE_EXECUTE.set(True)
            # Whether the budget below is what ran out.
            #
            # `asyncio.timeout` raises TimeoutError when it expires, and so
            # does every bounded wait INSIDE the skill. Catching the type and
            # reporting "skill timed out" makes those two indistinguishable,
            # and they send a reader to different clocks. Live 2026-08-30:
            # os_automation declares 90s, failed at 35.6s, and the log said it
            # timed out — so the search went to the skill's budget, the
            # engine's sizing and the executive constraints, none of which had
            # anything to do with it. The context manager knows which it was.
            own_budget = None
            try:
                # Execute with timeout
                async with asyncio.timeout(effective_timeout) as own_budget:
                    if inspect.iscoroutinefunction(self.execute):
                        result = await self.execute(params, context)
                    else:
                        sync_execute = cast(
                            Callable[[Any, dict[str, Any]], Any],
                            self.execute,
                        )
                        # to_thread copies the current context, so the flag
                        # reaches the worker thread and the guard sees it.
                        result = await asyncio.to_thread(sync_execute, params, context)
                        if inspect.isawaitable(result):
                            result = await result
                if hasattr(breaker, "record_success"):
                    breaker.record_success()
                break  # Success! Exit loop

            except asyncio.CancelledError:
                raise
            except TimeoutError as e:
                error_class = "transient"
                last_err = e
                mine = bool(getattr(own_budget, "expired", lambda: False)())
                if mine:
                    logger.warning(
                        "⏱️ Skill '%s' ran out its own %.1fs budget (attempt %d/%d)",
                        self.name,
                        effective_timeout,
                        attempt + 1,
                        max_attempts,
                    )
                else:
                    from core.runtime.errors import _raise_site

                    logger.warning(
                        "⏱️ Skill '%s' stopped on a timeout INSIDE it after %.1fs "
                        "of its %.1fs budget, raised at %s (attempt %d/%d)",
                        self.name,
                        time.monotonic() - start,
                        effective_timeout,
                        _raise_site(e),
                        attempt + 1,
                        max_attempts,
                    )

            except PermissionError as e:
                error_class = "permanent"
                last_err = e
                logger.warning("🔒 Skill '%s' permission denied: %s", self.name, e)

            except _SKILL_RECOVERABLE_ERRORS as e:
                _record_skill_degradation(e, action="retried or failed skill execution after database/os/runtime error")
                error_class = self._classify_error(e)
                last_err = e
                if error_class == "permanent":
                    logger.error("💥 Skill '%s' crashed with permanent error (attempt %d/%d): %s", self.name, attempt + 1, max_attempts, e, exc_info=attempt==max_attempts-1)
                else:
                    logger.warning("⚠️ Skill '%s' encountered transient error (attempt %d/%d): %s", self.name, attempt + 1, max_attempts, e)

            finally:
                _INSIDE_SAFE_EXECUTE.reset(inside_token)

            if error_class == "permanent" or attempt == max_attempts - 1:
                self._total_failures += 1
                return self._error_result(
                    f"Skill error: {type(last_err).__name__}: {last_err}",
                    time.monotonic() - start
                )
            # Sleep with exponential backoff on transient
            await asyncio.sleep(base_delay * (2 ** attempt))

        # Standardize the result
        duration_ms = (time.monotonic() - start) * 1000

        normalized_result: dict[str, Any]
        if isinstance(result, dict):
            normalized_result = dict(result)
        else:
            normalized_result = {"ok": True, "content": str(result)}

        # Inject standard fields
        normalized_result["ok"] = _infer_ok_flag(normalized_result)
        normalized_result["skill"] = self.name
        normalized_result["duration_ms"] = round(duration_ms, 1)
        deferred = str(normalized_result.get("status", "") or "").lower() == "deferred"
        # A failure with no stated cause is undiagnosable and un-learnable.
        # The live soak recorded `Task web_search failed:` with an EMPTY
        # message, and the surprise engine banked
        # `{'status': 'failed', 'error': ''}` — a maximal-surprise learning
        # signal carrying zero information about what to do differently.
        # Every skill failure names its cause or says plainly that the skill
        # returned none; silence is never an acceptable failure surface.
        if not normalized_result["ok"] and not deferred:
            stated_cause = str(normalized_result.get("error", "") or "").strip()
            if not stated_cause:
                status_hint = str(normalized_result.get("status", "") or "").strip()
                reason_hint = str(normalized_result.get("reason", "") or "").strip()
                normalized_result["error"] = (
                    f"{self.name} reported failure without a cause"
                    + (f" (status={status_hint})" if status_hint else "")
                    + (f" (reason={reason_hint})" if reason_hint else "")
                )

        if deferred:
            logger.info(
                "Skill '%s' deferred in %.0fms: %s",
                self.name,
                duration_ms,
                normalized_result.get("reason", "policy_deferred"),
            )
        elif normalized_result.get("ok"):
            logger.info(
                "✅ Skill '%s' completed in %.0fms",
                self.name, duration_ms
            )
        else:
            self._total_failures += 1
            logger.warning(
                "⚠️ Skill '%s' returned error in %.0fms: %s",
                self.name, duration_ms, normalized_result.get("error", "unknown")
            )

        return normalized_result

    def _error_result(self, error: str, elapsed: float) -> dict[str, Any]:
        """Build a standardized error result. The cause is never empty."""
        stated = str(error or "").strip()
        return {
            "ok": False,
            "skill": self.name,
            "error": stated or f"{self.name} failed without reporting a cause",
            "duration_ms": round(elapsed * 1000, 1)
        }

    def get_schema(self) -> dict[str, Any]:
        """Generate JSON schema for the skill's input parameters."""
        if self.input_model:
            return dict(self.input_model.model_json_schema())
        return {}

    def get_stats(self) -> dict[str, Any]:
        """Return execution statistics for this skill."""
        self._ensure_stats_initialized()
        return {
            "name": self.name,
            "executions": self._total_executions,
            "failures": self._total_failures,
            "success_rate": (
                round(1 - self._total_failures / max(1, self._total_executions), 3)
            ),
            "metabolic_cost": self.metabolic_cost
        }

    def match(self, goal: dict[str, Any]) -> bool:
        """Check if this skill can handle the given goal.

        Default implementation returns False. Skills that want to be
        auto-matched by the skill router should override this.
        """
        return False

    def __repr__(self) -> str:
        return f"<Skill:{self.name} cost={self.metabolic_cost}>"
