# core/brain/execution.py
import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.brain.trace_logger import TraceLogger
from core.runtime.errors import FallbackClassification, record_degradation

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
        self, trace: TraceLogger, safe_mode: bool = True, dangerous_whitelist: set | None = None
    ):
        self.trace = trace
        self.safe_mode = safe_mode
        self.dangerous_whitelist = dangerous_whitelist or set()

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
        **legacy_kwargs: Any,
    ) -> ExecResult:
        if "timeout" in legacy_kwargs:
            timeout_seconds = float(legacy_kwargs.pop("timeout"))
        if legacy_kwargs:
            raise TypeError(f"Unsupported execution options: {sorted(legacy_kwargs)}")
        if metadata is None:
            metadata = {}
        if timeout_seconds <= 0:
            raise ValueError("Execution timeout must be positive.")
        retries = max(1, int(retries))
        # safety gating
        if self.safe_mode and not allow_danger and action_name in self.dangerous_whitelist:
            msg = f"Action '{action_name}' denied by safe_mode"
            self.trace.log(
                {
                    "type": "execution_denied",
                    "action": action_name,
                    "reason": msg,
                    "context": context[:200],
                }
            )
            return ExecResult(ok=False, error=msg, duration=0.0, metadata=metadata)

        start = time.time()
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                res = await asyncio.wait_for(_invoke_action(action_fn), timeout=timeout_seconds)
                dur = time.time() - start
                self.trace.log(
                    {
                        "type": "execution",
                        "action": action_name,
                        "ok": True,
                        "duration": dur,
                        "attempt": attempt,
                    }
                )
                return ExecResult(ok=True, result=res, duration=dur, metadata=metadata)
            except TimeoutError:
                last_err = "timeout"
                self.trace.log(
                    {
                        "type": "execution_timeout",
                        "action": action_name,
                        "attempt": attempt,
                        "timeout": timeout_seconds,
                    }
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
                self.trace.log(
                    {
                        "type": "execution_exception",
                        "action": action_name,
                        "attempt": attempt,
                        "error": last_err,
                    }
                )
            # retry backoff
            if attempt < retries:
                await asyncio.sleep(retry_delay)
        dur = time.time() - start
        return ExecResult(ok=False, error=last_err, duration=dur, metadata=metadata)


async def _invoke_action(action_fn: Callable[..., Any]) -> Any:
    if inspect.iscoroutinefunction(action_fn):
        return await action_fn()
    result = await asyncio.to_thread(action_fn)
    if inspect.isawaitable(result):
        return await result
    return result
