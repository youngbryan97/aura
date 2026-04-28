# core/brain/execution.py
from core.runtime.errors import record_degradation
import asyncio
import time
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass

from core.brain.trace_logger import TraceLogger

@dataclass
class ExecResult:
    ok: bool
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict = None

class ExecutionManager:
    """
    Responsible for executing actions safely.
    - action_fn: Callable[[str], Any] or async
    - safety_check(action_name, context) -> bool (allow/deny)
    - supports timeouts, retries, and safe-mode gating for dangerous actions
    """

    def __init__(self, trace: TraceLogger, safe_mode: bool = True, dangerous_whitelist: Optional[set] = None):
        self.trace = trace
        self.safe_mode = safe_mode
        self.dangerous_whitelist = dangerous_whitelist or set()

    async def execute(self, action_name: str, action_fn: Callable[..., Any], context: str = "", timeout: float = 30.0, retries: int = 1, retry_delay: float = 1.0, allow_danger: bool = False, metadata: Optional[Dict] = None) -> ExecResult:
        if metadata is None:
            metadata = {}
        # safety gating
        if self.safe_mode and not allow_danger and action_name in self.dangerous_whitelist:
            msg = f"Action '{action_name}' denied by safe_mode"
            self.trace.log({"type": "execution_denied", "action": action_name, "reason": msg, "context": context[:200]})
            return ExecResult(ok=False, error=msg, duration=0.0, metadata=metadata)

        start = time.time()
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                # run either async or sync in executor
                if asyncio.iscoroutinefunction(action_fn):
                    coro = action_fn()
                else:
                    # wrap sync call
                    async def _wrap(): return action_fn()
                    coro = _wrap()
                res = await asyncio.wait_for(coro, timeout=timeout)
                dur = time.time() - start
                self.trace.log({"type": "execution", "action": action_name, "ok": True, "duration": dur, "attempt": attempt})
                return ExecResult(ok=True, result=res, duration=dur, metadata=metadata)
            except asyncio.TimeoutError:
                last_err = "timeout"
                self.trace.log({"type": "execution_timeout", "action": action_name, "attempt": attempt, "timeout": timeout})
            except Exception as e:
                record_degradation('execution', e)
                last_err = str(e)
                self.trace.log({"type": "execution_exception", "action": action_name, "attempt": attempt, "error": last_err})
            # retry backoff
            if attempt < retries:
                await asyncio.sleep(retry_delay)
        dur = time.time() - start
        return ExecResult(ok=False, error=last_err, duration=dur, metadata=metadata)
