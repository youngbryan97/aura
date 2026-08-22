"""Terminal execution motor: the body's general-execution actuator.

The docstring here said "Executes shell commands and processes safely" and
the module contained no safety at all — no allowlist, no isolation, and no
call to the execution gate every other general-execution surface goes
through. A motor named ``terminal`` that any planner can actuate with a
command string is the same reach as ``sovereign_terminal``, so it asks the
same question of the same authority and refuses the same way.
"""
import logging
import shlex
from subprocess import SubprocessError
from typing import Any, Dict

from core.body.motor_controller import BaseMotor
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.security.execution_authority import (
    KIND_SHELL,
    authorize_execution,
    release_execution,
)

logger = logging.getLogger("Body.TerminalMotor")

_TERMINAL_MOTOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class TerminalMotor(BaseMotor):
    """Actuator for running commands in the system shell."""

    @property
    def name(self) -> str:
        return "terminal"

    async def actuate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        command = params.get("argv") or params.get("command")
        cwd = params.get("cwd", ".")
        timeout = params.get("timeout", 5.0)

        if not command:
            return {"status": "error", "message": "Missing command string"}

        argv = [str(part) for part in command] if isinstance(command, (list, tuple)) else shlex.split(str(command))
        if not argv:
            return {"status": "error", "message": "Command produced no argv entries"}

        verdict = await authorize_execution(
            KIND_SHELL,
            argv,
            source="core.body.terminal_motor",
            cwd=str(cwd),
            extra={"actuator": self.name},
        )
        if not verdict.approved:
            return {"status": "error", "message": verdict.reason, "governance": verdict.receipt()}

        logger.info("Executing terminal command: %s (cwd=%s)", argv, cwd)

        succeeded = False
        failure = ""
        try:
            res = await get_subprocess_gateway().run_async(
                argv,
                cwd=cwd,
                timeout=timeout,
                source="body.terminal_motor",
                accelerator_capability="auto",
            )
            succeeded = res.returncode == 0
            return {
                "status": "success",
                "exit_code": res.returncode,
                "stdout": res.stdout[:5000],  # Truncate overly long outputs
                "stderr": res.stderr[:2000],
                "governance": verdict.receipt(),
            }
        except TimeoutError:
            failure = f"timed out after {timeout}s"
            return {"status": "timeout", "message": failure}
        except _TERMINAL_MOTOR_ERRORS as e:
            failure = str(e)
            record_degradation("body.terminal_motor", e)
            logger.error("Terminal motor execution failed: %s", e)
            return {"status": "error", "message": str(e)}
        finally:
            release_execution(
                verdict,
                source="core.body.terminal_motor",
                success=succeeded,
                error=failure,
            )
