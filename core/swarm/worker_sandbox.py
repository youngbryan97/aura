"""core/swarm/worker_sandbox.py — Swarm Sandboxed Environment.

Enforces absolute isolation of task executions to safeguard external actuation.
"""
from __future__ import annotations

import ast
import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger("Aura.WorkerSandbox")

class WorkerSandbox:
    """Restricts directory access and execution scope for workers."""

    def __init__(self, allowed_directory: str) -> None:
        self.allowed_directory = os.path.abspath(allowed_directory)

    def is_safe_path(self, target_path: str) -> bool:
        """Validate if a file path stays strictly inside the sandbox directory."""
        abs_target = os.path.abspath(target_path)
        return abs_target.startswith(self.allowed_directory)

    def execute_code_sandboxed(self, code_str: str, globals_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python snippets in a child interpreter, not the live process."""
        logger.warning("Executing sandboxed python payload in isolated child process")
        try:
            json.dumps(globals_dict)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"sandbox globals must be JSON-serializable: {exc}"}

        try:
            from core.sandbox.runner import run_untrusted

            preload = "\n".join(
                f"{name} = {json.dumps(value)}" for name, value in globals_dict.items()
            )
            result_marker = "__AURA_WORKER_RESULT__"
            result_capture = (
                "\ntry:\n"
                f"    print({result_marker!r} + repr(result))\n"
                "except NameError:\n"
                "    pass\n"
            )
            payload = f"{preload}\n{code_str}{result_capture}" if preload else f"{code_str}{result_capture}"
            result = run_untrusted(payload, timeout=5)
            if result.get("status") != "ok":
                return {
                    "ok": False,
                    "error": str(result.get("stderr") or result.get("repr") or result.get("status")),
                    "details": result,
                }
            stdout = str(result.get("stdout", ""))
            worker_result: Any = None
            for line in stdout.splitlines():
                if line.startswith(result_marker):
                    try:
                        worker_result = ast.literal_eval(line[len(result_marker) :])
                    except (SyntaxError, ValueError):
                        worker_result = line[len(result_marker) :]
            return {"ok": True, "result": worker_result, "stdout": stdout}
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error("Sandbox execution failed: %s", e)
            return {"ok": False, "error": str(e)}
