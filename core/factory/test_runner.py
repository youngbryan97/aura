"""core/factory/test_runner.py — Test Suite Executor.

Executes tests in localized sub-environments, capturing stdout/stderr,
pass/fail counts, and timing.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.TestRunner")


class TestRunner:
    """Executes project test suites and captures structured results."""

    async def run_tests(
        self,
        repo_path: str,
        *,
        test_command: str = "python -m pytest --tb=short -q",
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Run the test suite and return structured results."""
        logger.info("🧪 TestRunner: executing tests in %s", repo_path)
        started = time.time()

        try:
            # Execute command via approved subprocess gateway to pass linter
            proc = get_subprocess_gateway().run(
                argv=test_command.split(),
                cwd=repo_path,
                timeout=timeout,
                source="test_runner",
                accelerator_capability="auto",
            )
            duration = time.time() - started
            stdout = proc.stdout if proc.stdout else ""
            stderr = proc.stderr if proc.stderr else ""

            # Parse pytest output for pass/fail counts
            passed = 0
            failed = 0
            for line in stdout.splitlines():
                if "passed" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "passed" and i > 0:
                            try:
                                passed = int(parts[i - 1])
                            except ValueError:
                                pass
                        if p == "failed" and i > 0:
                            try:
                                failed = int(parts[i - 1])
                            except ValueError:
                                pass

            return {
                "all_passed": proc.returncode == 0,
                "return_code": proc.returncode,
                "passed": passed,
                "failed": failed,
                "duration_s": round(duration, 2),
                "summary": stdout.splitlines()[-1] if stdout.strip() else "no output",
                "stderr_tail": stderr[-500:],
            }

        except TimeoutError:
            return {"all_passed": False, "error": "timeout", "duration_s": timeout}
        except (OSError, RuntimeError) as e:
            record_degradation("test_runner", e, action="test execution failed")
            return {"all_passed": False, "error": str(e), "duration_s": time.time() - started}
