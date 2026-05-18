from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation
from core.tasks.managed_command import ManagedCommandResult, run_project_command

try:
    from core.utils.aura_logging import core_logger
except ImportError:
    core_logger = logging.getLogger("Aura.Core")

logger = logging.getLogger("Aura.CodeGuardian")
_CODE_GUARDIAN_RECOVERABLE_ERRORS = (
    FileNotFoundError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
CommandRunner = Callable[[tuple[str, ...], float], ManagedCommandResult]


@dataclass
class ValidationReport:
    success: bool
    ruff_output: str | None = None
    mypy_output: str | None = None
    error_message: str | None = None

class CodeGuardian:
    """
    Aura's QA Sandbox. Runs Ruff and Mypy on generated Python code.
    Prevents NameError and TypeError from reaching production.
    """
    
    @staticmethod
    def _get_bin_path(tool_name: str) -> str:
        """Finds the absolute path to a tool, preferring the active venv if available."""
        import sys
        venv_bin = Path(sys.prefix) / "bin" / tool_name
        if venv_bin.exists():
            return str(venv_bin)
        return tool_name

    @classmethod
    def validate_code(cls, filepath: Path, command_runner: CommandRunner | None = None) -> ValidationReport:
        """
        Runs a full QA battery on the given file.
        Returns a ValidationReport.
        """
        if not filepath.exists():
            return ValidationReport(success=False, error_message=f"File not found: {filepath}")

        # 1. Run Ruff (Linter for Syntax and NameErrors)
        ruff_path = cls._get_bin_path("ruff")
        logger.info("🛡️ CodeGuardian: Running %s on %s", ruff_path, filepath.name)
        try:
            ruff_res = cls._run_command((ruff_path, "check", str(filepath)), 10.0, command_runner)
            if ruff_res.timed_out:
                return ValidationReport(success=False, error_message="Ruff check timed out.")
            if not ruff_res.ok:
                logger.warning("❌ CodeGuardian: Ruff check failed for %s", filepath.name)
                return ValidationReport(
                    success=False,
                    ruff_output=ruff_res.stdout + ruff_res.stderr,
                    error_message="Syntax or NameError detected by Ruff.",
                )
        except _CODE_GUARDIAN_RECOVERABLE_ERRORS as e:
            record_degradation('code_guardian', e)
            return ValidationReport(success=False, error_message=f"Ruff execution error ({ruff_path}): {e}")

        # 2. Run Mypy (Static Type Checker)
        mypy_path = cls._get_bin_path("mypy")
        logger.info("🛡️ CodeGuardian: Running %s on %s", mypy_path, filepath.name)
        try:
            mypy_res = cls._run_command(
                (mypy_path, "--ignore-missing-imports", "--follow-imports=silent", str(filepath)),
                20.0,
                command_runner,
            )
            if mypy_res.timed_out:
                return ValidationReport(success=False, error_message="Mypy check timed out.")
            if not mypy_res.ok:
                logger.warning("❌ CodeGuardian: Mypy check failed for %s", filepath.name)
                return ValidationReport(
                    success=False,
                    mypy_output=mypy_res.stdout + mypy_res.stderr,
                    error_message="TypeError detected by Mypy.",
                )
        except _CODE_GUARDIAN_RECOVERABLE_ERRORS as e:
            record_degradation('code_guardian', e)
            return ValidationReport(success=False, error_message=f"Mypy execution error ({mypy_path}): {e}")

        logger.info("✅ CodeGuardian: %s passed all checks.", filepath.name)
        return ValidationReport(success=True)

    @staticmethod
    def _run_command(
        command: tuple[str, ...],
        timeout_s: float,
        command_runner: CommandRunner | None,
    ) -> ManagedCommandResult:
        runner = command_runner or (lambda cmd, limit: run_project_command(cmd, timeout_s=limit))
        return runner(command, timeout_s)


if __name__ == "__main__":
    # Self-test logic
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        report = CodeGuardian.validate_code(Path(sys.argv[1]))
        print(report)
