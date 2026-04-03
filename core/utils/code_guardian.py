import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

try:
    from core.utils.aura_logging import core_logger
except ImportError:
    core_logger = logging.getLogger("Aura.Core")

logger = logging.getLogger("Aura.CodeGuardian")

@dataclass
class ValidationReport:
    success: bool
    ruff_output: Optional[str] = None
    mypy_output: Optional[str] = None
    error_message: Optional[str] = None

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
    def validate_code(cls, filepath: Path) -> ValidationReport:
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
            ruff_res = subprocess.run(
                [ruff_path, "check", str(filepath)], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if ruff_res.returncode != 0:
                logger.warning("❌ CodeGuardian: Ruff check failed for %s", filepath.name)
                return ValidationReport(
                    success=False, 
                    ruff_output=ruff_res.stdout + ruff_res.stderr,
                    error_message="Syntax or NameError detected by Ruff."
                )
        except subprocess.TimeoutExpired:
            return ValidationReport(success=False, error_message="Ruff check timed out.")
        except Exception as e:
            return ValidationReport(success=False, error_message=f"Ruff execution error ({ruff_path}): {e}")

        # 2. Run Mypy (Static Type Checker)
        mypy_path = cls._get_bin_path("mypy")
        logger.info("🛡️ CodeGuardian: Running %s on %s", mypy_path, filepath.name)
        try:
            # We use --ignore-missing-imports and --follow-imports=silent to avoid noise
            mypy_res = subprocess.run(
                [mypy_path, "--ignore-missing-imports", "--follow-imports=silent", str(filepath)], 
                capture_output=True, 
                text=True, 
                timeout=20
            )
            if mypy_res.returncode != 0:
                logger.warning("❌ CodeGuardian: Mypy check failed for %s", filepath.name)
                return ValidationReport(
                    success=False, 
                    mypy_output=mypy_res.stdout + mypy_res.stderr,
                    error_message="TypeError detected by Mypy."
                )
        except subprocess.TimeoutExpired:
            return ValidationReport(success=False, error_message="Mypy check timed out.")
        except Exception as e:
            return ValidationReport(success=False, error_message=f"Mypy execution error ({mypy_path}): {e}")

        logger.info("✅ CodeGuardian: %s passed all checks.", filepath.name)
        return ValidationReport(success=True)

if __name__ == "__main__":
    # Self-test logic
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        report = CodeGuardian.validate_code(Path(sys.argv[1]))
        print(report)
