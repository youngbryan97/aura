import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("Aura.Integrity")

_IMMUTABLE_RELATIVE = [
    "core/prime_directives.py",
    "prime_directives.py",
    "core/identity_base.txt",
    "core/identity_prompt.txt",
    "core/values_engine.py",
    "core/soul.py",
    "core/constitution.py",
    "core/config.py",
    "data/identity/identity_base.txt",
]

def _get_project_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "core").is_dir() and (
            (candidate / "aura_main.py").exists() or (candidate / "run_aura.py").exists()
        ):
            return candidate
        candidate = candidate.parent
    return Path.cwd()

class SafetyGate:
    """Manages code validation and rollback mechanisms."""

    _project_root: Path = None
    _immutable_resolved: Set[Path] = None

    @classmethod
    def _ensure_resolved(cls):
        if cls._immutable_resolved is not None:
            return
        root = _get_project_root()
        cls._project_root = root
        resolved = set()
        for rel in _IMMUTABLE_RELATIVE:
            full = (root / rel).resolve()
            resolved.add(full)
        cls._immutable_resolved = resolved
        logger.debug("SafetyGate protecting %d files under %s", len(resolved), root)

    @staticmethod
    def validate_code(code_string: str) -> Tuple[bool, str]:
        try:
            ast.parse(code_string)
            return True, "Valid"
        except SyntaxError as e:
            return False, f"{e.msg} line {e.lineno}"
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"

    @classmethod
    def is_allowed_file(cls, file_path_raw) -> bool:
        cls._ensure_resolved()
        try:
            resolved = Path(file_path_raw).resolve()
        except Exception:
            logger.critical("Access denied: cannot resolve path '%s'", file_path_raw)
            return False

        try:
            resolved.relative_to(cls._project_root)
        except ValueError:
            logger.critical("Access denied: '%s' is outside project root '%s'", resolved, cls._project_root)
            return False

        if resolved in cls._immutable_resolved:
            logger.critical("Access denied: '%s' is a protected immutable file.", resolved)
            return False
        return True

    @staticmethod
    def create_backup(file_path: Path) -> None:
        try:
            if file_path.exists():
                shutil.copy(file_path, file_path.with_suffix(".bak"))
        except IOError as e:
            logger.error("Backup failed for %s: %s", file_path, e)

    @staticmethod
    def rollback(file_path: Path) -> bool:
        bak_path = file_path.with_suffix(".bak")
        if bak_path.exists():
            logger.critical("Initiating rollback protocol...")
            try:
                shutil.copy(bak_path, file_path)
                logger.info("Restored from backup. Restarting process...")
                os.execv(sys.executable, ['python'] + sys.argv)
                return True
            except Exception as e:
                logger.critical("Fatal: Restart failed during rollback: %s", e)
                return False
        else:
            logger.critical("No backup found. Rollback impossible.")
            return False