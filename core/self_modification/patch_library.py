import logging
import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Optimizer.PatchLibrary")


def _flag_enabled(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _proposal_dir() -> Path:
    from core.utils.paths import aura_data_dir

    target = aura_data_dir() / "repair_proposals"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_repair_proposal(*, kind: str, failure_reason: str, command: list[str], rationale: str) -> Path:
    from core.runtime.file_write_gateway import get_file_write_gateway

    safe_kind = re.sub(r"[^a-zA-Z0-9_.-]+", "-", kind).strip("-") or "repair"
    target = _proposal_dir() / f"{safe_kind}-{abs(hash((failure_reason, tuple(command))))}.md"
    text = "\n".join(
        [
            "# Aura repair proposal",
            "",
            f"Kind: {kind}",
            f"Rationale: {rationale}",
            "",
            "Failure signature:",
            "```text",
            failure_reason[:4000],
            "```",
            "",
            "Proposed command:",
            "```bash",
            " ".join(command),
            "```",
            "",
            "Status: not executed automatically. Operator approval or explicit runtime flag required.",
            "",
        ]
    )
    get_file_write_gateway().write_text(
        target,
        text,
        source="maintenance_tooling:patch_library.repair_proposal",
    )
    return target


class PatchStrategy(ABC):
    name = "base_patch"

    def match(self, failure_reason: str) -> bool:
        return False

    @abstractmethod
    async def apply(self, failure_reason: str) -> bool:
        """Applies the fix. Returns True if successful, False otherwise."""
        raise NotImplementedError


class GitInitPatch(PatchStrategy):
    name = "git_init_fix"

    def match(self, failure_reason: str) -> bool:
        return "not a git repository" in failure_reason.lower()

    async def apply(self, failure_reason: str) -> bool:
        command = ["git", "init"]
        if not _flag_enabled("AURA_ALLOW_AUTONOMIC_GIT_REPAIR"):
            proposal = _write_repair_proposal(
                kind=self.name,
                failure_reason=failure_reason,
                command=command,
                rationale="Repository metadata is missing; automatic git repair is disabled by default.",
            )
            logger.warning("Git repair requires operator approval; proposal saved to %s", proposal)
            return False

        logger.warning("Autonomic Core engaging 'git init' self-repair...")
        try:
            await get_subprocess_gateway().run_async(
                command,
                check=True,
                capture_output=True,
                offline_tooling=True,
                source="maintenance_tooling:patch_library.git_init",
                accelerator_capability="auto",
            )
            logger.info("Autonomic Core initialized local Git metadata; commit/add remain operator-governed.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error("Git repair failed: %s", e.stderr.decode() if e.stderr else e)
            return False


class PipInstallPatch(PatchStrategy):
    name = "pip_install_fix"

    def match(self, failure_reason: str) -> bool:
        return "modulenotfounderror" in failure_reason.lower()

    async def apply(self, failure_reason: str) -> bool:
        # Extract module name matches "No module named 'xyz'"
        match = re.search(r"No module named ['\"]([A-Za-z0-9_]+)['\"]", failure_reason)
        if match:
            module = match.group(1)

            import_to_pip: dict[str, str] = {
                "aiohttp": "aiohttp",
                "google": "google-generativeai",
                "pydantic": "pydantic",
                "structlog": "structlog",
                "psutil": "psutil",
                "webrtcvad": "webrtcvad",
                "pyaudio": "PyAudio",
                "numpy": "numpy",
            }
            
            if module not in import_to_pip:
                logger.error("SECURITY: Blocked autonomous installation of '%s'", module)
                return False

            pip_package = import_to_pip[module]
            command = [sys.executable, "-m", "pip", "install", pip_package]
            if not _flag_enabled("AURA_ALLOW_AUTONOMIC_PIP_INSTALL"):
                proposal = _write_repair_proposal(
                    kind=f"{self.name}.{module}",
                    failure_reason=failure_reason,
                    command=command,
                    rationale="Missing dependency can be installed, but live runtime package mutation is disabled by default.",
                )
                logger.warning("Dependency repair requires operator approval; proposal saved to %s", proposal)
                return False

            logger.warning("Autonomic Core attempting to install missing module: %s (as %s)", module, pip_package)
            try:
                await get_subprocess_gateway().run_async(
                    command,
                    check=True,
                    capture_output=True,
                    offline_tooling=True,
                    source="maintenance_tooling:patch_library.pip_install",
                    accelerator_capability="auto",
                )
                logger.info("Autonomic Core successfully installed missing package '%s'", pip_package)
                return True
            except subprocess.CalledProcessError as e:
                logger.error("Autonomic Core failed to install '%s': %s", pip_package, e.stderr.decode() if e.stderr else e)
                return False
        return False


def get_patches() -> list[PatchStrategy]:
    return [GitInitPatch(), PipInstallPatch()]


AVAILABLE_PATCHES: list[PatchStrategy] = get_patches()


__all__ = [
    "AVAILABLE_PATCHES",
    "GitInitPatch",
    "PipInstallPatch",
    "PatchStrategy",
    "get_patches",
]
