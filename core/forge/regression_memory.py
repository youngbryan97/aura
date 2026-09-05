"""core/forge/regression_memory.py — Regression Memory."""
from __future__ import annotations

import logging

logger = logging.getLogger("Aura.RegressionMemory")


class RegressionMemory:
    """Remembers past patch failures and bugs to avoid duplicate regression patterns."""

    def __init__(self) -> None:
        self.failed_patches: list[dict[str, str]] = []

    def record_failure(self, module: str, patch_hash: str, error_details: str) -> None:
        """Saves a failed patch fingerprint to regression memory."""
        self.failed_patches.append({
            "module": module,
            "patch_hash": patch_hash,
            "error_details": error_details,
        })
        logger.info("Recorded failed patch fingerprint for module %s in regression memory", module)

    def is_known_failure(self, code: str) -> bool:
        """Checks if the proposed code matches a previously failed code pattern."""
        import hashlib
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        for item in self.failed_patches:
            if item["patch_hash"] == code_hash:
                logger.warning("⚠️ Proposed patch matches a known historically failed patch hash!")
                return True
        return False


_memory_instance: RegressionMemory | None = None


def get_regression_memory() -> RegressionMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = RegressionMemory()
    return _memory_instance
