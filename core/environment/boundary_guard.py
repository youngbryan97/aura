"""Runtime information boundary enforcement.

Prevents contamination from forbidden channels:
- Save file inspection
- Process memory / RNG seed access
- Oracle map / hidden state
- Prior trace leakage across eval splits

This is environment-agnostic. Adapters register their forbidden paths/patterns.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.BoundaryGuard")


@dataclass
class BoundaryViolationError(Exception):
    """Raised when an operation attempts to breach the environment information boundary."""
    operation: str
    reason: str


FORBIDDEN_OBSERVATION_KEYS = frozenset({
    "seed", "rng_state", "full_map", "hidden_map", "hidden_items",
    "monster_internals", "oracle_state", "save_path", "process_memory",
})


@dataclass
class BoundaryConfig:
    """Configures what is forbidden for a given environment."""
    forbidden_file_patterns: list[str] = field(default_factory=list)
    forbidden_env_vars: list[str] = field(default_factory=list)
    forbidden_observation_keys: frozenset[str] = FORBIDDEN_OBSERVATION_KEYS
    trace_split: str = "train"  # "train", "dev", "eval"


@dataclass
class IntegrityReport:
    """Summary of boundary integrity for a run."""
    run_id: str
    mode: str
    verdict: str = "CLEAN"  # CLEAN or CONTAMINATED
    violations: list[dict[str, str]] = field(default_factory=list)
    contamination_count: int = 0
    simulated: bool = False
    adapter_integrity_hash: str = ""
    source_commit: str = ""
    policy_version: str = ""


class BoundaryGuard:
    """Intercepts and blocks forbidden operations (e.g. cheating, inspecting internal memory)."""

    def __init__(self, config: BoundaryConfig | None = None):
        self.log = logging.getLogger(__name__)
        self.config = config or BoundaryConfig()
        # Legacy compat
        self.allowed_channels = {"stdout", "stderr", "display", "audio"}
        self.blocked_operations = {
            "read_save_file",
            "inspect_memory",
            "oracle_metadata",
            "inject_state",
        }
        self.violations: list[dict[str, str]] = []
        self.contaminated = False

    def check_operation(self, operation: str, channel: str | None = None) -> None:
        """Validates if an operation is permitted across the boundary."""
        if operation in self.blocked_operations:
            self._record_violation(operation, "Operation is explicitly blocked by boundary guard.")
            raise BoundaryViolationError(operation, "Operation is explicitly blocked by boundary guard.")

        if channel and channel not in self.allowed_channels:
            self._record_violation(operation, f"Channel '{channel}' is not permitted.")
            raise BoundaryViolationError(operation, f"Channel '{channel}' is not permitted.")

    def check_file_access(self, path: str | Path) -> None:
        """Raises BoundaryViolationError if path matches a forbidden pattern."""
        path_str = str(path)
        for pattern in self.config.forbidden_file_patterns:
            if pattern in path_str:
                self._record_violation("save_file_inspection", f"path={path_str}")
                raise BoundaryViolationError("save_file_inspection", f"path={path_str}")

    def check_observation_metadata(self, metadata: dict[str, Any]) -> None:
        """Ensures observation metadata contains no oracle fields."""
        for key in metadata:
            if key in self.config.forbidden_observation_keys:
                self._record_violation("oracle_field_in_observation", f"key={key}")
                raise BoundaryViolationError("oracle_field_in_observation", f"key={key}")

    def check_env_access(self, var_name: str) -> None:
        """Prevents reading forbidden environment variables (e.g., RNG seeds)."""
        if var_name in self.config.forbidden_env_vars:
            self._record_violation("forbidden_env_var", f"var={var_name}")
            raise BoundaryViolationError("forbidden_env_var", f"var={var_name}")

    def check_trace_split(self, trace_id: str, requested_split: str) -> None:
        """Prevents eval traces from being used during training."""
        if requested_split == "eval" and self.config.trace_split != "eval":
            self._record_violation("eval_trace_leakage", f"trace={trace_id}")
            raise BoundaryViolationError("eval_trace_leakage", f"trace={trace_id}")

    def check_process_memory_access(self) -> None:
        """Prevents direct process memory inspection."""
        self._record_violation("process_memory_access", "")
        raise BoundaryViolationError("process_memory_access", "Blocked")

    def _record_violation(self, channel: str, detail: str) -> None:
        self.violations.append({"channel": channel, "detail": detail})
        self.contaminated = True
        logger.warning(f"BoundaryViolation: {channel} {detail}")

    def get_integrity_report(self, run_id: str, mode: str) -> IntegrityReport:
        """Generate integrity report for the run."""
        return IntegrityReport(
            run_id=run_id,
            mode=mode,
            verdict="CONTAMINATED" if self.contaminated else "CLEAN",
            violations=list(self.violations),
            contamination_count=len(self.violations),
        )

    def enforce_strict_real(self, mode: str) -> None:
        """Validates that mode is strict_real for real benchmark claims."""
        if mode != "strict_real":
            raise ValueError(f"Benchmark claims require strict_real mode, got {mode}")


__all__ = [
    "BoundaryViolationError", "BoundaryConfig", "BoundaryGuard",
    "IntegrityReport", "FORBIDDEN_OBSERVATION_KEYS",
]

