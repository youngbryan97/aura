"""core/grounding/observation_schema.py
==================================
Universal structured observation schema and specialized environmental subclasses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    """Universal structured representation of sensory input from a virtual environment."""
    source: str = ""
    event_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    severity: float = 0.0  # 0.0 (safe/informational) to 1.0 (highly critical/destructive)
    suggested_affordances: List[str] = field(default_factory=list)  # Action possibilities
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the observation to a dictionary."""
        return {
            "source": self.source,
            "event_type": self.event_type,
            "data": self.data,
            "severity": self.severity,
            "suggested_affordances": self.suggested_affordances,
            "timestamp": self.timestamp
        }


@dataclass
class FileObservation(Observation):
    """Specialized observation for file system activity."""
    path: str = ""
    operation: str = ""  # "read", "write", "delete", "create", "chmod"
    size_bytes: Optional[int] = None
    success: bool = True

    def __post_init__(self) -> None:
        self.source = "filesystem"
        self.event_type = f"file_{self.operation}"
        self.data.update({
            "path": self.path,
            "operation": self.operation,
            "size_bytes": self.size_bytes,
            "success": self.success
        })
        if not self.success:
            self.severity = max(self.severity, 0.5)
        if self.operation in ("write", "delete"):
            self.suggested_affordances.extend(["file_read", "git_diff", "run_test"])
        else:
            self.suggested_affordances.extend(["file_write", "file_delete"])


@dataclass
class ProcessObservation(Observation):
    """Specialized observation for terminal and subprocess shell execution."""
    command: str = ""
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    cpu_percent: Optional[float] = None
    memory_rss_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        self.source = "terminal"
        self.event_type = "command_executed"
        self.data.update({
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout_len": len(self.stdout),
            "stderr_len": len(self.stderr),
            "cpu_percent": self.cpu_percent,
            "memory_rss_bytes": self.memory_rss_bytes
        })
        if self.exit_code != 0:
            self.severity = max(self.severity, 0.4)
            self.suggested_affordances.extend(["diagnose_error", "run_test"])
        else:
            self.suggested_affordances.extend(["command_execute", "reflect"])


@dataclass
class UnitTestObservation(Observation):
    """Specialized observation for test suite running results."""
    test_suite: str = ""
    passed_count: int = 0
    failed_count: int = 0
    duration_seconds: float = 0.0
    failures: List[Dict[str, Any]] = field(default_factory=list)  # Elements: {"test_name": str, "message": str, "traceback": str}

    def __post_init__(self) -> None:
        self.source = "pytest"
        self.event_type = "test_run_complete"
        self.data.update({
            "test_suite": self.test_suite,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "duration_seconds": self.duration_seconds,
            "failures_count": len(self.failures)
        })
        if self.failed_count > 0:
            # Scale severity up with failed tests, capped at 0.9
            self.severity = min(0.9, 0.3 + 0.1 * self.failed_count)
            self.suggested_affordances.extend(["file_read", "patch_code", "run_test"])
        else:
            self.suggested_affordances.extend(["run_test", "commit_code"])
