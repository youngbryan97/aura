"""Diagnostic Hub: Universal Error Orchestration
Part of Aura's Neural Neuro-Surgeon (Phase 29).
"""

from core.runtime.errors import record_degradation
import logging
import json
import subprocess
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("Aura.Resilience.NeuroSurgeon")

class ErrorCategory(Enum):
    SYNTAX = "syntax"
    TYPE = "type"
    NAME = "name"
    ASYNC_STALL = "async_stall"
    OOM = "oom"
    LOGIC_TIMEOUT = "logic_timeout"
    UNKNOWN = "unknown"

class DiagnosticHub:
    """Orchestrates diagnostic tools to classify and reproduction system failures."""
    
    def __init__(self, code_base: Optional[Path] = None):
        from core.config import config
        self.code_base = code_base or config.paths.project_root
        self.tools = {
            "ruff": self._run_ruff,
            "pyright": self._run_pyright,
        }

    def classify_error(self, error_msg: str) -> ErrorCategory:
        """Categorize the error based on signature patterns."""
        error_msg = error_msg.lower()
        
        if "syntaxerror" in error_msg or "invalid syntax" in error_msg:
            return ErrorCategory.SYNTAX
        if "nameerror" in error_msg or "is not defined" in error_msg:
            return ErrorCategory.NAME
        if "typeerror" in error_msg or "incompatible type" in error_msg:
            return ErrorCategory.TYPE
        if "event loop is closed" in error_msg or "stall detected" in error_msg:
            return ErrorCategory.ASYNC_STALL
        if "memoryerror" in error_msg or "oom" in error_msg:
            return ErrorCategory.OOM
        if "timeout" in error_msg or "asyncio.timeouterror" in error_msg:
            return ErrorCategory.LOGIC_TIMEOUT
            
        return ErrorCategory.UNKNOWN

    async def run_deep_diagnostic(self, file_path: str) -> Dict[str, Any]:
        """Run multiple tools to find hidden issues in a specific file."""
        abs_path = self.code_base / file_path
        results = {
            "ruff": await self._run_ruff(abs_path),
            "pyright": await self._run_pyright(abs_path),
        }
        return results

    async def _run_ruff(self, target: Path) -> Dict[str, Any]:
        """Run Ruff linter and return issues."""
        try:
            cmd = ["ruff", "check", str(target), "--format", "json"]
            result = await asyncio.to_thread(lambda: subprocess.run(cmd, capture_output=True, text=True))
            if result.stdout:
                return {"ok": False, "issues": json.loads(result.stdout)}
            return {"ok": True, "issues": []}
        except Exception as e:
            record_degradation('diagnostic_hub', e)
            return {"ok": False, "error": str(e)}

    async def _run_pyright(self, target: Path) -> Dict[str, Any]:
        """Run Pyright type checker and return issues."""
        try:
            cmd = ["pyright", str(target), "--outputjson"]
            result = await asyncio.to_thread(lambda: subprocess.run(cmd, capture_output=True, text=True))
            if result.stdout:
                data = json.loads(result.stdout)
                return {"ok": False, "issues": data.get("generalDiagnostics", [])}
            return {"ok": True, "issues": []}
        except Exception as e:
            record_degradation('diagnostic_hub', e)
            return {"ok": False, "error": str(e)}

# Global Instance
_hub = None

def get_diagnostic_hub() -> DiagnosticHub:
    global _hub
    if _hub is None:
        _hub = DiagnosticHub()
    return _hub
