"""Active Coding Skill
Allows Aura to write and execute code in a sandbox to solve problems, 
analyze data, or test hypotheses.
"""
from core.runtime.errors import record_degradation
import logging
from pathlib import Path
from typing import Any, Dict

from core.config import config
from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

from ..sovereign.local_sandbox import LocalSandbox

# Singleton Sandbox for this session
_sandbox = None


def _sandbox_work_dir() -> str:
    base_dir = Path(getattr(config.paths, "base_dir", "."))
    return str(base_dir / ".aura_runtime" / "active_coding")


def get_sandbox():
    global _sandbox
    if not _sandbox or not getattr(_sandbox, 'is_alive', lambda: True)():
        if _sandbox and hasattr(_sandbox, 'stop'):
            try:
                _sandbox.stop()
            except Exception:
                pass  # no-op: intentional
        _sandbox = LocalSandbox(_sandbox_work_dir())
        _sandbox.start()
    return _sandbox

logger = logging.getLogger("Skills.RunCode")

class RunCodeParams(BaseModel):
    code: str = Field(..., description="Python code to execute.")
    stateful: bool = Field(True, description="Keep variables and functions in memory for the next run.")

class RunCodeSkill(BaseSkill):
    name = "run_code"
    description = "Executes Python code in a secure sandbox. Use for calculation, data processing, or testing."
    input_model = RunCodeParams
    
    def __init__(self):
        super().__init__()

        
    async def execute(self, params: RunCodeParams, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute python code.
        """
        # Legacy support
        if isinstance(params, dict):
            try:
                params = RunCodeParams(**params)
            except Exception as e:
                record_degradation('active_coding', e)
                return {"ok": False, "error": f"Invalid input: {e}"}
                
        code = params.code
        
        if not code:
            return {"ok": False, "error": "No code provided"}
            
        try:
            sandbox = get_sandbox()
            import asyncio
            if params.stateful:
                result = await sandbox.run_stateful_code(code)
            else:
                # Need to use an async wrapper for run_code natively if it is async
                result = await sandbox.run_code(code)
            
            # If the output is massive, truncate it smartly
            out_str = result.stdout
            err_str = result.stderr
            if len(out_str) > 5000:
                out_str = out_str[:2500] + "\n... [TRUNCATED] ...\n" + out_str[-2500:]
            if len(err_str) > 5000:
                err_str = err_str[:2500] + "\n... [TRUNCATED] ...\n" + err_str[-2500:]
            
            return {
                "ok": result.exit_code == 0,
                "stdout": out_str,
                "stderr": err_str,
                "exit_code": result.exit_code,
                "stateful": params.stateful,
                "summary": self._build_summary(out_str, err_str, result.exit_code, params.stateful),
            }
        except Exception as e:
            record_degradation('active_coding', e)
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _build_summary(stdout: str, stderr: str, exit_code: int, stateful: bool) -> str:
        signal = ""
        for candidate in (stderr, stdout):
            for raw_line in str(candidate or "").splitlines():
                line = " ".join(raw_line.split())
                if line:
                    signal = line
                    break
            if signal:
                break
        mode = "stateful" if stateful else "stateless"
        status = "ok" if exit_code == 0 else "failed"
        summary = f"python snippet ({mode}) -> {status}"
        if signal:
            summary = f"{summary} ({signal[:140]})"
        return summary[:220]
