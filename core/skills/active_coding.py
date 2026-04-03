"""Active Coding Skill
Allows Aura to write and execute code in a sandbox to solve problems, 
analyze data, or test hypotheses.
"""
import logging
from typing import Any, Dict

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

from ..sovereign.local_sandbox import LocalSandbox

# Singleton Sandbox for this session
_sandbox = None

def get_sandbox():
    global _sandbox
    if not _sandbox:
        _sandbox = LocalSandbox("aura_main")
        _sandbox.start()
    return _sandbox

logger = logging.getLogger("Skills.RunCode")

class RunCodeParams(BaseModel):
    code: str = Field(..., description="Python code to execute.")

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
                return {"ok": False, "error": f"Invalid input: {e}"}
                
        code = params.code
        
        if not code:
            return {"ok": False, "error": "No code provided"}
            
        try:
            sandbox = get_sandbox()
            import asyncio
            result = await asyncio.to_thread(sandbox.run_code, code)
            
            return {
                "ok": result.exit_code == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}