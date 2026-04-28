from core.runtime.errors import record_degradation
import asyncio
import asyncio.subprocess
import logging
import sys
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

from core.skills.base_skill import BaseSkill

# Prevent basic unsafe operations inside the sandbox
SECURITY_PREAMBLE = """
import sys
import builtins
import os

try:
    import resource
    # Cap Memory at 512 MB
    MAX_MEM = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEM, MAX_MEM))
    # Cap CPU time at 10 seconds
    MAX_CPU = 10
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU, MAX_CPU))
except ImportError:
    pass # Windows fallback

# Block dangerous builtins
_forbidden_builtins = ['eval', 'exec', 'open']
for b in _forbidden_builtins:
    if hasattr(builtins, b):
        setattr(builtins, b, None)

# Disable os.system and subprocess
if hasattr(os, 'system'):
    os.system = None
if hasattr(os, 'popen'):
    os.popen = None
    
import subprocess
subprocess.Popen = None
subprocess.run = None
"""

from pydantic import BaseModel, Field

logger = logging.getLogger("Skills.InternalSandbox")

class SandboxInput(BaseModel):
    code: Optional[str] = Field(None, description="Python code to execute immediately.")
    notes: Optional[str] = Field(None, description="Text to store in temporary scratchpad.")

class SandboxSkill(BaseSkill):
    name = "internal_sandbox"
    description = "An invisible scratchpad/terminal to test Python code or write notes purely for internal thought processing. Data here is ephemeral."
    input_model = SandboxInput
    
    # Safety limits
    MAX_EXECUTION_TIME = 30  # seconds
    MAX_OUTPUT_SIZE = 10000  # characters

    def __init__(self):
        self.scratchpad = ""

    async def execute(self, params: SandboxInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sandboxed code or notes."""
        if isinstance(params, dict):
            try:
                params = SandboxInput(**params)
            except Exception as e:
                record_degradation('internal_sandbox', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        code = params.code
        notes = params.notes
        
        if notes:
            self.scratchpad += f"\n--- note ---\n{notes}\n"
            return {"ok": True, "summary": "Notes added to internal scratchpad."}

        if code:
            return await self.execute_code_safely(code)
                
        return {"ok": True, "result": self.scratchpad, "summary": "Viewed scratchpad."}

    async def execute_code_safely(self, code: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute code in a subprocess with timeout (Async).
        A security preamble blocks dangerous operations before exec.
        
        Can also be used directly by the SelfModification engine.
        """
        cwd = cwd or tempfile.gettempdir()
        
        try:
            # Prepend security preamble to user code
            sandboxed_code = SECURITY_PREAMBLE + "\n" + code

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(sandboxed_code)
                temp_path = f.name
            
            try:
                # Run in subprocess with timeout
                process = await asyncio.create_subprocess_exec(
                    sys.executable, temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd
                )
                
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=self.MAX_EXECUTION_TIME)
                    stdout = stdout_b.decode()[:self.MAX_OUTPUT_SIZE] if stdout_b else ""
                    stderr = stderr_b.decode()[:self.MAX_OUTPUT_SIZE] if stderr_b else ""
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except Exception as e:
                        record_degradation('internal_sandbox', e)
                        logger.debug("Failed to kill sandboxed process: %s", e)
                    return {"ok": False, "error": f"Code execution timed out after {self.MAX_EXECUTION_TIME}s"}

                output = f"Stdout:\n{stdout}"
                if stderr:
                    output += f"\nStderr:\n{stderr}"
                
                if process.returncode != 0:
                    return {
                        "ok": False, 
                        "error": f"Code exited with code {process.returncode}",
                        "result": output,
                        "summary": "Code execution failed."
                    }
                    
                return {"ok": True, "result": output, "summary": "Code executed in sandbox."}
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    record_degradation('internal_sandbox', e)
                    logger.debug("Failed to delete temp sandbox file %s: %s", temp_path, e)

        except Exception as e:
            record_degradation('internal_sandbox', e)
            return {"ok": False, "error": f"Sandbox Exception: {e}"}