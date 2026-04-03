import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

from core.container import ServiceContainer

logger = logging.getLogger("Skills.TestGen")

class TestGeneratorParams(BaseModel):
    __test__ = False
    target_file: str = Field(..., description="The path to the Python file or module to generate tests for.")

class TestGeneratorSkill(BaseSkill):
    __test__ = False
    """Test Generator Skill v2.0
    Generates unit tests using the brain and executes them to verify code integrity.
    """

    name = "test_generator"
    description = "Generates and runs unit tests for a specific file or module."
    input_model = TestGeneratorParams
    
    def __init__(self, brain=None):
        super().__init__()
        self.brain = brain
        
    async def execute(self, params: TestGeneratorParams, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test generation and execution.
        """
        # Legacy support
        if isinstance(params, dict):
            try:
                params = TestGeneratorParams(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        target_file = params.target_file
        
        # Issue 82: Lazy resolve brain
        brain = self.brain or ServiceContainer.get("cognitive_engine")
        if not brain:
            from core.brain.cognitive_engine import cognitive_engine
            brain = cognitive_engine
            
        if not brain:
            return {"ok": False, "error": "Cognitive engine unavailable for test generation."}
        
        if not target_file:
            return {"ok": False, "error": "No target_file provided."}
            
        target_path = Path(target_file)
        if not target_path.exists():
            return {"ok": False, "error": f"Target file {target_file} not found."}

        # 1. Read the target code for context
        try:
            with open(target_path, 'r') as f:
                code_content = f.read()
            
            # 2. Generate Test Code via CognitiveEngine
            prompt = f"""
            YOU ARE A QA ENGINEER FOR AN AGI.
            Generate a COMPREHENSIVE pytest-based unit test for the following Python file:
            File: {target_file}
            
            CODE:
            ```python
            {code_content}
            ```
            
            Respond ONLY with the Python code for the test. Focus on edge cases and functional correctness.
            """
            
            # Issue 82: Fix brain.think parameter 'objective' -> 'prompt' and 'mode' -> ThinkingMode
            from core.brain.cognitive_engine import ThinkingMode
            thought = await brain.think(
                prompt=prompt,
                context={"role": "qa_engineer", "target": target_file},
                mode=ThinkingMode.DEEP
            )
            
            test_code = thought.content
            # Cleanup code block if the LLM included it
            test_code = re.sub(r"```python\n|```", "", test_code).strip()
            
            # 3. Write to temporary test file if not exists or override
            test_file = target_path.parent / f"test_{target_path.name}"
            # For safety, we'll use a temporary file if we don't want to pollute the source
            # But for AGI evolution, we might want to keep the tests.
            
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            logger.info("✨ Tests generated and saved to %s", test_file)
            
            # 4. Run tests in Sandbox
            try:
                from core.sovereign.local_sandbox import LocalSandbox
                sandbox = LocalSandbox(sandbox_id=f"test_gen_{target_path.stem}")
                sandbox.start()
                
                # Write files to sandbox
                sandbox.write_file("target_code.py", code_content)
                sandbox.write_file("test_script.py", test_code)
                
                # Execute pytest in sandbox (non-blocking event loop)
                import asyncio
                res = await asyncio.to_thread(sandbox.run_command, "pytest test_script.py", timeout=45)
                
                # Cleanup (optional, but keep for result analysis)
                return {
                    "ok": res.exit_code == 0,
                    "output": res.stdout,
                    "error": res.stderr,
                    "sandbox_id": sandbox.id,
                    "test_file": str(test_file)
                }
            except Exception as sandbox_err:
                logger.error("Sandbox execution failed: %s", sandbox_err)
                return {"ok": False, "error": f"Sandbox error: {sandbox_err}"}
        except Exception as e:
            logger.error("Test generation/execution failed: %s", e)
            return {"ok": False, "error": str(e)}
