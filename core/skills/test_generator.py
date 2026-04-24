import json
import logging
import os
import re
import sys
import tempfile
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

    def _resolve_brain(self):
        brain = self.brain or ServiceContainer.get("cognitive_engine", default=None)
        if brain is not None:
            self.brain = brain
        return brain

    @staticmethod
    def _fallback_test_code(target_path: Path) -> str:
        module_name = target_path.stem
        return (
            "import importlib.util\n"
            "from pathlib import Path\n\n"
            f"TARGET = Path(__file__).resolve().parent / '{target_path.name}'\n\n"
            "def test_target_compiles_and_loads():\n"
            "    source = TARGET.read_text(encoding='utf-8')\n"
            f"    code = compile(source, '{target_path.name}', 'exec')\n"
            "    namespace = {'__name__': '__test_target__'}\n"
            "    exec(code, namespace)\n"
            f"    assert namespace is not None\n"
            f"    assert isinstance('{module_name}', str)\n"
        )
        
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
        read_only = bool((context or {}).get("read_only"))
        
        # Issue 82: Lazy resolve brain
        brain = self._resolve_brain()
        
        if not target_file:
            return {"ok": False, "error": "No target_file provided."}
            
        target_path = Path(target_file)
        if not target_path.exists():
            return {"ok": False, "error": f"Target file {target_file} not found."}

        # 1. Read the target code for context
        try:
            with open(target_path, 'r') as f:
                code_content = f.read()
            
            # 2. Generate Test Code via CognitiveEngine, or fall back to a deterministic smoke test.
            test_code = ""
            if brain:
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
                try:
                    from core.brain.cognitive_engine import ThinkingMode

                    thought = await brain.think(
                        objective=prompt,
                        context={"role": "qa_engineer", "target": target_file},
                        origin="test_generator",
                        mode=ThinkingMode.DEEP
                    )
                    test_code = getattr(thought, "content", str(thought or ""))
                    test_code = re.sub(r"```python\n|```", "", test_code).strip()
                except Exception as e:
                    logger.warning("LLM-based test generation unavailable for %s: %s", target_file, e)

            if not test_code:
                test_code = self._fallback_test_code(target_path)
            
            # 3. Write to temporary test file if not exists or override
            if read_only:
                temp_dir = Path(tempfile.mkdtemp(prefix="aura_test_generator_"))
                test_file = temp_dir / f"test_{target_path.name}"
            else:
                test_file = target_path.parent / f"test_{target_path.name}"
            
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            logger.info("✨ Tests generated and saved to %s", test_file)
            
            # 4. Run tests in Sandbox
            try:
                from core.sovereign.local_sandbox import LocalSandbox
                sandbox = LocalSandbox()
                sandbox.start()
                
                # Write files to sandbox
                sandbox.write_file(target_path.name, code_content)
                sandbox.write_file(test_file.name, test_code)
                
                # Execute pytest in sandbox (non-blocking event loop)
                command = f"{sys.executable} -m pytest -q {test_file.name}"
                res = await sandbox.run_command(command, timeout=45)
                sandbox.stop()
                
                # Cleanup (optional, but keep for result analysis)
                return {
                    "ok": res.exit_code == 0,
                    "output": res.stdout,
                    "error": res.stderr,
                    "test_file": str(test_file)
                }
            except Exception as sandbox_err:
                logger.error("Sandbox execution failed: %s", sandbox_err)
                return {"ok": False, "error": f"Sandbox error: {sandbox_err}"}
        except Exception as e:
            logger.error("Test generation/execution failed: %s", e)
            return {"ok": False, "error": str(e)}
