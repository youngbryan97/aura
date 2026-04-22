"""Auto-Refactor Skill
Periodically scans the codebase for bad patterns, high complexity, or TODOs.
"""
import asyncio
import ast
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.AutoRefactor")

class AutoRefactorParams(BaseModel):
    path: str = Field(".", description="The directory path to scan for code issues.")
    run_tests: bool = Field(False, description="Whether to run pytest dynamically inside the sandbox to verify the codebase.")

class AutoRefactorSkill(BaseSkill):
    name = "auto_refactor"
    description = "Scans codebase for complexity and initiates self-improvement proposals."
    input_model = AutoRefactorParams
    
    def __init__(self):
        super().__init__()
        self.root_path = Path(".")
        
    async def execute(self, params: AutoRefactorParams, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the auto-refactor scan."""
        # Legacy support
        if isinstance(params, dict):
            try:
                params = AutoRefactorParams(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        target_path = params.path
        
        # 1. Scan for Weaknesses
        report = await asyncio.to_thread(self._scan_codebase, target_path)
        
        # 2. Limit to top 3 issues for reporting
        top_issues = report[:3]
        
        # 3. Dynamic Compilation Context (SWE-agent emulation)
        test_results = None
        if params.run_tests:
            try:
                from core.skills.active_coding import get_sandbox
                sandbox = get_sandbox()
                # Run tests in the ephemeral sandbox rather than local shell
                pytest_res = await sandbox.run_command(f"python3 -m pytest {target_path}")
                test_results = {
                    "ok": pytest_res.exit_code == 0,
                    "stdout": pytest_res.stdout[-2000:], # keep tail
                    "stderr": pytest_res.stderr[-2000:]
                }
            except Exception as e:
                logger.warning("Dynamic test execution failed: %s", e)
                test_results = {"ok": False, "error": str(e)}
        
        # 4. Autonomous Refactoring (v14.5 Stage: Reporting Only)
        # Publish to EventBus for Orchestrator review
        self._publish_proposals(top_issues)
                
        return {
            "ok": True,
            "issues_found": len(report),
            "top_issues": top_issues,
            "test_results": test_results,
            "message": f"Scanned {target_path}. Found {len(report)} issues." + (" Tests passed." if test_results and test_results.get("ok") else "")
        }

    def _scan_codebase(self, path_str: str) -> List[Dict[str, Any]]:
        """Static analysis for complexity."""
        issues = []
        path = Path(path_str)
        
        for file_path in path.rglob("*.py"):
            if "venv" in str(file_path): continue
            
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                # Check 1: Long Functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        length = node.end_lineno - node.lineno
                        if length > 50:
                            issues.append({
                                "file": str(file_path),
                                "line": node.lineno,
                                "type": "complexity",
                                "message": f"Function '{node.name}' is too long ({length} lines)."
                            })
                            
                # Check 2: TODOs
                for i, line in enumerate(content.splitlines(), 1):
                    if "TODO" in line:
                        issues.append({
                            "file": str(file_path),
                            "line": i,
                            "type": "todo",
                            "message": f"Found TODO: {line.strip()}"
                        })
                        
            except Exception as e:
                logger.warning("Failed to scan %s: %s", file_path, e)
                
        return sorted(issues, key=lambda x: 0 if x['type'] == 'complexity' else 1)

    def _publish_proposals(self, issues: List[Dict[str, Any]]):
        """Publishes refactor proposals to the EventBus."""
        from core.event_bus import get_event_bus
        bus = get_event_bus()
        for issue in issues:
            try:
                bus.publish_threadsafe(
                    "refactor_proposal",
                    {
                        "source": "AutoRefactorSkill",
                        "issue": issue
                    }
                )
            except Exception as e:
                logger.error("Failed to publish refactor proposal: %s", e)
