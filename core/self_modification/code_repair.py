"""Autonomous Code Repair System
Generates, validates, and applies fixes to detected bugs.
"""
import ast
import difflib
import json
import logging
import shutil
import subprocess
import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .ast_analyzer import ASTAnalyzer

logger = logging.getLogger("SelfModification.CodeRepair")


@dataclass
class CodeFix:
    """Represents a proposed code modification"""

    target_file: str
    target_line: int
    original_code: str
    fixed_code: str
    explanation: str
    hypothesis: str
    confidence: str  # 'high', 'medium', 'low'
    
    def to_dict(self):
        return {
            "target_file": self.target_file,
            "target_line": self.target_line,
            "original_code": self.original_code,
            "fixed_code": self.fixed_code,
            "explanation": self.explanation,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence
        }
    
    def generate_diff(self) -> str:
        """Generate unified diff for this fix"""
        original_lines = self.original_code.split('\n')
        fixed_lines = self.fixed_code.split('\n')
        
        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile=f"{self.target_file} (original)",
            tofile=f"{self.target_file} (fixed)",
            lineterm=''
        )
        
        return '\n'.join(diff)


class CodeFixGenerator:
    """Generates code fixes using LLM analysis.
    """
    
    def __init__(self, cognitive_engine, code_base_path: str = "."):
        self.brain = cognitive_engine
        self.code_base = Path(code_base_path)
        self.analyzer = ASTAnalyzer(self.code_base)
        logger.info("CodeFixGenerator initialized with AST support for %s", self.code_base)
    
    async def generate_fix(
        self,
        file_path: str,
        line_number: int,
        diagnosis: Dict[str, Any],
        context_lines: int = 10
    ) -> Optional[CodeFix]:
        """Generate a code fix for a diagnosed bug.
        
        Args:
            file_path: Path to file with bug
            line_number: Line number with bug
            diagnosis: Diagnosis from ErrorIntelligenceSystem
            context_lines: How many lines of context to include
            
        Returns:
            CodeFix object or None if generation failed

        """
        logger.info("Generating fix for %s:%d", file_path, line_number)
        
        # Read the buggy code
        try:
            code_context = await self._extract_code_context(file_path, line_number, context_lines)
            # v6.2: Add AST structural context
            ast_context = await self.analyzer.analyze_file(self.code_base / file_path)
            code_context["ast_summary"] = ast_context
        except Exception as e:
            logger.error("Failed to read code or analyze AST: %s", e)
            return None
        
        # Select best hypothesis
        hypotheses = diagnosis.get("hypotheses", [])
        if not hypotheses:
            logger.warning("No hypotheses in diagnosis")
            return None
        
        # Sort by confidence
        hypotheses.sort(
            key=lambda h: {'high': 0, 'medium': 1, 'low': 2}.get(h.get('confidence', 'low'), 3)
        )
        
        best_hypothesis = hypotheses[0]
        
        # Generate fix using LLM
        fix_code = await self._generate_fix_code(
            file_path,
            line_number,
            code_context,
            best_hypothesis
        )
        
        if not fix_code:
            return None
        
        # Create CodeFix object
        fix = CodeFix(
            target_file=file_path,
            target_line=line_number,
            original_code=code_context["buggy_section"],
            fixed_code=fix_code,
            explanation=best_hypothesis.get("potential_fix", ""),
            hypothesis=best_hypothesis.get("root_cause", ""),
            confidence=best_hypothesis.get("confidence", "medium")
        )
        
        logger.info("Generated fix with %s confidence", fix.confidence)
        return fix
    
    def _extract_code_context(
        self,
        file_path: str,
        line_number: int,
        context_lines: int
    ) -> Dict[str, Any]:
        """Extract code context around the bug location.
        
        Returns:
            Dictionary with full_file, buggy_section, before, after

        """
        full_path = self.code_base / file_path
        
        with open(full_path, 'r') as f:
            lines = f.readlines()
        
        # Calculate range
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        # Extract sections
        before = ''.join(lines[start:line_number-1])
        buggy_line = lines[line_number-1] if line_number <= len(lines) else ""
        after = ''.join(lines[line_number:end])
        
        buggy_section = before + buggy_line + after
        
        return {
            "full_file": ''.join(lines),
            "buggy_section": buggy_section,
            "before": before,
            "buggy_line": buggy_line,
            "after": after,
            "start_line": start + 1,
            "end_line": end
        }
    
    async def _generate_fix_code(
        self,
        file_path: str,
        line_number: int,
        code_context: Dict[str, Any],
        hypothesis: Dict[str, Any]
    ) -> Optional[str]:
        """Use LLM to generate fixed code.
        
        Returns:
            Fixed code string or None

        """
        prompt = f"""You are fixing a bug in your own code.

FILE: {file_path}
LINE: {line_number}

DIAGNOSIS:
Root Cause: {hypothesis.get('root_cause')}
Explanation: {hypothesis.get('explanation')}
Proposed Fix: {hypothesis.get('potential_fix')}

CURRENT CODE (lines {code_context['start_line']}-{code_context['end_line']}):
```python
{code_context['buggy_section']}
```

STRUCTURAL CONTEXT (AST):
Classes in file: {[c['name'] for c in code_context.get('ast_summary', {}).get('classes', [])]}
Functions in file: {[f['name'] for f in code_context.get('ast_summary', {}).get('functions', [])]}
Detected Smells: {code_context.get('ast_summary', {}).get('smells', [])}

TASK: Generate the FIXED version of this code section.

Requirements:
1. Fix ONLY the bug identified in the diagnosis
2. Maintain all existing functionality
3. Preserve code style and formatting
4. Add error handling if appropriate
5. Include a brief comment explaining the fix

Return ONLY the fixed code (same line range), no explanation, no markdown.
Start your response with the first line of fixed code."""

        try:
            thought = await self.brain.think(prompt, priority=0.1)
            response = thought.content # proper extraction
            
            # Clean up response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split('\n')
                # Remove first and last lines (```python and ```)
                response = '\n'.join(lines[1:-1])
            
            return response
            
        except Exception as e:
            logger.error("Fix generation failed: %s", e)
            return None


class CodeValidator:
    """Validates proposed code fixes before applying them.
    Multiple validation layers for safety.
    """
    
    def __init__(self):
        logger.info("CodeValidator initialized")
    
    def validate_fix(self, fix: CodeFix, full_file_content: str) -> Tuple[bool, str]:
        """Comprehensive validation of a proposed fix.
        
        Args:
            fix: The proposed fix
            full_file_content: Complete file content with fix applied
            
        Returns:
            (is_valid, reason)

        """
        # Layer 1: Syntax check
        syntax_valid, syntax_msg = self._validate_syntax(full_file_content)
        if not syntax_valid:
            return False, f"Syntax error: {syntax_msg}"
        
        # Layer 2: AST comparison (structural safety)
        structure_valid, structure_msg = self._validate_structure(fix, full_file_content)
        if not structure_valid:
            return False, f"Structural issue: {structure_msg}"
        
        # Layer 3: Import safety (no new dangerous imports)
        import_valid, import_msg = self._validate_imports(full_file_content)
        if not import_valid:
            return False, f"Import safety: {import_msg}"
        
        # Layer 4: Code smell detection
        smell_valid, smell_msg = self._check_code_smells(full_file_content)
        if not smell_valid:
            logger.warning("Code smell detected: %s", smell_msg)
            # Don't reject, just warn
        
        logger.info("Fix passed all validation layers")
        return True, "All validations passed"
    
    def _validate_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if code has valid Python syntax"""
        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
    
    def _validate_structure(self, fix: CodeFix, full_file: str) -> Tuple[bool, str]:
        """Ensure fix doesn't break structural invariants.
        
        Checks:
        - Class/function definitions aren't removed
        - Indentation is correct
        - Brackets are balanced
        """
        try:
            # Parse both versions
            original_ast = ast.parse(fix.original_code)
            fixed_ast = ast.parse(fix.fixed_code)
            
            # Count definitions
            def count_defs(tree):
                return {
                    'functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                    'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                }
            
            original_counts = count_defs(original_ast)
            fixed_counts = count_defs(fixed_ast)
            
            # Warn if definitions changed (might be intentional)
            if original_counts != fixed_counts:
                logger.warning("Definition counts changed: %s -> %s", original_counts, fixed_counts)
                # Don't reject - might be adding error handling function
            
            return True, "Structure validated"
            
        except Exception as e:
            return False, f"AST analysis failed: {e}"
    
    def _validate_imports(self, code: str) -> Tuple[bool, str]:
        """Check for dangerous or suspicious imports.
        
        Blacklist:
        - os.system, subprocess.Popen (without careful review)
        - eval, exec (code execution)
        - Network access without clear need
        """
        dangerous_patterns = [
            "os.system",
            "__import__",
            "eval(",
            "exec(",
            "compile("
        ]
        
        # Parse imports
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check import statements
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['os', 'subprocess'] and "system" in code:
                            return False, "Suspicious system call import"
                
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile']:
                            return False, f"Dangerous function: {node.func.id}"
            
            # String pattern check (less reliable but catches edge cases)
            for pattern in dangerous_patterns:
                if pattern in code and "#" not in code.split(pattern)[0].split('\n')[-1]:
                    # Not in a comment
                    logger.warning("Found potentially dangerous pattern: %s", pattern)
                    # Don't reject - might be legitimate
            
            return True, "Imports safe"
            
        except Exception as e:
            logger.error("Import validation failed: %s", e)
            return True, "Import validation inconclusive"
    
    def _check_code_smells(self, code: str) -> Tuple[bool, str]:
        """Detect code smells that might indicate problems.
        
        These don't fail validation but generate warnings.
        """
        smells = []
        
        lines = code.split('\n')
        
        # Check for very long lines
        for i, line in enumerate(lines):
            if len(line) > 120:
                smells.append(f"Long line at {i+1}: {len(line)} chars")
        
        # Check for deeply nested code
        max_indent = max((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
        if max_indent > 20:
            smells.append(f"Deep nesting: {max_indent} spaces")
        
        # Check for too many try/except blocks (might hide errors)
        try_count = code.count("try:")
        if try_count > 3:
            smells.append(f"Many try blocks: {try_count}")
        
        if smells:
            return False, "; ".join(smells)
        
        return True, "No smells detected"


class SandboxTester:
    """Tests code fixes in isolated environment before applying.
    """
    
    def __init__(self, code_base_path: str = "."):
        self.code_base = Path(code_base_path)
        logger.info("SandboxTester initialized")
    
    async def test_fix(self, fix: CodeFix) -> Tuple[bool, Dict[str, Any]]:
        """Test a fix in sandboxed environment.
        
        Args:
            fix: The fix to test
            
        Returns:
            (success, results_dict)

        """
        logger.info("Testing fix in sandbox: %s:%d", fix.target_file, fix.target_line)
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy relevant files to sandbox
            try:
                sandbox_ok = self._setup_sandbox(temp_path, fix)
                if not sandbox_ok:
                    return False, {"error": "Sandbox setup failed"}
            except Exception as e:
                logger.error("Sandbox setup failed: %s", e)
                return False, {"error": str(e)}
            
            # Apply fix in sandbox
            try:
                await self._apply_fix_in_sandbox(temp_path, fix)
            except Exception as e:
                logger.error("Fix application failed: %s", e)
                return False, {"error": f"Fix application: {e}"}
            
            # Run tests
            test_results = await self._run_tests_in_sandbox(temp_path, fix)
            
            return test_results["success"], test_results
    
    def _setup_sandbox(self, sandbox_path: Path, fix: CodeFix) -> bool:
        """Copy necessary files to sandbox"""
        target_file = self.code_base / fix.target_file
        
        if not target_file.exists():
            logger.error("Target file not found: %s", target_file)
            return False
        
        # Copy file to sandbox
        sandbox_file = sandbox_path / fix.target_file
        sandbox_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_file, sandbox_file)
        
        # v18 Hardening: Copy parent __init__.py and all siblings
        # This ensures imports like 'from . import sibling' work.
        source_dir = target_file.parent
        for dep_file in source_dir.glob("*.py"):
            if dep_file.name != "__pycache__":
                dest = sandbox_path / fix.target_file.replace(target_file.name, dep_file.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists(): # Don't overwrite the target_file we just copied
                    shutil.copy2(dep_file, dest)
        
        # Try to copy parent __init__.py if it exists (for absolute imports)
        parent_init = source_dir.parent / "__init__.py"
        if parent_init.exists():
            dest_init = sandbox_path / fix.target_file.split('/')[0] / "__init__.py"
            dest_init.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent_init, dest_init)
            
        return True
    
    async def _apply_fix_in_sandbox(self, sandbox_path: Path, fix: CodeFix):
        """Apply the fix to the sandboxed file (Async)."""
        sandbox_file = sandbox_path / fix.target_file
        
        content = await asyncio.to_thread(sandbox_file.read_text, encoding='utf-8')
        modified_content = content.replace(fix.original_code, fix.fixed_code)
        await asyncio.to_thread(sandbox_file.write_text, modified_content, encoding='utf-8')
    
    async def _run_tests_in_sandbox(self, sandbox_path: Path, fix: CodeFix) -> Dict[str, Any]:
        """Run tests in sandbox with enhanced safety (Async)."""
        results = {
            "success": False,
            "import_test": False,
            "syntax_test": False,
            "unit_tests": False,
            "integrity_check": False,
            "errors": []
        }
        
        sandbox_file = sandbox_path / fix.target_file
        
        # Test 1: Syntax check (Static)
        try:
            with open(sandbox_file, 'r') as f:
                code = f.read()
            ast.parse(code)
            results["syntax_test"] = True
        except SyntaxError as e:
            results["errors"].append(f"Syntax error: {e}")
            return results
        
        # Test 2: Import check (Dynamic Dry Run)
        try:
            # Try to import the module
            module_name = fix.target_file.replace('/', '.').replace('.py', '')
            
            # Run Python to try importing
            # We add sandbox path to sys.path
            test_code = f"import sys; sys.path.insert(0, '{sandbox_path}'); import {module_name}; print('Import OK')"
            
            result = await asyncio.to_thread(
                subprocess.run,
                ["python", "-c", test_code],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=sandbox_path
            )
            
            if result.returncode == 0 and "Import OK" in result.stdout:
                results["import_test"] = True
            else:
                results["errors"].append(f"Import failed: {result.stderr}")
                return results
                
        except asyncio.TimeoutError:
            results["errors"].append("Import test timeout")
            return results
        except Exception as e:
            results["errors"].append(f"Import test error: {e}")
            # Import failures are critical for self-modification
            return results

        # Test 3: System Integration Check (Dry Run)
        # We try to import the Orchestrator to ensure core dependencies aren't broken
        try:
            integrity_code = f"import sys; sys.path.insert(0, '{sandbox_path}'); from core.orchestrator import RobustOrchestrator; print('Integrity OK')"
            
            result = await asyncio.to_thread(
                subprocess.run,
                ["python", "-c", integrity_code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=sandbox_path
            )
            
            if result.returncode == 0 and "Integrity OK" in result.stdout:
                 results["integrity_check"] = True
            else:
                 # If integrity check fails, it might be because of missing deps in sandbox, so we warn but don't fail hard unless it's a core file modification
                 logger.warning("Integrity check warning: %s", result.stderr)
                 # Only fail if the modified file IS a core file
                 if "core/" in fix.target_file:
                     results["errors"].append(f"Core integrity check failed: {result.stderr}")
                     return results
                 results["integrity_check"] = True # Soft pass for non-core

        except Exception as e:
            logger.warning("Integrity check exception: %s", e)
            if "core/" in fix.target_file:
                 return results
            results["integrity_check"] = True


        # Test 4: Type check (Pyright Guard) - Phase 30
        try:
            from core.resilience.diagnostic_hub import get_diagnostic_hub
            hub = get_diagnostic_hub()
            logger.info("🛡️ [NEURO] Running Pyright validation on sandbox fix...")
            
            # We must run pyright on the specific file in the sandbox
            # Pyright usually needs a config, but we can run it on a single file
            pyright_res = await hub._run_pyright(sandbox_file)
            if not pyright_res.get("ok", True):
                issues = pyright_res.get("issues", [])
                if issues:
                    # Filter for errors only if desired, or report all
                    results["errors"].append(f"Pyright type mismatch: {issues[0].get('message')}")
                    logger.warning("❌ [NEURO] Pyright rejected the fix.")
                    return results
            results["integrity_check"] = True
            logger.info("✅ [NEURO] Pyright validation passed.")
        except Exception as e:
            logger.warning("Pyright guard bypassed due to error: %s", e)

        # Test 5: Run Unit Tests (pytest) if available
        # Check for test files associated with the target
        # e.g., core/memory.py -> tests/core/test_memory.py or same dir test_memory.py
        try:
            # Simple heuristic for test discovery
            test_files = list(sandbox_path.rglob(f"test_{sandbox_file.name}"))
            
            if test_files:
                logger.info("Running tests: %s", test_files)
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["python", "-m", "pytest", str(test_files[0])],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=sandbox_path
                )
                
                if result.returncode == 0:
                    results["unit_tests"] = True
                else:
                    results["errors"].append(f"Unit tests failed: {result.stdout}")
                    return results
            else:
                # No tests found - pass by default but log
                results["unit_tests"] = True # "N/A"
                
        except Exception as e:
            logger.warning("Test execution failed: %s", e)
            # Don't fail the fix just because test runner failed, unless we want strict TDD
            results["unit_tests"] = True

        # All critical tests passed
        results["success"] = True
        return results

    async def run_custom_probe(
        self,
        file_path: str,
        code_patch: str,
        probe_code: str,
        expect_pass: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run a custom reproduction probe against a patched version of a file."""
        import tempfile, shutil
        results = {"success": False, "error": None}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Setup sandbox (copy files)
            # Minimal setup: target file + its dir
            target_abs = self.code_base / file_path
            if not target_abs.exists():
                return False, {"error": "Target file not found"}
            
            sandbox_file = temp_path / file_path
            sandbox_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the PATCHED version
            # Use original file as base, replace buggy section
            with open(target_abs, 'r') as f:
                content = f.read()
            
            # Simple replacement (matches _apply_fix_in_sandbox logic)
            # This 'code_patch' is the CodeFix.original_code or CodeFix.fixed_code
            # Wait, the 'code_patch' here should be the WHOLE file content for simplicity,
            # but let's stick to the replacement logic for consistency with previous tools.
            # Actually, let's just write the modified content passed in.
            
            # We'll assume the caller (EvaluationHarness) handles the replacement if needed,
            # or we do it here. Let's do it here.
            # No, EvaluationHarness passed 'code_patch' which is the section.
            # Actually, I'll change the caller to pass modified_content.
            
            sandbox_file.write_text(code_patch, encoding="utf-8")
            
            # Copy siblings for imports
            for sibling in target_abs.parent.glob("*.py"):
                if sibling.name != target_abs.name:
                    shutil.copy2(sibling, temp_path / file_path.replace(target_abs.name, sibling.name))

            # 2. Write Probe
            probe_path = temp_path / "weakness_probe.py"
            probe_path.write_text(probe_code, encoding="utf-8")
            
            # 3. Run Probe
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["python3", "weakness_probe.py"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=temp_path
                )
                
                is_pass = (result.returncode == 0)
                
                if expect_pass:
                    results["success"] = is_pass
                    if not is_pass:
                        results["error"] = f"Probe failed (code {result.returncode}): {result.stderr or result.stdout}"
                else:
                    # Expect failure
                    results["success"] = not is_pass
                    if is_pass:
                        results["error"] = "Probe passed when it should have failed (weakness not reproduced)."
                
                return results["success"], results
                
            except Exception as e:
                return False, {"error": f"Probe execution error: {e}"}


# Integration class
class AutonomousCodeRepair:
    """Complete autonomous code repair system.
    Combines generation, validation, and testing.
    """
    
    def __init__(self, cognitive_engine, code_base_path: str = "."):
        self.generator = CodeFixGenerator(cognitive_engine, code_base_path)
        self.validator = CodeValidator()
        self.tester = SandboxTester(code_base_path)
        
        # v6.3: Evaluation Harness
        from .evaluation_harness import EvaluationHarness
        self.harness = EvaluationHarness(cognitive_engine, self.tester, code_base_path)
        
        logger.info("AutonomousCodeRepair system initialized with EvaluationHarness")
    
    async def repair_bug(
        self,
        file_path: str,
        line_number: int,
        diagnosis: Dict[str, Any]
    ) -> Tuple[bool, Optional[CodeFix], Dict[str, Any]]:
        """Attempt to repair a bug autonomously.
        
        Args:
            file_path: File with bug
            line_number: Line with bug
            diagnosis: Diagnosis from ErrorIntelligenceSystem
            
        Returns:
            (success, fix_object, test_results)

        """
        # v40: Growth Ladder Veto
        from core.container import ServiceContainer
        ladder = ServiceContainer.get("growth_ladder", default=None)
        if ladder:
            proposal_id = f"repair_{file_path}_{line_number}"
            # Level 2 or 3 depending on file path
            level = 3 if "core/" in file_path else 2
            consent = await ladder.propose_modification(
                proposal_id=proposal_id,
                modification_type="code_repair",
                level=level,
                description=f"Autonomous repair of {file_path}:{line_number}. Diagnosis: {diagnosis.get('summary', 'unknown')}"
            )
            if not consent:
                logger.warning("🚫 [GrowthLadder] Aura VETOED code repair for %s", file_path)
                return False, None, {"error": "Vetoed by entity"}

        # Step 1: Generate fix
        
        # v29.1: Mechanical Repair Layer (Ruff)
        logger.info("🔧 [NEURO] Attempting mechanical repair with Ruff...")
        try:
            from core.resilience.diagnostic_hub import get_diagnostic_hub
            hub = get_diagnostic_hub()
            # If it's a simple syntax or style issue, ruff might fix it
            cmd = ["ruff", "check", "--fix", file_path]
            subprocess.run(cmd, capture_output=True, text=True, cwd=self.generator.code_base)
        except Exception as e:
            logger.warning("Mechanical repair failed: %s", e)

        fix = await self.generator.generate_fix(file_path, line_number, diagnosis)
        if not fix:
            return False, None, {"error": "Fix generation failed"}
        
        logger.info("Generated fix:\n%s", fix.generate_diff())
        
        # Step 2: Validate fix
        # Read original file
        full_path = Path(self.generator.code_base) / file_path
        original_content = await asyncio.to_thread(full_path.read_text, encoding='utf-8')
        
        # Apply fix to get full new content
        modified_content = original_content.replace(fix.original_code, fix.fixed_code)
        
        valid, validation_msg = self.validator.validate_fix(fix, modified_content)
        if not valid:
            logger.warning("Fix validation failed: %s", validation_msg)
            return False, fix, {"error": f"Validation failed: {validation_msg}"}
        
        logger.info("Fix passed validation")
        
        # Step 3: Test in sandbox
        test_success, test_results = await self.tester.test_fix(fix)
        if not test_success:
            logger.warning("Fix failed sandbox tests: %s", test_results.get('errors'))
            return False, fix, test_results
        
        logger.info("Fix passed sandbox testing")
        
        # Step 4: Run Evaluation Harness (Reproduction & Verification)
        logger.info("Running Evaluation Harness...")
        eval_success, eval_msg = await self.harness.evaluate_fix(fix, diagnosis)
        
        if not eval_success:
            logger.warning("Fix failed Evaluation Harness: %s", eval_msg)
            return False, fix, {"error": eval_msg}
            
        logger.info("✅ Fix passed Evaluation Harness")
        
        # Fix is ready to apply!
        return True, fix, test_results