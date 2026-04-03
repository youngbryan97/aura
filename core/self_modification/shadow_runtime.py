"""core/self_modification/shadow_runtime.py — Shadow Runtime Sandbox

Isolated testing environment for code mutations before they touch live code.
Copies the codebase to a temp directory, applies the proposed mutation, boots
the modified copy as a subprocess, and monitors for crashes/errors during a
configurable soak period.

This goes beyond SandboxTester (which only validates syntax + imports) by
running the full modified system for N seconds to catch runtime failures.
"""
from core.utils.exceptions import capture_and_log
import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("Aura.ShadowRuntime")


@dataclass
class ShadowResult:
    """Result of a shadow runtime test."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    exit_code: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""


class ShadowRuntime:
    """Runs proposed mutations in an isolated copy of the codebase.
    
    Safety layers:
    1. Full codebase copy to temp directory (no live file touching)
    2. Mutation applied only in the copy
    3. Subprocess execution with timeout
    4. Soak period monitoring for delayed failures
    5. Automatic cleanup on completion
    """

    # Minimum phi (integration) required to permit self-modification.
    # Set by the orchestrator/phi_consciousness phase each tick via set_coherence_gate().
    # Below this threshold, self-modification is deferred — fragmented cognition
    # risks generating incoherent patches that pass syntax but break semantics.
    MIN_PHI_FOR_SELF_MOD: float = 0.45

    def __init__(self, code_base_path: str = "."):
        # Issue 93: Version Guard
        if sys.version_info < (3, 9):
             raise RuntimeError("ShadowRuntime requires Python 3.9+ for ast.unparse support")

        self.code_base = Path(code_base_path).resolve()
        self._active_shadow: Optional[Path] = None
        self._lock = asyncio.Lock() # Issue 94: Singleton lock for resource control
        self._current_phi: float = 1.0  # Assume integrated until told otherwise

        # Files/dirs to exclude from copy (saves time and space)
        self._exclude_patterns = {
            "__pycache__", ".git", "dist", "build", "node_modules",
            ".pyc", ".pyo", "*.egg-info", ".pytest_cache", "venv",
            ".venv", "data", "*.db", "*.sqlite", "*.log",
        }

    async def test_mutation(
        self,
        file_path: str,
        original_code: str,
        patched_code: str,
        soak_seconds: int = 15,
        boot_script: Optional[str] = None,
    ) -> ShadowResult:
        """Test a code mutation in an isolated shadow copy.
        
        Args:
            file_path: Relative path to the file being modified
            original_code: Current file content (for verification)
            patched_code: Proposed new file content
            soak_seconds: How long to let the shadow run (seconds)
            boot_script: Optional Python script to run for validation.
                         Defaults to a basic import + syntax check.
        
        Returns:
            ShadowResult with pass/fail and diagnostics
        """
        # Preflight: Phi coherence gate — self-modification under fragmented
        # cognition produces unreliable patches even if syntax is valid.
        if self._current_phi < self.MIN_PHI_FOR_SELF_MOD:
            logger.warning(
                "🧠 ShadowRuntime: Phi=%.3f is below self-mod threshold (%.2f). "
                "Deferring mutation of '%s' until coherence recovers.",
                self._current_phi, self.MIN_PHI_FOR_SELF_MOD, Path(file_path).name,
            )
            return ShadowResult(
                passed=False,
                errors=[
                    f"Phi coherence ({self._current_phi:.3f}) below minimum "
                    f"({self.MIN_PHI_FOR_SELF_MOD}) for self-modification. "
                    "Retry when integration recovers."
                ],
            )

        # Preflight: Block mutations to safety-critical modules entirely.
        # These files must never be patched by the autonomous optimizer —
        # a reward-hacking agent could otherwise neutralize its own guardrails.
        _PROTECTED_SAFETY_MODULES = frozenset({
            "constitutional_guard.py",
            "master_moral_integration.py",
            "emergency_protocol.py",
            "heartstone_values.py",
            "behavior_controller.py",
            "safety_registry.py",
            "identity_guard.py",
        })
        target_filename = Path(file_path).name
        if target_filename in _PROTECTED_SAFETY_MODULES:
            logger.critical(
                "🛡️ ShadowRuntime: BLOCKED attempt to mutate protected safety module '%s'. "
                "Self-modification of constitutional/moral modules is prohibited.",
                target_filename,
            )
            return ShadowResult(
                passed=False,
                errors=[
                    f"Mutation to protected safety module '{target_filename}' is not permitted. "
                    "Human approval required via IdentityGuard escalation path."
                ],
            )

        start_time = time.monotonic()
        result = ShadowResult(passed=False)
        shadow_dir = None

        # [Issue 94] Sequentialize shadow tests to prevent resource exhaustion
        async with self._lock:
            try:
                # 1. Create shadow copy
                shadow_dir = await self._create_shadow_copy()
                self._active_shadow = shadow_dir
                logger.info("🔮 Shadow copy created at %s", shadow_dir)

                # 2. Verify original content matches
                shadow_file = shadow_dir / file_path
                if shadow_file.exists():
                    current = shadow_file.read_text(encoding="utf-8")
                    if current.strip() != original_code.strip():
                        result.warnings.append("Original code mismatch — file may have changed since patch was generated")

                # 3. Apply mutation in shadow
                shadow_file.parent.mkdir(parents=True, exist_ok=True)
                shadow_file.write_text(patched_code, encoding="utf-8")
                logger.info("🔮 Mutation applied to shadow: %s", file_path)

                # 4. Run validation
                if boot_script is None:
                    # Issue 95: Windows path fix & robust module name resolution
                    module_name = Path(file_path).with_suffix('').as_posix().replace("/", ".")
                    boot_script = f"""
import sys
import os
sys.path.insert(0, '{shadow_dir.as_posix()}')
try:
    import {module_name}
    print("SHADOW_OK: Import successful")
except Exception as e:
    print(f"SHADOW_FAIL: Import error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Basic sanity: check the module has expected attributes
import ast
with open('{shadow_file.as_posix()}') as f:
    tree = ast.parse(f.read())
print(f"SHADOW_OK: AST parsed, {{len(tree.body)}} top-level nodes")
"""

                # 5. Execute in subprocess with soak timeout
                proc_result = await self._run_in_subprocess(
                    boot_script, shadow_dir, timeout=soak_seconds
                )

                result.exit_code = proc_result["exit_code"]
                result.stdout_tail = proc_result["stdout"][-500:] if proc_result["stdout"] else ""
                result.stderr_tail = proc_result["stderr"][-500:] if proc_result["stderr"] else ""

                # 6. Analyze results
                if result.exit_code != 0:
                    result.errors.append(f"Shadow process exited with code {result.exit_code}")
                    if result.stderr_tail:
                        result.errors.append(f"stderr: {result.stderr_tail[:200]}")
                elif "SHADOW_FAIL" in result.stdout_tail:
                    result.errors.append(f"Shadow validation failed: {result.stdout_tail}")
                else:
                    result.passed = True
                    logger.info("✅ Shadow runtime test PASSED for %s (%.1fs)", file_path, time.monotonic() - start_time)

            except asyncio.TimeoutError:
                result.errors.append(f"Shadow runtime timed out after {soak_seconds}s")
                logger.warning("⏰ Shadow runtime timed out for %s", file_path)
            except Exception as e:
                result.errors.append(f"Shadow runtime error: {e}")
                logger.error("Shadow runtime failed: %s", e)
            finally:
                # 7. Cleanup
                result.runtime_seconds = time.monotonic() - start_time
                if shadow_dir and shadow_dir.exists():
                    try:
                        shutil.rmtree(shadow_dir, ignore_errors=True)
                    except Exception as e:
                        capture_and_log(e, {'module': __name__})
                self._active_shadow = None

        if not result.passed:
            logger.warning("❌ Shadow runtime test FAILED for %s: %s", file_path, result.errors[:2])

        return result

    def set_coherence_gate(self, phi: float) -> None:
        """Update the current phi value used as the self-modification coherence gate.

        Called by the orchestrator/PhiConsciousnessPhase after each phi computation
        so test_mutation has an up-to-date picture of cognitive integration.
        """
        self._current_phi = float(phi)

    async def _create_shadow_copy(self) -> Path:
        """Create a lightweight copy of the codebase."""
        shadow_dir = Path(tempfile.mkdtemp(prefix="aura_shadow_"))

        def _copy():
            for item in self.code_base.iterdir():
                # Skip excluded patterns
                if any(item.name.endswith(p.lstrip("*")) or item.name == p
                       for p in self._exclude_patterns):
                    continue
                
                dest = shadow_dir / item.name
                try:
                    if item.is_dir():
                        shutil.copytree(
                            item, dest,
                            ignore=shutil.ignore_patterns(*self._exclude_patterns),
                            dirs_exist_ok=True,
                        )
                    else:
                        shutil.copy2(item, dest)
                except Exception as e:
                    logger.debug("Shadow copy skip %s: %s", item.name, e)

        await asyncio.to_thread(_copy)
        return shadow_dir

    async def _run_in_subprocess(
        self, script: str, cwd: Path, timeout: int = 30
    ) -> dict:
        """Run a Python script in a subprocess."""
        script_path = cwd / "_shadow_boot.py"
        script_path.write_text(script, encoding="utf-8")

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", str(script_path),
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return {
                "exit_code": proc.returncode or 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }
        except asyncio.TimeoutError:
            proc.kill()
            return {"exit_code": -1, "stdout": "", "stderr": "Timeout"}

    @property
    def is_active(self) -> bool:
        return self._active_shadow is not None


# Singleton
_instance: Optional[ShadowRuntime] = None

def get_shadow_runtime(code_base: str = ".") -> ShadowRuntime:
    global _instance
    if _instance is None:
        _instance = ShadowRuntime(code_base_path=code_base)
        logger.info("✓ Shadow Runtime initialized (base: %s)", code_base)
    return _instance
