import asyncio
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.TestGen")

_TESTGEN_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    PermissionError,
)


def _record_testgen_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    metadata["stage"] = stage
    try:
        record_degradation(
            "test_generator",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata,
        )
    except TypeError:
        record_degradation(
            "test_generator",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


async def _read_text(path: Path) -> str:
    return await asyncio.to_thread(path.read_text, encoding="utf-8")


async def _write_text(path: Path, content: str) -> None:
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")


class TestGeneratorParams(BaseModel):
    __test__ = False
    target_file: str = Field(
        ..., description="The path to the Python file or module to generate tests for."
    )


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
    def _effective_timeout(context: dict[str, Any] | None, default: float = 20.0) -> float:
        ctx = context or {}
        timeout_raw = (
            ctx.get("timeout_s")
            or (ctx.get("executive_constraints", {}) or {}).get("timeout_s")
            or default
        )
        try:
            return max(5.0, float(timeout_raw))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_read_only(context: dict[str, Any] | None) -> bool:
        ctx = context or {}
        if ctx.get("read_only") is not None:
            return bool(ctx.get("read_only"))
        return bool((ctx.get("executive_constraints", {}) or {}).get("read_only"))

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

    async def execute(self, params: TestGeneratorParams, context: dict[str, Any]) -> dict[str, Any]:
        """Execute test generation and execution."""
        # Legacy support
        if isinstance(params, dict):
            try:
                params = TestGeneratorParams(**params)
            except _TESTGEN_RECOVERABLE_ERRORS as e:
                _record_testgen_degradation(
                    e,
                    action="rejected invalid test generation input before filesystem effects",
                    stage="input_validation",
                    severity="warning",
                )
                return {"ok": False, "error": f"Invalid input: {e}"}

        target_file = params.target_file
        read_only = self._is_read_only(context)
        timeout_s = self._effective_timeout(context)
        llm_timeout = max(5.0, min(12.0, timeout_s * 0.35))
        command_timeout = max(5, min(45, int(timeout_s)))
        prefer_deterministic = read_only or bool((context or {}).get("prefer_deterministic"))

        # Issue 82: Lazy resolve brain
        brain = self._resolve_brain()

        if not target_file:
            return {"ok": False, "error": "No target_file provided."}

        target_path = await asyncio.to_thread(lambda: Path(target_file).expanduser().resolve())
        if not await asyncio.to_thread(target_path.exists):
            return {"ok": False, "error": f"Target file {target_file} not found."}
        if not await asyncio.to_thread(target_path.is_file):
            return {"ok": False, "error": f"Target file {target_file} is not a file."}

        # 1. Read the target code for context
        try:
            code_content = await _read_text(target_path)

            # 2. Generate Test Code via CognitiveEngine, or fall back to a deterministic smoke test.
            test_code = ""
            generated_with_brain = False
            used_fallback_rerun = False
            if brain and not prefer_deterministic:
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

                    thought = await asyncio.wait_for(
                        brain.think(
                            objective=prompt,
                            context={"role": "qa_engineer", "target": target_file},
                            origin="test_generator",
                            mode=ThinkingMode.DEEP,
                        ),
                        timeout=llm_timeout,
                    )
                    test_code = getattr(thought, "content", str(thought or ""))
                    test_code = re.sub(r"```python\n|```", "", test_code).strip()
                    generated_with_brain = bool(test_code)
                except _TESTGEN_RECOVERABLE_ERRORS as e:
                    _record_testgen_degradation(
                        e,
                        action="used deterministic smoke test after llm test generation failed",
                        stage="llm_generation",
                        severity="warning",
                        extra={"target_file": str(target_path)},
                    )
                    logger.warning(
                        "LLM-based test generation unavailable for %s: %s", target_file, e
                    )

            if not test_code:
                test_code = self._fallback_test_code(target_path)

            # 3. Write to temporary test file if not exists or override
            if read_only:
                temp_dir = Path(tempfile.mkdtemp(prefix="aura_test_generator_"))
                test_file = temp_dir / f"test_{target_path.name}"
                await _write_text(temp_dir / target_path.name, code_content)
            else:
                test_file = target_path.parent / f"test_{target_path.name}"

            await _write_text(test_file, test_code)

            logger.info("✨ Tests generated and saved to %s", test_file)

            # 4. Run tests — try sandbox first, fall back to subprocess
            sandbox = None
            try:
                from core.sovereign.local_sandbox import LocalSandbox

                sandbox = LocalSandbox()
                sandbox.start()

                # Write files to sandbox
                sandbox.write_file(target_path.name, code_content)
                sandbox.write_file(test_file.name, test_code)

                # Execute pytest in sandbox (non-blocking event loop)
                command = f"{sys.executable} -m pytest -q {test_file.name}"
                res = await sandbox.run_command(command, timeout=command_timeout)
                if res.exit_code != 0 and generated_with_brain:
                    logger.warning(
                        "LLM-generated sandbox tests failed for %s; retrying deterministic smoke test.",
                        target_file,
                    )
                    test_code = self._fallback_test_code(target_path)
                    await _write_text(test_file, test_code)
                    sandbox.write_file(test_file.name, test_code)
                    res = await sandbox.run_command(command, timeout=command_timeout)
                    used_fallback_rerun = True

                return {
                    "ok": res.exit_code == 0,
                    "output": res.stdout,
                    "error": res.stderr,
                    "test_file": str(test_file),
                    "fallback_used": used_fallback_rerun,
                }
            except _TESTGEN_RECOVERABLE_ERRORS as sandbox_err:
                # LocalSandbox unavailable or failed — run via subprocess directly
                if not isinstance(sandbox_err, ImportError):
                    _record_testgen_degradation(
                        sandbox_err,
                        action="fell back to subprocess test execution after local sandbox failed",
                        stage="sandbox_execution",
                        severity="warning",
                        extra={"target_file": str(target_path)},
                    )
                    logger.warning(
                        "Sandbox execution failed for %s, falling back to subprocess: %s",
                        target_file,
                        sandbox_err,
                    )
                else:
                    logger.info(
                        "LocalSandbox unavailable, running tests via subprocess for %s", target_file
                    )
                try:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        str(test_file),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(target_path.parent),
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=command_timeout
                        )
                    except TimeoutError as timeout_error:
                        proc.kill()
                        stdout, stderr = await proc.communicate()
                        raise TimeoutError(
                            f"pytest subprocess timed out after {command_timeout}s"
                        ) from timeout_error
                    return {
                        "ok": proc.returncode == 0,
                        "output": stdout.decode(errors="replace"),
                        "error": stderr.decode(errors="replace"),
                        "test_file": str(test_file),
                        "fallback_used": not generated_with_brain,
                        "sandbox": "subprocess",
                    }
                except _TESTGEN_RECOVERABLE_ERRORS as sp_err:
                    _record_testgen_degradation(
                        sp_err,
                        action="returned explicit failure after subprocess test execution failed",
                        stage="subprocess_execution",
                        severity="degraded",
                        extra={"target_file": str(target_path)},
                    )
                    return {"ok": False, "error": f"Subprocess test run failed: {sp_err}"}
            finally:
                if sandbox is not None:
                    try:
                        sandbox.stop()
                    except _TESTGEN_RECOVERABLE_ERRORS as cleanup_error:
                        _record_testgen_degradation(
                            cleanup_error,
                            action="left local sandbox cleanup to operator after stop failed",
                            stage="sandbox_cleanup",
                            severity="warning",
                            extra={"target_file": str(target_path)},
                        )
                        logger.debug("Sandbox cleanup skipped for %s", target_file)
        except _TESTGEN_RECOVERABLE_ERRORS as e:
            _record_testgen_degradation(
                e,
                action="returned explicit test generation failure payload",
                stage="execute",
                severity="degraded",
                extra={"target_file": target_file},
            )
            logger.error("Test generation/execution failed: %s", e)
            return {"ok": False, "error": str(e)}
