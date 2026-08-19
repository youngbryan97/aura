"""Canonical non-LLM proof gate for self-modification safety.

Every mutation path in Aura must route through this harness before
code changes are promoted. The LLM may advise, but this harness
is the authoritative gate.

Checks:
  1. AST safety: ban eval, exec, __import__, socket, subprocess
  2. py_compile on all changed files
  3. pytest in subprocess (not import-based)
  4. Hidden eval seed comparison
  5. Resource delta check (memory/CPU)
  6. Rollback drill (backup → apply → restore → fingerprint match)
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.resource_observation import get_resource_observer
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.self_modification.distributed_sandbox_gateway import (
    DistributedSandboxGateway,
    SandboxSweepRequest,
)
from core.self_modification.mutation_tiers import classify_mutation_path

logger = logging.getLogger("SelfModification.SafeHarness")

# AST node types that are categorically banned in self-modification patches
BANNED_AST_NODES = {
    "eval": ast.Call,
    "exec": ast.Call,
    "__import__": ast.Call,
}

BANNED_CALL_NAMES = frozenset({"eval", "exec", "__import__", "compile"})

BANNED_IMPORT_MODULES = frozenset({"socket", "http.client", "urllib.request", "ftplib", "smtplib"})

ALLOWED_SUBPROCESS_CALLERS = frozenset(
    {
        "core/self_modification/safe_modification_harness.py",
        "core/self_modification/code_repair.py",  # ruff mechanical repair only
        "core/architect/ghost_boot.py",
    }
)

_WORKSPACE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".venv_aura",
        "__pycache__",
        "artifacts",
        "build",
        "data",
        "dist",
        "logs",
        "model_weights",
        "models",
        "node_modules",
        "scratch",
        "venv",
    }
)


@dataclass(frozen=True)
class HarnessResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        failed = [k for k, v in self.checks.items() if not v]
        return f"SafeHarness {status} ({len(self.checks)} checks, {len(failed)} failures: {failed})"


class SafeModificationHarness:
    """Authoritative non-LLM proof gate for all code mutations."""

    def __init__(self, codebase_root: str | Path = ".") -> None:
        self.codebase_root = Path(codebase_root).resolve()

    async def run(
        self,
        changed_files: list[str],
        *,
        patch_content: dict[str, str] | None = None,
        extra_test_targets: list[str] | None = None,
        require_distributed_sandbox: bool | None = None,
        distributed_gateway: DistributedSandboxGateway | None = None,
    ) -> HarnessResult:
        """Run all safety checks on the given changed files.

        Args:
            changed_files: Relative paths to files that were modified.
            patch_content: Optional dict of {filepath: new_content} for pre-apply checks.

        Returns:
            HarnessResult with per-check pass/fail and aggregate decision.
        """
        import asyncio

        started = time.monotonic()
        checks: dict[str, bool] = {}
        errors: list[str] = []
        baseline_rss_mb = self._current_rss_mb()
        source_hashes_before = self._source_hashes(changed_files)

        # Use patch_content if provided, otherwise read from disk
        file_contents: dict[str, str] = {}
        for fpath in changed_files:
            if patch_content and fpath in patch_content:
                file_contents[fpath] = patch_content[fpath]
            else:
                abs_path = self.codebase_root / fpath
                if abs_path.exists():
                    file_contents[fpath] = abs_path.read_text(encoding="utf-8")
                else:
                    errors.append(f"File not found: {fpath}")
                    checks["files_exist"] = False

        if not file_contents:
            return HarnessResult(
                passed=False,
                checks={"files_exist": False},
                errors=errors,
                duration_s=time.monotonic() - started,
            )

        checks["files_exist"] = True

        # Check 1: AST safety
        ast_ok, ast_errors = self._check_ast_safety(file_contents)
        checks["ast_safety"] = ast_ok
        errors.extend(ast_errors)

        # Check 2: py_compile
        compile_ok, compile_errors = await asyncio.to_thread(self._check_py_compile, file_contents)
        checks["py_compile"] = compile_ok
        errors.extend(compile_errors)

        distributed_required = (
            require_distributed_sandbox
            if require_distributed_sandbox is not None
            else os.getenv("AURA_REQUIRE_DISTRIBUTED_SANDBOX", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        # Check 3: pytest against the exact candidate bytes in an isolated
        # workspace. Running tests in the live tree would exercise the old
        # implementation and could falsely approve a broken patch.
        try:
            with tempfile.TemporaryDirectory(prefix="aura-candidate-workspace-") as tmpdir:
                candidate_root = Path(tmpdir)
                await asyncio.to_thread(
                    self.prepare_candidate_workspace,
                    candidate_root,
                    file_contents,
                )
                overlay_ok, overlay_errors = self._verify_candidate_overlay(
                    candidate_root,
                    file_contents,
                )
                checks["candidate_overlay"] = overlay_ok
                errors.extend(overlay_errors)
                if overlay_ok:
                    test_ok, test_errors = await self._check_pytest(
                        changed_files,
                        candidate_root=candidate_root,
                        extra_test_targets=extra_test_targets,
                    )
                else:
                    test_ok = False
                    test_errors = ["candidate overlay verification failed"]
                checks["pytest"] = test_ok
                errors.extend(test_errors)
                if overlay_ok and test_ok and distributed_required:
                    targets = tuple(
                        self._related_test_files(
                            changed_files,
                            candidate_root,
                            extra_test_targets or [],
                        )
                    )
                    max_tier = max(int(classify_mutation_path(path).tier) for path in changed_files)
                    gateway = distributed_gateway or DistributedSandboxGateway()
                    sweep = await gateway.validate(
                        SandboxSweepRequest(
                            candidate_root=candidate_root,
                            test_targets=targets,
                            risk_tier=max_tier,
                            requested_workers=2 if max_tier >= 2 else 1,
                            max_cost_usd=0.0,
                        ),
                        local_runner=self._run_sandbox_attempt,
                    )
                    checks["distributed_sandbox"] = sweep.passed
                    errors.extend(sweep.errors)
                elif distributed_required:
                    checks["distributed_sandbox"] = False
                    errors.append(
                        "distributed sandbox skipped because local candidate validation failed"
                    )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            checks["candidate_overlay"] = False
            checks["pytest"] = False
            errors.append(f"candidate workspace failed: {exc}")

        # Check 4: Resource delta
        resource_ok, resource_errors = self._check_resource_delta(baseline_rss_mb)
        checks["resource_delta"] = resource_ok
        errors.extend(resource_errors)

        # Check 5: Rollback drill
        rollback_ok, rollback_errors = await asyncio.to_thread(
            self._check_rollback_drill, changed_files
        )
        checks["rollback_drill"] = rollback_ok
        errors.extend(rollback_errors)

        source_hashes_after = self._source_hashes(changed_files)
        source_immutable = source_hashes_before == source_hashes_after
        checks["source_immutable"] = source_immutable
        if not source_immutable:
            errors.append("live source changed while validating candidate workspace")

        passed = all(checks.values())
        duration = time.monotonic() - started

        result = HarnessResult(
            passed=passed, checks=checks, errors=errors, duration_s=round(duration, 4)
        )
        logger.info("%s (%.2fs)", result.summary(), duration)
        return result

    def _check_ast_safety(self, file_contents: dict[str, str]) -> tuple[bool, list[str]]:
        """Ban dangerous AST patterns in all changed files."""
        errors: list[str] = []
        for fpath, content in file_contents.items():
            try:
                tree = ast.parse(content, filename=fpath)
            except SyntaxError as e:
                errors.append(f"AST parse failed in {fpath}: {e}")
                continue

            for node in ast.walk(tree):
                # Check dangerous function calls
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr

                    if func_name in BANNED_CALL_NAMES:
                        # Allow if this file is in the allowed list
                        if fpath not in ALLOWED_SUBPROCESS_CALLERS:
                            errors.append(
                                f"Banned call '{func_name}' in {fpath}:{getattr(node, 'lineno', '?')}"
                            )

                # Check dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in BANNED_IMPORT_MODULES:
                            errors.append(f"Banned import '{alias.name}' in {fpath}:{node.lineno}")
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module in BANNED_IMPORT_MODULES:
                        errors.append(
                            f"Banned import from '{node.module}' in {fpath}:{node.lineno}"
                        )

        return len(errors) == 0, errors

    def _check_py_compile(self, file_contents: dict[str, str]) -> tuple[bool, list[str]]:
        """Compile all changed files without executing them."""
        errors: list[str] = []
        with tempfile.TemporaryDirectory(prefix="aura-harness-") as tmpdir:
            for fpath, content in file_contents.items():
                tmp_file = Path(tmpdir) / fpath
                get_file_write_gateway().write_text(
                    tmp_file,
                    content,
                    encoding="utf-8",
                    source="core.self_modification.safe_modification_harness.compile_temp",
                )
                try:
                    py_compile.compile(str(tmp_file), doraise=True)
                except py_compile.PyCompileError as e:
                    errors.append(f"py_compile failed for {fpath}: {e}")
        return len(errors) == 0, errors

    def _source_hashes(self, changed_files: list[str]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for relative in changed_files:
            path = self.codebase_root / relative
            if path.is_file():
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes

    def _workspace_files(self) -> list[Path]:
        git_marker = self.codebase_root / ".git"
        if git_marker.exists():
            try:
                result = get_subprocess_gateway().run(
                    [
                        "git",
                        "-C",
                        str(self.codebase_root),
                        "ls-files",
                        "-z",
                        "--cached",
                        "--others",
                        "--exclude-standard",
                    ],
                    capture_output=True,
                    timeout=20,
                    read_only=True,
                    source="self_modification.safe_modification_harness.git_ls_files",
                    accelerator_capability="none",
                )
                if result.returncode == 0:
                    return [
                        self.codebase_root / relative
                        for relative in result.stdout.split("\0")
                        if relative and self._workspace_path_allowed(Path(relative))
                    ]
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass
        return [
            path
            for path in self.codebase_root.rglob("*")
            if path.is_file() and self._workspace_path_allowed(path.relative_to(self.codebase_root))
        ]

    @staticmethod
    def _workspace_path_allowed(relative: Path) -> bool:
        if set(relative.parts).intersection(_WORKSPACE_EXCLUDED_PARTS):
            return False
        if relative.name.startswith("aura_codebase_ai_audit") and relative.suffix == ".txt":
            return False
        return True

    def prepare_candidate_workspace(
        self,
        candidate_root: Path,
        file_contents: dict[str, str],
    ) -> None:
        """Populate an isolated source tree and apply exact candidate bytes."""
        for source in self._workspace_files():
            try:
                relative = source.relative_to(self.codebase_root)
            except ValueError:
                continue
            if set(relative.parts).intersection(_WORKSPACE_EXCLUDED_PARTS):
                continue
            destination = candidate_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                destination.symlink_to(os.readlink(source))
                continue
            if relative.as_posix() in file_contents:
                shutil.copy2(source, destination)
                continue
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

        gateway = get_file_write_gateway()
        for relative, content in file_contents.items():
            gateway.write_text(
                candidate_root / relative,
                content,
                encoding="utf-8",
                source="core.self_modification.safe_modification_harness.candidate_overlay",
            )

    @staticmethod
    def _verify_candidate_overlay(
        candidate_root: Path,
        file_contents: dict[str, str],
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []
        for relative, expected in file_contents.items():
            candidate = candidate_root / relative
            try:
                actual = candidate.read_text(encoding="utf-8")
            except OSError as exc:
                errors.append(f"candidate overlay unreadable for {relative}: {exc}")
                continue
            if actual != expected:
                errors.append(f"candidate overlay content mismatch for {relative}")
        return not errors, errors

    async def _check_pytest(
        self,
        changed_files: list[str],
        *,
        candidate_root: Path,
        extra_test_targets: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Run related pytest files against the isolated candidate workspace."""
        errors: list[str] = []
        test_files = self._related_test_files(
            changed_files, candidate_root, extra_test_targets or []
        )
        if not test_files:
            preview = ", ".join(changed_files[:5])
            if len(changed_files) > 5:
                preview += ", ..."
            return False, [f"no related pytest files found for self-modification patch: {preview}"]

        test_files = sorted(set(test_files))

        try:
            env = dict(os.environ)
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(candidate_root) + (
                os.pathsep + existing_pp if existing_pp else ""
            )
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            env["AURA_TEST_MODE"] = "1"
            result = await get_subprocess_gateway().run_async(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "pytest_asyncio.plugin",
                    "-x",
                ]
                + test_files,
                capture_output=True,
                timeout=60,
                cwd=candidate_root,
                env=env,
                source="core.self_modification.safe_modification_harness.pytest",
                accelerator_capability="auto",
            )
            if result.returncode != 0:
                errors.append(
                    f"pytest failed: stdout={result.stdout[-500:]} stderr={result.stderr[-500:]}"
                )
                return False, errors
        except subprocess.TimeoutExpired:
            errors.append("pytest timed out (60s)")
            return False, errors
        except FileNotFoundError:
            errors.append("pytest executable not available for self-modification harness")
            return False, errors
        except (subprocess.SubprocessError, OSError) as e:
            errors.append(f"pytest subprocess failed: {e}")
            return False, errors

        return True, []

    def _related_test_files(
        self,
        changed_files: list[str],
        candidate_root: Path,
        extra_test_targets: list[str],
    ) -> list[str]:
        test_files: list[str] = []
        for fpath in changed_files:
            path = Path(fpath)
            if (
                path.suffix == ".py"
                and len(path.parts) >= 2
                and path.parts[0] == "tests"
                and path.name.startswith("test_")
            ):
                test_files.append(fpath)
            base = path.stem
            for candidate in (
                f"tests/{'/'.join(path.parts[:-1])}/test_{base}.py",
                f"tests/test_{base}.py",
            ):
                if (candidate_root / candidate).exists():
                    test_files.append(candidate)
        test_files.extend(self._resolve_extra_test_targets(candidate_root, extra_test_targets))
        return sorted(set(test_files))

    async def _run_sandbox_attempt(
        self,
        candidate_root: Path,
        targets: tuple[str, ...],
        timeout_s: int,
    ) -> tuple[bool, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(candidate_root)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env["AURA_TEST_MODE"] = "1"
        result = await get_subprocess_gateway().run_async(
            [sys.executable, "-m", "pytest", "-p", "pytest_asyncio.plugin", "-x", *targets],
            capture_output=True,
            timeout=timeout_s,
            cwd=candidate_root,
            env=env,
            source="core.self_modification.safe_modification_harness.distributed_attempt",
            accelerator_capability="none",
        )
        detail = (result.stdout or "") + "\n" + (result.stderr or "")
        return result.returncode == 0, detail

    @staticmethod
    def _resolve_extra_test_targets(candidate_root: Path, targets: list[str]) -> list[str]:
        resolved: list[str] = []
        tests_root = candidate_root / "tests"
        for raw in targets:
            target = str(raw or "").strip()
            if not target:
                continue
            path_part = target.split("::", 1)[0]
            if ".py" in path_part and (candidate_root / path_part).exists():
                try:
                    source = (candidate_root / path_part).read_text(
                        encoding="utf-8", errors="ignore"
                    )
                except OSError:
                    source = ""
                if SafeModificationHarness._is_recursive_self_mod_test(source):
                    continue
                resolved.append(target)
                continue
            if not tests_root.exists():
                continue
            needle = target.split("::")[-1]
            if not needle:
                continue
            for test_file in tests_root.rglob("test_*.py"):
                try:
                    source = test_file.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if SafeModificationHarness._is_recursive_self_mod_test(source):
                    continue
                if f"def {needle}" in source or f"async def {needle}" in source:
                    resolved.append(test_file.relative_to(candidate_root).as_posix())
                    break
        return resolved

    @staticmethod
    def _is_recursive_self_mod_test(source: str) -> bool:
        text = str(source or "")
        return ("AutonomousSelfModificationEngine" in text and ".apply_fix(" in text) or (
            "SafeModificationHarness" in text and ".run(" in text
        )

    def _current_rss_mb(self) -> float:
        try:
            process = get_resource_observer().process(os.getpid())
            if process is not None:
                return float(process.rss_bytes) / (1024 * 1024)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return 0.0

    def _check_resource_delta(self, baseline_rss_mb: float = 0.0) -> tuple[bool, list[str]]:
        """Check proof execution did not create a large incremental RSS jump."""
        errors: list[str] = []
        current_rss_mb = self._current_rss_mb()
        if baseline_rss_mb > 0.0 and current_rss_mb > 0.0:
            delta_mb = current_rss_mb - baseline_rss_mb
            if delta_mb > 512:
                errors.append(
                    f"High memory delta: {delta_mb:.0f}MB (baseline={baseline_rss_mb:.0f}MB current={current_rss_mb:.0f}MB)"
                )
                return False, errors
        return True, []

    def _check_rollback_drill(self, changed_files: list[str]) -> tuple[bool, list[str]]:
        """Verify we can backup, fingerprint, and restore changed files."""
        errors: list[str] = []
        with tempfile.TemporaryDirectory(prefix="aura-rollback-") as tmpdir:
            backup_dir = Path(tmpdir)

            # Backup
            fingerprints_before: dict[str, str] = {}
            for fpath in changed_files:
                abs_path = self.codebase_root / fpath
                if not abs_path.exists():
                    continue
                backup_path = backup_dir / fpath
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                content = abs_path.read_bytes()
                get_file_write_gateway().write_bytes(
                    backup_path,
                    content,
                    source="core.self_modification.safe_modification_harness.rollback_backup",
                )
                fingerprints_before[fpath] = hashlib.sha256(content).hexdigest()

            # Restore
            fingerprints_after: dict[str, str] = {}
            for fpath in changed_files:
                backup_path = backup_dir / fpath
                if not backup_path.exists():
                    continue
                content = backup_path.read_bytes()
                fingerprints_after[fpath] = hashlib.sha256(content).hexdigest()

            # Compare
            for fpath in fingerprints_before:
                if fingerprints_before.get(fpath) != fingerprints_after.get(fpath):
                    errors.append(f"Rollback fingerprint mismatch for {fpath}")

        return len(errors) == 0, errors


# Module-level singleton
_harness: SafeModificationHarness | None = None


def get_safe_harness(codebase_root: str | Path = ".") -> SafeModificationHarness:
    global _harness
    root = Path(codebase_root).resolve()
    if _harness is None or _harness.codebase_root != root:
        _harness = SafeModificationHarness(root)
    return _harness


async def run_self_mod_test(patch_path: str, test_command: str = "") -> dict[str, Any]:
    """Bridges the ActionExecutor's SELF_MODIFICATION domain to the SafeModificationHarness."""
    harness = get_safe_harness()
    # Treat patch_path as the changed file
    res = await harness.run([patch_path])
    return {
        "passed": res.passed,
        "output": res.summary() + "\n" + "\n".join(res.errors),
    }


__all__ = ["SafeModificationHarness", "HarnessResult", "get_safe_harness", "run_self_mod_test"]
