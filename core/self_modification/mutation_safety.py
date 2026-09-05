
"""Typed evaluation of candidate code mutations with quarantine.

The original sandbox path checked only ``exit_code != 0`` to decide
whether a mutation was bad.  That silently lumps together very
different failure modes — a SyntaxError, an ImportError, a runtime
TypeError, a failed assertion, a timeout, and an OOM kill — and
returns the same uninformative signal.  Worse, a malformed mutation
that crashed the runner could in some paths take down the parent.

This module reframes the question.  A mutation produces one of seven
typed outcomes:

    COMPILE_FAIL        SyntaxError or other compile-time error
    IMPORT_FAIL         ImportError / ModuleNotFoundError
    RUNTIME_EXCEPTION   raised at runtime (any non-AssertionError)
    ASSERTION_FAIL      tests asserted False
    TIMEOUT             exceeded wall-clock budget
    OOM                 memory limit hit (rlimit / signal 9)
    PASSED              compiled, imported, no exception, tests passed

Evaluation runs in a *subprocess* with rlimits, so the parent process
can never crash because of a bad mutation.  Any non-PASSED outcome is
written to a quarantine directory along with the source, the test
source, stdout, stderr, and a structured diagnostics blob.  Callers
inspect ``MutationDiagnostics.outcome`` to decide whether to retry,
escalate, or discard.

The module has no dependency on the existing ``self_modification``
engine and is safe to call from anywhere a candidate code change needs
to be vetted before being applied.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("core.self_modification.mutation_safety")


# Exit codes the bootstrap uses to signal each typed outcome.  Chosen so
# they don't collide with Python's built-in exit codes (0 = success,
# 1 = uncaught exception, 2 = CLI error).
_BOOTSTRAP_EXIT = {
    "passed": 0,
    "compile_fail": 11,
    "import_fail": 12,
    "assertion_fail": 13,
    "runtime_exception": 14,
}


class MutationOutcome(StrEnum):
    PASSED = "passed"
    COMPILE_FAIL = "compile_fail"
    IMPORT_FAIL = "import_fail"
    RUNTIME_EXCEPTION = "runtime_exception"
    ASSERTION_FAIL = "assertion_fail"
    TIMEOUT = "timeout"
    OOM = "oom"


@dataclass
class MutationDiagnostics:
    outcome: MutationOutcome
    runtime_seconds: float
    exit_code: int
    signal_number: int | None = None
    stdout: str = ""
    stderr: str = ""
    traceback_text: str = ""
    quarantine_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


# ---------------------------------------------------------------------------
# bootstrap (runs inside the child process)
# ---------------------------------------------------------------------------
_BOOTSTRAP_SOURCE = textwrap.dedent(
    '''
    """Bootstrap that compiles, imports, and tests a candidate mutation.

    The parent passes the source path and (optional) test path via
    argv.  Outcomes are signalled via well-known exit codes plus a
    JSON line on the last stdout line so the parent can recover full
    diagnostics even when the child terminates abnormally.
    """
    import json
    import sys
    import traceback
    import types

    EXIT_PASSED = 0
    EXIT_COMPILE_FAIL = 11
    EXIT_IMPORT_FAIL = 12
    EXIT_ASSERTION_FAIL = 13
    EXIT_RUNTIME_EXCEPTION = 14

    _COMPILE_RECOVERABLE = (
        MemoryError,
        RecursionError,
        SyntaxError,
        SystemError,
        TypeError,
        ValueError,
    )
    _RUNTIME_RECOVERABLE = (
        ArithmeticError,
        AttributeError,
        BufferError,
        EOFError,
        LookupError,
        MemoryError,
        NameError,
        OSError,
        ReferenceError,
        RuntimeError,
        StopAsyncIteration,
        StopIteration,
        SystemError,
        TypeError,
        ValueError,
        Warning,
    )

    def _emit(outcome, *, traceback_text="", extra=None):
        payload = {"outcome": outcome, "traceback": traceback_text, "extra": extra or {}}
        sys.stdout.write("__MUTATION_RESULT__:" + json.dumps(payload) + "\\n")
        sys.stdout.flush()

    def _emit_uncaught(exc_type, exc, tb):
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            sys.__excepthook__(exc_type, exc, tb)
            return
        if issubclass(exc_type, (ImportError, ModuleNotFoundError)):
            outcome = "import_fail"
        elif issubclass(exc_type, AssertionError):
            outcome = "assertion_fail"
        else:
            outcome = "runtime_exception"
        _emit(
            outcome,
            traceback_text="".join(traceback.format_exception(exc_type, exc, tb)),
            extra={"uncaught": True, "exception_type": getattr(exc_type, "__name__", str(exc_type))},
        )

    sys.excepthook = _emit_uncaught

    def _run(source_path, test_path):
        try:
            with open(source_path, "r", encoding="utf-8") as fh:
                source = fh.read()
        except (OSError, IOError) as e:
            _emit("compile_fail", traceback_text=f"could not read source: {e}")
            sys.exit(EXIT_COMPILE_FAIL)

        try:
            code_obj = compile(source, source_path, "exec")
        except SyntaxError as e:
            _emit("compile_fail", traceback_text=traceback.format_exc(), extra={"err": str(e)})
            sys.exit(EXIT_COMPILE_FAIL)
        except _COMPILE_RECOVERABLE:
            _emit("compile_fail", traceback_text=traceback.format_exc())
            sys.exit(EXIT_COMPILE_FAIL)

        module = types.ModuleType("aura_mutation_under_test")
        module.__file__ = source_path
        try:
            exec(code_obj, module.__dict__)
        except (ImportError, ModuleNotFoundError):
            _emit("import_fail", traceback_text=traceback.format_exc())
            sys.exit(EXIT_IMPORT_FAIL)
        except AssertionError:
            _emit("assertion_fail", traceback_text=traceback.format_exc())
            sys.exit(EXIT_ASSERTION_FAIL)
        except _RUNTIME_RECOVERABLE:
            _emit("runtime_exception", traceback_text=traceback.format_exc())
            sys.exit(EXIT_RUNTIME_EXCEPTION)

        if test_path:
            try:
                with open(test_path, "r", encoding="utf-8") as fh:
                    test_source = fh.read()
            except (OSError, IOError) as e:
                _emit("runtime_exception", traceback_text=f"could not read test: {e}")
                sys.exit(EXIT_RUNTIME_EXCEPTION)
            try:
                test_code = compile(test_source, test_path, "exec")
            except SyntaxError:
                _emit("compile_fail", traceback_text=traceback.format_exc())
                sys.exit(EXIT_COMPILE_FAIL)
            test_module = types.ModuleType("aura_mutation_test")
            test_module.__dict__.update(module.__dict__)
            try:
                exec(test_code, test_module.__dict__)
            except (ImportError, ModuleNotFoundError):
                _emit("import_fail", traceback_text=traceback.format_exc())
                sys.exit(EXIT_IMPORT_FAIL)
            except AssertionError:
                _emit("assertion_fail", traceback_text=traceback.format_exc())
                sys.exit(EXIT_ASSERTION_FAIL)
            except _RUNTIME_RECOVERABLE:
                _emit("runtime_exception", traceback_text=traceback.format_exc())
                sys.exit(EXIT_RUNTIME_EXCEPTION)

        _emit("passed")
        sys.exit(EXIT_PASSED)

    if __name__ == "__main__":
        if len(sys.argv) < 2:
            _emit("runtime_exception", traceback_text="bootstrap: missing source path")
            sys.exit(EXIT_RUNTIME_EXCEPTION)
        source_path = sys.argv[1]
        test_path = sys.argv[2] if len(sys.argv) > 2 else ""
        _run(source_path, test_path)
    '''
).strip()


# ---------------------------------------------------------------------------
# quarantine
# ---------------------------------------------------------------------------
class QuarantineStore:
    """Writes failed-mutation artifacts to an isolated directory tree.

    Quarantine entries are immutable once written: the parent's job is
    only to triage them.  Each entry gets a uuid-based directory.
    """

    def __init__(self, root: Path | None = None):
        self.root = (
            Path(root)
            if root is not None
            else state_root() / "data" / "mutation_quarantine"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def quarantine(
        self,
        *,
        source: str,
        test_source: str | None,
        diagnostics: MutationDiagnostics,
    ) -> Path:
        entry_id = f"mut-{uuid.uuid4()}"
        entry_dir = self.root / entry_id
        entry_dir.mkdir(parents=True, exist_ok=False)
        file_gateway = get_file_write_gateway()
        file_gateway.write_text(
            entry_dir / "source.py",
            source,
            encoding="utf-8",
            source="core.self_modification.mutation_safety.quarantine_source",
        )
        if test_source:
            file_gateway.write_text(
                entry_dir / "test.py",
                test_source,
                encoding="utf-8",
                source="core.self_modification.mutation_safety.quarantine_test",
            )
        file_gateway.write_text(
            entry_dir / "stdout.log",
            diagnostics.stdout,
            encoding="utf-8",
            source="core.self_modification.mutation_safety.quarantine_stdout",
        )
        file_gateway.write_text(
            entry_dir / "stderr.log",
            diagnostics.stderr,
            encoding="utf-8",
            source="core.self_modification.mutation_safety.quarantine_stderr",
        )
        file_gateway.write_text(
            entry_dir / "result.json",
            json.dumps(diagnostics.to_dict(), indent=2, default=str),
            encoding="utf-8",
            source="core.self_modification.mutation_safety.quarantine_result",
        )
        return entry_dir

    def list_entries(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(p for p in self.root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------
class SafeMutationEvaluator:
    """Subprocess-based evaluator that returns typed mutation outcomes.

    The evaluator never raises on a malformed mutation: any failure is
    folded into the returned ``MutationDiagnostics``.  Passing a
    mutation also writes nothing to quarantine; failures do.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        memory_mb: int = 512,
        quarantine: QuarantineStore | None = None,
    ):
        self.timeout_seconds = float(timeout_seconds)
        self.memory_mb = int(memory_mb)
        self.quarantine = quarantine or QuarantineStore()

    def evaluate(
        self,
        source: str,
        *,
        test_source: str | None = None,
    ) -> MutationDiagnostics:
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="aura_mutation_") as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "candidate.py"
            test_path = tmp / "test.py" if test_source else None
            bootstrap_path = tmp / "_bootstrap.py"
            file_gateway = get_file_write_gateway()
            file_gateway.write_text(
                source_path,
                source,
                encoding="utf-8",
                source="core.self_modification.mutation_safety.candidate_source",
            )
            if test_path is not None:
                file_gateway.write_text(
                    test_path,
                    test_source or "",
                    encoding="utf-8",
                    source="core.self_modification.mutation_safety.test_source",
                )
            file_gateway.write_text(
                bootstrap_path,
                _BOOTSTRAP_SOURCE,
                encoding="utf-8",
                source="core.self_modification.mutation_safety.bootstrap_source",
            )

            cmd = [
                sys.executable,
                *self._python_startup_flags(source, test_source),
                str(bootstrap_path),
                str(source_path),
            ]
            if test_path is not None:
                cmd.append(str(test_path))

            try:
                proc = get_subprocess_gateway().spawn(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(tmp),
                    env=self._safe_env(),
                    text=False,
                    preexec_fn=self._set_rlimits if hasattr(os, "fork") else None,
                    source="core.self_modification.mutation_safety.evaluator_subprocess",
                    accelerator_capability="auto",
                )
            except (subprocess.SubprocessError, OSError) as e:  # pragma: no cover - subprocess setup is platform-bound
                diag = MutationDiagnostics(
                    outcome=MutationOutcome.RUNTIME_EXCEPTION,
                    runtime_seconds=0.0,
                    exit_code=-1,
                    stderr=f"failed to spawn evaluator: {e}",
                )
                self._maybe_quarantine(source, test_source, diag)
                return diag

            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
                runtime = time.monotonic() - start
                exit_code = proc.returncode
                signal_number = (
                    -exit_code if exit_code is not None and exit_code < 0 else None
                )
                diag = self._classify(
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    exit_code=exit_code,
                    signal_number=signal_number,
                    runtime=runtime,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    stdout, stderr = proc.communicate(timeout=2.0)
                except (RuntimeError, AttributeError, TypeError, ValueError):
                    stdout = b""
                    stderr = b""
                diag = MutationDiagnostics(
                    outcome=MutationOutcome.TIMEOUT,
                    runtime_seconds=time.monotonic() - start,
                    exit_code=proc.returncode if proc.returncode is not None else -1,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                )
            except (subprocess.SubprocessError, OSError) as e:  # noqa: BLE001 - last-resort safety net
                diag = MutationDiagnostics(
                    outcome=MutationOutcome.RUNTIME_EXCEPTION,
                    runtime_seconds=time.monotonic() - start,
                    exit_code=-1,
                    stderr=f"evaluator host raised: {e}",
                )

        self._maybe_quarantine(source, test_source, diag)
        return diag

    # ------------------------------------------------------------------
    def _classify(
        self,
        *,
        stdout: str,
        stderr: str,
        exit_code: int | None,
        signal_number: int | None,
        runtime: float,
    ) -> MutationDiagnostics:
        # 1) Look for the bootstrap's structured marker first; it is the
        # most reliable signal when the child reached its emit.
        marker = self._extract_marker(stdout)
        if marker is not None:
            outcome_str = str(marker.get("outcome", ""))
            outcome = self._coerce_outcome(outcome_str)
            return MutationDiagnostics(
                outcome=outcome,
                runtime_seconds=runtime,
                exit_code=exit_code if exit_code is not None else -1,
                signal_number=signal_number,
                stdout=stdout,
                stderr=stderr,
                traceback_text=str(marker.get("traceback", "")),
                extra=dict(marker.get("extra", {}) or {}),
            )

        # 2) No marker: child died without emitting.  OOM and SIGKILL
        # land here.  On Linux/macOS rlimit AS triggers MemoryError
        # (caught by the bootstrap), but rlimit DATA / external killer
        # send SIGKILL with no python frame to catch it.
        if signal_number == signal.SIGKILL:
            outcome = MutationOutcome.OOM
        elif exit_code in {_BOOTSTRAP_EXIT[k] for k in _BOOTSTRAP_EXIT}:
            outcome = self._coerce_outcome_from_code(exit_code)
        else:
            outcome = MutationOutcome.RUNTIME_EXCEPTION
        return MutationDiagnostics(
            outcome=outcome,
            runtime_seconds=runtime,
            exit_code=exit_code if exit_code is not None else -1,
            signal_number=signal_number,
            stdout=stdout,
            stderr=stderr,
            traceback_text="",
        )

    @staticmethod
    def _extract_marker(stdout: str) -> dict[str, Any] | None:
        marker = "__MUTATION_RESULT__:"
        for line in reversed(stdout.splitlines()):
            if line.startswith(marker):
                try:
                    return json.loads(line[len(marker) :])
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _coerce_outcome(value: str) -> MutationOutcome:
        try:
            return MutationOutcome(value)
        except ValueError:
            return MutationOutcome.RUNTIME_EXCEPTION

    @staticmethod
    def _coerce_outcome_from_code(exit_code: int | None) -> MutationOutcome:
        if exit_code is None:
            return MutationOutcome.RUNTIME_EXCEPTION
        for name, code in _BOOTSTRAP_EXIT.items():
            if code == exit_code:
                return MutationOutcome(name)
        return MutationOutcome.RUNTIME_EXCEPTION

    @staticmethod
    def _python_startup_flags(source: str, test_source: str | None) -> list[str]:
        if SafeMutationEvaluator._source_needs_site_packages(source):
            return []
        if test_source and SafeMutationEvaluator._source_needs_site_packages(test_source):
            return []
        # Import-free candidate mutations do not need Python's site-package
        # bootstrap. Skipping it keeps the subprocess sandbox isolated while
        # shaving a large, noisy CI startup cost from the hot pass path.
        return ["-S"]

    @staticmethod
    def _source_needs_site_packages(source: str) -> bool:
        try:
            tree = ast.parse(source or "")
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
            ):
                return True
        return False

    def _maybe_quarantine(
        self,
        source: str,
        test_source: str | None,
        diag: MutationDiagnostics,
    ) -> None:
        if diag.outcome is MutationOutcome.PASSED:
            return
        try:
            entry = self.quarantine.quarantine(
                source=source,
                test_source=test_source,
                diagnostics=diag,
            )
            diag.quarantine_path = str(entry)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            diag.extra["quarantine_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            logger.warning(
                "Mutation quarantine failed after %s outcome: %s",
                diag.outcome.value,
                exc,
            )
            diag.quarantine_path = None

    # ------------------------------------------------------------------
    def _set_rlimits(self) -> None:  # pragma: no cover - runs in the child
        # Address-space limit so a runaway allocation hits MemoryError
        # rather than swap-thrashing.
        bytes_limit = self.memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        except (ValueError, OSError) as _exc:
            logger.debug(
                "Mutation evaluator address-space rlimit unavailable: %s: %s",
                type(_exc).__name__,
                _exc,
            )
        # CPU-time fence at 2x the wall-clock budget, in case wall-clock
        # measurement is unreliable (e.g. the host is suspended).
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(self.timeout_seconds * 2) + 1, int(self.timeout_seconds * 2) + 2),
            )
        except (ValueError, OSError) as _exc:
            logger.debug(
                "Mutation evaluator CPU rlimit unavailable: %s: %s",
                type(_exc).__name__,
                _exc,
            )

    @staticmethod
    def _safe_env() -> dict[str, str]:
        env = dict(os.environ)
        for key in list(env):
            up = key.upper()
            if any(s in up for s in ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL", "AUTH")):
                env.pop(key, None)
        return env
