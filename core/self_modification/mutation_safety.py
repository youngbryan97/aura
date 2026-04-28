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

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


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


class MutationOutcome(str, Enum):
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
    signal_number: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    traceback_text: str = ""
    quarantine_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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

    def _emit(outcome, *, traceback_text="", extra=None):
        payload = {"outcome": outcome, "traceback": traceback_text, "extra": extra or {}}
        sys.stdout.write("__MUTATION_RESULT__:" + json.dumps(payload) + "\\n")
        sys.stdout.flush()

    def _run(source_path, test_path):
        try:
            with open(source_path, "r", encoding="utf-8") as fh:
                source = fh.read()
        except Exception as e:
            _emit("compile_fail", traceback_text=f"could not read source: {e}")
            sys.exit(EXIT_COMPILE_FAIL)

        try:
            code_obj = compile(source, source_path, "exec")
        except SyntaxError as e:
            _emit("compile_fail", traceback_text=traceback.format_exc(), extra={"err": str(e)})
            sys.exit(EXIT_COMPILE_FAIL)
        except Exception:
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
        except Exception:
            _emit("runtime_exception", traceback_text=traceback.format_exc())
            sys.exit(EXIT_RUNTIME_EXCEPTION)

        if test_path:
            try:
                with open(test_path, "r", encoding="utf-8") as fh:
                    test_source = fh.read()
            except Exception as e:
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
            except Exception:
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

    def __init__(self, root: Optional[Path] = None):
        self.root = (
            Path(root)
            if root is not None
            else Path.home() / ".aura" / "data" / "mutation_quarantine"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def quarantine(
        self,
        *,
        source: str,
        test_source: Optional[str],
        diagnostics: MutationDiagnostics,
    ) -> Path:
        entry_id = f"mut-{uuid.uuid4()}"
        entry_dir = self.root / entry_id
        entry_dir.mkdir(parents=True, exist_ok=False)
        (entry_dir / "source.py").write_text(source, encoding="utf-8")
        if test_source:
            (entry_dir / "test.py").write_text(test_source, encoding="utf-8")
        (entry_dir / "stdout.log").write_text(diagnostics.stdout, encoding="utf-8")
        (entry_dir / "stderr.log").write_text(diagnostics.stderr, encoding="utf-8")
        (entry_dir / "result.json").write_text(
            json.dumps(diagnostics.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return entry_dir

    def list_entries(self) -> List[Path]:
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
        quarantine: Optional[QuarantineStore] = None,
    ):
        self.timeout_seconds = float(timeout_seconds)
        self.memory_mb = int(memory_mb)
        self.quarantine = quarantine or QuarantineStore()

    def evaluate(
        self,
        source: str,
        *,
        test_source: Optional[str] = None,
    ) -> MutationDiagnostics:
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="aura_mutation_") as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "candidate.py"
            test_path = tmp / "test.py" if test_source else None
            bootstrap_path = tmp / "_bootstrap.py"
            source_path.write_text(source, encoding="utf-8")
            if test_path is not None:
                test_path.write_text(test_source or "", encoding="utf-8")
            bootstrap_path.write_text(_BOOTSTRAP_SOURCE, encoding="utf-8")

            cmd = [sys.executable, str(bootstrap_path), str(source_path)]
            if test_path is not None:
                cmd.append(str(test_path))

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(tmp),
                    env=self._safe_env(),
                    preexec_fn=self._set_rlimits if hasattr(os, "fork") else None,
                )
            except Exception as e:  # pragma: no cover - subprocess setup is platform-bound
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
                except Exception:
                    stdout = b""
                    stderr = b""
                diag = MutationDiagnostics(
                    outcome=MutationOutcome.TIMEOUT,
                    runtime_seconds=time.monotonic() - start,
                    exit_code=proc.returncode if proc.returncode is not None else -1,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                )
            except Exception as e:  # noqa: BLE001 - last-resort safety net
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
        exit_code: Optional[int],
        signal_number: Optional[int],
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
    def _extract_marker(stdout: str) -> Optional[Dict[str, Any]]:
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
    def _coerce_outcome_from_code(exit_code: Optional[int]) -> MutationOutcome:
        if exit_code is None:
            return MutationOutcome.RUNTIME_EXCEPTION
        for name, code in _BOOTSTRAP_EXIT.items():
            if code == exit_code:
                return MutationOutcome(name)
        return MutationOutcome.RUNTIME_EXCEPTION

    def _maybe_quarantine(
        self,
        source: str,
        test_source: Optional[str],
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
        except Exception:
            # Quarantine failures must not change the outcome that the
            # parent observed.  We swallow here so a malformed mutation
            # cannot escalate via the quarantine layer.
            diag.quarantine_path = None

    # ------------------------------------------------------------------
    def _set_rlimits(self) -> None:  # pragma: no cover - runs in the child
        # Address-space limit so a runaway allocation hits MemoryError
        # rather than swap-thrashing.
        bytes_limit = self.memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        except (ValueError, OSError):
            pass
        # CPU-time fence at 2x the wall-clock budget, in case wall-clock
        # measurement is unreliable (e.g. the host is suspended).
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(self.timeout_seconds * 2) + 1, int(self.timeout_seconds * 2) + 2),
            )
        except (ValueError, OSError):
            pass

    @staticmethod
    def _safe_env() -> Dict[str, str]:
        env = dict(os.environ)
        for key in list(env):
            up = key.upper()
            if any(s in up for s in ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL", "AUTH")):
                env.pop(key, None)
        return env
