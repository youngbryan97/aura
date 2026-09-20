"""Does anything notice when a protection stops protecting?

Not general mutation testing — that would be hopeless across 24,900 tests and
would mostly find unimportant survivors. This targets ONE question, the one
that matters for a system whose gates are its main claim to maturity:

    if I disable this guard, does any test go red?

A guard is a function whose job is to REFUSE something: it raises, or it
returns False/None to deny. The mutation makes it always allow. That is the
exact failure mode that matters — a check that silently stopped checking — and
it is the failure mode already found by hand three times today (the mycelium
corruption tests, the agent-loop `ok` log, held_out_challenge).

Verdicts:
  KILLED    — a test failed. The protection is genuinely covered.
  SURVIVED  — every test still passed with the guard disabled. The coverage is
              vacuous: the protection could be deleted and nothing would say so.
  BROKEN    — mutation made the module unimportable; inconclusive, not a pass.

Safety: the original bytes are captured before any edit and rewritten in a
finally, then verified by hash. A run that cannot restore a file aborts loudly
rather than leaving the tree mutated.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = os.environ.get("PYTHON") or sys.executable


class _Neutralize(ast.NodeTransformer):
    """Replace refusals with acceptance inside one target function."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.in_target = False
        self.changes = 0

    def _visit_func(self, node):
        if node.name != self.target:
            # Do not descend into unrelated nested functions.
            return node
        self.in_target = True
        self.generic_visit(node)
        self.in_target = False
        return node

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Raise(self, node: ast.Raise):
        if not self.in_target:
            return node
        self.changes += 1
        return ast.Pass()

    def visit_Return(self, node: ast.Return):
        if not self.in_target:
            return node
        v = node.value
        # `return False` / `return None` are denials in a guard.
        if isinstance(v, ast.Constant) and v.value in (False, None):
            self.changes += 1
            return ast.Return(value=ast.Constant(value=True))
        return node


@dataclass
class Result:
    module: str
    function: str
    selector: str
    verdict: str
    changes: int = 0
    detail: str = ""
    failed_tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "function": self.function,
            "selector": self.selector,
            "verdict": self.verdict,
            "changes": self.changes,
            "detail": self.detail,
            "failed_tests": self.failed_tests[:6],
        }


def mutate_source(source: str, function: str) -> tuple[str | None, int]:
    tree = ast.parse(source)
    t = _Neutralize(function)
    tree = t.visit(tree)
    if not t.changes:
        return None, 0
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), t.changes


def run_tests(selector: str, timeout: int) -> tuple[bool, list[str], str]:
    """Returns (all_passed, failed_test_ids, detail)."""
    env = dict(os.environ)
    env["AURA_LOG_DIR"] = str(Path(__file__).parent / "logs")
    try:
        proc = subprocess.run(
            [PYTHON, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:randomly", *selector.split()],
            cwd=REPO, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return False, [], "timeout"
    out = proc.stdout + proc.stderr
    failed = [ln.split(" ")[0] for ln in out.splitlines() if ln.startswith("FAILED")]
    if "no tests ran" in out or "ERROR" in out and "collected 0" in out:
        return True, [], "no_tests_collected"
    return proc.returncode == 0, failed, ""


def _git_dirty(module: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "status", "--porcelain", "--", module],
        cwd=REPO, capture_output=True, text=True,
    )
    return proc.stdout.strip()


def assert_pristine(module: str) -> None:
    """Refuse to mutate a file that is not already git-clean.

    Learned by causing it, on this tool's own sweep. A run that outlived its
    shell kept mutating in the background; a later chunk then read one of those
    mutated files, captured it as "original", and faithfully restored it TO THE
    MUTATED STATE. Three files were left with their guards replaced by `pass`,
    and every individual probe had restored exactly what it read, so nothing
    reported a failure.

    "Original" is only trustworthy if the file was clean when it was read, and
    git is the one source of truth that survives a killed process.
    """
    dirty = _git_dirty(module)
    if dirty:
        raise SystemExit(
            f"REFUSING to mutate {module}: it is not git-clean.\n"
            f"  {dirty}\n"
            "A previous run may have died mid-probe. Restore it first:\n"
            f"  git checkout -- {module}"
        )


def probe(module: str, function: str, selector: str, timeout: int) -> Result:
    path = REPO / module
    assert_pristine(module)
    original = path.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    try:
        mutated, changes = mutate_source(original.decode("utf-8"), function)
        if mutated is None:
            return Result(module, function, selector, "NO_GUARD",
                          detail="no raise/return-False found in that function")
        path.write_text(mutated, encoding="utf-8")
        # Import check first: an unparseable/unimportable mutant proves nothing.
        mod = module.replace("/", ".")[:-3]
        chk = subprocess.run([PYTHON, "-c", f"import {mod}"], cwd=REPO,
                             capture_output=True, text=True, timeout=180)
        if chk.returncode != 0:
            return Result(module, function, selector, "BROKEN", changes,
                          detail=(chk.stderr or "").strip().splitlines()[-1][:160] if chk.stderr else "import failed")
        passed, failed, detail = run_tests(selector, timeout)
        if detail == "no_tests_collected":
            return Result(module, function, selector, "NO_TESTS", changes, detail=detail)
        if passed:
            return Result(module, function, selector, "SURVIVED", changes, detail=detail)
        return Result(module, function, selector, "KILLED", changes, detail=detail,
                      failed_tests=failed)
    finally:
        path.write_bytes(original)
        restored = hashlib.sha256(path.read_bytes()).hexdigest()
        if restored != original_hash:
            sys.stderr.write(f"\n!!! FAILED TO RESTORE {module} — ABORTING\n")
            raise SystemExit(2)
        # Belt and braces: a restore that faithfully rewrote the bytes we held
        # is still wrong if the bytes we held were already mutated.
        if _git_dirty(module):
            sys.stderr.write(
                f"\n!!! {module} IS DIRTY AFTER RESTORE — ABORTING\n"
                f"    run: git checkout -- {module}\n"
            )
            raise SystemExit(2)


def tests_referencing(module: str, function: str) -> str:
    """Every test file that mentions the function OR its module.

    Deriving the selector from the module NAME instead cost two false
    survivors on this tool's first run: validate_reference_artifact is covered
    by test_frontier_gap.py and recoverable_answer by
    test_turn_ledger_is_live_in_the_response_path.py, neither of which a
    name-based guess would find. A mutation tool that reports phantom findings
    gets distrusted, which is worse than not having one.
    """
    stem = Path(module).stem
    files: set[str] = set()
    for pattern in (function, stem):
        proc = subprocess.run(
            ["grep", "-rl", pattern, "tests/"], cwd=REPO, capture_output=True, text=True
        )
        files.update(f for f in proc.stdout.split() if f.endswith(".py"))
    return " ".join(sorted(files))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="JSON file: [[module, function, selector], ...]")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    targets = json.loads(Path(args.targets).read_text())
    results: list[Result] = []
    for target in targets:
        module, function = target[0], target[1]
        selector = target[2] if len(target) > 2 and target[2] else tests_referencing(module, function)
        if not selector:
            print(f"   no-tests   {function:<38} {module}", flush=True)
            results.append(Result(module, function, "", "NO_TESTS"))
            continue
        r = probe(module, function, selector, args.timeout)
        results.append(r)
        mark = {"SURVIVED": "!! SURVIVED", "KILLED": "   killed  ",
                "BROKEN": "   broken  ", "NO_GUARD": "   no-guard",
                "NO_TESTS": "   no-tests"}.get(r.verdict, r.verdict)
        print(f"{mark}  {function:<38} {module}", flush=True)
        if r.verdict == "SURVIVED":
            print(f"             ^ {r.changes} refusal(s) disabled, tests still green: {selector}", flush=True)

    survived = [r for r in results if r.verdict == "SURVIVED"]
    print(f"\n{'='*72}")
    print(f"probed={len(results)}  SURVIVED={len(survived)}  "
          f"killed={sum(1 for r in results if r.verdict=='KILLED')}  "
          f"broken={sum(1 for r in results if r.verdict=='BROKEN')}  "
          f"no_guard={sum(1 for r in results if r.verdict=='NO_GUARD')}  "
          f"no_tests={sum(1 for r in results if r.verdict=='NO_TESTS')}")
    if survived:
        print("\nVACUOUSLY COVERED PROTECTIONS:")
        for r in survived:
            print(f"  {r.module}::{r.function}")
    if args.out:
        Path(args.out).write_text(json.dumps([r.to_dict() for r in results], indent=2))


if __name__ == "__main__":
    main()
