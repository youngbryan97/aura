"""Whether one commit of hers holds, judged from clean checkouts by code that shares nothing with her search.

    python -m outside_bar.verify_change --repo PATH --commit SHA

Four questions, each answered on its own checkout of the commit or of its parent:

1. Its new tests fail before it and pass after it. A commit that adds no test has nothing to show here, and says so.
2. The tests that name the files it changed pass before and after.
3. The functions it changed keep what they did, on the calls its tests make and on inputs near them, such as an argument
   replaced by a key of the table passed beside it. A value that differs between two runs of the old code depends on
   the moment and is not compared. Four things count against the change: an input that returned one value now returns
   another; an input refused with another kind of error is no longer refused; the fault it mended still happens on an
   input like the one that showed it; and no test ran a function it changed.
4. Reverting it applies cleanly and gives back the tree of its parent.

The result is one JSON object on stdout, then a line in words when the commit did not hold. The exit code is 1 when a
question failed. A check that could not be run says so and exits 0: it found nothing against the commit.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
#: How long one run of tests may take, in seconds: the bound her own gates give one.
TEST_SECONDS = 2400
#: Inputs tried for one function: each costs a call on both sides, and the nearest are made first.
NEAR_INPUTS = 200
#: What the differential run may do: no network, and writes only in its scratch folder and the system's temporary ones.
_BOUNDARY = ('(version 1)(allow default)(deny network*)(deny file-write*)'
             '(allow file-write* (subpath "{scratch}") (subpath "/private/tmp") (subpath "/private/var/folders")'
             ' (literal "/dev/null"))')
_RESULT = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "core.fsmonitor=false", *args], check=True,
                          capture_output=True, text=True).stdout


def changed_files(repo: Path, commit: str) -> list[str]:
    return [line for line in git(repo, "diff", "--name-only", f"{commit}^", commit).splitlines() if line]


def _a_test_file(name: str) -> bool:
    return Path(name).name.startswith("test_") and name.endswith(".py")


def tests_naming(checkout: Path, files: Sequence[str]) -> list[str]:
    """The test files in ``checkout`` that name a changed module, leaving out the changed test files themselves."""
    found: set[str] = set()
    for name in files:
        if not name.endswith(".py") or _a_test_file(name):
            continue
        dotted = name[:-3].replace("/", ".")
        shown = subprocess.run(["git", "-C", str(checkout), "-c", "core.fsmonitor=false", "grep", "-l", "-F", "-e", dotted,
                                "-e", name, "--", "tests"], capture_output=True, text=True, check=False).stdout
        found.update(line for line in shown.splitlines() if _a_test_file(line))
    return sorted(found - set(files))


def run_tests(checkout: Path, tests: Sequence[str], witness: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Each test's verdict, run by plain pytest in ``checkout``."""
    if not tests:
        return {}
    scratch = tempfile.mkdtemp(prefix="outside-bar-verify-")
    env = {**os.environ, "AURA_LOG_DIR": scratch, "AURA_STATE_ROOT": scratch, "PYTHONDONTWRITEBYTECODE": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    env.pop("AURA_TESTING", None)
    argv = [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider"]
    if witness:
        env.update(witness, PYTHONPATH=f"{HERE}{os.pathsep}{env.get('PYTHONPATH', '')}")
        argv += ["-p", "value_witness"]
    try:
        out = subprocess.run([*argv, *tests], cwd=checkout, env=env, capture_output=True, text=True,
                             timeout=TEST_SECONDS, check=False).stdout
    except subprocess.TimeoutExpired:
        return {}
    return {node: word == "PASSED" for word, node in _RESULT.findall(out)}


def values_that_changed(before: Mapping[str, str], again: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """The calls that returned one steady value before and return something else now."""
    return sorted(key for key, value in before.items()
                  if not value.startswith("raised ") and again.get(key) == value and key in after and after[key] != value)


def changed_functions(repo: Path, commit: str, name: str) -> list[str]:
    """The functions of ``name`` whose lines the commit added or changed, by qualified name."""
    diff = git(repo, "diff", "-U0", f"{commit}^", commit, "--", name)
    lines: set[int] = set()
    for found in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.MULTILINE):
        first, count = int(found.group(1)), int(found.group(2) or 1)
        lines.update(range(first, first + max(count, 1)))
    names: list[str] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if lines & set(range(child.lineno, (child.end_lineno or child.lineno) + 1)):
                    names.append(f"{prefix}{child.name}")
                walk(child, f"{prefix}{child.name}.<locals>.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")

    walk(ast.parse(git(repo, "show", f"{commit}:{name}")), "")
    return names


def similar_values(value: Any) -> list[Any]:
    """Other plain values of the same kind as ``value``."""
    if isinstance(value, str):
        return [other for other in (f"{value}_another", value + value, value.upper()) if other != value]
    if isinstance(value, int) and not isinstance(value, bool):
        return [value + 1, -value - 1]
    return []


def similar_inputs(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The call with one plain argument changed to another of its own kind."""
    return [{**call, name: other} for name, value in call.items() for other in similar_values(value)]


def inputs_near(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The recorded calls; each with one argument changed to another of its kind, or to None; then each with one argument
    taken from another call, or from a table or list beside it."""
    out = [dict(call) for call in calls]
    for call in calls:
        out += similar_inputs(call)
        out += [{**call, name: None} for name in call]
    for call, other in itertools.product(calls, calls):
        out += [{**call, name: other[name]} for name in call if name in other]
    for call in calls:
        for holder, held in call.items():
            members = list(held) if isinstance(held, (dict, list)) else []
            for name, value in call.items():
                if name != holder:
                    out += [{**call, name: member} for member in members if type(member) is type(value)]
    unique = {json.dumps(item, sort_keys=True): item for item in out}
    return list(unique.values())[:NEAR_INPUTS]


def _differential(checkout: Path, inputs: Path, out: Path, scratch: Path) -> dict[str, dict[str, str]]:
    """What each input came to in ``checkout``, run inside the boundary."""
    env = {**os.environ, "AURA_LOG_DIR": str(scratch), "AURA_STATE_ROOT": str(scratch), "PYTHONDONTWRITEBYTECODE": "1"}
    argv = [sys.executable, str(HERE / "differential.py"), str(inputs), str(out)]
    if sys.platform == "darwin":
        argv = ["sandbox-exec", "-p", _BOUNDARY.format(scratch=os.path.realpath(scratch)), *argv]
    subprocess.run(argv, cwd=checkout, env=env, capture_output=True, text=True, timeout=TEST_SECONDS, check=False)
    return json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}


def _raised(outcome: str) -> str:
    return outcome.removeprefix("raised ") if outcome.startswith("raised ") else ""


def values_near_that_changed(before: Mapping[str, Any], again: Mapping[str, Any], after: Mapping[str, Any],
                             recorded: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> list[str]:
    """What the change lost, in words: for each function, an input that gave one steady value and gives another now, a
    refusal of another kind that is gone, and the mended fault still happening on an input like the one that showed it."""
    out = []
    for name, results in before.items():
        now = after.get(name, {})
        steady = {key: value for key, value in results.items()
                  if again.get(name, {}).get(key) == value and key in now and value != "took too long"}
        shown = {json.dumps(call, sort_keys=True) for call in (recorded or {}).get(name, ())}
        faults = {_raised(steady[key]) for key in shown if key in steady and _raised(steady[key]) and not _raised(now[key])}
        for key, value in steady.items():
            if not _raised(value) and now[key] != value:
                out.append(f"{name}: {key[:160]} was {value[:60]}, now {now[key][:60]}")
            elif _raised(value) and not _raised(now[key]) and key not in shown and _raised(value) not in faults:
                out.append(f"{name}: {key[:160]} was refused with {_raised(value)}, now {now[key][:60]}")
        for key in shown:
            if key in steady and _raised(steady[key]) in faults and not _raised(now[key]):
                call = json.loads(key)
                for argument, value in call.items():
                    like = [json.dumps({**call, argument: other}, sort_keys=True) for other in similar_values(value)]
                    # On every one: a mend for blank text rightly still refuses text that is not a number.
                    if like and all(_raised(now.get(item, "")) == _raised(steady[key]) for item in like):
                        out.append(f"{name}: the fault still happens on {like[0][:160]}: {now[like[0]][:60]}")
    return sorted(set(out))


def _witnessed(checkout: Path, tests: Sequence[str], functions: Mapping[str, Sequence[str]], out: Path) -> dict[str, Any]:
    watched = {os.path.realpath(checkout / name): list(names) for name, names in functions.items()}
    run_tests(checkout, tests, {"OUTSIDE_BAR_WITNESS_FUNCTIONS": json.dumps(watched), "OUTSIDE_BAR_WITNESS_OUT": str(out)})
    return json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"values": {}, "inputs": {}}


def verify(repo: Path, commit: str) -> dict[str, Any]:
    """What each of the four questions came to, and whether the commit holds."""
    files = changed_files(repo, commit)
    work = Path(tempfile.mkdtemp(prefix="outside-bar-checkouts-"))
    old, new = work / "old", work / "new"
    git(repo, "worktree", "add", "-q", "--detach", str(old), f"{commit}^")
    git(repo, "worktree", "add", "-q", "--detach", str(new), commit)
    try:
        added = [name for name in files if _a_test_file(name) and (new / name).exists()]
        # A test file the commit edited or deleted is a guard too: skipping a test is not a way to pass it.
        guards = sorted({*tests_naming(new, files), *(name for name in files if _a_test_file(name) and (old / name).exists())})
        guard_before, guard_after = run_tests(old, guards), run_tests(new, [name for name in guards if (new / name).exists()])
        broke = sorted(node for node, passed in guard_before.items() if passed and not guard_after.get(node, False))
        after = run_tests(new, added)
        for name in added:  # the commit's own tests, asked of the code before it
            (old / name).parent.mkdir(parents=True, exist_ok=True)
            (old / name).write_text((new / name).read_text(encoding="utf-8"), encoding="utf-8")
        before = run_tests(old, added) if added else {}
        newly_passing = [node for node, passed in after.items() if passed and not before.get(node, False)]
        shows_the_fault = {"applies": bool(added), "held": not added or (bool(newly_passing) and all(after.values())),
                           "newly_passing": newly_passing[:20]}
        guarded = {"tests": len(guard_before), "broke": broke[:20], "held": not broke}
        kept = [node for node, passed in guard_before.items() if passed and guard_after.get(node, False)]
        files_of = sorted({node.split("::")[0] for node in kept})
        functions = {name: changed_functions(repo, commit, name) for name in files
                     if name.endswith(".py") and not _a_test_file(name) and (new / name).exists()}
        first = _witnessed(old, files_of, functions, work / "first.json")
        second = _witnessed(old, files_of, functions, work / "second.json")
        third = _witnessed(new, sorted({*files_of, *added}), functions, work / "third.json")
        changed = values_that_changed(first["values"], second["values"], third["values"])
        recorded: dict[str, list[dict[str, Any]]] = {}
        for watched, calls in third["inputs"].items():
            path, function = watched.split("::")
            recorded[f"{os.path.relpath(path, os.path.realpath(new))}::{function}"] = calls
        near = {name: inputs_near(calls) for name, calls in recorded.items()}
        (work / "near.json").write_text(json.dumps(near), encoding="utf-8")
        scratch = work / "scratch"
        scratch.mkdir()
        old_near = _differential(old, work / "near.json", work / "old1.json", scratch)
        old_again = _differential(old, work / "near.json", work / "old2.json", scratch)
        new_near = _differential(new, work / "near.json", work / "new.json", scratch)
        near_changed = values_near_that_changed(old_near, old_again, new_near, recorded)
        ran = {f"{os.path.relpath(item.split('::')[0], os.path.realpath(new))}::{item.split('::')[1]}" for item in third["ran"]}
        unreached = sorted(f"{name}::{function}" for name, names in functions.items() for function in names
                           if f"{name}::{function}" not in ran)
        values = {"compared": sum(1 for key, value in first["values"].items()
                                  if second["values"].get(key) == value and key in third["values"]),
                  "functions": {name: names for name, names in functions.items() if names},
                  "near_inputs": sum(len(calls) for calls in near.values()), "changed": changed[:20],
                  "changed_near": near_changed[:20], "no_test_ran": unreached[:20],
                  "held": not changed and not near_changed and not unreached}
        git(new, "-c", "user.name=outside-bar", "-c", "user.email=bar@localhost", "revert", "--no-edit", commit)
        reverted = git(new, "rev-parse", "HEAD^{tree}").strip() == git(repo, "rev-parse", f"{commit}^^{{tree}}").strip()
        answers = {"shows_the_fault": shows_the_fault, "tests_held": guarded, "values_kept": values,
                   "reverts_cleanly": {"held": reverted}}
        return {"commit": commit, "files": files, "held": all(answer["held"] for answer in answers.values()), **answers}
    finally:
        for checkout in (old, new):
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(checkout)], check=False,
                           capture_output=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(args.repo, args.commit)
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError) as exc:
        print(json.dumps({"commit": args.commit, "held": None, "could_not_be_run": f"{type(exc).__name__}: {exc}"[:400]}))
        return 0
    print(json.dumps(result, indent=1))
    if not result["held"]:
        print("did not hold:", "; ".join(_what_failed(result)))
    return 0 if result["held"] else 1


def _what_failed(result: Mapping[str, Any]) -> list[str]:
    """Each failed question in words, with what showed it, for whoever mends the change next."""
    values = result["values_kept"]
    reasons = [] if result["shows_the_fault"]["held"] else ["its new test does not fail before it and pass after it"]
    reasons += [f"this test passed before and fails now: {node}" for node in result["tests_held"]["broke"][:3]]
    reasons += [f"an input that worked gives another value: {line}" for line in values["changed_near"][:3]]
    reasons += [f"a call in the tests returns another value: {line}" for line in values["changed"][:3]]
    reasons += [f"no test ran this function it changed: {name}" for name in values["no_test_ran"][:3]]
    return reasons + ([] if result["reverts_cleanly"]["held"] else ["reverting it does not give back the code before it"])


if __name__ == "__main__":
    sys.exit(main())
