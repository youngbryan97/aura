"""Whether one commit of hers holds, judged from clean checkouts by code that shares nothing with her search.

    python -m outside_bar.verify_change --repo PATH --commit SHA

Three verdicts. HELD: nothing counts against the commit. REFUSED: it demonstrably lost something that worked.
ASKS A PERSON: its behaviour changed in a way behaviour alone cannot settle, such as a new special case or a corrected
value, so someone who knows what the code is for has to say.

1. Its new tests fail before it and pass after it. A commit that adds no test has nothing to show here, and says so.
2. The tests that reach the files it changed pass before and after, and a test it edited still passes as it was written.
3. The functions it changed keep what they did, on the calls its tests make and on inputs near them. A value that differs
   between two runs of the old code depends on the moment and is not compared.
4. Reverting it applies cleanly and gives back the tree of its parent.
5. It leaves alone the machinery that decides how tests run.

The code under test runs inside a boundary that can write only to its own checkout and a scratch folder. Its results come
back over a pipe, and nothing in its environment names this check. Code built to detect the check and behave differently
under it can still go unseen: no test run can rule that out from inside the process being tested.

The result is one JSON object on stdout, then a line in words when the commit did not hold. The exit code is 1 when it did
not hold. A check the machine could not run says so and exits 0: it found nothing against the commit.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
#: How long one run of tests may take, in seconds: the bound her own gates give one.
TEST_SECONDS = 2400
#: Inputs tried for one function: each costs a call on both sides, and the nearest are made first.
NEAR_INPUTS = 200
#: The chance of a slowdown this consistent by luck below which it is reported, the convention the outside bar judges by.
ALPHA = 0.05
#: The added time per call below which a slowdown is not reported: a branch costs nanoseconds, and nobody waits on those.
NOTICEABLE_SECONDS = 0.001
#: Files that decide how tests are collected and run. A commit that changes them changes the judge.
_MACHINERY = frozenset({"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", ".coveragerc",
                        "sitecustomize.py", "noxfile.py"})
#: What code under test may do: write only in its own checkout and scratch folder.
_BOUNDARY = ('(version 1)(allow default){network}(deny file-write*)(allow file-write* (subpath "{checkout}") '
             '(subpath "{scratch}") (literal "/dev/null") (literal "/dev/zero") (regex #"^/dev/(tty|fd/)"))')
#: Loaded at the start of every Python process a test starts, so a function a test runs in a child process counts as run.
_RAN = '''import os, sys
_W, _D = {watched!r}, {folder!r}
def _s(code, _o):
    if code.co_qualname in _W.get(code.co_filename, ()):
        try:
            with open(os.path.join(_D, str(os.getpid())), "a", encoding="utf-8") as h:
                h.write(code.co_filename + "::" + code.co_qualname + "\\n")
        except OSError:
            pass
    return sys.monitoring.DISABLE
sys.monitoring.use_tool_id(sys.monitoring.COVERAGE_ID, "outside_bar_ran")
sys.monitoring.register_callback(sys.monitoring.COVERAGE_ID, sys.monitoring.events.PY_START, _s)
sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, sys.monitoring.events.PY_START)
'''


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "core.fsmonitor=false", *args], check=True,
                          capture_output=True, text=True).stdout


def changed_files(repo: Path, commit: str) -> list[str]:
    return [line for line in git(repo, "diff", "--name-only", f"{commit}^", commit).splitlines() if line]


def _a_test_file(name: str) -> bool:
    stem = Path(name).name
    return name.endswith(".py") and (stem.startswith("test_") or stem.endswith("_test.py"))


def _machinery(name: str) -> bool:
    return Path(name).name in _MACHINERY or Path(name).name.startswith("pytest_")


def _packages_that_pass_it_on(checkout: Path, name: str) -> list[str]:
    """The packages above a module whose ``__init__.py`` imports it, so that importing the package reaches it."""
    out, stem = [], Path(name).stem
    for parent in Path(name).parents:
        init = checkout / parent / "__init__.py"
        if str(parent) == "." or not init.exists():
            continue
        try:
            tree = ast.parse(init.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and stem in {(node.module or "").rsplit(".", 1)[-1],
                                                             *(alias.name for alias in node.names)}:
                out.append(parent.as_posix().replace("/", "."))
                break
    return out


def tests_naming(checkout: Path, files: Sequence[str]) -> list[str]:
    """The test files anywhere in ``checkout`` that name a changed module or a package that passes it on."""
    found: set[str] = set()
    for name in files:
        if not name.endswith(".py") or _a_test_file(name):
            continue
        dotted = name[:-3].removesuffix("/__init__").replace("/", ".")
        patterns = [dotted, name, f"import {Path(name).stem}", f"from {Path(name).stem} import"]
        for package in _packages_that_pass_it_on(checkout, name):
            patterns += [f"from {package} import", f"import {package}"]
        argv = ["git", "-C", str(checkout), "-c", "core.fsmonitor=false", "grep", "-l", "-F"]
        shown = subprocess.run([*argv, *itertools.chain(*(("-e", item) for item in patterns)), "--", "*.py"],
                               capture_output=True, text=True, check=False).stdout
        found.update(line for line in shown.splitlines() if _a_test_file(line))
    return sorted(found - set(files))


def _bounded(argv: Sequence[str], checkout: Path, scratch: Path, *, network: bool) -> list[str]:
    if sys.platform != "darwin":
        return list(argv)
    profile = _BOUNDARY.format(network="" if network else "(deny network*)", checkout=os.path.realpath(checkout),
                               scratch=os.path.realpath(scratch))
    return ["sandbox-exec", "-p", profile, *argv]


def _child_environment(scratch: Path, path: str = "") -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != "AURA_TESTING" and not key.startswith("OUTSIDE_BAR")}
    env.update(AURA_LOG_DIR=str(scratch), AURA_STATE_ROOT=str(scratch), TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED="0", PYTHONPATH=path, VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1")
    return env


def _read_all(descriptor: int, into: list[str]) -> threading.Thread:
    def read() -> None:
        with os.fdopen(descriptor, encoding="utf-8", errors="replace") as pipe:
            into.append(pipe.read())

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    return thread


def run_tests(checkout: Path, tests: Sequence[str], work: Path,
              functions: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Each test's verdict and what the changed functions did, from a pytest run inside the boundary."""
    empty: dict[str, Any] = {"verdicts": {}, "values": {}, "inputs": {}, "ran": []}
    if not tests:
        return empty
    token = secrets.token_hex(8)
    private, scratch = work / f"p{token}", work / f"s{token}"
    ran = scratch / "ran"
    ran.mkdir(parents=True)
    private.mkdir()
    watched = {os.path.realpath(checkout / name): list(names) for name, names in (functions or {}).items()}
    read_end, write_end = os.pipe()
    config = json.dumps({"functions": watched, "pipe": write_end, "mark": token})
    plugin = (HERE / "value_witness.py").read_text(encoding="utf-8").replace(
        "CONFIG: dict[str, Any] = {}", f"CONFIG: dict[str, Any] = json.loads({config!r})")
    (private / f"run_{token}.py").write_text(plugin, encoding="utf-8")
    (private / "sitecustomize.py").write_text(_RAN.format(watched=watched, folder=str(ran)), encoding="utf-8")
    argv = _bounded([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", f"run_{token}", *tests],
                    checkout, scratch, network=True)
    got: list[str] = []
    reader = _read_all(read_end, got)
    process = subprocess.Popen(argv, cwd=checkout, env=_child_environment(scratch, str(private)), pass_fds=(write_end,),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(write_end)
    try:
        process.wait(timeout=TEST_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    reader.join(timeout=30)
    try:
        report = json.loads(got[0]) if got else {}
    except ValueError:
        report = {}
    if report.get("mark") != token:
        return empty
    names = {line for path in ran.iterdir() for line in path.read_text(encoding="utf-8").splitlines()}
    return {**report, "ran": sorted(names)}


def _differential(checkout: Path, inputs: Mapping[str, Any], work: Path) -> tuple[dict[str, dict[str, str]], dict[str, float]]:
    """What each input came to in ``checkout``, and how long it took, from a process that cannot reach this one's files."""
    token = secrets.token_hex(8)
    private, scratch = work / f"p{token}", work / f"s{token}"
    private.mkdir()
    scratch.mkdir()
    shutil.copyfile(HERE / "differential.py", private / "main.py")
    read_end, write_end = os.pipe()
    got: list[str] = []
    reader = _read_all(read_end, got)
    process = subprocess.Popen(_bounded([sys.executable, str(private / "main.py")], checkout, scratch, network=False),
                               cwd=checkout, env=_child_environment(scratch), pass_fds=(write_end,), text=True,
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(write_end)
    try:
        process.communicate(json.dumps({"inputs": inputs, "pipe": write_end, "mark": token}), timeout=TEST_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    reader.join(timeout=30)
    outcomes: dict[str, dict[str, str]] = {}
    seconds: dict[str, float] = {}
    for line in "".join(got).splitlines():
        try:
            mark, name, key, outcome, took = json.loads(line)
        except ValueError:
            continue
        if mark == token:
            outcomes.setdefault(name, {})[key] = outcome
            seconds[f"{name}\n{key}"] = float(took)
    return outcomes, seconds


def values_that_changed(before: Mapping[str, str], again: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """The calls that returned one steady value before and return something else now."""
    return sorted(key for key, value in before.items()
                  if not value.startswith("raised ") and again.get(key) == value and key in after and after[key] != value)


def _changed_lines(repo: Path, commit: str, name: str) -> set[int]:
    diff = git(repo, "diff", "-U0", f"{commit}^", commit, "--", name)
    lines: set[int] = set()
    for found in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.MULTILINE):
        first, count = int(found.group(1)), int(found.group(2) or 1)
        lines.update(range(first, first + max(count, 1)))
    return lines


def constants_it_added(repo: Path, commit: str, name: str) -> list[Any]:
    """The plain constants on the lines a commit added to a file. A new branch on one value is reached only by that value,
    so each is tried as an input (``data == b"xyz"`` is found by no input near the ones the tests use)."""
    lines = _changed_lines(repo, commit, name)
    found = [node.value for node in ast.walk(ast.parse(git(repo, "show", f"{commit}:{name}")))
             if isinstance(node, ast.Constant) and getattr(node, "lineno", 0) in lines
             and isinstance(node.value, (bool, int, float, str, bytes))]
    return list({repr(value): value for value in found}.values())


def changed_functions(repo: Path, commit: str, name: str) -> list[str]:
    """The functions of ``name`` the commit changed: those whose lines it touched, and those that read a module-level
    name it touched."""
    lines = _changed_lines(repo, commit, name)
    tree = ast.parse(git(repo, "show", f"{commit}:{name}"))
    touched = {target.id for statement in tree.body
               if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign))
               and lines & set(range(statement.lineno, (statement.end_lineno or statement.lineno) + 1))
               for target in ast.walk(statement) if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)}
    names: list[str] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                reads = {item.id for item in ast.walk(child) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}
                if lines & set(range(child.lineno, (child.end_lineno or child.lineno) + 1)) or reads & touched:
                    names.append(f"{prefix}{child.name}")
                walk(child, f"{prefix}{child.name}.<locals>.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")

    walk(tree, "")
    return names


def _like_it(value: Any) -> list[Any]:
    """Other values of the very kind ``value`` is: more of the same side of zero, or the same text carried further."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        side = -1 if value < 0 else 1
        return [other for other in (value * 2, value + side, value + side * 2) if other != value]
    if isinstance(value, str):
        return [other for other in (f"{value}_another", value + value, value.upper()) if other != value]
    if isinstance(value, bytes):
        return [other for other in (value + b"_another", value + value) if other != value]
    return []


def _beside_it(value: Any) -> list[Any]:
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, (int, float)):
        return [value + 1, -value - 1]
    if isinstance(value, (list, tuple, dict, set)) and value:
        return [type(value)()]
    return []


def _with(call: Mapping[str, Any], name: str, value: Any) -> dict[str, Any]:
    return {**call, "args": {**call["args"], name: repr(value)}}


def _key(call: Mapping[str, Any]) -> str:
    return json.dumps(call, sort_keys=True)


def inputs_near(calls: Sequence[Mapping[str, Any]], constants: Sequence[Any] = ()) -> list[dict[str, Any]]:
    """The recorded calls; each with an argument set to a constant the commit added; each with one argument changed to one
    like it, beside it, or None; then each with an argument taken from another call, or from a table or list beside it."""
    out: list[dict[str, Any]] = [dict(call) for call in calls]
    for call in calls:
        for name, text in call["args"].items():
            value = ast.literal_eval(text)
            out += [_with(call, name, other) for other in constants if type(other) is type(value)]
            out += [_with(call, name, other) for other in [*_like_it(value), *_beside_it(value), None]]
    for call, other in itertools.product(calls, calls):
        out += [{**call, "args": {**call["args"], name: other["args"][name]}} for name in call["args"] if name in other["args"]]
    for call in calls:
        for holder, text in call["args"].items():
            held = ast.literal_eval(text)
            members = list(held) if isinstance(held, (dict, list, tuple, set)) else []
            for name, own in call["args"].items():
                if name != holder:
                    out += [_with(call, name, member) for member in members
                            if type(member) is type(ast.literal_eval(own))]
    return list({_key(item): item for item in out}.values())[:NEAR_INPUTS]


def _raised(outcome: str) -> str:
    return outcome.split(";", 1)[0] if outcome.startswith("raised ") else ""


def _kind(refusal: str) -> str:
    return refusal.removeprefix("raised ").split(":", 1)[0]


def judged(before: Mapping[str, Any], again: Mapping[str, Any], after: Mapping[str, Any],
           pinned: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[str], list[str]]:
    """(what the change demonstrably lost, what a person has to settle), from what each input came to before and after.

    ``pinned`` holds the calls the commit's own tests make: those it has said what they should do.
    """
    lost, ask = [], []
    for name, results in before.items():
        now = after.get(name, {})
        steady = {key: value for key, value in results.items() if again.get(name, {}).get(key) == value and key in now
                  and value != "took too long" and not value.startswith("unavailable")}
        shown = {_key(call): call for call in pinned.get(name, ())}
        mended = {_raised(steady[key]) for key in shown if key in steady and _raised(steady[key]) and not _raised(now[key])}
        respecified = any(key in steady and not _raised(steady[key]) and now[key] != steady[key] for key in shown)
        for key, value in steady.items():
            if key in shown or now[key] == value:
                continue
            if not _raised(value):
                (ask if respecified else lost).append(f"{name}: {key[:160]} was {value[:60]}, now {now[key][:60]}")
            elif not _raised(now[key]) and _raised(value) not in mended:
                lost.append(f"{name}: {key[:160]} was refused with {_kind(_raised(value))}, now {now[key][:60]}")
            elif _raised(now[key]) and _kind(_raised(now[key])) != _kind(_raised(value)) and "None" not in key:
                lost.append(f"{name}: {key[:160]} was refused with {_kind(_raised(value))}, now {_raised(now[key])[:60]}")
        for key, call in shown.items():
            if key not in steady or not _raised(steady[key]) or _raised(now[key]):
                continue
            for argument, text in call["args"].items():
                value = ast.literal_eval(text)
                like = [_key(_with(call, argument, other)) for other in _like_it(value)]
                if like and all(_raised(now.get(item, "")) == _raised(steady[key]) for item in like):
                    line = f"{name}: the fault still happens on {like[0][:160]}: {now[like[0]][:60]}"
                    (lost if isinstance(value, (int, float)) else ask).append(line)
    return sorted(set(lost)), sorted(set(ask))


def slower(before: Mapping[str, float], again: Mapping[str, float], after: Mapping[str, float]) -> str:
    """Words for a slowdown too consistent to be luck and larger than two runs of the old code differ, or nothing."""
    shared = [key for key in before if key in again and key in after]
    if not shared:
        return ""
    worse = sum(after[key] > max(before[key], again[key]) for key in shared)
    chance = sum(math.comb(len(shared), k) * 2 ** (len(shared) - k) for k in range(worse, len(shared) + 1)) / 3 ** len(shared)
    old, older, new = (statistics.median(times[key] for key in shared) for times in (before, again, after))
    if chance < ALPHA and new - max(old, older) > max(abs(old - older), NOTICEABLE_SECONDS):
        return f"each call takes about {new:.3f} s, and took {max(old, older):.3f} s, on {worse} of {len(shared)} inputs"
    return ""


def _a_passing_test_made_it(key: str, kept: set[str]) -> bool:
    return key.rsplit("|", 3)[0] in kept


def _seeds(found: Mapping[str, Any], checkout: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for where, calls in found.items():
        path, function = where.split("::")
        out[f"{os.path.relpath(path, os.path.realpath(checkout))}::{function}"] = calls
    return out


def verify(repo: Path, commit: str) -> dict[str, Any]:
    """What each question came to, and the verdict."""
    files = changed_files(repo, commit)
    work = Path(tempfile.mkdtemp(prefix="outside-bar-checkouts-"))
    old, new = work / secrets.token_hex(6) / "checkout", work / secrets.token_hex(6) / "checkout"
    git(repo, "worktree", "add", "-q", "--detach", str(old), f"{commit}^")
    git(repo, "worktree", "add", "-q", "--detach", str(new), commit)
    try:
        added = [name for name in files if _a_test_file(name) and (new / name).exists()]
        edited = [name for name in files if _a_test_file(name) and (old / name).exists()]
        guards = sorted({*tests_naming(new, files), *edited})
        present = [name for name in guards if (new / name).exists()]
        functions = {name: changed_functions(repo, commit, name) for name in files
                     if name.endswith(".py") and not _a_test_file(name) and not _machinery(name) and (new / name).exists()}
        first, second = run_tests(old, guards, work, functions), run_tests(old, guards, work, functions)
        third = run_tests(new, present, work, functions)
        broke = sorted(node for node, passed in first["verdicts"].items() if passed and not third["verdicts"].get(node, False))
        for name in edited:
            (new / name).parent.mkdir(parents=True, exist_ok=True)
            (new / name).write_text(git(repo, "show", f"{commit}^:{name}"), encoding="utf-8")
        as_written = run_tests(new, edited, work)["verdicts"]
        for name in edited:
            if name in added:
                git(new, "checkout", "-q", commit, "--", name)
            else:
                (new / name).unlink()
        rewritten = sorted(node for node, passed in first["verdicts"].items() if passed and node.split("::")[0] in edited
                           and node not in broke and not as_written.get(node, False))
        spec = run_tests(new, added, work, functions)
        for name in added:  # the commit's own tests, asked of the code before it
            (old / name).parent.mkdir(parents=True, exist_ok=True)
            (old / name).write_text((new / name).read_text(encoding="utf-8"), encoding="utf-8")
        before = run_tests(old, added, work)["verdicts"]
        newly_passing = [node for node, passed in spec["verdicts"].items() if passed and not before.get(node, False)]
        shows_the_fault = {"applies": bool(added), "newly_passing": newly_passing[:20],
                           "held": not added or (bool(newly_passing) and all(spec["verdicts"].values()))}
        kept = {node for node, passed in first["verdicts"].items() if passed and third["verdicts"].get(node, False)}
        steady = {key: value for key, value in first["values"].items() if _a_passing_test_made_it(key, kept)}
        changed = values_that_changed(steady, second["values"], third["values"])
        pinned = _seeds(spec["inputs"], new)
        seeds = {name: [*calls, *pinned.get(name, ())] for name, calls in _seeds(third["inputs"], new).items()}
        seeds.update({name: calls for name, calls in pinned.items() if name not in seeds})
        constants = {name: constants_it_added(repo, commit, name) for name in functions}
        near = {name: inputs_near(calls, constants.get(name.split("::")[0], ())) for name, calls in seeds.items()}
        old_near, old_times = _differential(old, near, work)
        again_near, again_times = _differential(old, near, work)
        new_near, new_times = _differential(new, near, work)
        lost, asks = judged(old_near, again_near, new_near, pinned)
        ran = {f"{os.path.relpath(item.split('::')[0], os.path.realpath(new))}::{item.split('::')[1]}"
               for item in {*third["ran"], *spec["ran"]}}
        unreached = sorted(f"{name}::{function}" for name, names in functions.items() for function in names
                           if f"{name}::{function}" not in ran)
        nested = sorted(f"{name}::{function}" for name, names in functions.items() for function in names if "<locals>" in function)
        slowdown = slower(old_times, again_times, new_times)
        machinery = [name for name in files if _machinery(name)]
        asks += [f"a test it edited fails as it was first written: {node}" for node in rewritten]
        asks += [f"it changes how tests run: {name}" for name in machinery]
        asks += [f"this function is reachable only from inside another, so no input of its own was tried: {name}" for name in nested]
        asks += [f"it is slower: {slowdown}"] if slowdown else []
        values = {"compared": sum(1 for key, value in steady.items() if second["values"].get(key) == value and key in third["values"]),
                  "functions": {name: names for name, names in functions.items() if names},
                  "near_inputs": sum(len(calls) for calls in near.values()), "changed": changed[:20],
                  "changed_near": lost[:20], "no_test_ran": unreached[:20],
                  "held": not changed and not lost and not unreached}
        try:
            git(new, "-c", "user.name=outside-bar", "-c", "user.email=bar@localhost", "revert", "--no-edit", commit)
            reverted = git(new, "rev-parse", "HEAD^{tree}").strip() == git(repo, "rev-parse", f"{commit}^^{{tree}}").strip()
        except subprocess.CalledProcessError:
            reverted = False
        answers = {"shows_the_fault": shows_the_fault, "tests_held": {"tests": len(first["verdicts"]), "broke": broke[:20], "held": not broke},
                   "values_kept": values, "reverts_cleanly": {"held": reverted}}
        refused = not all(answer["held"] for answer in answers.values())
        verdict = "REFUSED" if refused else "ASKS A PERSON" if asks else "HELD"
        return {"commit": commit, "files": files, "verdict": verdict, "held": verdict == "HELD", "asks": asks[:20], **answers}
    finally:
        for checkout in (old, new):
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(checkout)], check=False,
                           capture_output=True)
        shutil.rmtree(work, ignore_errors=True)


def _what_failed(result: Mapping[str, Any]) -> list[str]:
    """Each failed question in words, with what showed it, for whoever mends the change next."""
    values = result["values_kept"]
    reasons = [] if result["shows_the_fault"]["held"] else ["its new test does not fail before it and pass after it"]
    reasons += [f"this test passed before and fails now: {node}" for node in result["tests_held"]["broke"][:3]]
    reasons += [f"an input that worked gives another value: {line}" for line in values["changed_near"][:3]]
    reasons += [f"a call in the tests returns another value: {line}" for line in values["changed"][:3]]
    reasons += [f"no test ran this function it changed: {name}" for name in values["no_test_ran"][:3]]
    reasons += [] if result["reverts_cleanly"]["held"] else ["reverting it does not give back the code before it"]
    return reasons + [f"a person has to settle this: {line}" for line in result["asks"][:3]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(args.repo, args.commit)
    except (OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"commit": args.commit, "held": None, "could_not_be_run": f"{type(exc).__name__}: {exc}"[:400]}))
        return 0
    except Exception as exc:  # noqa: BLE001 - a commit that breaks the check does not pass it
        print(json.dumps({"commit": args.commit, "held": False, "verdict": "REFUSED", "unreadable": f"{type(exc).__name__}: {exc}"[:400]}))
        print(f"did not hold: the check could not read this change: {type(exc).__name__}: {exc}"[:400])
        return 1
    print(json.dumps(result, indent=1))
    if not result["held"]:
        print("did not hold:", "; ".join(_what_failed(result)))
    return 0 if result["held"] else 1


if __name__ == "__main__":
    sys.exit(main())
