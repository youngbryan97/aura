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
back over a pipe, and nothing in its environment names this check. A commit whose code looks into the interpreter running
it goes to a person. Code built to detect the check by other means can still go unseen: no test run can rule that out from
inside the process being tested.

The result is one JSON object on stdout, then a line in words when the commit did not hold. The exit code is 1 when it did
not hold. Only a commit the machine could not check out exits 0 unjudged: a commit that breaks the check does not pass it.
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
#: What in the interpreter a commit's code has no ordinary reason to read: the frames, hooks and raw exits a test run stands on.
_THE_INTERPRETER_ITSELF = frozenset({"sys._getframe", "sys.monitoring", "sys.settrace", "sys.setprofile", "sys.addaudithook",
                                     "gc.get_objects", "gc.get_referrers", "inspect.currentframe", "inspect.stack", "os._exit"})
#: The constructors of what holds threads apart.
_LOCK_KINDS = frozenset({"Lock", "RLock", "Semaphore", "BoundedSemaphore", "Condition", "checked_lock", "checked_async_lock"})
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
sys.monitoring.use_tool_id(sys.monitoring.COVERAGE_ID, "coverage")
sys.monitoring.register_callback(sys.monitoring.COVERAGE_ID, sys.monitoring.events.PY_START, _s)
sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, sys.monitoring.events.PY_START)
'''


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "core.fsmonitor=false", *args], check=True,
                          capture_output=True, text=True, errors="replace").stdout


def shown_at(repo: Path, commit: str, name: str) -> bytes:
    """A file as a commit holds it, in the bytes it was written in."""
    return subprocess.run(["git", "-C", str(repo), "-c", "core.fsmonitor=false", "show", f"{commit}:{name}"], check=True,
                          capture_output=True).stdout


class CouldNotStart(Exception):
    """The commit could not be checked out, so nothing about it was judged."""


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
            tree = ast.parse(init.read_bytes())
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and stem in {(node.module or "").rsplit(".", 1)[-1],
                                                             *(alias.name for alias in node.names)}:
                out.append(parent.as_posix().replace("/", "."))
                break
    return out


def _modules_that_hand_it_on(checkout: Path, name: str, dotted: str) -> list[str]:
    """The modules that import a changed module only to offer its names as their own."""
    stem = dotted.rsplit(".", 1)[-1]
    argv = ["git", "-C", str(checkout), "-c", "core.fsmonitor=false", "grep", "-l", "-F", "-e", dotted, "-e", f".{stem} import",
            "--", "*.py"]
    out = []
    for path in subprocess.run(argv, capture_output=True, text=True, check=False).stdout.splitlines():
        if path == name or _a_test_file(path):
            continue
        try:
            tree = ast.parse((checkout / path).read_bytes())
        except (OSError, SyntaxError, ValueError):
            continue
        brought = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                   and (node.module or "").endswith(dotted.rsplit(".", 1)[-1]) for alias in node.names}
        used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        if brought and not brought & used:
            out.append(path[:-3].removesuffix("/__init__").replace("/", "."))
    return out


def tests_naming(checkout: Path, files: Sequence[str]) -> list[str]:
    """The test files anywhere in ``checkout`` that name a changed module or a module that passes it on."""
    found: set[str] = set()
    for name in files:
        if not name.endswith(".py") or _a_test_file(name):
            continue
        dotted = name[:-3].removesuffix("/__init__").replace("/", ".")
        patterns = [dotted, name, f"import {Path(name).stem}", f"from {Path(name).stem} import"]
        for package in [*_packages_that_pass_it_on(checkout, name), *_modules_that_hand_it_on(checkout, name, dotted)]:
            patterns += [f"from {package} import", f"import {package}"]
        argv = ["git", "-C", str(checkout), "-c", "core.fsmonitor=false", "grep", "-l", "-F"]
        shown = subprocess.run([*argv, *itertools.chain(*(("-e", item) for item in patterns)), "--", "*.py"],
                               capture_output=True, text=True, check=False).stdout
        found.update(line for line in shown.splitlines() if _a_test_file(line))
    return sorted(found - set(files))


def _bounded(argv: Sequence[str], checkout: Path, scratch: Path, *, network: bool, also: Path | None = None) -> list[str]:
    if sys.platform != "darwin":
        return list(argv)
    profile = _BOUNDARY.format(network="" if network else "(deny network*)", checkout=os.path.realpath(checkout),
                               scratch=os.path.realpath(scratch))
    if also is not None:
        profile += f'(allow file-write* (subpath "{os.path.realpath(also)}"))'
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
    private, scratch, ran = work / f"p{token}", work / f"s{token}", work / secrets.token_hex(8)
    for folder in (private, scratch, ran):
        folder.mkdir()
    watched = {os.path.realpath(checkout / name): list(names) for name, names in (functions or {}).items()}
    read_end, write_end = os.pipe()
    config = json.dumps({"functions": watched, "pipe": write_end, "mark": token})
    plugin = (HERE / "value_witness.py").read_text(encoding="utf-8").replace(
        "CONFIG: dict[str, Any] = {}", f"CONFIG: dict[str, Any] = json.loads({config!r})")
    (private / f"run_{token}.py").write_text(plugin, encoding="utf-8")
    (private / "sitecustomize.py").write_text(_RAN.format(watched=watched, folder=str(ran)), encoding="utf-8")
    argv = _bounded([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", f"run_{token}", *tests],
                    checkout, scratch, network=True, also=ran)
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
    names: set[str] = set()
    for path in ran.iterdir():
        try:
            names.update(path.read_text(encoding="utf-8").splitlines())
        except (OSError, ValueError):
            continue  # what a test left there says nothing about what ran
    known = {f"{where}::{function}" for where, functions_there in watched.items() for function in functions_there}
    return {**report, "ran": sorted(names & known)}


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
    found = [node.value for node in ast.walk(ast.parse(shown_at(repo, commit, name)))
             if isinstance(node, ast.Constant) and getattr(node, "lineno", 0) in lines
             and isinstance(node.value, (bool, int, float, str, bytes))]
    return list({repr(value): value for value in found}.values())


def _span(node: Any) -> set[int]:
    first = min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", ()))])
    return set(range(first, (node.end_lineno or node.lineno) + 1))


def changed_functions(repo: Path, commit: str, name: str) -> list[str]:
    """The functions of ``name`` the commit changed: those whose lines or decorators it touched, and those that read a
    module-level name or class attribute it touched."""
    lines = _changed_lines(repo, commit, name)
    tree = ast.parse(shown_at(repo, commit, name))
    assignments = (ast.Assign, ast.AnnAssign, ast.AugAssign)
    def assigned(body: Sequence[ast.stmt]) -> set[str]:
        return {target.id for statement in body if isinstance(statement, assignments) and lines & _span(statement)
                for target in ast.walk(statement) if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)}

    touched = assigned(tree.body)
    attributes = {name for node in ast.walk(tree) if isinstance(node, ast.ClassDef) for name in assigned(node.body)}
    names: list[str] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                reads = {item.id for item in ast.walk(child) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}
                reached = {item.attr for item in ast.walk(child) if isinstance(item, ast.Attribute)}
                if lines & _span(child) or reads & touched or reached & attributes:
                    names.append(f"{prefix}{child.name}")
                walk(child, f"{prefix}{child.name}.<locals>.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")

    walk(tree, "")
    return names


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else ""


def reaches_into_the_interpreter(repo: Path, commit: str, name: str) -> list[str]:
    """Where the lines a commit added read the frames, hooks or raw exits of the process they run in."""
    lines = _changed_lines(repo, commit, name)
    tree = ast.parse(shown_at(repo, commit, name))
    return sorted({f"{name}:{node.lineno}: {_dotted(node)}" for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                   and node.lineno in lines and _dotted(node) in _THE_INTERPRETER_ITSELF})


def _locks_held(source: bytes) -> dict[str, int]:
    """How many ``with`` blocks each function holds on something built to keep threads apart."""
    tree = ast.parse(source)
    locks = {target.id if isinstance(target, ast.Name) else target.attr for node in ast.walk(tree)
             if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call)
             and (getattr(node.value.func, "attr", "") in _LOCK_KINDS or getattr(node.value.func, "id", "") in _LOCK_KINDS)
             for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
             if isinstance(target, (ast.Name, ast.Attribute))}
    held: dict[str, int] = {}
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            count = sum(1 for node in ast.walk(function) if isinstance(node, (ast.With, ast.AsyncWith)) for item in node.items
                        if (getattr(item.context_expr, "id", "") in locks or getattr(item.context_expr, "attr", "") in locks))
            if count:
                held[function.name] = count
    return held


def _raises(source: bytes) -> dict[str, int]:
    """How many ``raise`` statements each function holds."""
    return {node.name: sum(isinstance(item, ast.Raise) for item in ast.walk(node)) for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _like_it(value: Any) -> list[list[Any]]:
    """Other values of the very kind ``value`` is, in groups: each side of zero is a kind of its own."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        if value == 0:
            return [[value + 1, value + 2], [value - 1, value - 2]]
        side = -1 if value < 0 else 1
        return [[other for other in (value * 2, value + side, value + side * 2) if other != value]]
    if isinstance(value, str):
        return [[other for other in (f"{value}_another", value + value, value.upper()) if other != value]]
    if isinstance(value, bytes):
        return [[other for other in (value + b"_another", value + value) if other != value]]
    return []


def _moved_together(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The call with every number moved by the same step, for a fault that lies in how two numbers relate."""
    numbers = {name: ast.literal_eval(text) for name, text in call["args"].items()}
    numbers = {name: value for name, value in numbers.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    if len(numbers) < 2:
        return []
    return [{**call, "args": {**call["args"], **{name: repr(value + step) for name, value in numbers.items()}}} for step in (1, 2)]


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


def _an_object(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 4 and value[0] == "__object__"


def _inside(call: Mapping[str, Any]) -> list[tuple[tuple[str, ...], Any]]:
    """Each plain value a call carries, by where it sits: an argument, a part of an object passed, or the object's own state."""
    found: list[tuple[tuple[str, ...], Any]] = []
    for field in ("args", "self"):
        for name, text in call.get(field, {}).items():
            value = ast.literal_eval(text)
            if _an_object(value):
                found += [((field, name, part), item) for part, item in value[3].items()]
            else:
                found.append(((field, name), value))
    return found


def _changed_at(call: Mapping[str, Any], where: tuple[str, ...], value: Any) -> dict[str, Any]:
    field, name, *part = where
    if not part:
        return {**call, field: {**call[field], name: repr(value)}}
    kind, module, qualname, parts = ast.literal_eval(call[field][name])
    return {**call, field: {**call[field], name: repr((kind, module, qualname, {**parts, part[0]: value}))}}


def _near_one(call: Mapping[str, Any], constants: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [dict(call), *_moved_together(call)]
    for where, value in _inside(call):
        others = [*(item for item in constants if type(item) is type(value)), *itertools.chain(*_like_it(value)),
                  *_beside_it(value), None]
        out += [_changed_at(call, where, other) for other in others]
    return out


def inputs_near(calls: Sequence[Mapping[str, Any]], constants: Sequence[Any] = (),
                pinned: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """The calls the commit's own tests pin and what lies near them come first, so a long test suite cannot crowd them out.
    Then the other recorded calls and what lies near them, an argument taken from another call, and a member of a table
    or list passed beside it."""
    out: list[dict[str, Any]] = [item for call in pinned for item in _near_one(call, constants)]
    out += [item for call in calls for item in _near_one(call, constants)]
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


def _given_none(key: str) -> bool:
    return any(value is None for _, value in _inside(json.loads(key)))


def judged(before: Mapping[str, Any], again: Mapping[str, Any], after: Mapping[str, Any],
           pinned: Mapping[str, Sequence[Mapping[str, Any]]],
           gave_up_a_raise: Sequence[str] = ()) -> tuple[list[str], list[str], bool]:
    """(what the change demonstrably lost, what a person has to settle, whether its tests pin a new value for an input that
    already had one), from what each input came to before and after.

    ``pinned`` holds the calls the commit's own tests make: those it has said what they should do. ``gave_up_a_raise`` names
    the functions it took a ``raise`` out of.
    """
    lost, ask = [], []
    any_respecified = False
    for name, results in before.items():
        now = after.get(name, {})
        steady = {key: value for key, value in results.items() if again.get(name, {}).get(key) == value and key in now
                  and value != "took too long" and not value.startswith("unavailable")}
        shown = {_key(call): call for call in pinned.get(name, ())}
        mended = {_raised(steady[key]) for key in shown if key in steady and _raised(steady[key]) and not _raised(now[key])}
        respecified = any(key in steady and not _raised(steady[key]) and now[key] != steady[key] for key in shown)
        any_respecified = any_respecified or respecified
        for key, value in steady.items():
            if key in shown or now[key] == value:
                continue
            refused_with = _kind(_raised(value))
            if not _raised(value):
                (ask if respecified else lost).append(f"{name}: {key[:160]} was {value[:60]}, now {now[key][:60]}")
            elif not _raised(now[key]) and _raised(value) not in mended:
                lost.append(f"{name}: {key[:160]} was refused with {refused_with}, now {now[key][:60]}")
            elif not _raised(now[key]) and name.split("::")[1].rsplit(".", 1)[-1] in gave_up_a_raise:
                ask.append(f"{name}: {key[:160]} was refused with {refused_with} by a check it removed, now {now[key][:60]}")
            elif _raised(now[key]) and _kind(_raised(now[key])) != refused_with and not _given_none(key):
                ask.append(f"{name}: {key[:160]} was refused with {refused_with}, now {_raised(now[key])[:60]}")
        for key, call in shown.items():
            if key not in steady or not _raised(steady[key]) or _raised(now[key]):
                continue
            groups = [[_key(_changed_at(call, where, other)) for other in group] for where, value in _inside(call)
                      for group in _like_it(value)]
            groups += [[_key(item) for item in _moved_together(call)]]
            for like in groups:
                if like and all(_raised(now.get(item, "")) == _raised(steady[key]) for item in like):
                    ask.append(f"{name}: the fault still happens on {like[0][:160]}: {now[like[0]][:60]}")
    return sorted(set(lost)), sorted(set(ask)), any_respecified


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
    work = Path(tempfile.mkdtemp())
    old, new = work / secrets.token_hex(6) / "checkout", work / secrets.token_hex(6) / "checkout"
    try:
        files = changed_files(repo, commit)
        git(repo, "worktree", "add", "-q", "--detach", str(old), f"{commit}^")
        git(repo, "worktree", "add", "-q", "--detach", str(new), commit)
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise CouldNotStart(f"{type(exc).__name__}: {exc}") from exc
    try:
        added = [name for name in files if _a_test_file(name) and (new / name).is_file()]
        edited = [name for name in files if _a_test_file(name) and (old / name).exists()]
        guards = sorted({*tests_naming(new, files), *edited})
        present = [name for name in guards if (new / name).is_file()]
        functions = {name: changed_functions(repo, commit, name) for name in files
                     if name.endswith(".py") and not _a_test_file(name) and not _machinery(name) and (new / name).exists()}
        first, second = run_tests(old, guards, work, functions), run_tests(old, guards, work, functions)
        third = run_tests(new, present, work, functions)
        for name in edited:
            if (new / name).is_dir():
                shutil.rmtree(new / name)
            (new / name).parent.mkdir(parents=True, exist_ok=True)
            (new / name).write_bytes(shown_at(repo, f"{commit}^", name))
        as_written = run_tests(new, edited, work)["verdicts"]
        git(new, "reset", "-q", "--hard", commit)
        git(new, "clean", "-fdq")
        # A test renamed where it stands still passes as it was first written, so it is not a test lost.
        broke = sorted(node for node, passed in first["verdicts"].items() if passed and not third["verdicts"].get(node, False)
                       and not (as_written.get(node, False) and (new / node.split("::")[0]).is_file()))
        rewritten = sorted(node for node, passed in first["verdicts"].items() if passed and node.split("::")[0] in edited
                           and node not in broke and not as_written.get(node, False))
        spec = run_tests(new, added, work, functions)
        for name in added:  # the commit's own tests, asked of the code before it
            if (old / name).parent.is_file():
                (old / name).parent.unlink()
            (old / name).parent.mkdir(parents=True, exist_ok=True)
            (old / name).write_bytes((new / name).read_bytes())
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
        near = {name: inputs_near(calls, constants.get(name.split("::")[0], ()), pinned.get(name, ())) for name, calls in seeds.items()}
        old_near, old_times = _differential(old, near, work)
        again_near, again_times = _differential(old, near, work)
        new_near, new_times = _differential(new, near, work)
        sources = [name for name in functions if (old / name).exists()]
        before_after = {name: (shown_at(repo, f"{commit}^", name), shown_at(repo, commit, name)) for name in sources}
        gave_up = sorted({function for was, now in before_after.values() for function, count in _raises(was).items()
                          if count > _raises(now).get(function, 0)})
        lost, asks, respecified = judged(old_near, again_near, new_near, pinned, gave_up)
        if respecified:  # its tests say a value that was there is wrong, so a call beside it may rightly change too
            asks += [f"a call in the tests returns another value: {key}" for key in changed]
            changed = []
        unlocked = sorted(f"{name}::{function}" for name, (was, now) in before_after.items()
                          for function, count in _locks_held(was).items() if _locks_held(now).get(function, 0) < count)
        reaching = [line for name in files if name.endswith(".py") and (new / name).is_file()
                    for line in reaches_into_the_interpreter(repo, commit, name)]
        ran = {f"{os.path.relpath(item.split('::')[0], os.path.realpath(new))}::{item.split('::')[1]}"
               for item in {*third["ran"], *spec["ran"]}}
        unreached = sorted(f"{name}::{function}" for name, names in functions.items() for function in names
                           if f"{name}::{function}" not in ran)
        nested = sorted(f"{name}::{function}" for name, names in functions.items() for function in names if "<locals>" in function)
        slowdown = slower(old_times, again_times, new_times)
        machinery = [name for name in files if _machinery(name)]
        asks += [f"a test it edited fails as it was first written: {node}" for node in rewritten]
        asks += [f"no test ran this function it changed: {name}" for name in unreached]
        asks += [f"it holds a lock in fewer places here: {name}" for name in unlocked]
        asks += [f"its code reads the interpreter running it: {line}" for line in reaching]
        asks += [f"it changes how tests run: {name}" for name in machinery]
        asks += [f"this function is reachable only from inside another, so no input of its own was tried: {name}" for name in nested]
        asks += [f"it is slower: {slowdown}"] if slowdown else []
        values = {"compared": sum(1 for key, value in steady.items() if second["values"].get(key) == value and key in third["values"]),
                  "functions": {name: names for name, names in functions.items() if names},
                  "near_inputs": sum(len(calls) for calls in near.values()), "changed": changed[:20],
                  "changed_near": lost[:20], "no_test_ran": unreached[:20],
                  "held": not changed and not lost}
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
    reasons += [] if result["reverts_cleanly"]["held"] else ["reverting it does not give back the code before it"]
    return reasons + [f"a person has to settle this: {line}" for line in result["asks"][:3]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(args.repo, args.commit)
    except CouldNotStart as exc:
        print(json.dumps({"commit": args.commit, "held": None, "could_not_be_run": str(exc)[:400]}))
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
