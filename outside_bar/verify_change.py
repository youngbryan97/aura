"""Whether one commit of hers holds, judged from clean checkouts by code that shares nothing with her search.

    python -m outside_bar.verify_change --repo PATH --commit SHA

Three verdicts, and none of them waits for a person. HELD: nothing counts against the commit. REFUSED: it lost something
that worked, or left her check without the evidence it needs, and goes back to her to mend. PROBATION: its behaviour changed
in a way behaviour alone cannot settle, such as a new special case or a corrected value. It may land, one such change at a
time, and it comes back out if the outside benchmarks fall while it stands.

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
was refused, and 2 when the repository or the commit could not be found. A commit that breaks the check does not pass it.
"""
from __future__ import annotations

import argparse
import ast
import builtins
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
import types
from collections import abc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
#: How long one run of tests may take, in seconds: the bound her own gates give one.
TEST_SECONDS = 2400
#: Inputs tried for one function: each costs a call on both sides, and the nearest are made first.
NEAR_INPUTS = 400
#: The chance of a slowdown this consistent by luck below which it is reported, the convention the outside bar judges by.
ALPHA = 0.05
#: The added time per call below which a slowdown is not reported: a branch costs nanoseconds, and nobody waits on those.
NOTICEABLE_SECONDS = 0.001
#: A slowdown this many times over is a loss, not a doubt: nobody waits ten times as long and calls it the same function.
ORDER_OF_MAGNITUDE = 10
#: Files that decide how tests are collected and run. A commit that changes them changes the judge.
_MACHINERY = frozenset({"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", ".coveragerc",
                        "sitecustomize.py", "noxfile.py"})
#: Names that mean the interpreter's own machinery on whatever object they are read from: frames, compiled code,
#: closures, hooks and the object graph, so ``exc.__traceback__.tb_frame`` counts as ``sys._getframe`` does.
_REFLECTION = frozenset({"_getframe", "_current_frames", "settrace", "setprofile", "addaudithook", "tb_frame", "f_back",
                         "f_locals", "f_globals", "f_builtins", "f_code", "gi_frame", "cr_frame", "ag_frame", "__closure__",
                         "cell_contents", "__globals__", "__code__", "__builtins__", "__subclasses__", "__getattribute__",
                         "__self__", "__func__", "get_objects", "get_referrers", "get_referents", "walk_stack", "walk_tb",
                         "getclosurevars"})
#: The modules a test run stands on. A commit may read ``os.path`` or ``sys.argv``; it may not pass these modules around,
#: and the names below are what in them it has no ordinary reason to touch.
_INTERPRETER = frozenset({"sys", "os", "inspect", "gc", "importlib", "traceback", "builtins", "__main__", "ctypes",
                          "_ctypes", "sitecustomize", "signal"})
#: Their module tables, monitoring, stacks, raw exits, signals and import machinery.
_IN_THE_INTERPRETER = frozenset({"modules", "monitoring", "stack", "currentframe", "_exit", "kill", "killpg", "abort",
                                 "meta_path", "path_hooks", "path_importer_cache", "__dict__", "reload", "getouterframes",
                                 "getinnerframes", "trace", "getframeinfo", "alarm", "signal", "setitimer", "pthread_kill",
                                 "raise_signal", "excepthook", "unraisablehook", "breakpointhook", "displayhook"})
#: Modules nothing a commit adds should import at all: the run's start-up module, the entry module, raw memory.
_NEVER_IMPORTED = frozenset({"__main__", "ctypes", "_ctypes", "sitecustomize", "builtins"})
#: Calls that reach an attribute by a name given as text, and names that run text as code.
_BY_NAME = frozenset({"getattr", "setattr", "delattr", "hasattr", "getattr_static", "attrgetter", "methodcaller"})
#: Calls that find a module or object by a dotted name given as text, and libraries that swap one object for another.
_RESOLVERS = frozenset({"locate", "resolve_name", "_resolve_name", "import_string", "find_class"})
_DOUBLES = frozenset({"mock", "unittest.mock"})
_RUNS_TEXT = frozenset({"eval", "exec", "compile", "__import__", "breakpoint", "__builtins__"})
#: Readers whose input can name any callable to run.
_UNPICKLERS = frozenset({"pickle", "_pickle", "cPickle", "marshal", "dill", "cloudpickle", "shelve", "jsonpickle"})
#: Files whose code the check cannot read: compiled modules, cached bytecode, and path files the interpreter runs.
_UNREADABLE = frozenset({".so", ".pyd", ".dylib", ".dll", ".pyc", ".pth"})
#: The methods by which Python's own mutable containers change, taken from the abstract classes that define them.
_MUTATORS = frozenset((set(dir(abc.MutableSequence)) - set(dir(abc.Sequence)))
                      | (set(dir(abc.MutableMapping)) - set(dir(abc.Mapping))) | (set(dir(abc.MutableSet)) - set(dir(abc.Set))))
#: The constructors of what holds threads apart.
_LOCK_KINDS = frozenset({"Lock", "RLock", "Semaphore", "BoundedSemaphore", "Condition", "checked_lock", "checked_async_lock"})
#: What code under test may do: write only in its own checkout and scratch folder.
_BOUNDARY = ('(version 1)(allow default){network}(deny file-write*)(allow file-write* (subpath "{checkout}") '
             '(subpath "{scratch}") (literal "/dev/null") (literal "/dev/zero") (regex #"^/dev/(tty|fd/)"))')
#: Loaded at the start of every Python process a test starts, so a function a test runs in a child process counts as run.
#: A code object holding a changed line has line events turned on when it first starts, and each changed line is written
#: down the first time it runs, then left alone.
_RAN = '''import sys
def _watching(watched, lines, folder):
    import os
    def write(text):
        try:
            with open(os.path.join(folder, str(os.getpid())), "a", encoding="utf-8") as handle:
                handle.write(text + "\\n")
        except OSError:
            pass
    def seen(code, _offset):
        if code.co_qualname in watched.get(code.co_filename, ()):
            write(code.co_filename + "::" + code.co_qualname)
        wanted = lines.get(code.co_filename)
        if wanted and any(number in wanted for _start, _end, number in code.co_lines()):
            sys.monitoring.set_local_events(sys.monitoring.COVERAGE_ID, code, sys.monitoring.events.LINE)
        return sys.monitoring.DISABLE
    def line(code, number):
        if number in lines.get(code.co_filename, ()):
            write(code.co_filename + "#" + str(number))
        return sys.monitoring.DISABLE
    return seen, line
sys.monitoring.use_tool_id(sys.monitoring.COVERAGE_ID, "coverage")
_seen, _line = _watching({watched!r}, {{name: frozenset(numbers) for name, numbers in {lines!r}.items()}}, {folder!r})
sys.monitoring.register_callback(sys.monitoring.COVERAGE_ID, sys.monitoring.events.PY_START, _seen)
sys.monitoring.register_callback(sys.monitoring.COVERAGE_ID, sys.monitoring.events.LINE, _line)
sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, sys.monitoring.events.PY_START)
del _watching, _seen, _line
sys.modules.pop("sitecustomize", None)
'''


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "core.fsmonitor=false", *args], check=True,
                          capture_output=True, text=True, errors="replace").stdout


def shown_at(repo: Path, commit: str, name: str) -> bytes:
    """A file as a commit holds it, in the bytes it was written in."""
    return subprocess.run(["git", "-C", str(repo), "-c", "core.fsmonitor=false", "show", f"{commit}:{name}"], check=True,
                          capture_output=True).stdout


class CouldNotStart(Exception):
    """The repository or the commit could not be found, so nothing about it was judged."""


class UncheckableCommit(Exception):
    """The commit exists but cannot be checked out cleanly beside its parent: it has no parent, or its tree cannot be
    written to disk. That is a fault of the commit, and it is refused."""


def _in_tree(repo: Path, commit: str, name: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "-c", "core.fsmonitor=false", "cat-file", "-e", f"{commit}:{name}"],
                          capture_output=True, check=False).returncode == 0


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


def _modules_that_hand_it_on(checkout: Path, name: str, dotted: str, changed: Sequence[str] = ()) -> list[str]:
    """The modules that import a changed module only to offer its names as their own, or that import a function the
    commit changed: a test reaching that function through them reaches the change."""
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
        inside = {node.id for definition in ast.walk(tree) if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef,
                  ast.ClassDef)) for node in ast.walk(definition) if isinstance(node, ast.Name)}
        if brought and (not brought & inside or brought & {function.split(".")[0] for function in changed}):
            out.append(path[:-3].removesuffix("/__init__").replace("/", "."))
    return out


def tests_naming(checkout: Path, files: Sequence[str], functions: Mapping[str, Sequence[str]] | None = None) -> list[str]:
    """The test files anywhere in ``checkout`` that name a changed module or a module that passes it on."""
    found: set[str] = set()
    for name in files:
        if not name.endswith(".py") or _a_test_file(name):
            continue
        dotted = name[:-3].removesuffix("/__init__").replace("/", ".")
        patterns = [dotted, name, f"import {Path(name).stem}", f"from {Path(name).stem} import"]
        handing = _modules_that_hand_it_on(checkout, name, dotted, (functions or {}).get(name, ()))
        for package in [*_packages_that_pass_it_on(checkout, name), *handing]:
            patterns += [f"from {package} import", f"import {package}"]
        argv = ["git", "-C", str(checkout), "-c", "core.fsmonitor=false", "grep", "-l", "-F"]
        shown = subprocess.run([*argv, *itertools.chain(*(("-e", item) for item in patterns)), "--", "*.py"],
                               capture_output=True, text=True, check=False).stdout
        found.update(line for line in shown.splitlines() if _a_test_file(line))
    return sorted(found - set(files))


def readers_of(checkout: Path, files: Sequence[str]) -> list[str]:
    """The modules that name a changed file that is not Python, by its file name: what they read changed with it."""
    names = {Path(name).name for name in files if not name.endswith(".py") and not _machinery(name)}
    if not names:
        return []
    argv = ["git", "-C", str(checkout), "-c", "core.fsmonitor=false", "grep", "-l", "-F",
            *itertools.chain(*(("-e", item) for item in sorted(names))), "--", "*.py"]
    shown = subprocess.run(argv, capture_output=True, text=True, check=False).stdout.splitlines()
    # A module beside the file with the same stem reads it by its own name: Path(__file__).with_suffix(".json").
    shown += [str(Path(name).with_suffix(".py")) for name in files if not name.endswith(".py") and not _machinery(name)
              and (checkout / Path(name).with_suffix(".py")).is_file()]
    return sorted({line for line in shown if line and not _a_test_file(line) and line not in files})


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
              functions: Mapping[str, Sequence[str]] | None = None,
              lines: Mapping[str, Sequence[int]] | None = None) -> dict[str, Any]:
    """Each test's verdict and what the changed functions did, from a pytest run inside the boundary. ``lines`` names,
    for each file, the lines whose running is written down; ``lines_ran`` gives back those that ran."""
    empty: dict[str, Any] = {"verdicts": {}, "values": {}, "inputs": {}, "seen": [], "ran": [], "finished": True,
                             "whole": True, "ran_here": [], "lines_ran": []}
    if not tests:
        return empty
    token = secrets.token_hex(8)
    private, scratch, ran = work / f"p{token}", work / f"s{token}", work / secrets.token_hex(8)
    for folder in (private, scratch, ran):
        folder.mkdir()
    watched = {os.path.realpath(checkout / name): list(names) for name, names in (functions or {}).items()}
    watched_lines = {os.path.realpath(checkout / name): sorted(numbers) for name, numbers in (lines or {}).items() if numbers}
    read_end, write_end = os.pipe()
    config = json.dumps({"functions": watched, "pipe": write_end, "mark": token})
    plugin = (HERE / "value_witness.py").read_text(encoding="utf-8").replace(
        "CONFIG: dict[str, Any] = {}", f"CONFIG: dict[str, Any] = json.loads({config!r})")
    (private / f"run_{token}.py").write_text(plugin, encoding="utf-8")
    (private / "sitecustomize.py").write_text(_RAN.format(watched=watched, lines=watched_lines, folder=str(ran)),
                                              encoding="utf-8")
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
    report: dict[str, Any] = {"verdicts": {}, "values": {}, "inputs": {}, "seen": []}
    finished, whole, process_id = False, False, None
    for line in "".join(got).splitlines():
        try:
            mark, kind, *rest = json.loads(line)
        except (ValueError, TypeError):
            continue
        if mark != token:
            continue
        if kind == "verdict":
            report["verdicts"].setdefault(rest[0], bool(rest[1]))  # the first word on a test is the one kept
        elif kind == "value":
            report["values"].setdefault(rest[0], rest[1])
        elif kind == "every":
            report["values"][rest[0]] = rest[1]  # a running digest: its last value covers every call
        elif kind == "input":
            report["inputs"].setdefault(rest[0], []).append(rest[1])
        elif kind == "seen":
            report["seen"].append(rest[0])
        elif kind == "process" and process_id is None:
            process_id = str(rest[0])
        elif kind == "whole":
            whole = bool(rest[0])
        elif kind == "done":
            finished = True
    names: set[str] = set()
    here: set[str] = set()
    for path in ran.iterdir():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, ValueError):
            continue  # what a test left there says nothing about what ran
        names.update(lines)
        here.update(lines if path.name == process_id else ())
    known = {f"{where}::{function}" for where, functions_there in watched.items() for function in functions_there}
    lines_known = {f"{where}#{number}" for where, numbers in watched_lines.items() for number in numbers}
    # A run that did not say it ended, or whose watch was replaced, is evidence of nothing, and its verdicts with it.
    return {**report, "ran": sorted(names & known), "ran_here": sorted(here & known), "finished": finished,
            "whole": whole, "lines_ran": sorted(names & lines_known)}


def _differential(checkout: Path, inputs: Mapping[str, Any], work: Path) -> dict[str, Any]:
    """What each input came to in ``checkout``, how long it took, which exceptions the code caught on the way, and whether
    the run said it ended, from a process that cannot reach this one's files."""
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
    caught: dict[str, dict[str, list[str]]] = {}
    finished = False
    for line in "".join(got).splitlines():
        try:
            mark, name, key, outcome, took, handled = json.loads(line)
        except (ValueError, TypeError):
            continue
        if mark != token:
            continue
        if not name and outcome == "done":
            finished = True
            continue
        outcomes.setdefault(name, {})[key] = outcome
        seconds[f"{name}\n{key}"] = float(took)
        caught.setdefault(name, {})[key] = list(handled)
    return {"outcomes": outcomes, "seconds": seconds, "caught": caught, "finished": finished}


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


def lines_it_wrote_that_run(repo: Path, commit: str, name: str) -> set[int]:
    """The lines ``commit`` added or changed in ``name`` that compile to something that runs: a comment, a blank line or
    a docstring asks nothing of a test."""
    diff = git(repo, "diff", "-U0", f"{commit}^", commit, "--", name)
    written: set[int] = set()
    for found in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.MULTILINE):
        first, count = int(found.group(1)), int(found.group(2) if found.group(2) is not None else 1)
        written.update(range(first, first + count))
    try:
        code = compile(shown_at(repo, commit, name), name, "exec")
    except (SyntaxError, ValueError):
        return set()
    runnable: set[int] = set()
    pending = [code]
    while pending:
        each = pending.pop()
        runnable.update(number for _start, _end, number in each.co_lines() if number is not None)
        pending.extend(constant for constant in each.co_consts if isinstance(constant, types.CodeType))
    return written & runnable


_FOLDS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
          ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b, ast.Div: lambda a, b: a / b,
          ast.LShift: lambda a, b: a << b if 0 <= b < 64 else None, ast.RShift: lambda a, b: a >> b,
          ast.BitAnd: lambda a, b: a & b, ast.BitOr: lambda a, b: a | b, ast.BitXor: lambda a, b: a ^ b}


def _folded(node: ast.AST) -> Any:
    """The number an expression of constants comes to, ``-50`` and ``6 * 7`` among them, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
        inner = _folded(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner if isinstance(node.op, ast.UAdd) else \
            ~inner if isinstance(inner, int) else None
    if isinstance(node, ast.BinOp) and type(node.op) in _FOLDS:
        left, right = _folded(node.left), _folded(node.right)
        try:
            return None if left is None or right is None else _FOLDS[type(node.op)](left, right)
        except (ArithmeticError, TypeError, ValueError):
            return None
    return None


def constants_it_added(repo: Path, commit: str, name: str) -> list[Any]:
    """The plain constants on the lines a commit added to a file, with each expression of constants worked out. A new
    branch on one value is reached only by that value, so each is tried as an input (``data == b"xyz"`` is found by no
    input near the ones the tests use). A data file is read as Python where it parses as Python, as JSON does."""
    lines = _changed_lines(repo, commit, name)
    try:
        tree = ast.parse(shown_at(repo, commit, name))
    except (SyntaxError, ValueError):
        return []
    found = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and getattr(node, "lineno", 0) in lines
             and isinstance(node.value, (bool, int, float, str, bytes))]
    found += [value for node in ast.walk(tree) if isinstance(node, (ast.UnaryOp, ast.BinOp))
              and getattr(node, "lineno", 0) in lines and (value := _folded(node)) is not None]
    return list({repr(value): value for value in found}.values())


def _span(node: Any) -> set[int]:
    first = min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", ()))])
    return set(range(first, (node.end_lineno or node.lineno) + 1))


def _header(node: ast.ClassDef) -> set[int]:
    """The lines of a class above its body: decorators, name, bases and keywords."""
    first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
    return set(range(first, node.body[0].lineno if node.body else (node.end_lineno or node.lineno) + 1))


def _assigned(body: Sequence[ast.stmt], lines: set[int]) -> set[str]:
    """The names at this level that the changed lines assign, delete or change in place, inside ``if`` and ``try`` too."""
    out: set[str] = set()
    for statement in body:
        if not lines & _span(statement) or isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            for target in ast.walk(statement):
                if isinstance(target, (ast.Name, ast.Subscript, ast.Attribute)) and isinstance(target.ctx, (ast.Store, ast.Del)):
                    base = target
                    while isinstance(base, (ast.Subscript, ast.Attribute)):
                        base = base.value
                    out.update([base.id] if isinstance(base, ast.Name) else [])
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) \
                and isinstance(statement.value.func, ast.Attribute):
            base = statement.value.func.value  # RATES.update(...), CONFIG["limits"].update(...)
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            out.update([base.id] if isinstance(base, ast.Name) else [])
        for inner in (*(getattr(statement, field, None) or [] for field in ("body", "orelse", "finalbody")),
                      *(handler.body for handler in getattr(statement, "handlers", None) or [])):
            out |= _assigned(inner, lines)
    return out


def names_it_changed(repo: Path, commit: str, name: str, everything: bool = False) -> set[str]:
    """The module-level names of ``name`` the commit changed, or every one of them when its data changed under it."""
    tree = ast.parse(shown_at(repo, commit, name))
    lines = set(range(1, len(shown_at(repo, commit, name).splitlines()) + 2)) if everything else \
        _changed_lines(repo, commit, name)
    return _assigned(tree.body, lines)


def changed_functions(repo: Path, commit: str, name: str, everything: bool = False, directly: bool = False) -> list[str]:
    """The functions of ``name`` the commit changed: those whose lines or decorators it touched, the methods of a class
    whose bases or decorators it touched, and those that read a module-level name or class attribute it touched.
    ``everything`` names every function, for a module whose data file changed under it; ``directly`` leaves out those
    that only read what changed."""
    lines = _changed_lines(repo, commit, name)
    tree = ast.parse(shown_at(repo, commit, name))
    touched = set() if directly else _assigned(tree.body, lines)
    attributes = set() if directly else {name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                                         for name in _assigned(node.body, lines)}
    names: list[str] = []

    def walk(node: ast.AST, prefix: str, all_of_it: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                reads = {item.id for item in ast.walk(child) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}
                reached = {item.attr for item in ast.walk(child) if isinstance(item, ast.Attribute)}
                if all_of_it or lines & _span(child) or reads & touched or reached & attributes:
                    names.append(f"{prefix}{child.name}")
                walk(child, f"{prefix}{child.name}.<locals>.", all_of_it)
            elif isinstance(child, ast.ClassDef):
                rebased = bool(lines & _header(child))
                walk(child, f"{prefix}{child.name}.", all_of_it or rebased)
                if rebased:  # what the class now inherits from a base defined here is what it does
                    bases = {base.id for base in child.bases if isinstance(base, ast.Name)}
                    names.extend(f"{base.name}.{item.name}" for base in tree.body if isinstance(base, ast.ClassDef)
                                 and base.name in bases for item in base.body
                                 if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
            elif isinstance(child, ast.Lambda) and lines & _span(child):
                names.append(f"{prefix}<lambda>")
            else:
                walk(child, prefix, all_of_it)

    walk(tree, "", everything)
    return list(dict.fromkeys(names))


def readers_elsewhere(checkout: Path, name: str, changed: set[str]) -> dict[str, list[str]]:
    """The functions in other modules of the checkout that read a module-level name ``name`` changed, through
    ``from module import NAME`` or ``import module`` and ``module.NAME``."""
    if not changed or not name.endswith(".py"):
        return {}
    dotted = name[:-3].removesuffix("/__init__").replace("/", ".")
    stem = dotted.rsplit(".", 1)[-1]
    argv = ["git", "-C", str(checkout), "-c", "core.fsmonitor=false", "grep", "-l", "-w", "-F", "-e", stem, "--", "*.py"]
    found: dict[str, list[str]] = {}
    for path in subprocess.run(argv, capture_output=True, text=True, check=False).stdout.splitlines():
        if path == name or _a_test_file(path):
            continue
        try:
            tree = ast.parse((checkout / path).read_bytes())
        except (OSError, SyntaxError, ValueError):
            continue
        brought = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                   and (node.module or "").rsplit(".", 1)[-1] == stem for alias in node.names if alias.name in changed}
        held = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names if alias.name in (dotted, stem)}
        names: list[str] = []

        def walk(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    reads = {item.id for item in ast.walk(child) if isinstance(item, ast.Name)}
                    through = {_dotted(item) for item in ast.walk(child) if isinstance(item, ast.Attribute)}
                    if reads & brought or {f"{module}.{item}" for module in held for item in changed} & through:
                        names.append(f"{prefix}{child.name}")
                    walk(child, f"{prefix}{child.name}.<locals>.")
                elif isinstance(child, ast.ClassDef):
                    walk(child, f"{prefix}{child.name}.")
                else:
                    walk(child, prefix)

        walk(tree, "")
        if names:
            found[path] = list(dict.fromkeys(names))
    return found


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else ""


def reaches_into_the_interpreter(repo: Path, commit: str, name: str, local: frozenset[str] = frozenset()) -> list[str]:
    """Where the lines a commit added reach the machinery of the process they run in: a name for frames, code, closures
    or hooks on any object; the module table, monitoring, exits or signals of a module the run stands on; one of those
    modules handed on as a value; an attribute or module named by text made at run time; text run as code; a reader
    that runs what a file names; or a name rebound inside a module it imported from outside ``local``."""
    lines = _changed_lines(repo, commit, name)
    tree = ast.parse(shown_at(repo, commit, name))
    modules: dict[str, str] = {}
    imported_as: dict[str, str] = {}
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            imported_as.update({alias.asname or alias.name: alias.name for alias in node.names})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    inner = {alias for alias, module in modules.items() if module.split(".")[0] in _INTERPRETER}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found: set[str] = set()

    def chain(node: ast.AST) -> list[str]:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        return [node.id if isinstance(node, ast.Name) else "", *reversed(parts)]

    def outside(alias: str) -> bool:
        return alias in modules and modules[alias].split(".")[0] not in local

    for node in ast.walk(tree):
        if getattr(node, "lineno", 0) not in lines:
            continue
        at = f"{name}:{getattr(node, 'lineno', 0)}: "
        if isinstance(node, ast.Attribute):
            parts = chain(node)
            through = parts[0] in inner or any(part in _INTERPRETER for part in parts[1:-1])
            if node.attr == "__dict__" and parts[0] in modules:
                found.add(at + f"it reads the module {parts[0]} as a table of names")
            if node.attr in _REFLECTION or through and node.attr in _IN_THE_INTERPRETER:
                found.add(at + ".".join(part for part in parts if part))
            elif node.attr in _INTERPRETER and parts[0] not in inner:
                found.add(at + f"{'.'.join(parts)} reaches the module {node.attr} through another")
            if isinstance(node.ctx, (ast.Store, ast.Del)) and outside(parts[0]):
                found.add(at + f"it rebinds {'.'.join(parts)} inside a module it imported")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            parent = parents.get(node)
            called = isinstance(parent, ast.Call) and parent.func is node
            if node.id in inner and not (isinstance(parent, ast.Attribute) and parent.value is node):
                found.add(at + f"it hands the module {node.id} on")
            elif node.id in _RUNS_TEXT and node.id not in bound:
                found.add(at + f"it runs text as code: {node.id}")
            elif imported_as.get(node.id, node.id) in _BY_NAME and node.id not in bound and not called:
                found.add(at + f"it hands {node.id} on")
        elif isinstance(node, ast.Import):
            found.update(at + f"import {alias.name}" for alias in node.names if alias.name.split(".")[0] in _NEVER_IMPORTED
                         or alias.name in _DOUBLES and not _a_test_file(name))
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            found.update(at + f"from {node.module} import {alias.name}" for alias in node.names
                         if top in _NEVER_IMPORTED or alias.name in _INTERPRETER or alias.name in _REFLECTION
                         or top in _INTERPRETER and (alias.name == "*" or alias.name in _IN_THE_INTERPRETER)
                         or alias.name in _RESOLVERS
                         or not _a_test_file(name) and (node.module in _DOUBLES or f"{node.module}.{alias.name}" in _DOUBLES))
        elif isinstance(node, ast.Call):
            named = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            callee = imported_as.get(named, named)
            if callee == "vars" and node.args and chain(node.args[0])[0] in modules:
                found.add(at + f"it reads the module {chain(node.args[0])[0]} as a table of names")
            if callee in _RESOLVERS:
                found.add(at + f"{callee} finds an object by a name given as text")
            texts = node.args[1:2] if callee in ("getattr", "setattr", "delattr", "hasattr", "getattr_static") else \
                list(node.args) if callee in ("attrgetter", "methodcaller") else []
            on_inner = bool(node.args) and chain(node.args[0])[0] in inner
            for text in texts:
                if not (isinstance(text, ast.Constant) and isinstance(text.value, str)):
                    found.add(at + f"{callee} with an attribute named at run time")
                elif set(text.value.split(".")) & (_REFLECTION | _INTERPRETER | (_IN_THE_INTERPRETER if on_inner else set())):
                    found.add(at + f"{callee} naming {text.value!r}")
            if callee in ("setattr", "delattr") and node.args and isinstance(node.args[0], ast.Name) and outside(node.args[0].id):
                found.add(at + f"it rebinds a name inside the module {node.args[0].id}")
            if callee == "import_module" and node.args and not (isinstance(node.args[0], ast.Constant)
                                                                and isinstance(node.args[0].value, str)
                                                                and node.args[0].value.split(".")[0] not in _INTERPRETER):
                found.add(at + "it imports a module named at run time, or one the run stands on")
            if isinstance(node.func, ast.Attribute) and chain(node.func)[0] in _UNPICKLERS \
                    and node.func.attr in ("load", "loads", "Unpickler", "open", "decode"):
                found.add(at + f"{'.'.join(chain(node.func))} runs whatever its input names")
    return sorted(found)


def writes_under_lock(source: bytes) -> dict[str, tuple[set[str], set[str]]]:
    """For each function, the shared state it writes while it holds a lock, and the shared state it writes holding none.

    Shared state outlives the call: an attribute, an item of an argument or a global, a name declared global, or a
    container changed through one of the methods Python's mutable containers define. A lock is held inside its ``with``
    block, and between an ``acquire()`` and a ``release()`` written at the same level."""
    tree = ast.parse(source)
    locks = {target.id if isinstance(target, ast.Name) else target.attr for node in ast.walk(tree)
             if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call)
             and (getattr(node.value.func, "attr", "") in _LOCK_KINDS or getattr(node.value.func, "id", "") in _LOCK_KINDS)
             for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
             if isinstance(target, (ast.Name, ast.Attribute))}
    imported = {alias.asname or alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}

    def a_lock(expr: ast.AST) -> bool:
        """A lock kept somewhere every call shares. A lock made in the ``with`` itself is new on each call and keeps no
        two threads apart."""
        return not isinstance(expr, ast.Call) and (getattr(expr, "id", "") in locks or getattr(expr, "attr", "") in locks)

    out: dict[str, tuple[set[str], set[str]]] = {}
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        declared = {name for node in ast.walk(function) if isinstance(node, (ast.Global, ast.Nonlocal)) for name in node.names}
        local = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        local -= declared

        def shared(expr: ast.AST, bare_counts: bool) -> str:
            while isinstance(expr, ast.Subscript):
                expr = expr.value
            root = expr
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if not isinstance(root, ast.Name) or root.id in imported or root.id in local or a_lock(expr):
                return ""
            return ast.unparse(expr) if bare_counts or expr is not root or root.id in declared else ""

        def writes(nodes: Sequence[ast.AST], targets: Sequence[ast.AST]) -> set[str]:
            found = {shared(node, isinstance(node, (ast.Attribute, ast.Subscript))) for target in targets
                     for node in ast.walk(target) if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript))
                     and isinstance(node.ctx, (ast.Store, ast.Del))}
            found |= {shared(node.func.value, True) for part in nodes for node in ast.walk(part)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATORS}
            return found - {""}

        guarded: set[str] = set()
        unguarded: set[str] = set()

        def lock_call(node: ast.AST, verb: str) -> bool:
            node = node.value if isinstance(node, ast.Expr) else node
            return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == verb \
                and a_lock(node.func.value)

        def ends(statements: Sequence[ast.stmt]) -> bool:
            return bool(statements) and isinstance(statements[-1], (ast.Raise, ast.Return, ast.Continue, ast.Break))

        def walk(statements: Sequence[ast.stmt], held: bool) -> bool:
            """Walk in order, and say whether the lock is held after: on a branch, only if every way through holds it."""
            for statement in statements:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if lock_call(statement, "acquire") or lock_call(statement, "release"):
                    held = lock_call(statement, "acquire")
                    continue
                if isinstance(statement, (ast.With, ast.AsyncWith)):
                    inside = walk(statement.body, held or any(a_lock(item.context_expr) for item in statement.items))
                    held = held if any(a_lock(item.context_expr) for item in statement.items) else inside
                    continue
                if isinstance(statement, ast.If):
                    test = statement.test
                    taken = isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and lock_call(test.operand, "acquire")
                    (guarded if held else unguarded).update(writes([test], []))
                    if lock_call(test, "acquire"):  # if lock.acquire(timeout=...): the body holds it
                        after = [walk(statement.body, True), walk(statement.orelse, held)]
                    elif taken and ends(statement.body):  # if not lock.acquire(...): raise -- past it, the lock is held
                        walk(statement.body, held)
                        after = [walk(statement.orelse, True)]
                    else:
                        after = [walk(body, held) for body in (statement.body, statement.orelse)
                                 if not ends(body)] or [held]
                    held = all(after)
                    continue
                inner = [*(getattr(statement, field, None) or [] for field in ("body", "orelse")),
                         *(part.body for part in getattr(statement, "handlers", None) or []),
                         *(part.body for part in getattr(statement, "cases", None) or [])]
                if any(inner) or getattr(statement, "finalbody", None):
                    heads = [getattr(statement, field) for field in ("test", "iter", "subject") if getattr(statement, field, None)]
                    (guarded if held else unguarded).update(writes(heads, []))
                    # A loop may run no times and a handler may never run, so each holds the lock after only if the way
                    # around it does too.
                    held = all([held, *(walk(body, held) for body in inner if body)])
                    held = walk(getattr(statement, "finalbody", None) or [], held)
                    continue
                targets = statement.targets if isinstance(statement, (ast.Assign, ast.Delete)) else \
                    [statement.target] if isinstance(statement, (ast.AugAssign, ast.AnnAssign)) else []
                (guarded if held else unguarded).update(writes([statement], targets))
            return held

        walk(function.body, False)
        out[function.name] = (guarded, unguarded)
    return out


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
    """The call with its numbers moved together, for a fault in how two numbers relate: every one by the same step, the
    first two by opposite steps (which keeps their sum), and every one doubled (which keeps their ratio)."""
    numbers = {name: ast.literal_eval(text) for name, text in call["args"].items()}
    numbers = {name: value for name, value in numbers.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    if len(numbers) < 2:
        return []
    first, second = list(numbers)[:2]
    moves = [{name: value + step for name, value in numbers.items()} for step in (1, 2)]
    moves += [{first: numbers[first] + step, second: numbers[second] - step} for step in (1, 2)]
    moves += [{name: value * 2 for name, value in numbers.items()}]
    return [{**call, "args": {**call["args"], **{name: repr(value) for name, value in move.items()}}} for move in moves]


def _around(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every input within three steps of the call along its first two numbers at once. A fault in how two numbers relate,
    such as ``2 * x - y == 3``, sits at more than one such point when the commit's own test shows it at one."""
    numbers = [(name, ast.literal_eval(text)) for name, text in call["args"].items()]
    numbers = [(name, value) for name, value in numbers if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if len(numbers) < 2:
        return []
    (first, x), (second, y) = numbers[:2]
    return [{**call, "args": {**call["args"], first: repr(x + right), second: repr(y + up)}}
            for right in range(-3, 4) for up in range(-3, 4) if right or up]


def _a_mark(value: Any) -> bool:
    """A value the witness wrote as a marked tuple: an object, a lock, a callable, compiled code."""
    return isinstance(value, tuple) and bool(value) and isinstance(value[0], str) and value[0].startswith("__")


def _beside_it(value: Any) -> list[Any]:
    """Values on the other side of the line ``value`` sits on: the other truth value, a number past zero from it, and the
    same text with a character of another kind added (a space after letters and digits)."""
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, (int, float)):
        return [value + 1, -value - 1]
    if isinstance(value, str):
        return [f"{value} "]
    if isinstance(value, bytes):
        return [value + b" "]
    if isinstance(value, (list, tuple, dict, set)) and value and not _a_mark(value):
        return [type(value)()]
    return []


def _with(call: Mapping[str, Any], name: str, value: Any) -> dict[str, Any]:
    return {**call, "args": {**call["args"], name: repr(value)}}


def _key(call: Mapping[str, Any]) -> str:
    return json.dumps(call, sort_keys=True)


def _an_object(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 4 and value[0] == "__object__"


#: Items of a list or tuple argument varied one by one: the first few, where a fault in handling one item shows.
_ITEMS_VARIED = 3


def _inside(call: Mapping[str, Any]) -> list[tuple[tuple[Any, ...], Any]]:
    """Each plain value a call carries, by where it sits: an argument, a part of an object passed, the object's own state,
    or one of the first items of a list or tuple passed."""
    found: list[tuple[tuple[Any, ...], Any]] = []
    for field in ("args", "self"):
        for name, text in call.get(field, {}).items():
            value = ast.literal_eval(text)
            if _an_object(value):
                found += [((field, name, part), item) for part, item in value[3].items()]
            elif type(value) in (list, tuple) and value and not _a_mark(value) \
                    and all(not isinstance(item, (list, tuple, dict, set)) for item in value):
                found.append(((field, name), value))
                found += [((field, name, place), item) for place, item in enumerate(value[:_ITEMS_VARIED])]
            else:
                found.append(((field, name), value))
    return found


def _changed_at(call: Mapping[str, Any], where: tuple[Any, ...], value: Any) -> dict[str, Any]:
    field, name, *part = where
    if not part:
        return {**call, field: {**call[field], name: repr(value)}}
    held = ast.literal_eval(call[field][name])
    if type(held) in (list, tuple) and not _a_mark(held):
        items = list(held)
        items[part[0]] = value
        return {**call, field: {**call[field], name: repr(type(held)(items))}}
    kind, module, qualname, parts = held
    return {**call, field: {**call[field], name: repr((kind, module, qualname, {**parts, part[0]: value}))}}


def _near_one(call: Mapping[str, Any], constants: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [dict(call), *_moved_together(call)]
    for where, value in _inside(call):
        bounds = [edge for item in constants if type(item) is type(value) and isinstance(item, (int, float))
                  and not isinstance(item, bool) for edge in (item - 1, item + 1)]
        others = [*(item for item in constants if type(item) is type(value)), *bounds, *itertools.chain(*_like_it(value)),
                  *_beside_it(value), None]
        out += [_changed_at(call, where, other) for other in others]
    return out


# ── inputs solved from the conditions a change wrote or removed ──────────────────────────────────────────────────────
#
# Near inputs vary the values a call carries. A special case written on a relation, ``if 3 * x + 7 * y == 100``, sits at
# none of them unless a test already stands on it, and a guard a change deleted is tried only where a test once crossed
# it. So each condition on a changed line, before the change and after it, is solved for the function's own parameters.

_ARITHMETIC = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
               ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b}


class _NotPlain(Exception):
    """An expression this cannot work out without running code the change wrote."""


def _worked_out(node: ast.AST, values: Mapping[str, Any]) -> Any:
    """The value of a plain expression over constants and parameters: arithmetic, a sign, ``len`` of a parameter. Nothing
    else is evaluated, so no code of the change runs here."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bytes)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _worked_out(node.operand, values)
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        left, right = _worked_out(node.left, values), _worked_out(node.right, values)
        if all(isinstance(side, (int, float)) and not isinstance(side, bool) for side in (left, right)):
            try:
                return _ARITHMETIC[type(node.op)](left, right)
            except (ArithmeticError, ValueError):
                raise _NotPlain from None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1 \
            and not node.keywords:
        inner = _worked_out(node.args[0], values)
        if isinstance(inner, (str, bytes, list, tuple, dict, set)):
            return len(inner)
    raise _NotPlain


def _names_in(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _a_root(difference: Any, around: Any) -> list[Any]:
    """Where a straight-line function of one number crosses zero, and one step to each side: worked out from its value at
    two points and checked at a third, so a bend is never taken for a line."""
    try:
        start, step = difference(around), difference(around + 1)
        slope = step - start
        if slope == 0 or difference(around + 2) - step != slope:
            return []
    except (_NotPlain, TypeError):
        return []
    root = around - start / slope
    whole = isinstance(around, int) and float(root).is_integer()
    if whole:
        root = int(root)
    elif isinstance(around, int):
        return [math.floor(root), math.ceil(root)]
    return [root - 1, root, root + 1]


#: The longest value made to stand at a length a condition names, in items. A bound on the check's own cost, as
#: TEST_SECONDS is: ``len(s) == 10**9`` asks for a gigabyte, while the buffers and limits code compares lengths with
#: (a page, a packet, a field) sit far below it.
_LONGEST_VALUE_MADE = 1 << 16

#: Text tests whose argument a value must start or end with to pass.
_AFFIXES = {"startswith": lambda held, part: [part + held, part[1:] + held],
            "endswith": lambda held, part: [held + part, held + part[:-1]]}


def _solved_for(condition: ast.expr, name: str, values: Mapping[str, Any]) -> list[Any]:
    """Values of parameter ``name`` on and beside the line ``condition`` draws, the other parameters held at ``values``."""
    held = values.get(name)
    if isinstance(condition, ast.Call):
        # ``name.startswith(P)``: one value that starts with P and one that misses it by a character.
        method = condition.func
        if not (isinstance(method, ast.Attribute) and method.attr in _AFFIXES and isinstance(method.value, ast.Name)
                and method.value.id == name and len(condition.args) == 1 and isinstance(held, (str, bytes))):
            return []
        try:
            part = _worked_out(condition.args[0], values)
        except _NotPlain:
            return []
        return _AFFIXES[method.attr](held, part) if isinstance(part, type(held)) and part else []
    if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
        return []
    left, right, op = condition.left, condition.comparators[0], condition.ops[0]
    found: list[Any] = []
    # A text that starts or ends with what the condition names, or holds it, and one that does not.
    for side, other in ((left, right), (right, left)):
        if isinstance(op, (ast.In, ast.NotIn)) and side is right and isinstance(side, ast.Name) and side.id == name \
                and isinstance(held, (str, bytes)):
            try:
                part = _worked_out(other, values)
            except _NotPlain:
                part = None
            if isinstance(part, type(held)) and part:
                found += [part, held + part, part[:-1]]
    # A length compared with a number: the parameter at that length and one either side, of its own kind.
    for side, other in ((left, right), (right, left)):
        if isinstance(side, ast.Call) and isinstance(side.func, ast.Name) and side.func.id == "len" and side.args \
                and isinstance(side.args[0], ast.Name) and side.args[0].id == name and isinstance(held, (str, bytes, list, tuple)):
            try:
                size = _worked_out(other, values)
            except _NotPlain:
                continue
            if isinstance(size, int) and not isinstance(size, bool) and 0 <= size <= _LONGEST_VALUE_MADE:
                unit = held[:1] or (" " if isinstance(held, str) else b" " if isinstance(held, bytes) else [0])
                found += [type(held)((held * (size // max(len(held), 1) + 2))[:length]) if held else unit * length
                          for length in (size - 1, size, size + 1) if length >= 0]
    # A relation of numbers: where it holds with the other parameters where the call put them.
    if isinstance(held, (int, float)) and not isinstance(held, bool) and name in _names_in(condition):
        def difference(value: Any) -> Any:
            here = {**values, name: value}
            return _worked_out(left, here) - _worked_out(right, here)
        found += _a_root(difference, held)
    return found


def _conditions(source: bytes, qualname: str, lines: set[int]) -> list[ast.expr]:
    """The comparisons, and the start and end tests of text, in function ``qualname`` of ``source`` on ``lines``."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    holder: list[ast.AST] = [tree]
    for part in qualname.split("."):
        holder = [child for node in holder for child in ast.iter_child_nodes(node)
                  if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part]
    out: list[ast.expr] = []
    for function in holder:
        for node in ast.walk(function):
            on_a_line = getattr(node, "lineno", 0) in lines
            if on_a_line and isinstance(node, ast.Compare):
                out.append(node)
            elif on_a_line and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in _AFFIXES:
                out.append(node)
    return out


def _removed_lines(repo: Path, commit: str, name: str) -> set[int]:
    """The lines of ``name`` before ``commit`` that it removed or changed."""
    diff = git(repo, "diff", "-U0", f"{commit}^", commit, "--", name)
    lines: set[int] = set()
    for found in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", diff, re.MULTILINE):
        first, count = int(found.group(1)), int(found.group(2) if found.group(2) is not None else 1)
        lines.update(range(first, first + count))
    return lines


def inputs_from_conditions(repo: Path, commit: str, name: str, qualname: str,
                           calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each recorded call moved onto and beside every line a condition on a changed line draws, before the change and
    after it, one parameter at a time."""
    sides = [(shown_at(repo, commit, name), _changed_lines(repo, commit, name))]
    if _in_tree(repo, f"{commit}^", name):
        sides.append((shown_at(repo, f"{commit}^", name), _removed_lines(repo, commit, name)))
    conditions = [condition for source, lines in sides for condition in _conditions(source, qualname, lines)]
    out: list[dict[str, Any]] = []
    for call in calls:
        try:
            values = {key: ast.literal_eval(text) for key, text in call["args"].items()}
        except (ValueError, SyntaxError, TypeError):
            continue
        for condition in conditions:
            for parameter in _names_in(condition) & set(values):
                out += [_with(call, parameter, value) for value in _solved_for(condition, parameter, values)
                        if type(value) is type(values[parameter]) or (isinstance(value, (int, float))
                                                                       and isinstance(values[parameter], (int, float)))]
    return list({_key(item): item for item in out}.values())


def inputs_near(calls: Sequence[Mapping[str, Any]], constants: Sequence[Any] = (),
                pinned: Sequence[Mapping[str, Any]] = (), solved: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """The calls the commit's own tests pin and what lies near them come first, with the inputs solved from the conditions
    it wrote or removed, so a long test suite cannot crowd them out. Then the other recorded calls and what lies near
    them, an argument taken from another call, and a member of a table or list passed beside it."""
    # The pinned calls, then what was solved from the conditions (few, and placed where the change draws its lines), then
    # what lies near the pinned calls, so neither of the first two is cut off by the third.
    own = list({_key(item): item for item in [*(dict(call) for call in pinned), *solved,
                                              *(item for call in pinned
                                                for item in [*_near_one(call, constants), *_around(call)])]
                }.values())[:NEAR_INPUTS // 2]
    out: list[dict[str, Any]] = [*own, *(item for call in calls for item in _near_one(call, constants))]
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


#: Exceptions that say the code broke, not that it refused: an input once refused by name that now crashes one of these was
#: lost. Python's own meaning of the builtins, not a list of words.
_CRASHES = frozenset({"ZeroDivisionError", "ArithmeticError", "OverflowError", "FloatingPointError", "AttributeError",
                      "TypeError", "NameError", "UnboundLocalError", "RecursionError", "AssertionError", "IndexError",
                      "NotImplementedError", "SystemError", "SystemExit", "KeyboardInterrupt", "GeneratorExit"})


def _deliberate(kind: str) -> bool:
    """A refusal the code meant: raised by a ``raise`` in its own code, or of a kind that is not a crash."""
    return kind.endswith(" by the code") or kind.split(" ")[0] not in _CRASHES


def _domain(value: Any) -> str:
    """Which side of a line an input sits on: the sign of a number, an empty text or a filled one, a container's kind."""
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return "negative" if value < 0 else "zero" if value == 0 else "positive"
    if isinstance(value, (str, bytes)):
        return f"{type(value).__name__}:{'empty' if not value else 'filled'}"
    return type(value).__name__ if not _an_object(value) else f"object:{value[2]}"


def _injected_none(call: Mapping[str, Any], recorded: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a None in this input was put there by the check, where no recorded call ever had one."""
    nones = {where for where, value in _inside(call) if value is None}
    return bool(nones - {where for seed in recorded for where, value in _inside(seed) if value is None})


def _same_side(mine: Any, theirs: Any) -> bool:
    """Two values on the same side of a line: the same sign, both empty or both filled, the same kind. Zero is a line of
    its own, so it sits with neither sign."""
    return mine is None or _domain(mine) == _domain(theirs)


def _a_boundary(value: Any) -> bool:
    """Zero or an empty value: the one place where a special case is the natural mend."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return isinstance(value, (str, bytes, list, tuple, dict, set)) and not value and not _a_mark(value)


def _beside_the_same(call: Mapping[str, Any], shown: Mapping[str, Any]) -> bool:
    mine = dict(_inside(call))
    return all(_same_side(mine.get(where), value) for where, value in _inside(shown) if mine.get(where) != value)


def _written_in(call: Mapping[str, Any], constants: Sequence[Any]) -> bool:
    """Whether every plain value a call passes is a constant the commit wrote into the code: the test's own input made a
    special case of, as ``if x == -5`` for a test of ``x=-5``."""
    plain = [value for _, value in _inside(call) if isinstance(value, (int, float, str, bytes)) and not isinstance(value, bool)]
    written = {repr(item) for item in constants}
    return bool(plain) and all(repr(value) in written for value in plain)


def _vouched(call: Mapping[str, Any], refusal: str, handled: Sequence[str],
             mends: Sequence[tuple[str, Mapping[str, Any], Sequence[str]]]) -> str:
    """How far the commit's own test speaks for a refusal now gone: "mended" when a test took away a refusal of this kind
    on the same side of every line; "beyond" when a test took away this very refusal and this input sits across a line
    from it, which a range guard done right and a swallow done wrong both do; "lost" otherwise.

    A refusal gone because the new code caught an exception is spoken for only by a test that went through a handler of
    the same exception. A test that sat on zero or an empty value speaks for that value alone."""
    verdict = "lost"
    mine = dict(_inside(call))
    for theirs, shown, their_handled in mends:
        if set(handled) - set(their_handled):
            continue
        if any(_a_boundary(value) for where, value in _inside(shown) if mine.get(where) != value):
            continue  # a test on zero or an empty value, of any kind, speaks for that value alone
        if _kind(theirs) == _kind(refusal) and _beside_the_same(call, shown):
            return "mended"
        if theirs == refusal:
            verdict = "beyond"
    return verdict


def difference(before: str, after: str | None) -> str:
    """What kind of difference one call shows between the code before a change and the change."""
    if after is None:
        return "came to nothing"
    if after == "took too long":
        return "no longer finishes"
    was, now = _raised(before), _raised(after)
    if was and now:
        return "another refusal"
    return "a refusal gone" if was else "now raises" if now else "another value"


def _counterexample(function: str, call: Any, before: str, after: str | None, counted: str) -> dict[str, Any]:
    """One input the change treats differently, as data a repair search can read: what it was, what it came to on
    each side, the kind of difference, and whether the check counted it as lost or as a doubt."""
    return {"function": function, "input": call, "before": before, "after": after, "kind": difference(before, after),
            "counted": counted}


def judged(before: Mapping[str, Any], again: Mapping[str, Any], after: Mapping[str, Any],
           pinned: Mapping[str, Sequence[Mapping[str, Any]]], near_the_pinned: Mapping[str, set[str]] | None = None,
           recorded: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
           caught: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
           constants: Mapping[str, Sequence[Any]] | None = None,
           shown_by: list[dict[str, Any]] | None = None) -> tuple[list[str], list[str], bool]:
    """(what the change demonstrably lost, what stands in doubt, whether its tests pin a new value for an input that already
    had one), from what each input came to before and after. ``shown_by``, when given, gets a counterexample for each
    input behind a line of the first two.

    ``pinned`` holds the calls the commit's own tests make: those it has said what they should do. ``recorded`` holds every
    call seen. ``caught`` holds, for each input, the exceptions the new code caught while answering it. ``constants``
    holds, for each function, the constants the commit wrote: a test whose input is among them made a special case of
    that input, and speaks for nothing else.
    """
    lost, doubted = [], []
    any_respecified = False
    shown_by = shown_by if shown_by is not None else []
    for name, results in before.items():
        now = after.get(name, {})
        handled = (caught or {}).get(name, {})
        steady = {key: value for key, value in results.items() if again.get(name, {}).get(key) == value
                  and value != "took too long" and not value.startswith("unavailable")}
        lost += [f"{name}: {key[:160]} came to nothing in the new code, whose run ended first" for key in steady if key not in now]
        shown_by += [_counterexample(name, json.loads(key), value, None, "lost") for key, value in steady.items() if key not in now]
        steady = {key: value for key, value in steady.items() if key in now}
        shown = {_key(call): call for call in pinned.get(name, ())}
        seen = list(recorded.get(name, ())) if recorded else list(shown.values())
        mends = [(_raised(steady[key]), call, handled.get(key, ())) for key, call in shown.items()
                 if key in steady and _raised(steady[key]) and not _raised(now[key])]
        written = list((constants or {}).get(name, ()))
        respecified = any(key in steady and not _raised(steady[key]) and now[key] != steady[key]
                          and not _written_in(call, written) for key, call in shown.items())
        any_respecified = any_respecified or respecified
        excused: set[str] = set()
        for key, value in steady.items():
            if key in shown or now[key] == value:
                continue
            call = json.loads(key)
            refused_with = _kind(_raised(value))
            counted = ""
            if now[key] == "took too long":  # no test says a function should stop answering
                lost.append(f"{name}: {key[:160]} was {value[:60]}, and now does not finish")
                counted = "lost"
            elif not _raised(value):
                # Its own test says an answer this function gave was wrong, so other answers may rightly change with it.
                # A function that now gives one answer to everything is caught below whatever its test says.
                (doubted if respecified else lost).append(f"{name}: {key[:160]} was {value[:60]}, now {now[key][:60]}")
                counted = "doubt" if respecified else "lost"
            elif not _raised(now[key]):
                spoken = _vouched(call, _raised(value), handled.get(key, ()), mends)
                if spoken == "mended":
                    excused.add(_raised(value))
                else:
                    (doubted if spoken == "beyond" else lost).append(
                        f"{name}: {key[:160]} was refused with {refused_with}, now {now[key][:60]}")
                    counted = "doubt" if spoken == "beyond" else "lost"
            elif _kind(_raised(now[key])) != refused_with and not _injected_none(call, seen):
                crashed = _deliberate(refused_with) and not _deliberate(_kind(_raised(now[key])))
                (lost if crashed else doubted).append(f"{name}: {key[:160]} was refused with {refused_with}, now {_raised(now[key])[:60]}")
                counted = "lost" if crashed else "doubt"
            if counted:
                shown_by.append(_counterexample(name, call, value, now[key], counted))
        # A refusal its own code raised on purpose, gone from every input that had it while its test spoke for one: the
        # guard may be gone, not mended.
        gone = {refusal for refusal in excused if _kind(refusal).endswith(" by the code")
                and not any(_raised(now[key]) == refusal for key in steady)}
        doubted += [f"{name}: it no longer refuses with {refusal[7:80]} anywhere, and its test speaks for one input"
                    for refusal in sorted(gone)]
        plain = [key for key in steady if not _injected_none(json.loads(key), seen)]
        answered_before = {steady[key] for key in plain if not _raised(steady[key])}
        answered_now = {now[key] for key in plain if not _raised(now[key])}
        if len(answered_before) > 1 and len(answered_now) == 1 and len(plain) > 2:
            lost.append(f"{name}: it gave {len(answered_before)} different answers and now gives one, {next(iter(answered_now))[:60]}")
        for key, call in shown.items():
            if key not in steady or not _raised(steady[key]) or _raised(now[key]):
                continue
            groups = [[_changed_at(call, where, other) for other in group] for where, value in _inside(call)
                      for group in _like_it(value)]
            for like in [*groups, _moved_together(call), _around(call)]:
                still = [item for item in like if _raised(now.get(_key(item), "")) == _raised(steady[key])]
                # The crash its test showed, still there beside it on the same side of every line, after a special case
                # of the test's own input: a mend of one value. A refusal the code still makes on purpose, or a crash
                # beside a mend that wrote no such case (an index past the end after an off-by-one fix), is a doubt.
                near = [item for item in still if _beside_the_same(item, call)
                        and not _deliberate(_kind(_raised(steady[key]))) and _written_in(call, written)]
                if near:
                    lost.append(f"{name}: the fault still happens on {_key(near[0])[:160]}: {now[_key(near[0])][:60]}")
                elif still:
                    doubted.append(f"{name}: the fault still happens on {_key(still[0])[:160]}: {now[_key(still[0])][:60]}")
                for item, counted in ((near[:1], "lost"), (still[:1] if not near else [], "doubt")):
                    shown_by += [{"function": name, "input": one, "before": steady[key], "after": now[_key(one)],
                                  "kind": "the fault still happens", "counted": counted} for one in item]
    return sorted(set(lost)), sorted(set(doubted)), any_respecified


def slower(before: Mapping[str, float], again: Mapping[str, float], after: Mapping[str, float]) -> str:
    """Words for a slowdown too consistent to be luck and larger than two runs of the old code differ, or nothing."""
    shared = [key for key in before if key in again and key in after]
    if not shared:
        return ""
    worse = sum(after[key] > max(before[key], again[key]) for key in shared)
    chance = sum(math.comb(len(shared), k) * 2 ** (len(shared) - k) for k in range(worse, len(shared) + 1)) / 3 ** len(shared)
    old, older, new = (statistics.median(times[key] for key in shared) for times in (before, again, after))
    if chance < ALPHA and new - max(old, older) > max(abs(old - older), NOTICEABLE_SECONDS):
        return (f"each call takes about {new:.3f} s, and took {max(old, older):.3f} s, on {worse} of {len(shared)} inputs"
                + ("; that is an order of magnitude" if new > ORDER_OF_MAGNITUDE * max(old, older, 1e-9) else ""))
    return ""


def checks_in(source: bytes) -> int:
    """How many checks a test file makes: ``assert`` statements, and calls that assert, expect a raise or fail."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return 0
    called = [getattr(node.func, "attr", getattr(node.func, "id", "")) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return sum(isinstance(node, ast.Assert) for node in ast.walk(tree)) + \
        sum(1 for name in called if name in ("raises", "warns", "fail") or name.startswith("assert"))


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
    try:
        git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    except (OSError, subprocess.SubprocessError) as exc:
        raise CouldNotStart(f"{type(exc).__name__}: {exc}") from exc
    work = Path(tempfile.mkdtemp())
    # Both checkouts carry the repository's own folder name, so the folder a test runs in does not say a check is running.
    folder = Path(repo).resolve().name or "repository"
    old, new = work / secrets.token_hex(6) / folder, work / secrets.token_hex(6) / folder
    try:
        files = changed_files(repo, commit)
        git(repo, "worktree", "add", "-q", "--detach", str(old), f"{commit}^")
        git(repo, "worktree", "add", "-q", "--detach", str(new), commit)
    except (OSError, subprocess.SubprocessError) as exc:
        for checkout in (old, new):
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(checkout)], check=False, capture_output=True)
        shutil.rmtree(work, ignore_errors=True)
        raise UncheckableCommit(f"{type(exc).__name__}: {exc}") from exc
    try:
        added = [name for name in files if _a_test_file(name) and (new / name).is_file()]
        edited = [name for name in files if _a_test_file(name) and (old / name).exists()]
        functions = {name: changed_functions(repo, commit, name) for name in files
                     if name.endswith(".py") and not _a_test_file(name) and not _machinery(name) and (new / name).exists()}
        readers = [name for name in readers_of(new, files) if (old / name).exists()]
        functions.update({name: changed_functions(repo, commit, name, everything=True) for name in readers})
        direct = {f"{name}::{function}" for name in functions if name not in readers
                  for function in changed_functions(repo, commit, name, directly=True)}
        # What another module reads of a changed name changed with it: a table in constants.py, a setting loaded from
        # policy.json. Those functions are tried with the constants the commit wrote where the name lives.
        source_of: dict[str, str] = {}
        for name in list(functions):
            for path, names in readers_elsewhere(new, name, names_it_changed(repo, commit, name, name in readers)).items():
                if (old / path).exists():
                    functions[path] = list(dict.fromkeys([*functions.get(path, []), *names]))
                    source_of.setdefault(path, name)
        guards = sorted({*tests_naming(new, [*files, *readers, *source_of], functions), *edited})
        present = [name for name in guards if (new / name).is_file()]
        first, second = run_tests(old, guards, work, functions), run_tests(old, guards, work, functions)
        written_lines = {name: sorted(lines_it_wrote_that_run(repo, commit, name)) for name in functions
                         if name not in readers and name not in source_of}
        third = run_tests(new, present, work, functions, written_lines)
        for name in edited:
            if (new / name).is_dir():
                shutil.rmtree(new / name)
            (new / name).parent.mkdir(parents=True, exist_ok=True)
            (new / name).write_bytes(shown_at(repo, f"{commit}^", name))
        written = run_tests(new, edited, work)
        as_written = written["verdicts"]
        git(new, "reset", "-q", "--hard", commit)
        git(new, "clean", "-fdq")
        # A test renamed where it stands still passes as it was first written, so it is not a test lost.
        broke = sorted(node for node, passed in first["verdicts"].items() if passed and not third["verdicts"].get(node, False)
                       and not (as_written.get(node, False) and (new / node.split("::")[0]).is_file()))
        rewritten = sorted(node for node, passed in first["verdicts"].items() if passed and node.split("::")[0] in edited
                           and node not in broke and not as_written.get(node, False))
        spec = run_tests(new, added, work, functions, written_lines)
        # The commit's own tests, asked of the code before it, with any helper file it added beside its tests.
        helpers = [name for name in files if not name.endswith(".py") or not functions.get(name)]
        helpers = [name for name in helpers if not _a_test_file(name) and (new / name).is_file() and not (old / name).exists()
                   and any(part in ("tests", "test") for part in Path(name).parts[:-1])]
        for name in [*added, *helpers]:
            if (old / name).parent.is_file():
                (old / name).parent.unlink()
            (old / name).parent.mkdir(parents=True, exist_ok=True)
            (old / name).write_bytes((new / name).read_bytes())
        asked_before = run_tests(old, added, work)
        before = asked_before["verdicts"]
        newly_passing = [node for node, passed in spec["verdicts"].items() if passed and not before.get(node, False)]
        shows_the_fault = {"applies": bool(added), "newly_passing": newly_passing[:20],
                           "held": not added or (bool(newly_passing) and all(spec["verdicts"].values()))}
        kept = {node for node, passed in first["verdicts"].items() if passed and third["verdicts"].get(node, False)}
        steady = {key: value for key, value in first["values"].items() if _a_passing_test_made_it(key, kept)}
        changed = values_that_changed(steady, second["values"], third["values"])
        pinned = _seeds(spec["inputs"], new)
        seeds = {name: [*calls, *pinned.get(name, ())] for name, calls in _seeds(third["inputs"], new).items()}
        seeds.update({name: calls for name, calls in pinned.items() if name not in seeds})
        constants = {name: [*constants_it_added(repo, commit, name),
                            *(constants_it_added(repo, commit, source_of[name]) if name in source_of else [])]
                     for name in functions}
        for name in readers:  # a module reading a changed data file is tried with the values the file now holds
            constants[name] += [item for data in files if not data.endswith(".py") for item in constants_it_added(repo, commit, data)]
        for path, source in source_of.items():
            constants[path] += constants.get(source, [])
        near = {name: inputs_near(calls, constants.get(name.split("::")[0], ()), pinned.get(name, ()),
                                  inputs_from_conditions(repo, commit, name.split("::")[0], name.split("::")[1],
                                                         [*pinned.get(name, ()), *calls]))
                for name, calls in seeds.items()}
        near_the_pinned = {name: {_key(item) for call in calls for item in _near_one(call, constants.get(name.split("::")[0], ()))}
                           for name, calls in pinned.items()}
        runs = [_differential(old, near, work), _differential(old, near, work), _differential(new, near, work)]
        (old_near, again_near, new_near), (old_times, again_times, new_times) = \
            [run["outcomes"] for run in runs], [run["seconds"] for run in runs]
        sources = [name for name in functions if _in_tree(repo, f"{commit}^", name)]
        before_after = {name: (shown_at(repo, f"{commit}^", name), shown_at(repo, commit, name)) for name in sources}
        # A passing test's call, as the watch keeps it: its arguments and values as digests, so "as" says so.
        shown_by: list[dict[str, Any]] = [
            {"function": key.split("|")[1], "input": key.split("|", 3)[3] if key.count("|") >= 3 else key,
             "test": key.split("|")[0], "before": steady[key], "after": third["values"][key], "as": "digests",
             "kind": difference(steady[key], third["values"][key]), "counted": "lost"}
            for key in changed if not key.endswith("|every call")]
        lost, doubts, respecified = judged(old_near, again_near, new_near, pinned, near_the_pinned, seeds, runs[2]["caught"],
                                           {name: constants.get(name.split("::")[0], []) for name in near}, shown_by)
        unlocked = sorted(f"{name}::{function}: {', '.join(sorted(shared))}" for name, (was, now) in before_after.items()
                          for function, (guarded, _) in writes_under_lock(was).items()
                          if (shared := guarded & writes_under_lock(now).get(function, (set(), set()))[1]))
        local = frozenset(Path(line).parts[0].removesuffix(".py") for line in git(new, "ls-files").splitlines() if line)
        reaching = [line for name in files if name.endswith(".py") and (new / name).is_file()
                    for line in reaches_into_the_interpreter(repo, commit, name, local)]
        ran = {f"{os.path.relpath(item.split('::')[0], os.path.realpath(new))}::{item.split('::')[1]}"
               for item in {*third["ran"], *spec["ran"]}}
        unreached = sorted(f"{name}::{function}" for name, names in functions.items() for function in names
                           if f"{name}::{function}" not in ran and f"{name}::{function}" in direct)
        lines_ran = {*third["lines_ran"], *spec["lines_ran"]}
        unrun_lines = {name: [number for number in numbers if f"{os.path.realpath(new / name)}#{number}" not in lines_ran]
                       for name, numbers in written_lines.items()}
        unrun_lines = {name: numbers for name, numbers in unrun_lines.items() if numbers}
        # A function the commit did not edit, that only reads a name it changed, is a doubt when no test runs it.
        unread = sorted(f"{name}::{function}" for name, names in functions.items() for function in names
                        if f"{name}::{function}" not in ran and f"{name}::{function}" not in direct)
        checks = [sum(checks_in(shown_at(repo, at, name)) if _in_tree(repo, at, name) else 0 for name in files
                      if _a_test_file(name)) for at in (f"{commit}^", commit)]
        nested = sorted(f"{name}::{function}" for name, names in functions.items() for function in names if "<locals>" in function)
        slowdown = slower(old_times, again_times, new_times)
        machinery = [name for name in files if _machinery(name)]
        unreadable = [name for name in files if Path(name).suffix in _UNREADABLE]
        runs_of_tests = {"the guarding tests on the code before": first, "the guarding tests on the code before, again": second,
                         "the guarding tests on this change": third, "the tests it edited, as first written": written,
                         "its own tests on this change": spec, "its own tests on the code before": asked_before}
        cut_short = [what for what, run in runs_of_tests.items() if not run["finished"]]
        cut_short += [what for what, run in zip(("the calls on the code before", "the calls on the code before, again",
                                                 "the calls on this change"), runs) if near and not run["finished"]]
        unwatched = [what for what, run in runs_of_tests.items() if run["finished"] and run["verdicts"] and not run["whole"]]
        recorded_values = " ".join({*third["values"], *spec["values"], *third["seen"], *spec["seen"]})
        unrecorded = sorted(item for item in {*third["ran_here"], *spec["ran_here"]}
                            if f"|{item.split('::')[1]}|" not in recorded_values)
        tried = {name for name, results in old_near.items()
                 if any(again_near.get(name, {}).get(key) == value and key in new_near.get(name, {})
                        and not value.startswith("unavailable") for key, value in results.items())}
        unexamined = sorted(item for item in ran if "<" not in item and item not in tried)
        must_fix = [f"a test it edited fails as it was first written: {node}" for node in rewritten]
        must_fix += [f"a run did not say it ended, so it is evidence of nothing: {what}" for what in cut_short]
        must_fix += [f"the watch on what the code did was switched off or replaced during: {what}" for what in unwatched]
        must_fix += [f"a test ran this function and the watch on its values saw nothing: {item}" for item in unrecorded]
        must_fix += [f"it adds code the check cannot read: {name}" for name in unreadable]
        must_fix += [f"no test ran this function it changed: {name}" for name in unreached]
        if checks[1] < checks[0]:
            must_fix.append(f"it takes checks out of the tests it changed: {checks[1]} where there were {checks[0]}")
        must_fix += [f"state it wrote holding a lock it now writes holding none: {name}" for name in unlocked]
        must_fix += [f"its code reads the interpreter running it: {line}" for line in reaching]
        must_fix += [f"it changes how tests run: {name}" for name in machinery]
        nested += sorted(f"{name}::{function}" for name, names in functions.items() for function in names if function.endswith("<lambda>"))
        doubts += [f"this function is reachable only from inside another, so no input of its own was tried: {name}" for name in nested]
        doubts += [f"a test ran this function, but no input of its own could be tried on both sides: {name}" for name in unexamined]
        doubts += [f"no test ran this function, which reads a name the commit changed: {name}" for name in unread]
        # Code no test runs says nothing either way, and a line added only to be counted is such code.
        doubts += [f"no test ran these lines it wrote in {name}: {', '.join(map(str, numbers[:20]))}"
                   for name, numbers in sorted(unrun_lines.items())]
        if files and not first["verdicts"] and not spec["verdicts"]:
            doubts.append("no test ran anything this change touches")
        if slowdown.endswith("an order of magnitude"):
            lost.append(f"it is slower: {slowdown}")
        elif slowdown:
            doubts.append(f"it is slower: {slowdown}")
        values = {"compared": sum(1 for key, value in steady.items() if second["values"].get(key) == value and key in third["values"]),
                  "functions": {name: names for name, names in functions.items() if names},
                  "near_inputs": sum(len(calls) for calls in near.values()), "changed": changed[:20],
                  "changed_near": lost[:20], "no_test_ran": unreached[:20],
                  "lines_no_test_ran": {name: numbers[:20] for name, numbers in unrun_lines.items()},
                  "held": not changed and not lost}
        try:
            git(new, "-c", "user.name=outside-bar", "-c", "user.email=bar@localhost", "revert", "--no-edit", commit)
            reverted = git(new, "rev-parse", "HEAD^{tree}").strip() == git(repo, "rev-parse", f"{commit}^^{{tree}}").strip()
        except subprocess.CalledProcessError:
            reverted = False
        if added and not shows_the_fault["held"] and spec["verdicts"] and all(spec["verdicts"].values()) \
                and not changed and not lost and not doubts and not respecified and not broke:
            # A test added beside a change that keeps every value: a refactor with a guard, not a mend that shows nothing.
            shows_the_fault = {**shows_the_fault, "held": True, "keeps_every_value": True}
        answers = {"shows_the_fault": shows_the_fault, "tests_held": {"tests": len(first["verdicts"]), "broke": broke[:20], "held": not broke},
                   "values_kept": values, "reverts_cleanly": {"held": reverted}}
        refused = not all(answer["held"] for answer in answers.values())
        verdict = "REFUSED" if refused or must_fix else "PROBATION" if doubts else "HELD"
        return {"commit": commit, "files": files, "verdict": verdict, "held": verdict == "HELD",
                "must_fix": must_fix[:20], "doubts": doubts[:20], **answers,
                # Every input behind a line above, as data: a repair search narrows a patch on these without reading prose.
                "counterexamples": shown_by}
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
    return reasons + list(result["must_fix"][:3])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--result", type=Path, default=None,
                        help="write the verdict here as JSON, counterexamples included: what a gate decides by, where "
                             "the printed words are for people")
    args = parser.parse_args(argv)

    def said(verdict: dict[str, Any]) -> None:
        """The verdict as data where a gate asked for it, and in words for whoever reads the run."""
        if args.result is not None:
            args.result.write_text(json.dumps(verdict), encoding="utf-8")

    try:
        result = verify(args.repo, args.commit)
    except CouldNotStart as exc:
        said({"commit": args.commit, "held": None, "verdict": "", "could_not_be_run": str(exc)[:400]})
        print(json.dumps({"commit": args.commit, "held": None, "could_not_be_run": str(exc)[:400]}))
        return 2
    except UncheckableCommit as exc:
        said({"commit": args.commit, "held": False, "verdict": "REFUSED", "uncheckable": str(exc)[:400]})
        print(json.dumps({"commit": args.commit, "held": False, "verdict": "REFUSED", "uncheckable": str(exc)[:400]}))
        print(f"did not hold: the commit could not be checked out cleanly beside its parent: {exc}"[:400])
        return 1
    except Exception as exc:  # noqa: BLE001 - a commit that breaks the check does not pass it
        said({"commit": args.commit, "held": False, "verdict": "REFUSED", "unreadable": f"{type(exc).__name__}: {exc}"[:400]})
        print(json.dumps({"commit": args.commit, "held": False, "verdict": "REFUSED", "unreadable": f"{type(exc).__name__}: {exc}"[:400]}))
        print(f"did not hold: the check could not read this change: {type(exc).__name__}: {exc}"[:400])
        return 1
    said(result)
    print(json.dumps(result, indent=1))
    if result["verdict"] == "REFUSED":
        print("did not hold:", "; ".join(_what_failed(result)))
        return 1
    if result["verdict"] == "PROBATION":
        print("on probation:", "; ".join(result["doubts"][:3]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
