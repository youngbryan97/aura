"""What a project offers as a way to run it, and what running it showed.

LIVE, 2026-08-22. `diagnose_repository` knew one experiment — run the test
suite — so a project with no tests got "no test runner was found" and nothing
else. A person with a symptom and no traceback was told the tool could not
look.

A diagnosis is an experiment, and which experiments are available is a property
of the project rather than of the tool. This finds them: a test runner, and the
scripts that are meant to be run. Both are read off the code. An entry point is
a top-level module nothing else imports which does work when loaded — that is
what a script IS, and it holds for a project laid out in any style.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Affordance", "Observation", "affordances", "observe"]

#: A project that takes longer than this to run is not one a chat turn waits on.
_RUN_TIMEOUT_S = 120.0

#: Statements that only declare something, so a file holding nothing else is a
#: library rather than a script.
_DECLARATIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                 ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Expr, ast.Pass)

_SKIP_PARTS = {"__pycache__", "node_modules", "site-packages", "build", "dist", ".venv"}


@dataclass(frozen=True, slots=True)
class Affordance:
    """One way this project can be run, discovered from the project itself."""

    kind: str
    argv: tuple[str, ...]
    described: str
    why: str = ""


@dataclass(frozen=True, slots=True)
class Observation:
    """What running it produced."""

    kind: str
    ran: str
    output: str
    exit_code: int
    ok: bool
    error: str = ""


def _worth_reading(path: Path) -> bool:
    """A source file of the project, not of something it vendored."""
    return not any(part.startswith(".") or part in _SKIP_PARTS for part in path.parts)


def _imported_modules(tree: ast.Module) -> set[str]:
    """Every module name this file imports, by its last component."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _does_work_when_loaded(tree: ast.Module) -> bool:
    """True when loading this module runs something, rather than only declaring."""
    for node in tree.body:
        if isinstance(node, ast.If):
            # `if __name__ == "__main__":` is the explicit form of the same thing.
            if "__name__" in ast.dump(node.test):
                return True
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # a docstring
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
        if not isinstance(node, _DECLARATIONS):
            return True
    return False


def _entry_points(root: Path) -> list[Affordance]:
    """Modules meant to be run: they do work when loaded, and nothing imports them."""
    parsed: dict[Path, ast.Module] = {}
    for path in sorted(root.rglob("*.py")):
        if not _worth_reading(path.relative_to(root)):
            continue
        try:
            parsed[path] = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
    imported: set[str] = set()
    for tree in parsed.values():
        imported |= _imported_modules(tree)
    found: list[Affordance] = []
    for path, tree in parsed.items():
        stem = path.stem
        if stem in imported or stem.startswith("test_") or stem.endswith("_test"):
            continue
        if stem != "__main__" and not _does_work_when_loaded(tree):
            continue
        shown = str(path.relative_to(root))
        found.append(
            Affordance(
                kind="entry point",
                argv=(shown,),
                described=f"python {shown}",
                why=f"nothing in this project imports {stem}, and loading it runs something",
            )
        )
    # A shallower path is the more likely front door.
    found.sort(key=lambda item: (item.argv[0].count("/"), item.argv[0]))
    return found


def _test_runner(root: Path) -> list[Affordance]:
    """How this project runs its tests, discovered rather than assumed."""
    if (root / "tests").is_dir() or list(root.glob("test_*.py")) or list(root.glob("*_test.py")):
        target = "tests" if (root / "tests").is_dir() else "."
        return [
            Affordance(
                kind="tests",
                argv=("-m", "pytest", target, "-q", "--no-header", "-p", "no:cacheprovider"),
                described=f"pytest {target}",
                why="this project has tests",
            )
        ]
    if (root / "manage.py").is_file():
        return [Affordance(kind="tests", argv=("manage.py", "test"), described="manage.py test",
                           why="this is a Django project")]
    return []


def affordances(root: str | Path) -> tuple[Affordance, ...]:
    """Every way this project can be run, tests first."""
    base = Path(str(root)).expanduser()
    if not base.is_dir():
        return ()
    return tuple(_test_runner(base) + _entry_points(base))


def observe(root: str | Path, affordance: Affordance) -> Observation:
    """Run it through the governed gateway and keep what it produced."""
    import sys

    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    base = Path(str(root)).expanduser()
    try:
        with local_internal_governed_scope("diagnosis.experiment"):
            done = get_subprocess_gateway().run(
                [sys.executable, *affordance.argv],
                cwd=str(base),
                capture_output=True,
                text=True,
                timeout=_RUN_TIMEOUT_S,
                read_only=False,
                check=False,
                source="diagnosis.experiment",
                accelerator_capability="none",
            )
    except (OSError, subprocess.SubprocessError, RuntimeError, ImportError) as exc:
        return Observation(
            kind=affordance.kind, ran=affordance.described, output="", exit_code=-1,
            ok=False, error=f"{type(exc).__name__}: {exc}"[:300],
        )
    output = f"{done.stdout or ''}\n{done.stderr or ''}".strip()
    return Observation(
        kind=affordance.kind,
        ran=affordance.described,
        output=output,
        exit_code=done.returncode,
        ok=done.returncode == 0,
    )
