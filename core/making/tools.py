"""Programs run from a command line, in whatever languages this machine has: built, run on cases, and held to what they print.

A case is a run a person could make: arguments, what is typed in, files that are there, and what comes out. It is written
from the rules before the program is. Every run is under the system's sandbox with no network and no writing but in a
folder of its own, because a model wrote what is run.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.making.trying import Result, Trial
from core.rebuilding.her_build_writes import governed

logger = logging.getLogger("Making.Tools")

__all__ = ["Case", "Cases", "Language", "languages", "try_tool"]

#: The longest one run may take, and the most of what it prints that is read.
RUN_S, MOST_PRINTED = 20.0, 20000


@dataclass(frozen=True)
class Language:
    name: str
    file: str
    #: How it is built, with ``{file}`` and ``{out}``; empty where it is run as it stands.
    build: tuple[str, ...]
    run: tuple[str, ...]
    has: str


def _candidates() -> tuple[tuple[str, Language], ...]:
    return (
        (sys.executable, Language("python", "main.py", (), (sys.executable, "{file}"), "Python 3.12's own library, numpy, and no more")),
        ("node", Language("javascript", "main.mjs", (), ("node", "{file}"), "Node's own modules")),
        ("swiftc", Language("swift", "main.swift", ("swiftc", "-O", "-o", "{out}", "{file}"), ("{out}",), "Foundation")),
        ("clang", Language("c", "main.c", ("clang", "-O2", "-std=c17", "-Wall", "-lm", "-o", "{out}", "{file}"), ("{out}",), "the standard library")),
        ("clang++", Language("c++", "main.cpp", ("clang++", "-O2", "-std=c++20", "-Wall", "-o", "{out}", "{file}"), ("{out}",), "the standard library")),
        ("rustc", Language("rust", "main.rs", ("rustc", "-O", "-o", "{out}", "{file}"), ("{out}",), "std")),
        ("ruby", Language("ruby", "main.rb", (), ("ruby", "{file}"), "Ruby's own library")),
        ("perl", Language("perl", "main.pl", (), ("perl", "{file}"), "Perl's own library")),
        ("bash", Language("bash", "main.sh", (), ("bash", "{file}"), "the system's own commands")),
        ("sqlite3", Language("sql", "main.sql", (), ("sqlite3", ":memory:", ".read {file}"), "SQLite")),
    )


@lru_cache(maxsize=1)
def languages() -> dict[str, Language]:
    """The languages this machine can build and run."""
    return {language.name: language for command, language in _candidates() if shutil.which(command) or Path(command).exists()}


class Case(BaseModel):
    rule: str = Field(description="the rule it tests")
    args: list[str] = Field(default_factory=list)
    typed: str = Field("", description="what is given on standard input")
    files: dict[str, str] = Field(default_factory=dict, description="files that are there, by name")
    prints: list[str] = Field(default_factory=list, description="each of these is in what it prints")
    not_prints: list[str] = Field(default_factory=list)
    leaves_with: int | None = Field(None, description="its exit code, where the rule says")
    writes: dict[str, str] = Field(default_factory=dict, description="files it makes, each with something it holds")


class Cases(BaseModel):
    cases: list[Case] = Field(default_factory=list)


_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{folder}") {more}(literal "/dev/null") (literal "/dev/dtracehelper") (literal "/dev/tty"))
"""
#: What a compiler keeps its own workings in, which a build may write and a run may not.
_BUILDS_IN = '(subpath "/private/var/folders") '


def _run(argv: list[str], folder: Path, typed: str = "", timeout: float = RUN_S, *, building: bool = False) -> tuple[int, str, str]:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    profile = folder / ".sandbox"
    governed("write_text", profile, _PROFILE.format(folder=folder, more=_BUILDS_IN if building else ""), source="making.tools")
    with local_internal_governed_scope("making.tools", domain="tool_execution"):
        done = get_subprocess_gateway().run(["sandbox-exec", "-f", str(profile), *argv], cwd=str(folder), input=typed,
                                            env={**os.environ, "TMPDIR": str(folder)}, timeout=timeout, source="making.tools", accelerator_capability="none")
    return int(done.returncode), str(done.stdout or "")[:MOST_PRINTED], str(done.stderr or "")[:4000]


def _try(code: str, language: Language, cases: list[Case]) -> Trial:
    with tempfile.TemporaryDirectory(prefix="made-") as made:
        folder = Path(made).resolve()
        main = folder / language.file
        governed("write_text", main, code, source="making.tools")
        out = folder / "program"
        say = {"file": str(main), "out": str(out)}
        trial = Trial()
        if language.build:
            try:
                left, printed, wrong = _run([part.format(**say) for part in language.build], folder, timeout=180.0, building=True)
            except Exception as exc:  # noqa: BLE001 - a build that cannot be run is a build that failed
                return Trial(generic=[Result("it builds", False, f"{type(exc).__name__}: {exc}"[:600])])
            if left != 0:
                return Trial(generic=[Result("it builds", False, (wrong or printed)[:900])])
            trial.generic.append(Result("it builds", True))
        argv = [part.format(**say) for part in language.run]
        if language.name != "sql":
            trial.generic.append(_told(lambda: _helps(argv, folder)))
        for n, case in enumerate(cases):
            trial.rules.append(_told(lambda case=case: _one(case, argv, folder / f"run{n}"), f"{case.rule} [case {n + 1}]"))
        return trial


def _told(act: Any, name: str = "it says how it is used") -> Result:
    try:
        why = act()
        return Result(name, not why, why or "")
    except Exception as exc:  # noqa: BLE001 - a run that breaks is a run that failed, and is told as one
        return Result(name, False, f"{type(exc).__name__}: {exc}"[:400])


def _helps(argv: list[str], folder: Path) -> str:
    left, printed, wrong = _run([*argv, "--help"], folder)
    if "Traceback" in wrong:
        return wrong[-400:]
    return "" if len((printed + wrong).strip()) > 20 and left == 0 else f"--help printed {len(printed)} characters and left with {left}"


def _one(case: Case, argv: list[str], folder: Path) -> str:
    governed("ensure_directory", folder, source="making.tools")
    for name, text in case.files.items():
        target = (folder / name).resolve()
        if folder.resolve() not in target.parents:
            return f"a case names a file outside its folder: {name}"
        governed("ensure_directory", target.parent, source="making.tools")
        governed("write_text", target, text, source="making.tools")
    left, printed, wrong = _run([*argv, *case.args], folder, case.typed)
    said = f"run with {case.args} and given {case.typed[:120]!r} it printed {printed[:300]!r}" + (f" and to standard error {wrong[-260:]!r}" if wrong else "")
    if case.leaves_with is not None and left != case.leaves_with:
        return f"it left with {left} and not {case.leaves_with}; {said}"
    if "Traceback" in wrong and (case.leaves_with in (None, 0)):
        return said
    both = printed + wrong
    for want in case.prints:
        if " ".join(want.split()) not in " ".join(both.split()):
            return f"{want!r} is not in what it printed; {said}"
    for bad in case.not_prints:
        if bad and bad in both:
            return f"{bad!r} is in what it printed; {said}"
    for name, holds in case.writes.items():
        made = folder / name
        if not made.is_file():
            return f"it made no {name}; {said}"
        if holds and holds not in made.read_text(encoding="utf-8", errors="replace"):
            return f"{name} does not hold {holds!r}"
    return ""


async def try_tool(code: str, language: Language, cases: list[Case]) -> Trial:
    """What building it and running it on its cases shows."""
    return await asyncio.to_thread(_try, code, language, cases)
