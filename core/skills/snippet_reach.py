"""How far a snippet can reach, read off the snippet.

`run_code` is rated high risk because sandboxed_compute "usually means running
code the MODEL wrote". That is true and it is a statement about authorship, not
about consequence. Rated that way, every snippet needs a confirmation the turn
has no way to ask for, so "read these docs and actually use the library" cannot
be answered at all — while the same sandbox runs the same import in 40ms with
no effect on anything.

The code is a string before it runs, so what it can touch is a question with an
answer. This reads it: a snippet that imports arithmetic and prints a number
reaches nothing, and a snippet that opens a socket, writes a file or spawns a
process reaches exactly as far as it did before.

Deliberately conservative in one direction. Anything that could resolve a name
at runtime — `__import__`, `eval`, `exec`, `getattr` on a module, an import
built from a string — is unknown reach, and unknown is treated as the widest,
because the whole point is that the reading has to be sound rather than
encouraging.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

__all__ = ["SnippetReach", "reach_of"]

#: Modules that touch the world outside the process.
_REACHES_OUT = frozenset(
    {
        "socket", "ssl", "http", "urllib", "urllib2", "urllib3", "requests",
        "httpx", "aiohttp", "ftplib", "smtplib", "imaplib", "poplib",
        "telnetlib", "xmlrpc", "webbrowser", "asyncio",
        "subprocess", "multiprocessing", "os", "sys", "shutil", "pathlib",
        "tempfile", "glob", "io", "fileinput", "sqlite3", "dbm", "shelve",
        "pickle", "ctypes", "mmap", "signal", "resource", "pty", "fcntl",
        "importlib", "runpy", "site", "gc", "inspect", "threading",
    }
)

#: Names that resolve something at runtime, so no reading of the source binds.
_UNBOUNDED = frozenset({"__import__", "eval", "exec", "compile", "globals", "locals", "vars"})

#: `sys` and `pathlib` are how a snippet reaches a named library, so their
#: read-only uses are separated from the rest below.
_PATH_SETUP = frozenset({"sys", "pathlib", "os"})


@dataclass(frozen=True, slots=True)
class SnippetReach:
    """What this snippet can touch."""

    parses: bool
    unbounded: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    writes_files: bool = False
    spawns: bool = False
    networks: bool = False

    @property
    def only_computes(self) -> bool:
        """True when nothing here can affect anything outside the process."""
        return (
            self.parses
            and not self.unbounded
            and not self.writes_files
            and not self.spawns
            and not self.networks
        )

    def why(self) -> str:
        """The reason it is not pure computation, or "" when it is."""
        if not self.parses:
            return "the snippet does not parse"
        reasons = []
        if self.unbounded:
            reasons.append("it resolves names at runtime (" + ", ".join(self.unbounded) + ")")
        if self.networks:
            reasons.append("it can reach the network")
        if self.writes_files:
            reasons.append("it can write files")
        if self.spawns:
            reasons.append("it can start a process")
        return "; ".join(reasons)


#: Network-capable module roots, kept apart so the reason can say which.
_NETWORK = frozenset(
    {
        "socket", "ssl", "http", "urllib", "urllib2", "urllib3", "requests",
        "httpx", "aiohttp", "ftplib", "smtplib", "imaplib", "poplib",
        "telnetlib", "xmlrpc", "webbrowser",
    }
)

_SPAWNS = frozenset({"subprocess", "multiprocessing", "pty", "ctypes"})

#: Open modes that change something on disk.
_WRITE_MODES = ("w", "a", "x", "+")


def _root(name: str) -> str:
    return str(name or "").split(".")[0]


def reach_of(code: object) -> SnippetReach:
    """What this snippet can touch, read off its syntax."""
    source = str(code or "")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return SnippetReach(parses=False)

    modules: set[str] = set()
    unbounded: set[str] = set()
    writes = spawns = networks = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(_root(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(_root(node.module))
            if node.level:
                # A relative import inside a snippet has no package to resolve
                # against, so what it would load cannot be read here.
                unbounded.add("relative import")
        elif isinstance(node, ast.Call):
            called = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", "")
            )
            if called in _UNBOUNDED:
                unbounded.add(str(called))
            if called == "getattr" and len(node.args) >= 2:
                if not isinstance(node.args[1], ast.Constant):
                    unbounded.add("getattr")
            if called == "open":
                mode = ""
                for index, argument in enumerate(node.args):
                    if index == 1 and isinstance(argument, ast.Constant):
                        mode = str(argument.value)
                for keyword in node.keywords:
                    if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                        mode = str(keyword.value.value)
                if any(flag in mode for flag in _WRITE_MODES):
                    writes = True
            if called in {
                "system", "popen", "spawn", "spawnv", "fork", "execv", "execvp",
                "run", "call", "check_output", "Popen",
            }:
                spawns = True
            if called in {"remove", "unlink", "rmdir", "rmtree", "rename", "replace",
                          "mkdir", "makedirs", "chmod", "chown", "symlink", "link",
                          "write_text", "write_bytes", "touch"}:
                writes = True

    networks = bool(modules & _NETWORK)
    spawns = spawns or bool(modules & _SPAWNS)
    return SnippetReach(
        parses=True,
        unbounded=tuple(sorted(unbounded)),
        modules=tuple(sorted(modules)),
        writes_files=writes,
        spawns=spawns,
        networks=networks,
    )
