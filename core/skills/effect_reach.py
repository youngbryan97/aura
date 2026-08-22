"""What a skill's module actually reaches for, measured from its source.

Every runtime skill declares an ``effect_scope`` and registration refuses one
that declares nothing recognised. That is the declaration half of the
contract, and until now it was the whole contract: a skill could declare
``pure_compute`` and open a socket, spawn a process and write to the user's
home directory, because nothing compared the declaration with the code.

Python cannot sandbox a module in-process — an imported skill has the
interpreter's authority and no wrapper changes that. What it can do is refuse
to load one whose reach exceeds what it declared. That is the achievable form
of isolation here, and it is checkable statically, which means it can be a
gate as well as a load-time refusal.

The mapping is deliberately coarse. It answers "does this module contain a
socket / a spawn / a write / a screen call" and nothing subtler, because a
finer analysis would be a static-analysis project with its own failure modes,
and this one only has to be right about the direction: a module that imports
``socket`` is not ``pure_compute``, whatever its docstring says.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

#: The two edges worth refusing, and the reason it is only two.
#:
#: `core.skills.catalog_policy` groups the ten effect scopes into authority
#: classes — observe, bounded_compute, state_write, artifact_write,
#: external_effect, foreground_control, privileged. Those are KINDS, not
#: levels: a skill that drives the screen is not "more" than one that writes a
#: file, it is a different thing, and a linear ranking over them invents an
#: ordering the policy does not have.
#:
#: What the policy does say is that `observe` promises no effect at all, and
#: `privileged` is the class with no ceiling. So the check refuses exactly two
#: shapes: a skill that promises to only observe and does something, and a
#: skill that does something privileged without saying so. Everything between
#: is a kind mismatch a static reader cannot adjudicate — a network skill that
#: caches to disk is not lying — and flagging it would make this a gate people
#: switch off.
OBSERVE_SCOPES: frozenset[str] = frozenset({"status", "read_only", "pure_compute"})
PRIVILEGED_SCOPE = "privileged_mutation"

#: Modules whose IMPORT is evidence on its own. Nothing in these libraries is
#: a read: importing `requests` and never calling it is dead code, not a
#: harmless one.
IMPORT_REACH: dict[str, str] = {
    "requests": "external_io",
    "httpx": "external_io",
    "aiohttp": "external_io",
    "websockets": "external_io",
    "ftplib": "external_io",
    "smtplib": "external_io",
    "subprocess": "privileged_mutation",
    "pty": "privileged_mutation",
    "pyautogui": "foreground_desktop_control",
    "Quartz": "foreground_desktop_control",
    "AppKit": "foreground_desktop_control",
    "pynput": "foreground_desktop_control",
    "mss": "foreground_desktop_control",
    "selenium": "foreground_browser_dialogue",
    "playwright": "foreground_browser_dialogue",
}

#: First-party gateways. Reaching one of these is the same evidence as
#: reaching the raw facility, and it is the shape this repository prefers, so
#: it must count.
#:
#: `core.security.execution_authority` is deliberately absent: importing the
#: gate is ASKING for permission, not taking it, and counting it would punish
#: exactly the surfaces that do the right thing. The spawn itself is the
#: evidence, and it goes through subprocess_gateway.
FIRST_PARTY_REACH: dict[str, str] = {
    "core.runtime.subprocess_gateway": "privileged_mutation",
    "core.runtime.file_write_gateway": "read_write_artifacts",
    "core.runtime.atomic_writer": "read_write_artifacts",
    "core.capabilities.host_automation": "foreground_desktop_control",
    "core.capabilities.browser_controller": "foreground_browser_dialogue",
    "core.sandbox.macos_sandbox": "sandboxed_compute",
    "security.sandbox": "sandboxed_compute",
}

#: Calls that are evidence, whatever was imported.
#:
#: `socket`, `tempfile`, `shutil` and `urllib` are keyed on the CALL rather
#: than the import, because each contains reads as well as effects and the
#: import cannot tell them apart. `socket.gethostname()` is how
#: `environment_info` reports the machine's name and
#: `tempfile.gettempdir()` is how three skills name a directory — both were
#: reported as network access and disk writes by an import-only rule, which is
#: the kind of false positive that gets a gate switched off.
CALL_REACH: dict[str, str] = {
    "os.system": "privileged_mutation",
    "os.popen": "privileged_mutation",
    "os.execv": "privileged_mutation",
    "os.remove": "read_write_artifacts",
    "os.unlink": "read_write_artifacts",
    "os.rename": "read_write_artifacts",
    "os.makedirs": "read_write_artifacts",
    "eval": "privileged_mutation",
    "exec": "privileged_mutation",
    "socket.socket": "external_io",
    "socket.create_connection": "external_io",
    "socket.getaddrinfo": "external_io",
    "socket.gethostbyname": "external_io",
    "urllib.request.urlopen": "external_io",
    "request.urlopen": "external_io",
    "urlopen": "external_io",
    "tempfile.mkstemp": "read_write_artifacts",
    "tempfile.mkdtemp": "read_write_artifacts",
    "tempfile.NamedTemporaryFile": "read_write_artifacts",
    "tempfile.TemporaryDirectory": "read_write_artifacts",
    "shutil.copy": "read_write_artifacts",
    "shutil.copy2": "read_write_artifacts",
    "shutil.copytree": "read_write_artifacts",
    "shutil.move": "read_write_artifacts",
    "shutil.rmtree": "read_write_artifacts",
}

#: Path methods that write.
WRITE_METHODS = frozenset({"write_text", "write_bytes", "mkdir", "unlink", "rmdir"})


@dataclass(frozen=True)
class Reach:
    """What a module's source justifies, and the lines that justify it."""

    scopes: frozenset[str]
    evidence: tuple[str, ...]

    @property
    def observes_only(self) -> bool:
        return not (self.scopes - OBSERVE_SCOPES)

    @property
    def privileged(self) -> bool:
        return PRIVILEGED_SCOPE in self.scopes


def violation(declared: str, reach: Reach) -> str:
    """The refusal, or "" when the declaration covers the reach.

    Two shapes only. See the note above OBSERVE_SCOPES for why.
    """
    declared = str(declared or "").strip().lower()
    if declared in OBSERVE_SCOPES and not reach.observes_only:
        beyond = ", ".join(sorted(reach.scopes - OBSERVE_SCOPES))
        return f"declares {declared} and reaches {beyond}"
    if reach.privileged and declared != PRIVILEGED_SCOPE:
        return f"declares {declared} and reaches {PRIVILEGED_SCOPE}"
    return ""


def _root(module: str) -> str:
    return module.split(".", 1)[0]


def measure_source(source: str, *, filename: str = "<skill>") -> Reach:
    """The reach of one module's source."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return Reach(frozenset(), ())

    found: dict[str, str] = {}

    def note(scope: str, why: str) -> None:
        if why not in found:
            found[why] = scope

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for candidate in (alias.name, _root(alias.name)):
                    if candidate in IMPORT_REACH:
                        note(IMPORT_REACH[candidate], f"import {candidate}")
                    if candidate in FIRST_PARTY_REACH:
                        note(FIRST_PARTY_REACH[candidate], f"import {candidate}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for candidate in (node.module, _root(node.module)):
                if candidate in IMPORT_REACH:
                    note(IMPORT_REACH[candidate], f"from {candidate}")
                if candidate in FIRST_PARTY_REACH:
                    note(FIRST_PARTY_REACH[candidate], f"from {candidate}")
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in CALL_REACH:
                note(CALL_REACH[name], f"{name}()")
            elif name.rsplit(".", 1)[-1] in WRITE_METHODS and "." in name:
                note("read_write_artifacts", f"{name}()")
            elif name == "open":
                for argument in list(node.args[1:2]) + [
                    keyword.value for keyword in node.keywords if keyword.arg == "mode"
                ]:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        if set(argument.value) & set("wxa+"):
                            note("read_write_artifacts", "open(mode=w)")

    return Reach(frozenset(found.values()), tuple(sorted(found)))


def measure_file(path: Path) -> Reach:
    try:
        return measure_source(path.read_text("utf-8", errors="ignore"), filename=str(path))
    except OSError:
        return Reach(frozenset(), ())


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))
