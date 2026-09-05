"""Static audit: which writes bypass the atomic write gateway.

CLAUDE.md states the rule plainly: "All consequential file writes go through
core/runtime/file_write_gateway.py". Nothing enforced it.

The measured answer turned out to be good news — core/ and interface/ carry
two direct writes, both deliberate (see EXEMPT_CRASH_DUMPS). A first, cruder
version of this scan reported 153, then 13; both numbers were its own false
positives, from failing to recognise `file_gateway.write_text(...)` and the
manual write-temp-then-os.replace idiom as compliant. The count is recorded
here because a scanner's early wrong answers are worth knowing about: this
audit is only useful while it is trusted, and it earns that by not crying
wolf.

So the value is the ratchet, not a cleanup. A direct write is not atomic. The old contents are truncated the moment the
file is opened, and the new contents land in pieces. A crash, an OOM kill or
a power loss between those two moments leaves a truncated or empty file where
durable state used to be — and this runtime is killed by a liveness sentinel
often enough that the window is not theoretical. The failure is also silent
in exactly the wrong way: the next boot reads a valid-looking short file and
carries on with half a state.

The gateway exists and does the right thing (write temp, fsync, os.replace).
What was missing is anything that notices when a new call site skips it.

This module answers the static question — *does this write go through the
gateway* — so a ratchet can freeze the existing debt and fail on additions.
It deliberately reports rather than judges: a scratch file, a log line and a
belief store all look the same to a parser, and the caller decides which
matter.
"""
from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Direct write calls that bypass the gateway.
_DIRECT_WRITE_METHODS = frozenset({"write_text", "write_bytes"})

#: Modes that truncate or extend a file. Read modes are irrelevant here.
_MUTATING_MODES = ("w", "a", "x", "+")

#: Directories scanned. tools/ and tests/ write freely on purpose — a
#: benchmark harness losing its scratch output is not a durability incident.
_DEFAULT_ROOTS = ("core", "interface")

#: Writes that must NOT go through the gateway, by design.
#:
#: These run while the process is dying — a memory-spike stack dump and a
#: loop-wedge dump before a forced exit. The gateway allocates, takes locks
#: and fsyncs; at that moment the loop is already wedged or the allocator is
#: already failing, which is exactly when those three things hang. A forensic
#: dump that blocks forever produces no forensics.
#:
#: They are also append-only crash logs, so the truncation window the gateway
#: protects against does not apply: nothing existing is at risk.
EXEMPT_CRASH_DUMPS: frozenset[str] = frozenset({
    "core/resilience/memory_watchdog.py::MemoryWatchdog._dump_thread_stacks::open",
    "core/resilience/stall_watchdog.py::StallWatchdog._force_exit_for_restart::open",
})

_SKIP_PARTS = frozenset({
    "__pycache__", ".venv", "node_modules", ".git", "build", "dist",
    ".claude", "artifacts",
})

#: Substrings marking a receiver that IS the durable lane, or that writes
#: somewhere a crash cannot corrupt.
#:
#: Substrings rather than exact names, learned the hard way: the first
#: version listed "gateway" and missed `file_gateway.write_text(...)`,
#: reporting eight compliant calls in mutation_safety.py as bypasses. A
#: scanner that cries wolf gets muted, which is worse than not having one.
_SAFE_RECEIVER_MARKERS = (
    "gateway",     # file_gateway, _write_gateway, get_file_write_gateway()
    "atomic",      # atomic_writer and friends
    "tmp",         # manual write-temp-then-os.replace, the atomic idiom
    "temp",
    "buffer",
    "stream",
    "handle",
    "stdout",
    "stderr",
    "stringio",
    "bytesio",
)


@dataclass(frozen=True)
class DirectWrite:
    """One write that does not go through the gateway."""

    path: str
    line: int
    call: str
    function: str = ""

    def key(self) -> str:
        """Stable identity for a baseline entry.

        Deliberately excludes the line number: a write does not become a new
        defect because something above it grew by three lines, and a baseline
        that churns on every edit is a baseline nobody trusts.
        """
        return f"{self.path}::{self.function}::{self.call}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "call": self.call,
            "function": self.function,
        }


@dataclass
class DurableWriteReport:
    writes: list[DirectWrite] = field(default_factory=list)

    @property
    def keys(self) -> set[str]:
        return {write.key() for write in self.writes}

    @property
    def enforceable(self) -> list[DirectWrite]:
        """Bypasses that are actually defects — crash dumps excluded."""
        return [w for w in self.writes if w.key() not in EXEMPT_CRASH_DUMPS]

    def new_since(self, baseline: Iterable[str]) -> list[DirectWrite]:
        known = set(baseline)
        return sorted(
            (w for w in self.writes if w.key() not in known),
            key=lambda w: (w.path, w.line),
        )

    def fixed_since(self, baseline: Iterable[str]) -> list[str]:
        """Baseline entries that no longer exist — the ratchet's other half.

        A baseline that only ever grows is a list of excuses. Entries that
        have been fixed must leave it, or the next real regression hides
        behind a stale allowance.
        """
        return sorted(set(baseline) - self.keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.durable_write_audit.v1",
            "total": len(self.writes),
            "writes": [w.to_dict() for w in self.writes],
        }


class _WriteVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.found: list[DirectWrite] = []
        self._scope: list[str] = []
        #: Names bound to a descriptor from ``os.open(..., O_EXCL, ...)``.
        #: Scoped, so a name reused in another function cannot inherit an
        #: exemption it did not earn.
        self._exclusive_fds: list[set[str]] = [set()]
        #: Names that are, or are derived from, a temporary directory.
        self._temp_names: list[set[str]] = [set()]

    def _enter(self, node: Any) -> None:
        self._scope.append(node.name)
        self._exclusive_fds.append(set())
        self._temp_names.append(set())
        self.generic_visit(node)
        self._temp_names.pop()
        self._exclusive_fds.pop()
        self._scope.pop()

    def visit_With(self, node: ast.With) -> None:  # noqa: N802 - ast API
        """Bind ``with tempfile.TemporaryDirectory() as x`` to a temp name.

        Everything under such a directory is deleted when the block exits, so a
        crash mid-write cannot corrupt durable state — there is no durable
        state. The scanner already trusts receivers *named* like temporaries;
        this extends the same judgement to paths demonstrably derived from one,
        which is how the compliant idiom actually reads.
        """
        for item in node.items:
            if _is_tempfile_call(item.context_expr) and isinstance(item.optional_vars, ast.Name):
                self._temp_names[-1].add(item.optional_vars.id)
        self.generic_visit(node)

    visit_AsyncWith = visit_With  # noqa: N815 - ast API casing

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast API
        """Notice ``fd = os.open(path, ... | os.O_EXCL, 0o600)``.

        This is the one shape of raw write that is not the hazard this scanner
        exists to find. The ratchet's premise is truncation: a plain open
        destroys the old contents before the new ones land. ``O_CREAT|O_EXCL``
        cannot do that — it *fails* when the file already exists, so there is
        no durable state for it to shorten.

        It also cannot be replaced by the gateway without losing the property
        it was chosen for. The gateway is temp-file-plus-replace, which is
        atomic but NOT exclusive: two processes minting a key would each write
        their own and the loser would return a value the file no longer holds.
        Key material needs the exclusivity, so the primitive is correct and
        flagging it would push callers toward a weaker write.
        """
        if _is_exclusive_open(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._exclusive_fds[-1].add(target.id)
        # `adapter_path = Path(temporary) / "adapter.safetensors"` — and
        # `scratch = tempfile.mkdtemp()` — are both still the temporary.
        if _is_tempfile_call(node.value) or self._is_temp_derived(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._temp_names[-1].add(target.id)
        self.generic_visit(node)

    def _is_temp_derived(self, node: ast.AST) -> bool:
        """Does this expression trace back to a temporary directory?"""
        if isinstance(node, ast.Name):
            return any(node.id in scope for scope in self._temp_names)
        if isinstance(node, ast.BinOp):
            # Path joins: judge the base, exactly as _receiver_is_safe does.
            return self._is_temp_derived(node.left)
        if isinstance(node, ast.Call):
            # Path(temporary), Path(temporary).resolve(), ...
            func = node.func
            if isinstance(func, ast.Attribute):
                return self._is_temp_derived(func.value)
            return any(self._is_temp_derived(arg) for arg in node.args)
        if isinstance(node, ast.Attribute):
            return self._is_temp_derived(node.value)
        return False

    visit_FunctionDef = _enter          # noqa: N815 - ast API casing
    visit_AsyncFunctionDef = _enter     # noqa: N815
    visit_ClassDef = _enter             # noqa: N815

    def _record(self, node: ast.Call, call: str) -> None:
        self.found.append(
            DirectWrite(
                path=self.rel_path,
                line=node.lineno,
                call=call,
                function=".".join(self._scope),
            )
        )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        func = node.func
        if isinstance(func, ast.Attribute):
            if (
                func.attr in _DIRECT_WRITE_METHODS
                and not _receiver_is_safe(func.value)
                and not self._is_temp_derived(func.value)
            ):
                self._record(node, func.attr)
        elif isinstance(func, ast.Name) and func.id == "open":
            if _opens_for_writing(node) and not self._wraps_exclusive_fd(node):
                self._record(node, "open")
        self.generic_visit(node)

    def _wraps_exclusive_fd(self, node: ast.Call) -> bool:
        """``open(fd, "wb")`` where fd came from an exclusive create.

        The first argument is a descriptor, not a path: this call adopts a file
        that ``os.open`` already created, it does not open one by name.
        """
        if not node.args:
            return False
        first = node.args[0]
        # Inline: open(os.open(path, ...O_EXCL, 0o600), "wb").
        if _is_exclusive_open(first):
            return True
        if not isinstance(first, ast.Name):
            return False
        return any(first.id in scope for scope in self._exclusive_fds)


def _receiver_is_safe(node: ast.AST) -> bool:
    """Is this write already going somewhere a crash cannot corrupt?"""
    name = ""
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    elif isinstance(node, ast.Call):
        inner = node.func
        name = inner.attr if isinstance(inner, ast.Attribute) else getattr(inner, "id", "")
    elif isinstance(node, ast.BinOp):
        # `entry_dir / "result.json"` — judge the base, not the join.
        return _receiver_is_safe(node.left)
    lowered = name.lower()
    return any(marker in lowered for marker in _SAFE_RECEIVER_MARKERS)


#: Factories whose result is deleted by the runtime, not kept as state.
_TEMPFILE_FACTORIES = frozenset(
    {"TemporaryDirectory", "NamedTemporaryFile", "mkdtemp", "mkstemp"}
)


def _is_tempfile_call(node: ast.AST) -> bool:
    """Is this a ``tempfile`` factory call?"""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in _TEMPFILE_FACTORIES
    if isinstance(func, ast.Name):
        return func.id in _TEMPFILE_FACTORIES
    return False


def _is_exclusive_open(node: ast.AST) -> bool:
    """Is this ``os.open(...)`` with ``O_EXCL`` among its flags?

    Matched structurally rather than by rendering the source, so
    ``os.O_EXCL``, a bare ``O_EXCL``, and any ``|`` combination of them all
    read the same. Requires the call to actually be ``os.open`` — a local
    helper named ``open`` must not inherit the exemption.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    is_os_open = (
        isinstance(func, ast.Attribute)
        and func.attr == "open"
        and isinstance(func.value, ast.Name)
        and func.value.id == "os"
    )
    if not is_os_open:
        return False
    return any(
        isinstance(sub, ast.Attribute) and sub.attr == "O_EXCL"
        or isinstance(sub, ast.Name) and sub.id == "O_EXCL"
        for arg in node.args[1:]
        for sub in ast.walk(arg)
    )


def _opens_for_writing(node: ast.Call) -> bool:
    modes = [arg for arg in node.args[1:2]]
    modes += [kw.value for kw in node.keywords if kw.arg == "mode"]
    for mode in modes:
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
            if any(flag in mode.value for flag in _MUTATING_MODES):
                return True
    return False


def scan_direct_writes(
    *,
    roots: Iterable[str] = _DEFAULT_ROOTS,
    repo_root: Path | None = None,
) -> DurableWriteReport:
    """Every gateway-bypassing write under the scanned roots."""
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    found: list[DirectWrite] = []
    for name in roots:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root)
            # Match the skip list against the path RELATIVE to the root. The
            # absolute path is the wrong thing to test: a worktree lives under
            # ``.claude/worktrees/`` — a skipped part — so every file in it
            # would be skipped, the scan would find nothing, and a ratchet
            # built on it would read that as "everything was fixed".
            if _SKIP_PARTS.intersection(rel.parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            visitor = _WriteVisitor(str(rel))
            visitor.visit(tree)
            found.extend(visitor.found)
    return DurableWriteReport(sorted(found, key=lambda w: (w.path, w.line)))


__all__ = [
    "EXEMPT_CRASH_DUMPS",
    "DirectWrite",
    "DurableWriteReport",
    "scan_direct_writes",
]
