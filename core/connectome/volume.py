"""core/connectome/volume.py — reconstructing Aura's anatomy from the tissue.

Electron microscopy gives connectomics an unarguable substrate: the tissue is
there, and every claim about a circuit has to survive being checked against it.
Aura's equivalent substrate is her source. It is complete, it is exact, and
nothing in it is a summary of something else.

The mapping is one-to-one where the biology has a real analogue:

======================  ==========================================
cell                    a function or method
neuropil                the module the arbour sits in
region                  the package above that module
contact (synapse)       one call site
connection strength     how many call sites join the same pair
axon initial segment    the guard that decides whether the body runs
cell class              measured from what the cell's exits do
======================  ==========================================

Two things about the reconstruction are worth stating plainly, because a
connectome that hides them is a drawing rather than a measurement.

**Resolution is lossy in the same way segmentation is.** A call to ``run()``
where forty modules define ``run`` cannot be resolved from the call site alone.
Those are recorded as ambiguous and left out of the graph rather than attached
to a guess, and the count is reported. It is the same trade every automated
reconstruction makes between merge errors and coverage.

**A call carries information both ways.** The call drives the callee, so the
drive edge runs caller to callee. The return value drives the caller, so when
the value is used there is a second edge running back, and its sign is negative
when the callee is a gate. A guard that calls a predicate and returns early on
False is an inhibitory contact onto the caller's initial segment, which is the
exact structure behind the failure mode where a gate's own failure keeps its
precondition true.
"""

from __future__ import annotations

import ast
import builtins
import logging
import os
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import (
    CellClass,
    Compartment,
    Connection,
    ConnectomeSnapshot,
    ContactSite,
    EdgeKind,
    Neuropil,
    Unit,
    stable_id,
)

logger = logging.getLogger("Aura.Connectome.Volume")

__all__ = [
    "ReconstructionConfig",
    "VolumeReconstructor",
    "AmbiguousSite",
    "reconstruct",
    "DEFAULT_ROOTS",
]

#: Packages that make up the body being reconstructed.
DEFAULT_ROOTS: tuple[str, ...] = (
    "core",
    "interface",
    "skills",
    "security",
    "llm",
    "executors",
)

#: Calls that maintain the cell without carrying computation downstream. A cell
#: whose whole output is these is glial: it keeps the tissue alive and does not
#: signal through it.
_MAINTENANCE_CALLS: frozenset[str] = frozenset(
    {
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "exception",
        "critical",
        "log",
        "getLogger",
        "record_degradation",
        "format_exc",
        "print_exc",
        "flush",
        "close",
        "gc",
        "collect",
        "sleep",
    }
)

#: Names that resolve outside the reconstructed volume. A call to one of these
#: is an axon leaving the imaged block, not a gap in the reconstruction.
_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))


def _outside_volume(module_path: str) -> bool:
    head = module_path.split(".", 1)[0]
    return head in _STDLIB_MODULES or (head and head not in DEFAULT_ROOTS)


_STDLIB_MODULES: frozenset[str] = frozenset(sys.stdlib_module_names)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Bounds on the sweep. Every one of these exists so a run terminates."""

    roots: tuple[str, ...] = DEFAULT_ROOTS
    max_files: int = 20_000
    max_file_bytes: int = 4_000_000
    skip_dirs: frozenset[str] = frozenset(
        {"__pycache__", ".git", ".venv", "node_modules", ".claude", "artifacts", "data"}
    )
    include_tests: bool = False
    #: A name resolving to more than this many candidates is left unresolved.
    #: Connectomics makes the same trade: attaching a process to a guess buys
    #: coverage and pays for it in merge errors, and merge errors are the ones
    #: that survive review because a wrong join looks like a real circuit.
    ambiguity_ceiling: int = 1


@dataclass
class _RawCall:
    """A call site before its target is known."""

    caller: str
    target_name: str
    qualifier: str | None
    locus: str
    value_used: bool
    in_guard: bool
    scope: str = ""
    local_types: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AmbiguousSite:
    """A call site with more than one plausible target.

    Segmentation faces the same choice at every uncertain boundary: attach and
    risk a merge error, or leave it and accept a split. Keeping the site with
    its candidates means the choice can be made once, at a threshold that was
    measured, instead of separately at every site by whoever wrote the code.
    """

    caller: str
    caller_module: str
    caller_region: str
    target_name: str
    qualifier: str | None
    locus: str
    value_used: bool
    in_guard: bool
    candidates: tuple[str, ...]
    imported: tuple[str, ...] = ()


@dataclass
class _ModuleScan:
    module: str
    region: str
    units: list[Unit] = field(default_factory=list)
    calls: list[_RawCall] = field(default_factory=list)
    imports: dict[str, str] = field(default_factory=dict)
    module_aliases: dict[str, str] = field(default_factory=dict)
    bases: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _module_name(path: Path, repo: Path) -> str:
    rel = path.relative_to(repo).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_suppressive_return(node: ast.Return) -> bool:
    value = node.value
    if value is None:
        return True
    if isinstance(value, ast.Constant):
        payload = value.value
        if payload is None or payload is False:
            return True
        if isinstance(payload, (str, bytes)) and not payload:
            return True
        if isinstance(payload, (int, float)) and payload == 0 and not isinstance(payload, bool):
            return False
        return False
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)) and not value.elts:
        return True
    if isinstance(value, ast.Dict) and not value.keys:
        return True
    return False


def _returns_only_boolean(returns: Sequence[ast.Return]) -> bool:
    seen = False
    for node in returns:
        value = node.value
        if value is None:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, bool):
            seen = True
            continue
        if isinstance(value, ast.Compare) or (
            isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
        ):
            seen = True
            continue
        if isinstance(value, ast.BoolOp):
            seen = True
            continue
        return False
    return seen


class _FunctionVisitor(ast.NodeVisitor):
    """Walks one function body and records what it does.

    Nested functions are folded into the enclosing cell rather than promoted:
    a closure has no identity a caller elsewhere can reach, so it is a branch
    of its parent's arbour rather than a cell of its own.
    """

    def __init__(self) -> None:
        self.returns: list[ast.Return] = []
        self.raises = 0
        self.guards = 0
        self.calls: list[tuple[str, str | None, int, int, bool, bool]] = []
        self.mutates_state = False
        self.maintenance_calls = 0
        self.total_calls = 0
        self.local_types: dict[str, str] = {}
        self._guard_depth = 0
        self._used_depth = 0

    # -- statements -----------------------------------------------------

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        self.returns.append(node)
        if node.value is not None:
            self._used_depth += 1
            self.generic_visit(node)
            self._used_depth -= 1

    def visit_Raise(self, node: ast.Raise) -> None:  # noqa: N802
        self.raises += 1
        self._used_depth += 1
        self.generic_visit(node)
        self._used_depth -= 1

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self.mutates_state = True

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        self.mutates_state = True

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                self.mutates_state = True
        # ``handle = SomeClass()`` is the one piece of type information a call
        # site gives away for free, and it resolves the common case where a
        # method is reached through a local rather than through self.
        if isinstance(node.value, ast.Call):
            builder, _ = _call_target(node.value.func)
            if builder and builder[:1].isupper():
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.local_types[target.id] = builder
        self._used_depth += 1
        self.generic_visit(node)
        self._used_depth -= 1

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        if isinstance(node.target, ast.Attribute):
            self.mutates_state = True
        self._used_depth += 1
        self.generic_visit(node)
        self._used_depth -= 1

    def visit_Expr(self, node: ast.Expr) -> None:  # noqa: N802
        # A bare expression statement discards its value: the call drives the
        # callee and nothing comes back that the caller reads.
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        if self._body_is_suppressive(node.body):
            self.guards += 1
            self._guard_depth += 1
            self._used_depth += 1
            self.visit(node.test)
            self._used_depth -= 1
            self._guard_depth -= 1
        else:
            self._used_depth += 1
            self.visit(node.test)
            self._used_depth -= 1
        for child in node.body:
            self.visit(child)
        for child in node.orelse:
            self.visit(child)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name, qualifier = _call_target(node.func)
        if name:
            self.total_calls += 1
            if name in _MAINTENANCE_CALLS:
                self.maintenance_calls += 1
            self.calls.append(
                (
                    name,
                    qualifier,
                    getattr(node, "lineno", 0),
                    getattr(node, "col_offset", 0),
                    self._used_depth > 0,
                    self._guard_depth > 0,
                )
            )
        self._used_depth += 1
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)
        self._used_depth -= 1
        self.visit(node.func)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        for child in node.body:
            self.visit(child)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        for child in node.body:
            self.visit(child)

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _body_is_suppressive(body: Sequence[ast.stmt]) -> bool:
        if not body:
            return False
        head = body[0]
        if isinstance(head, ast.Raise):
            return True
        if isinstance(head, ast.Return):
            return _is_suppressive_return(head)
        if isinstance(head, (ast.Continue, ast.Pass)) and len(body) == 1:
            return True
        return False


def _call_target(func: ast.expr) -> tuple[str, str | None]:
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Name):
            return func.attr, base.id
        if isinstance(base, ast.Attribute):
            return func.attr, base.attr
        return func.attr, None
    return "", None


class VolumeReconstructor:
    """Turns a source tree into a connectome.

    One pass parses every file and records cells plus unresolved call sites.
    A second pass resolves the call sites against the index of cells the first
    pass built, which is the only order that works: a call's sign depends on
    the class of the cell it lands on, and that class is not known until every
    cell has been seen.
    """

    def __init__(self, repo: Path, config: ReconstructionConfig | None = None) -> None:
        self.repo = Path(repo).resolve()
        self.config = config or ReconstructionConfig()
        self.scans: list[_ModuleScan] = []
        self.parse_failures: list[tuple[str, str]] = []
        self.files_read = 0
        self.bytes_read = 0
        self.ambiguous_calls = 0
        self.unresolved_calls = 0
        self.resolved_calls = 0
        self.out_of_volume_calls = 0
        #: Sites the first pass refused to attach, kept with their candidates so
        #: an agglomeration step can be swept and scored rather than guessed at.
        self.ambiguous_sites: list[AmbiguousSite] = []

    # -- discovery ------------------------------------------------------

    def iter_files(self) -> Iterator[Path]:
        seen = 0
        for root in self.config.roots:
            base = self.repo / root
            if not base.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(d for d in dirnames if d not in self.config.skip_dirs)
                for filename in sorted(filenames):
                    if not filename.endswith(".py"):
                        continue
                    if not self.config.include_tests and filename.startswith("test_"):
                        continue
                    if seen >= self.config.max_files:
                        return
                    seen += 1
                    yield Path(dirpath) / filename

    # -- pass one -------------------------------------------------------

    def scan(self) -> None:
        for path in self.iter_files():
            try:
                raw = path.read_bytes()
            except OSError as exc:
                self.parse_failures.append((str(path), f"read: {exc}"))
                continue
            if len(raw) > self.config.max_file_bytes:
                self.parse_failures.append((str(path), "oversize"))
                continue
            self.files_read += 1
            self.bytes_read += len(raw)
            try:
                tree = ast.parse(raw.decode("utf-8", "replace"), filename=str(path))
            except SyntaxError as exc:
                self.parse_failures.append((str(path), f"syntax: {exc.lineno}"))
                continue
            self.scans.append(self._scan_module(path, tree))

    def _scan_module(self, path: Path, tree: ast.Module) -> _ModuleScan:
        module = _module_name(path, self.repo)
        region = module.split(".")[1] if module.count(".") >= 1 else module.split(".")[0]
        scan = _ModuleScan(module=module, region=region)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    scan.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                    scan.module_aliases.setdefault(alias.asname or alias.name, node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    scan.imports[alias.asname or alias.name] = alias.name
                    scan.module_aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ClassDef):
                bases: list[str] = []
                for base in node.bases:
                    name, _ = _call_target(base) if isinstance(base, ast.Call) else ("", None)
                    if isinstance(base, ast.Name):
                        name = base.id
                    elif isinstance(base, ast.Attribute):
                        name = base.attr
                    if name:
                        bases.append(name)
                scan.bases[node.name] = tuple(bases)
        self._collect(tree, scan, prefix="")
        return scan

    def _collect(self, node: ast.AST, scan: _ModuleScan, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._collect(child, scan, prefix=f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._record_unit(child, scan, prefix)

    def _record_unit(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        scan: _ModuleScan,
        prefix: str,
    ) -> None:
        qualname = f"{prefix}{node.name}"
        uid = stable_id(scan.module, qualname)
        visitor = _FunctionVisitor()
        for stmt in node.body:
            visitor.visit(stmt)
        suppressive = sum(1 for r in visitor.returns if _is_suppressive_return(r)) + visitor.raises
        productive = sum(1 for r in visitor.returns if not _is_suppressive_return(r))
        unit = Unit(
            uid=uid,
            name=f"{scan.module}:{qualname}",
            neuropil=scan.module,
            region=scan.region,
            line=node.lineno,
            size=max(1, (node.end_lineno or node.lineno) - node.lineno + 1),
            guards=visitor.guards,
            exits_suppressive=suppressive,
            exits_productive=productive,
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )
        unit.cell_class = _classify(unit, visitor)
        scan.units.append(unit)
        scope = prefix[:-1] if prefix.endswith(".") else prefix
        unit.attrs["scope"] = scope
        for name, qualifier, lineno, col, used, in_guard in visitor.calls:
            scan.calls.append(
                _RawCall(
                    caller=uid,
                    target_name=name,
                    qualifier=qualifier,
                    locus=f"{scan.module}:{lineno}:{col}",
                    value_used=used,
                    in_guard=in_guard,
                    scope=scope,
                    local_types=visitor.local_types,
                )
            )

    # -- pass two -------------------------------------------------------

    def build(self) -> ConnectomeSnapshot:
        if not self.scans:
            self.scan()
        units: dict[str, Unit] = {}
        neuropils: dict[str, Neuropil] = {}
        by_leaf: dict[str, list[str]] = {}
        by_module_qual: dict[str, dict[str, str]] = {}
        by_class: dict[tuple[str, str], dict[str, str]] = {}
        class_home: dict[str, list[str]] = {}
        bases: dict[tuple[str, str], tuple[str, ...]] = {}

        for scan in self.scans:
            neuropils[scan.module] = Neuropil(name=scan.module, parent=scan.region)
            qual_index = by_module_qual.setdefault(scan.module, {})
            for class_name, base_names in scan.bases.items():
                bases[(scan.module, class_name)] = base_names
                class_home.setdefault(class_name, []).append(scan.module)
            for unit in scan.units:
                units[unit.uid] = unit
                qualname = unit.name.rsplit(":", 1)[1]
                leaf = qualname.rsplit(".", 1)[-1]
                by_leaf.setdefault(leaf, []).append(unit.uid)
                qual_index[qualname] = unit.uid
                scope = str(unit.attrs.get("scope") or "")
                if scope:
                    by_class.setdefault((scan.module, scope), {})[leaf] = unit.uid
                else:
                    qual_index.setdefault(leaf, unit.uid)

        index = _ResolutionIndex(
            units=units,
            by_leaf=by_leaf,
            by_module_qual=by_module_qual,
            by_class=by_class,
            class_home=class_home,
            bases=bases,
        )

        contacts: list[ContactSite] = []
        for scan in self.scans:
            for call in scan.calls:
                target = self._resolve(call, scan, index)
                if target is None:
                    continue
                contacts.append(
                    ContactSite(
                        pre=call.caller,
                        post=target,
                        locus=call.locus,
                        compartment=Compartment.SOMA,
                        sign=1,
                        kind=EdgeKind.DRIVE,
                    )
                )
                if call.value_used:
                    callee = units[target]
                    sign = -1 if callee.cell_class is CellClass.INHIBITORY else 1
                    compartment = (
                        Compartment.AXON_INITIAL_SEGMENT if call.in_guard else Compartment.DENDRITE
                    )
                    contacts.append(
                        ContactSite(
                            pre=target,
                            post=call.caller,
                            locus=f"{call.locus}:r",
                            compartment=compartment,
                            sign=sign,
                            kind=EdgeKind.RETURN,
                        )
                    )

        connections = _aggregate(contacts)
        snapshot = ConnectomeSnapshot(
            version=1,
            units=units,
            connections=connections,
            neuropils=neuropils,
            built_at=time.time(),
            source=str(self.repo),
        )
        total_sites = (
            self.resolved_calls
            + self.ambiguous_calls
            + self.unresolved_calls
            + self.out_of_volume_calls
        )
        in_volume = max(1, total_sites - self.out_of_volume_calls)
        snapshot.attrs.update(
            {
                "files_read": self.files_read,
                "bytes_read": self.bytes_read,
                "parse_failures": len(self.parse_failures),
                "call_sites": total_sites,
                "calls_resolved": self.resolved_calls,
                "calls_ambiguous": self.ambiguous_calls,
                "calls_unresolved": self.unresolved_calls,
                "calls_out_of_volume": self.out_of_volume_calls,
                "in_volume_coverage": round(self.resolved_calls / in_volume, 4),
                "roots": list(self.config.roots),
            }
        )
        return snapshot

    def _resolve(
        self,
        call: _RawCall,
        scan: _ModuleScan,
        index: _ResolutionIndex,
    ) -> str | None:
        """Attach one call site to a cell, or say honestly that it could not be.

        A site that leaves the volume — a builtin, the standard library, a
        dependency — is not a failure of the reconstruction any more than an
        axon leaving the imaged block is. It is counted separately so coverage
        means what it says.
        """
        name = call.target_name
        qualifier = call.qualifier
        module = scan.module

        if qualifier in (None, "self", "cls"):
            if call.scope:
                hit = index.method(module, call.scope, name)
                if hit:
                    return self._hit(hit)
            if qualifier is None:
                qual = index.by_module_qual.get(module, {})
                hit = qual.get(name)
                if hit:
                    return self._hit(hit)
                imported = scan.imports.get(name)
                if imported:
                    home, _, leaf = imported.rpartition(".")
                    hit = index.by_module_qual.get(home, {}).get(leaf)
                    if hit:
                        return self._hit(hit)
                    if _outside_volume(home):
                        self.out_of_volume_calls += 1
                        return None
                if name in _BUILTIN_NAMES:
                    self.out_of_volume_calls += 1
                    return None
        else:
            alias = scan.module_aliases.get(qualifier)
            imported = scan.imports.get(qualifier)
            if imported and imported != alias:
                home, _, leaf = imported.rpartition(".")
                hit = index.method(home, leaf, name)
                if hit:
                    return self._hit(hit)
                hit = index.by_module_qual.get(imported, {}).get(name)
                if hit:
                    return self._hit(hit)
            if alias:
                hit = index.by_module_qual.get(alias, {}).get(name)
                if hit:
                    return self._hit(hit)
                if _outside_volume(alias):
                    self.out_of_volume_calls += 1
                    return None
            local_type = call.local_types.get(qualifier)
            if local_type:
                hit = index.method(module, local_type, name)
                if hit:
                    return self._hit(hit)
                homes = index.class_home.get(local_type, ())
                if len(homes) == 1:
                    hit = index.method(homes[0], local_type, name)
                    if hit:
                        return self._hit(hit)

        candidates = index.by_leaf.get(name, [])
        if not candidates:
            if name in _BUILTIN_NAMES:
                self.out_of_volume_calls += 1
            else:
                self.unresolved_calls += 1
            return None
        same_module = [uid for uid in candidates if index.units[uid].neuropil == module]
        if len(same_module) == 1:
            return self._hit(same_module[0])
        if len(candidates) <= self.config.ambiguity_ceiling:
            return self._hit(candidates[0])
        self.ambiguous_calls += 1
        self.ambiguous_sites.append(
            AmbiguousSite(
                caller=call.caller,
                caller_module=module,
                caller_region=scan.region,
                target_name=name,
                qualifier=qualifier,
                locus=call.locus,
                value_used=call.value_used,
                in_guard=call.in_guard,
                candidates=tuple(candidates),
                imported=tuple(sorted(set(scan.module_aliases.values()) | set(scan.imports.values()))),
            )
        )
        return None

    def _hit(self, uid: str) -> str:
        self.resolved_calls += 1
        return uid


@dataclass(frozen=True)
class _ResolutionIndex:
    """Everything pass two needs to attach a call site to a cell."""

    units: dict[str, Unit]
    by_leaf: dict[str, list[str]]
    by_module_qual: dict[str, dict[str, str]]
    by_class: dict[tuple[str, str], dict[str, str]]
    class_home: dict[str, list[str]]
    bases: dict[tuple[str, str], tuple[str, ...]]

    def method(self, module: str, class_name: str, name: str, depth: int = 0) -> str | None:
        """Look a method up on a class, then on its bases.

        Depth is bounded because a base list read out of source can contain a
        cycle that the interpreter would never build, and a reconstruction that
        follows one hangs instead of reporting.
        """
        hit = self.by_class.get((module, class_name), {}).get(name)
        if hit:
            return hit
        if depth >= 4:
            return None
        for base in self.bases.get((module, class_name), ()):
            hit = self.method(module, base, name, depth + 1)
            if hit:
                return hit
            homes = self.class_home.get(base, ())
            if len(homes) == 1 and homes[0] != module:
                hit = self.method(homes[0], base, name, depth + 1)
                if hit:
                    return hit
        return None


def _classify(unit: Unit, visitor: _FunctionVisitor) -> CellClass:
    """Assign a class from measured behaviour, never from the name.

    Order matters. Maintenance is checked first because a logger that returns
    nothing would otherwise read as a gate, and a gate is checked before the
    default because most cells produce something and the interesting minority
    is the one that refuses.
    """
    productive = unit.exits_productive
    if (
        visitor.total_calls
        and visitor.maintenance_calls / visitor.total_calls >= 0.6
        and productive == 0
    ):
        return CellClass.GLIAL
    if productive == 0 and visitor.mutates_state:
        return CellClass.MODULATORY
    if _returns_only_boolean(visitor.returns) and productive > 0:
        return CellClass.INHIBITORY
    if unit.exit_count and unit.suppression >= 0.5:
        return CellClass.INHIBITORY
    return CellClass.EXCITATORY


def _aggregate(contacts: Iterable[ContactSite]) -> dict[tuple[str, str, str], Connection]:
    """Collapse contact sites into pairs, keeping drive and return apart."""
    counts: dict[tuple[str, str, str], list[Any]] = {}
    for contact in contacts:
        key = (contact.pre, contact.post, str(contact.kind))
        entry = counts.get(key)
        if entry is None:
            counts[key] = [1, contact.sign, {contact.compartment}]
        else:
            entry[0] += 1
            entry[2].add(contact.compartment)
            if contact.sign < 0:
                entry[1] = -1
    return {
        key: Connection(
            pre=key[0],
            post=key[1],
            contacts=int(value[0]),
            sign=int(value[1]),
            compartments=tuple(sorted(value[2], key=str)),
            kind=EdgeKind(key[2]),
        )
        for key, value in counts.items()
    }


def reconstruct(
    repo: str | Path | None = None,
    config: ReconstructionConfig | None = None,
) -> ConnectomeSnapshot:
    """Build the connectome once. Offline, deterministic, no imports executed."""
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    reconstructor = VolumeReconstructor(root, config)
    reconstructor.scan()
    return reconstructor.build()
