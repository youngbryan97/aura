"""Deterministic, evidence-bearing architecture quality scoring.

The scorer analyzes repository-local Python dependencies without importing the
code. Reports distinguish unconditional, optional, type-only, deferred, and
dynamic dependencies; retain complete debt populations; and are immutable once
constructed so a gate evaluates exactly the snapshot that was scored.
"""

from __future__ import annotations

import ast
import hashlib
import tokenize
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock

from .dependency_analysis import ImportInventory, extract_import_inventory
from .models import (
    ArchitectureQualityFinding,
    ArchitectureQualityMetrics,
    ArchitectureQualityReport,
    ModuleStructure,
    normalize_repository_path,
)

DEFAULT_INCLUDE_ROOTS = ("core", "interface", "infrastructure", "slo", "tools")
DEFAULT_EXCLUDE_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "venv",
    ".venv",
}

@dataclass(frozen=True)
class _ParsedModule:
    """Content-addressed syntax evidence shared by repeated exact scans."""

    structure: ModuleStructure
    inventory: ImportInventory
    exports: frozenset[str]
    has_dynamic_exports: bool


_ANALYSIS_CACHE_MAX_ENTRIES = 8192
_ANALYSIS_CACHE: OrderedDict[tuple[str, str, str], _ParsedModule] = OrderedDict()
_ANALYSIS_CACHE_LOCK = RLock()


def score_codebase(
    root: str | Path,
    *,
    include_roots: Iterable[str] = DEFAULT_INCLUDE_ROOTS,
    exclude_parts: Iterable[str] = DEFAULT_EXCLUDE_PARTS,
    overlay_content: Mapping[str, str | None] | None = None,
    god_file_threshold: int = 1500,
    max_cycles_reported: int | None = None,
) -> ArchitectureQualityReport:
    """Score a Python source tree without importing it.

    Overlay keys must be canonical repository-relative paths. A string replaces
    or adds a module; ``None`` is an explicit deletion tombstone. The historical
    ``max_cycles_reported`` argument is accepted for compatibility but no longer
    truncates evidence.
    """

    if god_file_threshold < 1:
        raise ValueError("god_file_threshold must be positive")
    if max_cycles_reported is not None and max_cycles_reported < 1:
        raise ValueError("max_cycles_reported must be positive when provided")

    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError(f"architecture root is not a directory: {root_path}")
    include_roots_tuple = _validated_include_roots(include_roots)
    exclude_set = {_validate_exclude_part(part) for part in exclude_parts}
    overlays = _validated_overlays(overlay_content or {})

    files = _collect_python_files(root_path, include_roots_tuple, exclude_set)
    for rel_path, content in overlays.items():
        if not _is_included(rel_path, include_roots_tuple):
            raise ValueError(f"overlay path is outside include roots: {rel_path}")
        if content is None:
            if rel_path not in files:
                raise ValueError(f"deletion overlay does not name an existing module: {rel_path}")
            del files[rel_path]
        else:
            candidate_parent = (root_path / rel_path).parent.resolve(strict=False)
            if not candidate_parent.is_relative_to(root_path):
                raise ValueError(f"overlay path escapes repository through a symlink: {rel_path}")
            files[rel_path] = root_path / rel_path

    module_to_path, path_to_module, ownership_conflicts = _build_module_maps(files)
    known_modules = set(module_to_path)
    line_counts: dict[str, int] = {}
    structures: dict[str, ModuleStructure] = {}
    imports: dict[str, ImportInventory] = {}
    module_exports: dict[str, set[str]] = {}
    dynamic_export_modules: set[str] = set()
    findings: list[ArchitectureQualityFinding] = []
    for module, paths in ownership_conflicts:
        findings.append(
            ArchitectureQualityFinding(
                severity="high",
                code="ambiguous_module_owner",
                modules=(module,),
                message=f"{module} has competing file and package owners: {', '.join(paths)}",
            )
        )

    for rel_path in sorted(files):
        module = path_to_module[rel_path]
        content = overlays.get(rel_path)
        if content is None:
            try:
                content = _read_source(files[rel_path])
            except (SyntaxError, UnicodeDecodeError) as exc:
                findings.append(
                    ArchitectureQualityFinding(
                        severity="critical",
                        code="source_decode_error",
                        path=rel_path,
                        message=f"{rel_path} cannot be decoded according to its Python encoding: {exc}",
                    )
                )
                content = files[rel_path].read_bytes().decode("utf-8", errors="replace")
        line_counts[rel_path] = _line_count(content)
        try:
            parsed = _parse_module_cached(
                content,
                module=module,
                rel_path=rel_path,
                current_is_package=(
                    rel_path.endswith("/__init__.py") or rel_path == "__init__.py"
                ),
            )
        except SyntaxError as exc:
            findings.append(
                ArchitectureQualityFinding(
                    severity="critical",
                    code="syntax_error",
                    path=rel_path,
                    message=f"{rel_path} does not parse: {exc.msg}",
                    value=exc.lineno,
                )
            )
            imports[module] = ImportInventory(
                runtime=frozenset(),
                type_only=frozenset(),
                optional=frozenset(),
                conditional=frozenset(),
                deferred=frozenset(),
                dynamic=frozenset(),
                required_runtime_modules=frozenset(),
                runtime_from_aliases=frozenset(),
                unresolved_dynamic=0,
                invalid_relative=0,
            )
            module_exports[module] = set()
            structures[rel_path] = _structure_without_ast(content)
            continue
        structures[rel_path] = parsed.structure
        module_exports[module] = set(parsed.exports)
        if parsed.has_dynamic_exports:
            dynamic_export_modules.add(module)
        imports[module] = parsed.inventory

    runtime_raw = {module: inventory.runtime for module, inventory in imports.items()}
    graph = _resolve_graph(runtime_raw, known_modules)
    type_graph = _resolve_graph({m: i.type_only for m, i in imports.items()}, known_modules)
    optional_graph = _resolve_graph({m: i.optional for m, i in imports.items()}, known_modules)
    conditional_graph = _resolve_graph({m: i.conditional for m, i in imports.items()}, known_modules)
    deferred_graph = _resolve_graph({m: i.deferred for m, i in imports.items()}, known_modules)
    dynamic_graph = _resolve_graph({m: i.dynamic for m, i in imports.items()}, known_modules)
    unresolved_local = _unresolved_local_imports(
        imports,
        known_modules=known_modules,
        known_namespaces=_known_namespace_packages(files),
        include_roots=include_roots_tuple,
        module_exports=module_exports,
        dynamic_export_modules=dynamic_export_modules,
    )
    return _finalize_report(
        root=str(root_path),
        include_roots=include_roots_tuple,
        exclude_parts=tuple(sorted(exclude_set)),
        god_file_threshold=god_file_threshold,
        line_counts=line_counts,
        module_to_path=module_to_path,
        graph=graph,
        module_structures=structures,
        type_checking_graph=type_graph,
        optional_graph=optional_graph,
        conditional_graph=conditional_graph,
        deferred_graph=deferred_graph,
        dynamic_graph=dynamic_graph,
        module_exports=module_exports,
        dynamic_export_modules=dynamic_export_modules,
        import_inventories=imports,
        unresolved_local=unresolved_local,
        retained_findings=findings,
    )


def _finalize_report(
    *,
    root: str,
    include_roots: tuple[str, ...],
    exclude_parts: tuple[str, ...],
    god_file_threshold: int,
    line_counts: Mapping[str, int],
    module_to_path: Mapping[str, str],
    graph: Mapping[str, Iterable[str]],
    module_structures: Mapping[str, ModuleStructure],
    type_checking_graph: Mapping[str, Iterable[str]],
    optional_graph: Mapping[str, Iterable[str]],
    conditional_graph: Mapping[str, Iterable[str]],
    deferred_graph: Mapping[str, Iterable[str]],
    dynamic_graph: Mapping[str, Iterable[str]],
    module_exports: Mapping[str, Iterable[str]],
    dynamic_export_modules: Iterable[str],
    import_inventories: Mapping[str, ImportInventory],
    unresolved_local: Mapping[str, tuple[str, ...]],
    retained_findings: Iterable[ArchitectureQualityFinding],
) -> ArchitectureQualityReport:
    graph_sets = {key: set(value) for key, value in graph.items()}
    type_sets = {key: set(value) for key, value in type_checking_graph.items()}
    optional_sets = {key: set(value) for key, value in optional_graph.items()}
    conditional_sets = {key: set(value) for key, value in conditional_graph.items()}
    deferred_sets = {key: set(value) for key, value in deferred_graph.items()}
    dynamic_sets = {key: set(value) for key, value in dynamic_graph.items()}
    executable_graph = _union_graphs(
        graph_sets,
        optional_sets,
        conditional_sets,
        deferred_sets,
    )
    reverse_graph = _reverse_graph(graph_sets)
    cycles = tuple(
        sorted(_strongly_connected_components(graph_sets), key=lambda item: (-len(item), item))
    )
    executable_cycles = tuple(
        sorted(
            _strongly_connected_components(executable_graph),
            key=lambda item: (-len(item), item),
        )
    )
    god_files = _structurally_oversized_modules(module_structures, god_file_threshold)
    findings = list(retained_findings)
    findings.extend(
        _build_findings(
            graph_sets,
            cycles,
            executable_cycles,
            god_files,
            module_structures,
            import_inventories,
            module_to_path,
            god_file_threshold,
            unresolved_local,
        )
    )
    out_degrees = [len(targets) for targets in graph_sets.values()]
    in_degrees = [len(sources) for sources in reverse_graph.values()]
    parse_errors = sum(
        finding.code in {"syntax_error", "source_decode_error"} for finding in findings
    )
    preliminary = ArchitectureQualityMetrics(
        module_count=len(module_to_path),
        dependency_edges=_edge_count(graph_sets),
        cycle_count=len(cycles),
        largest_cycle_size=max((len(cycle) for cycle in cycles), default=0),
        god_file_count=len(god_files),
        max_file_lines=max(line_counts.values(), default=0),
        max_out_degree=max(out_degrees, default=0),
        max_in_degree=max(in_degrees, default=0),
        dependency_concentration_pct=_dependency_concentration(out_degrees),
        parse_error_count=parse_errors,
        type_only_dependency_edges=_edge_count(type_sets),
        optional_dependency_edges=_edge_count(optional_sets),
        conditional_dependency_edges=_edge_count(conditional_sets),
        deferred_dependency_edges=_edge_count(deferred_sets),
        dynamic_dependency_edges=_edge_count(dynamic_sets),
        unresolved_dynamic_imports=sum(
            inventory.unresolved_dynamic for inventory in import_inventories.values()
        ),
        unresolved_local_imports=sum(len(targets) for targets in unresolved_local.values()),
        invalid_relative_imports=sum(
            inventory.invalid_relative for inventory in import_inventories.values()
        ),
        executable_dependency_edges=_edge_count(executable_graph),
        executable_cycle_count=len(executable_cycles),
        largest_executable_cycle_size=max(
            (len(cycle) for cycle in executable_cycles),
            default=0,
        ),
        max_code_lines=max((item.code_lines for item in module_structures.values()), default=0),
        max_complexity=max((item.complexity for item in module_structures.values()), default=0),
        max_symbol_count=max((item.symbol_count for item in module_structures.values()), default=0),
        cyclic_module_count=(
            len(set().union(*(set(cycle) for cycle in cycles))) if cycles else 0
        ),
        executable_cyclic_module_count=(
            len(set().union(*(set(cycle) for cycle in executable_cycles)))
            if executable_cycles
            else 0
        ),
    )
    debt = _architecture_debt(preliminary, god_file_threshold=god_file_threshold)
    metrics = ArchitectureQualityMetrics(**{**preliminary.__dict__, "architecture_debt": debt})
    score = 0.0 if parse_errors else 100.0 / (1.0 + debt / 100.0)
    return ArchitectureQualityReport(
        root=root,
        include_roots=include_roots,
        exclude_parts=exclude_parts,
        god_file_threshold=god_file_threshold,
        metrics=metrics,
        score=score,
        line_counts=line_counts,
        module_to_path=module_to_path,
        graph=_sorted_graph(graph_sets),
        reverse_graph=_sorted_graph(reverse_graph),
        cycles=cycles,
        findings=tuple(findings),
        module_structures=module_structures,
        type_checking_graph=_sorted_graph(type_sets),
        optional_graph=_sorted_graph(optional_sets),
        conditional_graph=_sorted_graph(conditional_sets),
        deferred_graph=_sorted_graph(deferred_sets),
        dynamic_graph=_sorted_graph(dynamic_sets),
        executable_graph=_sorted_graph(executable_graph),
        executable_cycles=executable_cycles,
        module_exports={key: tuple(sorted(value)) for key, value in module_exports.items()},
        dynamic_export_modules=tuple(sorted(dynamic_export_modules)),
    )


def _validated_include_roots(include_roots: Iterable[str]) -> tuple[str, ...]:
    roots: list[str] = []
    seen: set[str] = set()
    for item in include_roots:
        root = normalize_repository_path(item, label="include root")
        if root.endswith(".py"):
            raise ValueError(f"include root must be a directory: {root}")
        if root not in seen:
            seen.add(root)
            roots.append(root)
    if not roots:
        raise ValueError("at least one include root is required")
    return tuple(roots)


def _validate_exclude_part(part: str | Path) -> str:
    value = str(part)
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"exclude part must be one path component: {value!r}")
    return value


def _validated_overlays(overlays: Mapping[str, str | None]) -> dict[str, str | None]:
    validated: dict[str, str | None] = {}
    raw_for_path: dict[str, str] = {}
    for raw_path, content in overlays.items():
        rel_path = normalize_repository_path(raw_path, label="overlay path")
        if not rel_path.endswith(".py"):
            raise ValueError(f"overlay path must name a Python module: {rel_path}")
        if content is not None and not isinstance(content, str):
            raise TypeError(f"overlay content for {rel_path} must be text or None")
        if rel_path in validated:
            raise ValueError(
                f"overlay paths alias each other: {raw_for_path[rel_path]!r} and {raw_path!r}"
            )
        validated[rel_path] = content
        raw_for_path[rel_path] = str(raw_path)
    return validated


def _collect_python_files(
    root: Path,
    include_roots: tuple[str, ...],
    exclude_parts: set[str],
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for include_root in include_roots:
        start = root / include_root
        resolved_start = start.resolve(strict=False)
        if not resolved_start.is_relative_to(root):
            raise ValueError(f"include root escapes repository through a symlink: {include_root}")
        if not start.exists():
            continue
        if not start.is_dir():
            raise ValueError(f"include root is not a directory: {include_root}")
        for path in start.rglob("*.py"):
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError(f"source path escapes repository through a symlink: {path}")
            rel_path = path.relative_to(root).as_posix()
            if any(part in exclude_parts for part in PurePosixPath(rel_path).parts):
                continue
            files[rel_path] = path
    return files


def _read_source(path: Path) -> str:
    with tokenize.open(path) as handle:
        return handle.read()


def _parse_module_cached(
    content: str,
    *,
    module: str,
    rel_path: str,
    current_is_package: bool,
) -> _ParsedModule:
    """Parse one module once per exact content identity.

    The cache only retains syntax-derived immutable evidence. Every scoring pass
    still recollects paths, resolves the complete dependency graph, recomputes
    findings, and attests the resulting report. A content digest prevents mtime
    aliasing and makes overlay replacement exact.
    """

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    key = (module, rel_path, digest)
    with _ANALYSIS_CACHE_LOCK:
        cached = _ANALYSIS_CACHE.get(key)
        if cached is not None:
            _ANALYSIS_CACHE.move_to_end(key)
            return cached

    tree = ast.parse(content, filename=rel_path)
    exports, has_dynamic_exports = _module_exports(tree)
    parsed = _ParsedModule(
        structure=_measure_structure(content, tree),
        inventory=extract_import_inventory(
            tree,
            _module_name_from_path(rel_path),
            current_is_package=current_is_package,
        ),
        exports=frozenset(exports),
        has_dynamic_exports=has_dynamic_exports,
    )
    with _ANALYSIS_CACHE_LOCK:
        existing = _ANALYSIS_CACHE.get(key)
        if existing is not None:
            _ANALYSIS_CACHE.move_to_end(key)
            return existing
        _ANALYSIS_CACHE[key] = parsed
        while len(_ANALYSIS_CACHE) > _ANALYSIS_CACHE_MAX_ENTRIES:
            _ANALYSIS_CACHE.popitem(last=False)
    return parsed


def _clear_analysis_cache() -> None:
    """Clear process-local parse evidence for deterministic test isolation."""

    with _ANALYSIS_CACHE_LOCK:
        _ANALYSIS_CACHE.clear()


def _resolve_graph(
    raw_by_module: Mapping[str, Iterable[str]],
    known_modules: set[str],
) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {module: set() for module in known_modules}
    for module, imports in raw_by_module.items():
        for raw_import in imports:
            target = _resolve_known_module(raw_import, known_modules)
            if target and target != module:
                graph[module].add(target)
    return graph


def _resolve_known_module(raw_import: str, known_modules: set[str]) -> str | None:
    parts = raw_import.split(".")
    for length in range(len(parts), 0, -1):
        candidate = ".".join(parts[:length])
        if candidate in known_modules:
            return candidate
    return None


def _unresolved_local_imports(
    imports: Mapping[str, ImportInventory],
    *,
    known_modules: set[str],
    known_namespaces: set[str],
    include_roots: tuple[str, ...],
    module_exports: Mapping[str, set[str]],
    dynamic_export_modules: set[str],
) -> dict[str, tuple[str, ...]]:
    local_prefixes = tuple(".".join(root.split("/")) for root in include_roots)
    unresolved: dict[str, tuple[str, ...]] = {}
    for module, inventory in imports.items():
        missing_modules = {
            target
            for target in inventory.required_runtime_modules
            if any(target == prefix or target.startswith(f"{prefix}.") for prefix in local_prefixes)
            and target not in known_modules
            and target not in known_namespaces
        }
        missing_aliases: set[str] = set()
        for base, alias in inventory.runtime_from_aliases:
            if not any(base == prefix or base.startswith(f"{prefix}.") for prefix in local_prefixes):
                continue
            candidate = f"{base}.{alias}"
            if candidate in known_modules or candidate in known_namespaces:
                continue
            if alias in module_exports.get(base, set()) or base in dynamic_export_modules:
                continue
            missing_aliases.add(candidate)
        missing = sorted(missing_modules | missing_aliases)
        if missing:
            unresolved[module] = tuple(missing)
    return unresolved


def _module_exports(tree: ast.Module) -> tuple[set[str], bool]:
    exports: set[str] = set()
    dynamic = False
    pending = list(reversed(tree.body))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.add(node.name)
            dynamic = dynamic or node.name == "__getattr__"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                exports.update(_bound_names(target))
            value = node.value
            if any(name == "__all__" for target in targets for name in _bound_names(target)):
                if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    exports.update(
                        item.value
                        for item in value.elts
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    )
        elif isinstance(node, ast.Import):
            exports.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            exports.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
            # A `with` has a body and no `else`: reading its orelse raised AttributeError, and her architecture gate,
            # which every fix she applies passes through, failed on any module with one at its top level.
            pending.extend(reversed((*node.body, *getattr(node, "orelse", ()))))
        elif isinstance(node, ast.Try):
            handler_bodies = tuple(item for handler in node.handlers for item in handler.body)
            pending.extend(reversed((*node.body, *handler_bodies, *node.orelse, *node.finalbody)))
        elif isinstance(node, ast.Match):
            pending.extend(reversed(tuple(item for case in node.cases for item in case.body)))
    return exports, dynamic


def _bound_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in node.elts:
            names.update(_bound_names(item))
        return names
    return set()


def _known_namespace_packages(files: Mapping[str, Path]) -> set[str]:
    namespaces: set[str] = set()
    for rel_path in files:
        parts = PurePosixPath(rel_path).parts[:-1]
        for length in range(1, len(parts) + 1):
            namespaces.add(".".join(parts[:length]))
    return namespaces


def _reverse_graph(graph: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = {module: set() for module in graph}
    for source, targets in graph.items():
        for target in targets:
            reverse.setdefault(target, set()).add(source)
    return reverse


def _union_graphs(*graphs: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    nodes = set().union(*(graph.keys() for graph in graphs))
    combined = {node: set() for node in nodes}
    for graph in graphs:
        for node, targets in graph.items():
            combined[node].update(targets)
    return combined


def _strongly_connected_components(
    graph: Mapping[str, Iterable[str]],
) -> tuple[tuple[str, ...], ...]:
    """Iterative Kosaraju decomposition; safe for arbitrarily deep graphs."""

    adjacency = {node: tuple(sorted(graph.get(node, ()))) for node in graph}
    reverse = _reverse_graph(adjacency)
    visited: set[str] = set()
    finish_order: list[str] = []

    for root in sorted(adjacency):
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            node, index = stack[-1]
            targets = adjacency[node]
            if index < len(targets):
                target = targets[index]
                stack[-1] = (node, index + 1)
                if target not in visited:
                    visited.add(target)
                    stack.append((target, 0))
                continue
            finish_order.append(node)
            stack.pop()

    assigned: set[str] = set()
    components: list[tuple[str, ...]] = []
    for root in reversed(finish_order):
        if root in assigned:
            continue
        assigned.add(root)
        component: list[str] = []
        stack = [root]
        while stack:
            node = stack.pop()
            component.append(node)
            for target in sorted(reverse.get(node, ()), reverse=True):
                if target not in assigned:
                    assigned.add(target)
                    stack.append(target)
        if len(component) > 1:
            components.append(tuple(sorted(component)))
    return tuple(components)


def _measure_structure(content: str, tree: ast.AST) -> ModuleStructure:
    code_lines, comment_lines = _token_line_classes(content)
    statements = sum(isinstance(node, ast.stmt) for node in ast.walk(tree))
    symbols = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    branch_points = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.comprehension)):
            branch_points += 1
        elif isinstance(node, ast.Try):
            branch_points += max(1, len(node.handlers))
        elif isinstance(node, ast.BoolOp):
            branch_points += max(0, len(node.values) - 1)
        elif isinstance(node, ast.Match):
            branch_points += len(node.cases)
    return ModuleStructure(
        source_lines=_line_count(content),
        code_lines=len(code_lines),
        comment_lines=len(comment_lines),
        statement_count=statements,
        symbol_count=symbols,
        branch_points=branch_points,
        max_nesting=_max_control_nesting(tree),
    )


def _structure_without_ast(content: str) -> ModuleStructure:
    code_lines, comment_lines = _token_line_classes(content)
    return ModuleStructure(
        source_lines=_line_count(content),
        code_lines=len(code_lines),
        comment_lines=len(comment_lines),
        statement_count=0,
        symbol_count=0,
        branch_points=0,
        max_nesting=0,
    )


def _token_line_classes(content: str) -> tuple[set[int], set[int]]:
    code_lines: set[int] = set()
    comment_lines: set[int] = set()
    for index, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment_lines.add(index)
        else:
            code_lines.add(index)
    return code_lines, comment_lines


def _max_control_nesting(tree: ast.AST) -> int:
    control_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
    maximum = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        next_depth = depth + 1 if isinstance(node, control_nodes) else depth
        maximum = max(maximum, next_depth)
        stack.extend((child, next_depth) for child in ast.iter_child_nodes(node))
    return maximum


def _structurally_oversized_modules(
    structures: Mapping[str, ModuleStructure],
    god_file_threshold: int,
) -> dict[str, ModuleStructure]:
    code_threshold = max(100, int(god_file_threshold * 0.65))
    return {
        path: item
        for path, item in structures.items()
        if item.source_lines > god_file_threshold
        or item.code_lines > code_threshold
        or item.complexity > 300
        or item.symbol_count > 120
    }


def _build_findings(
    graph: Mapping[str, Iterable[str]],
    cycles: tuple[tuple[str, ...], ...],
    executable_cycles: tuple[tuple[str, ...], ...],
    god_files: Mapping[str, ModuleStructure],
    structures: Mapping[str, ModuleStructure],
    imports: Mapping[str, ImportInventory],
    module_to_path: Mapping[str, str],
    god_file_threshold: int,
    unresolved_local: Mapping[str, tuple[str, ...]],
) -> list[ArchitectureQualityFinding]:
    findings: list[ArchitectureQualityFinding] = []
    for cycle in cycles:
        findings.append(
            ArchitectureQualityFinding(
                severity="high",
                code="import_cycle",
                modules=cycle,
                message=f"Import cycle across {len(cycle)} modules",
                value=len(cycle),
            )
        )
    runtime_cycles = {tuple(sorted(cycle)) for cycle in cycles}
    for cycle in executable_cycles:
        if tuple(sorted(cycle)) in runtime_cycles:
            continue
        findings.append(
            ArchitectureQualityFinding(
                severity="high",
                code="executable_import_cycle",
                modules=cycle,
                message=(
                    f"Potential executable cycle across {len(cycle)} modules "
                    "through deferred, conditional, or optional dependencies"
                ),
                value=len(cycle),
            )
        )
    for path, item in sorted(
        god_files.items(), key=lambda pair: (-pair[1].source_lines, pair[0])
    ):
        findings.append(
            ArchitectureQualityFinding(
                severity="medium",
                code="structurally_oversized_module",
                path=path,
                message=(
                    f"{path} has {item.source_lines} source lines, {item.code_lines} code lines, "
                    f"complexity {item.complexity}, and {item.symbol_count} symbols "
                    f"(source threshold {god_file_threshold})"
                ),
                value=item.source_lines,
            )
        )
    for module, targets in sorted(graph.items(), key=lambda item: (-len(tuple(item[1])), item[0])):
        degree = len(tuple(targets))
        if degree >= 40:
            findings.append(
                ArchitectureQualityFinding(
                    severity="medium",
                    code="dependency_fanout",
                    path=module_to_path.get(module),
                    modules=(module,),
                    message=f"{module} imports {degree} unconditional local modules",
                    value=degree,
                )
            )
    for module, inventory in sorted(imports.items()):
        if inventory.unresolved_dynamic:
            findings.append(
                ArchitectureQualityFinding(
                    severity="medium",
                    code="unresolved_dynamic_import",
                    path=module_to_path.get(module),
                    modules=(module,),
                    message=(
                        f"{module} contains {inventory.unresolved_dynamic} dynamic import call(s) "
                        "whose target is not a literal and cannot be included in the graph"
                    ),
                    value=inventory.unresolved_dynamic,
                )
            )
        if inventory.invalid_relative:
            findings.append(
                ArchitectureQualityFinding(
                    severity="high",
                    code="invalid_relative_import",
                    path=module_to_path.get(module),
                    modules=(module,),
                    message=(
                        f"{module} contains {inventory.invalid_relative} relative import(s) "
                        "that escape above the package root"
                    ),
                    value=inventory.invalid_relative,
                )
            )
    for module, targets in sorted(unresolved_local.items()):
        findings.append(
            ArchitectureQualityFinding(
                severity="high",
                code="unresolved_local_import",
                path=module_to_path.get(module),
                modules=(module, *targets),
                message=(
                    f"{module} unconditionally imports missing local module(s): "
                    + ", ".join(targets)
                ),
                value=len(targets),
            )
        )
    return findings


def _architecture_debt(
    metrics: ArchitectureQualityMetrics,
    *,
    god_file_threshold: int,
) -> float:
    """Return uncapped debt; every measured regression remains observable."""

    debt = 0.0
    debt += metrics.parse_error_count * 1000.0
    debt += metrics.cyclic_module_count * 0.1
    debt += metrics.largest_cycle_size / 100.0
    debt += metrics.god_file_count * 0.75
    debt += max(0, metrics.max_out_degree - 35) * 0.2
    debt += max(0, metrics.max_in_degree - 45) * 0.15
    debt += max(0.0, metrics.dependency_concentration_pct - 30.0) * 0.2
    debt += metrics.unresolved_dynamic_imports * 0.25
    debt += metrics.unresolved_local_imports * 5.0
    debt += metrics.invalid_relative_imports * 5.0
    debt += max(0, metrics.executable_dependency_edges - metrics.dependency_edges) * 0.01
    debt += metrics.executable_cyclic_module_count * 0.05
    debt += metrics.largest_executable_cycle_size / 200.0
    debt += max(0, metrics.max_code_lines - int(god_file_threshold * 0.65)) * 0.002
    debt += max(0, metrics.max_complexity - 300) * 0.02
    debt += max(0, metrics.max_symbol_count - 120) * 0.05
    return debt


def _dependency_concentration(out_degrees: list[int]) -> float:
    total = sum(out_degrees)
    if total <= 0:
        return 0.0
    top_n = max(1, len(out_degrees) // 20)
    return (sum(sorted(out_degrees, reverse=True)[:top_n]) / total) * 100.0


def _module_name_from_path(rel_path: str) -> str:
    module = rel_path.removesuffix(".py").replace("/", ".")
    return module.removesuffix(".__init__")


def _build_module_maps(
    paths: Mapping[str, Path],
) -> tuple[dict[str, str], dict[str, str], tuple[tuple[str, tuple[str, ...]], ...]]:
    module_to_path: dict[str, str] = {}
    path_to_module: dict[str, str] = {}
    conflicts: list[tuple[str, tuple[str, ...]]] = []
    for rel_path in sorted(paths):
        module = _module_name_from_path(rel_path)
        if module in module_to_path:
            existing_path = module_to_path[module]
            conflicts.append((module, tuple(sorted((existing_path, rel_path)))))
            if rel_path.endswith("/__init__.py"):
                file_module = f"{module}.__file__"
                module_to_path[file_module] = existing_path
                path_to_module[existing_path] = file_module
                module_to_path[module] = rel_path
                path_to_module[rel_path] = module
                continue
            module = f"{module}.__file__"
        module_to_path[module] = rel_path
        path_to_module[rel_path] = module
    return module_to_path, path_to_module, tuple(conflicts)


def _line_count(content: str) -> int:
    if not content:
        return 0
    return content.count("\n") + (0 if content.endswith("\n") else 1)


def _is_included(rel_path: str, include_roots: tuple[str, ...]) -> bool:
    return any(rel_path == root or rel_path.startswith(f"{root}/") for root in include_roots)


def _edge_count(graph: Mapping[str, Iterable[str]]) -> int:
    return sum(len(tuple(targets)) for targets in graph.values())


def _sorted_graph(graph: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    return {key: tuple(sorted(value)) for key, value in sorted(graph.items())}
