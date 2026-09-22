"""Immutable evidence models and scope identities for architecture scoring."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 2
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def normalize_repository_path(path: str | Path, *, label: str = "path") -> str:
    """Return a canonical repository-relative path or reject the input."""

    raw = str(path)
    if not raw or "\x00" in raw:
        raise ValueError(f"{label} must be a non-empty path")
    if "\\" in raw:
        raise ValueError(f"{label} must use canonical '/' separators: {raw}")
    if raw.startswith("/") or _WINDOWS_DRIVE.match(raw):
        raise ValueError(f"{label} must be repository-relative: {raw}")
    pure = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} must be canonical and cannot contain dot segments: {raw}")
    canonical = pure.as_posix()
    if canonical != raw or canonical in {"", "."}:
        raise ValueError(f"{label} must be canonical: {raw}")
    return canonical


@dataclass(frozen=True)
class ArchitectureQualityFinding:
    """A concrete architecture-quality finding."""

    severity: str
    code: str
    message: str
    path: str | None = None
    modules: tuple[str, ...] = ()
    value: int | float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "modules": list(self.modules),
            "value": self.value,
        }


@dataclass(frozen=True)
class ModuleStructure:
    """Syntax-derived module size and responsibility evidence."""

    source_lines: int
    code_lines: int
    comment_lines: int
    statement_count: int
    symbol_count: int
    branch_points: int
    max_nesting: int

    @property
    def complexity(self) -> int:
        return self.branch_points + self.max_nesting

    def to_dict(self) -> dict[str, int]:
        return {
            "source_lines": self.source_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "statement_count": self.statement_count,
            "symbol_count": self.symbol_count,
            "branch_points": self.branch_points,
            "max_nesting": self.max_nesting,
            "complexity": self.complexity,
        }


@dataclass(frozen=True)
class ArchitectureQualityMetrics:
    """Complete metrics used by CI and self-modification gates."""

    module_count: int
    dependency_edges: int
    cycle_count: int
    largest_cycle_size: int
    god_file_count: int
    max_file_lines: int
    max_out_degree: int
    max_in_degree: int
    dependency_concentration_pct: float
    parse_error_count: int = 0
    type_only_dependency_edges: int = 0
    optional_dependency_edges: int = 0
    conditional_dependency_edges: int = 0
    deferred_dependency_edges: int = 0
    dynamic_dependency_edges: int = 0
    unresolved_dynamic_imports: int = 0
    unresolved_local_imports: int = 0
    invalid_relative_imports: int = 0
    executable_dependency_edges: int = 0
    executable_cycle_count: int = 0
    largest_executable_cycle_size: int = 0
    max_code_lines: int = 0
    max_complexity: int = 0
    max_symbol_count: int = 0
    architecture_debt: float = 0.0
    cyclic_module_count: int = 0
    executable_cyclic_module_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_count": self.module_count,
            "dependency_edges": self.dependency_edges,
            "cycle_count": self.cycle_count,
            "largest_cycle_size": self.largest_cycle_size,
            "god_file_count": self.god_file_count,
            "max_file_lines": self.max_file_lines,
            "max_out_degree": self.max_out_degree,
            "max_in_degree": self.max_in_degree,
            "dependency_concentration_pct": round(self.dependency_concentration_pct, 4),
            "parse_error_count": self.parse_error_count,
            "type_only_dependency_edges": self.type_only_dependency_edges,
            "optional_dependency_edges": self.optional_dependency_edges,
            "conditional_dependency_edges": self.conditional_dependency_edges,
            "deferred_dependency_edges": self.deferred_dependency_edges,
            "dynamic_dependency_edges": self.dynamic_dependency_edges,
            "unresolved_dynamic_imports": self.unresolved_dynamic_imports,
            "unresolved_local_imports": self.unresolved_local_imports,
            "invalid_relative_imports": self.invalid_relative_imports,
            "executable_dependency_edges": self.executable_dependency_edges,
            "executable_cycle_count": self.executable_cycle_count,
            "largest_executable_cycle_size": self.largest_executable_cycle_size,
            "max_code_lines": self.max_code_lines,
            "max_complexity": self.max_complexity,
            "max_symbol_count": self.max_symbol_count,
            "architecture_debt": round(self.architecture_debt, 6),
            "cyclic_module_count": self.cyclic_module_count,
            "executable_cyclic_module_count": self.executable_cyclic_module_count,
        }


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def _freeze_graph(graph: Mapping[str, Iterable[str]]) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({key: tuple(sorted(value)) for key, value in graph.items()})


@dataclass(frozen=True)
class ArchitectureQualityReport:
    """Immutable architecture-quality evidence for one source snapshot."""

    root: str
    include_roots: tuple[str, ...]
    god_file_threshold: int
    metrics: ArchitectureQualityMetrics
    score: float
    line_counts: Mapping[str, int] = field(default_factory=dict)
    module_to_path: Mapping[str, str] = field(default_factory=dict)
    graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    reverse_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    cycles: tuple[tuple[str, ...], ...] = ()
    findings: tuple[ArchitectureQualityFinding, ...] = ()
    exclude_parts: tuple[str, ...] = ()
    module_structures: Mapping[str, ModuleStructure] = field(default_factory=dict)
    type_checking_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    optional_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    conditional_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    deferred_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    dynamic_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    executable_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    executable_cycles: tuple[tuple[str, ...], ...] = ()
    module_exports: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    dynamic_export_modules: tuple[str, ...] = ()
    findings_complete: bool = True
    findings_omitted: int = 0
    schema_version: int = SCHEMA_VERSION
    _attestation_sha256: str = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "include_roots", tuple(self.include_roots))
        object.__setattr__(self, "exclude_parts", tuple(self.exclude_parts))
        object.__setattr__(self, "cycles", tuple(tuple(cycle) for cycle in self.cycles))
        object.__setattr__(
            self,
            "executable_cycles",
            tuple(tuple(cycle) for cycle in self.executable_cycles),
        )
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(
            self,
            "dynamic_export_modules",
            tuple(sorted(self.dynamic_export_modules)),
        )
        object.__setattr__(self, "line_counts", _freeze_mapping(self.line_counts))
        object.__setattr__(self, "module_to_path", _freeze_mapping(self.module_to_path))
        object.__setattr__(self, "module_structures", _freeze_mapping(self.module_structures))
        object.__setattr__(self, "module_exports", _freeze_graph(self.module_exports))
        for name in (
            "graph",
            "reverse_graph",
            "type_checking_graph",
            "optional_graph",
            "conditional_graph",
            "deferred_graph",
            "dynamic_graph",
            "executable_graph",
        ):
            object.__setattr__(self, name, _freeze_graph(getattr(self, name)))
        if self.findings_omitted < 0:
            raise ValueError("findings_omitted cannot be negative")
        if self.findings_complete and self.findings_omitted:
            raise ValueError("a complete report cannot omit findings")
        attested = self._to_dict(include_attestation=False)
        attested.pop("root", None)
        payload = json.dumps(
            attested,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "_attestation_sha256", hashlib.sha256(payload).hexdigest())

    def changed_module_for_path(self, path: str) -> str | None:
        try:
            normalized = normalize_repository_path(path, label="changed path")
        except ValueError as exc:
            logger.debug(
                "a changed path did not normalize, so it maps to no module (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return None
        module = normalized.removesuffix(".py").replace("/", ".")
        if module in self.module_to_path:
            return module
        if module.endswith(".__init__"):
            package = module.removesuffix(".__init__")
            if package in self.module_to_path:
                return package
        return None

    @property
    def attestation_sha256(self) -> str:
        return self._attestation_sha256

    @property
    def analysis_scope_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "include_roots": self.include_roots,
            "exclude_parts": self.exclude_parts,
            "god_file_threshold": self.god_file_threshold,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _to_dict(self, *, include_attestation: bool) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "root": self.root,
            "include_roots": list(self.include_roots),
            "exclude_parts": list(self.exclude_parts),
            "analysis_scope_sha256": self.analysis_scope_sha256,
            "god_file_threshold": self.god_file_threshold,
            "metrics": self.metrics.to_dict(),
            "score": round(self.score, 6),
            "line_counts": dict(sorted(self.line_counts.items())),
            "module_to_path": dict(sorted(self.module_to_path.items())),
            "module_structures": {
                key: value.to_dict() for key, value in sorted(self.module_structures.items())
            },
            "module_exports": {
                key: list(value) for key, value in sorted(self.module_exports.items())
            },
            "dynamic_export_modules": list(self.dynamic_export_modules),
            "graph": {key: list(value) for key, value in sorted(self.graph.items())},
            "reverse_graph": {
                key: list(value) for key, value in sorted(self.reverse_graph.items())
            },
            "type_checking_graph": {
                key: list(value) for key, value in sorted(self.type_checking_graph.items())
            },
            "optional_graph": {
                key: list(value) for key, value in sorted(self.optional_graph.items())
            },
            "conditional_graph": {
                key: list(value) for key, value in sorted(self.conditional_graph.items())
            },
            "deferred_graph": {
                key: list(value) for key, value in sorted(self.deferred_graph.items())
            },
            "dynamic_graph": {
                key: list(value) for key, value in sorted(self.dynamic_graph.items())
            },
            "executable_graph": {
                key: list(value) for key, value in sorted(self.executable_graph.items())
            },
            "cycles": [list(cycle) for cycle in self.cycles],
            "executable_cycles": [list(cycle) for cycle in self.executable_cycles],
            "findings": [finding.to_dict() for finding in self.findings],
            "findings_complete": self.findings_complete,
            "findings_omitted": self.findings_omitted,
        }
        if include_attestation:
            data["attestation_sha256"] = self.attestation_sha256
        return data

    def to_dict(self) -> dict[str, Any]:
        return self._to_dict(include_attestation=True)

    def summary(self) -> str:
        metrics = self.metrics
        return (
            f"score={self.score:.1f}; debt={metrics.architecture_debt:.2f}; "
            f"modules={metrics.module_count}; edges={metrics.dependency_edges}; "
            f"cycles={metrics.cycle_count}; parse_errors={metrics.parse_error_count}; "
            f"god_files={metrics.god_file_count}; max_lines={metrics.max_file_lines}"
        )
