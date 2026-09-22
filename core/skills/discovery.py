"""Deterministic, import-free skill source discovery and catalog validation.

The source pass parses only trusted skill roots and never imports candidate
modules.  A separate bounded child process proves import, construction, schema,
and execution-contract validity before a declaration can enter the live
registry.  The optional Rust extension canonicalizes the same candidate payload
and is compared with Python on every Rust-enabled boot.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from core.skills.catalog_policy import (
    INTERNAL_ONLY_SKILLS,
    class_exclusion_reason,
    resolve_skill_policy,
)

_MAX_SKILL_SOURCE_BYTES = 4 * 1024 * 1024
_SKILL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_SKILL_DECORATORS = frozenset({"aura_skill", "capability_skill", "register_skill", "skill"})
_BASE_SKILL_NAMES = frozenset(
    {
        "core.skills.base_skill.BaseSkill",
        "infrastructure.BaseSkill",
        "infrastructure.base_skill.BaseSkill",
    }
)
_METADATA_FIELDS = frozenset(
    {
        "abstract",
        "constructor_dependencies",
        "description",
        "effect_scope",
        "enabled",
        "execution_profile",
        "is_core_personality",
        "memory_mb_estimate",
        "metabolic_cost",
        "name",
        "timeout_seconds",
    }
)
_MISSING = object()

#: Every value `build_catalog` can report for parity between the Rust and
#: Python implementations, and the ones that mean they agree.
#:
#: `prove_clean_live_skill_boot` asked `parity_status == "ready"`, which this
#: module has never produced, so `catalog_parity_ready` was False on every
#: boot whatever the catalogs actually said — and its unit test passed
#: because the fixture hand-wrote "ready" into a health dict. A vocabulary
#: nobody can read from one place is a vocabulary two files will disagree
#: about, so it is named here, beside the code that assigns it.
#:
#: `python_only` agrees by construction: Rust was never asked, which is the
#: Rust-absent install `skill-portability-audit` exists to support.
#: `unavailable` does NOT agree — Rust was asked for and could not be loaded,
#: so nothing checked the Python catalog.
PARITY_STATES = frozenset({
    "unavailable",
    "python_only",
    "diverged",
    "failed",
    "matched",
    "canonicalizer_matched",
})
PARITY_STATES_THAT_AGREE = frozenset({
    "python_only",
    "matched",
    "canonicalizer_matched",
})


@dataclass(frozen=True, slots=True)
class SkillSourceRoot:
    path: Path
    module_prefix: str
    source_kind: str


@dataclass(frozen=True, slots=True)
class CatalogIssue:
    code: str
    severity: str
    detail: str
    module_path: str = ""
    class_name: str = ""
    source_path: str = ""
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SkillDeclaration:
    name: str
    description: str
    module_path: str
    class_name: str
    source_kind: str
    source_path: str
    source_sha256: str
    line: int
    effect_scope: str
    authority_class: str
    constructor_dependencies: tuple[str, ...] = ()
    decorated: bool = False
    inherited_metadata: bool = False
    exclusion_reason: str = ""
    catalog_id: str = ""

    def __post_init__(self) -> None:
        if not self.catalog_id:
            identity = f"{self.module_path}:{self.class_name}:{self.name}:{self.line}"
            object.__setattr__(self, "catalog_id", hashlib.sha256(identity.encode()).hexdigest()[:20])

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["constructor_dependencies"] = list(self.constructor_dependencies)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillDeclaration:
        data = dict(payload)
        data["constructor_dependencies"] = tuple(data.get("constructor_dependencies") or ())
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    accepted: tuple[SkillDeclaration, ...]
    excluded: tuple[SkillDeclaration, ...]
    issues: tuple[CatalogIssue, ...]
    backend: str
    parity_status: str
    digest: str
    source_file_count: int
    candidate_count: int
    duplicate_count: int

    @property
    def blocking_issues(self) -> tuple[CatalogIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def ok(self) -> bool:
        return not self.blocking_issues

    def canonical_payload(self) -> dict[str, Any]:
        """Return backend-independent catalog content for parity and replay proofs."""

        return {
            "accepted": [item.to_dict() for item in self.accepted],
            "candidate_count": self.candidate_count,
            "duplicate_count": self.duplicate_count,
            "excluded": [item.to_dict() for item in self.excluded],
            "issues": [item.to_dict() for item in self.issues],
            "source_file_count": self.source_file_count,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "accepted": len(self.accepted),
            "backend": self.backend,
            "blocking_issue_count": len(self.blocking_issues),
            "candidate_count": self.candidate_count,
            "digest": self.digest,
            "duplicate_count": self.duplicate_count,
            "excluded": len(self.excluded),
            "issue_count": len(self.issues),
            "parity_status": self.parity_status,
            "source_file_count": self.source_file_count,
        }


@dataclass(slots=True)
class _ParsedClass:
    module_path: str
    class_name: str
    source_kind: str
    source_path: str
    source_sha256: str
    line: int
    bases: tuple[str, ...]
    metadata: dict[str, Any]
    metadata_errors: tuple[str, ...]
    decorated: bool
    meaningful_body: bool

    @property
    def qualified_name(self) -> str:
        return f"{self.module_path}.{self.class_name}"


@dataclass(slots=True)
class _ModuleRecord:
    path: Path
    module_path: str
    source_kind: str
    source_path: str
    source_sha256: str
    tree: ast.Module
    aliases: dict[str, str] = field(default_factory=dict)
    local_classes: set[str] = field(default_factory=set)


def default_skill_roots(project_root: Path | None = None) -> tuple[SkillSourceRoot, ...]:
    if project_root is None:
        from core.config import config

        project_root = Path(config.paths.base_dir)
    root = Path(project_root).expanduser().resolve()
    return (
        SkillSourceRoot(root / "core" / "skills", "core.skills", "core"),
        SkillSourceRoot(root / "skills", "skills", "project"),
    )


def _module_path_for(root: SkillSourceRoot, path: Path) -> str:
    relative = path.relative_to(root.path)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = path.stem
    suffix = ".".join(parts)
    return root.module_prefix if not suffix else f"{root.module_prefix}.{suffix}"


def _display_source_path(root: SkillSourceRoot, path: Path) -> str:
    try:
        repository_root = root.path.parents[1] if root.module_prefix.startswith("core.") else root.path.parent
        return path.relative_to(repository_root).as_posix()
    except (IndexError, ValueError):
        return f"{root.source_kind}:{path.relative_to(root.path).as_posix()}"


def _iter_python_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    resolved_root = root.resolve()
    for current, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in {"__pycache__", "tests"} and not name.startswith(".")
        )
        current_path = Path(current)
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            if filename != "__init__.py" and filename.startswith("_"):
                continue
            if filename.endswith("_test.py"):
                continue
            path = current_path / filename
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(resolved_root)
            except (OSError, RuntimeError, ValueError):
                continue
            if resolved.is_file():
                yield resolved


def _resolve_from_import(module_path: str, path: Path, node: ast.ImportFrom) -> str:
    if not node.level:
        return str(node.module or "")
    package = module_path if path.name == "__init__.py" else module_path.rpartition(".")[0]
    relative = f"{'.' * node.level}{node.module or ''}"
    try:
        return importlib.util.resolve_name(relative, package)
    except (ImportError, ValueError):
        return str(node.module or "")


def _attribute_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _resolve_symbol(raw: str, module: _ModuleRecord) -> str:
    if not raw:
        return raw
    first, separator, remainder = raw.partition(".")
    if first in module.aliases:
        target = module.aliases[first]
        return f"{target}.{remainder}" if separator else target
    if not separator and first in module.local_classes:
        return f"{module.module_path}.{first}"
    return raw


def _literal_metadata(node: ast.expr | None) -> Any:
    if node is None:
        return _MISSING
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return _MISSING


def _decorator_metadata(node: ast.expr) -> tuple[bool, dict[str, Any], list[str]]:
    target = node.func if isinstance(node, ast.Call) else node
    decorator_name = _attribute_name(target).rpartition(".")[2]
    if decorator_name not in _SKILL_DECORATORS:
        return False, {}, []
    metadata: dict[str, Any] = {}
    errors: list[str] = []
    if isinstance(node, ast.Call):
        if node.args:
            value = _literal_metadata(node.args[0])
            if isinstance(value, str):
                metadata["name"] = value
            else:
                errors.append("decorator positional name must be a string literal")
        for keyword in node.keywords:
            if keyword.arg not in _METADATA_FIELDS:
                continue
            value = _literal_metadata(keyword.value)
            if value is _MISSING:
                errors.append(f"decorator metadata {keyword.arg!r} must be literal")
            else:
                metadata[keyword.arg] = value
    return True, metadata, errors


def _class_metadata(node: ast.ClassDef) -> tuple[dict[str, Any], tuple[str, ...], bool]:
    metadata: dict[str, Any] = {}
    errors: list[str] = []
    decorated = False
    for decorator in node.decorator_list:
        recognized, values, decorator_errors = _decorator_metadata(decorator)
        decorated = decorated or recognized
        metadata.update(values)
        errors.extend(decorator_errors)

    for item in node.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(item, ast.Assign):
            targets = list(item.targets)
            value = item.value
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
            value = item.value
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in _METADATA_FIELDS:
                continue
            literal = _literal_metadata(value)
            if literal is _MISSING:
                errors.append(f"class metadata {target.id!r} must be literal")
            else:
                metadata[target.id] = literal
    return metadata, tuple(errors), decorated


def _has_meaningful_body(node: ast.ClassDef) -> bool:
    for item in node.body:
        if isinstance(item, ast.Pass):
            continue
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(
            item.value.value, str
        ):
            continue
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            if all(isinstance(target, ast.Name) and target.id in _METADATA_FIELDS for target in targets):
                continue
        return True
    return False


def _read_modules(
    roots: Iterable[SkillSourceRoot],
) -> tuple[list[_ModuleRecord], list[CatalogIssue]]:
    modules: list[_ModuleRecord] = []
    issues: list[CatalogIssue] = []
    for root in roots:
        root_path = Path(root.path).expanduser().resolve()
        normalized_root = replace(root, path=root_path)
        for path in _iter_python_files(root_path):
            module_path = _module_path_for(normalized_root, path)
            source_path = _display_source_path(normalized_root, path)
            try:
                size = path.stat().st_size
                if size > _MAX_SKILL_SOURCE_BYTES:
                    raise ValueError(f"source exceeds {_MAX_SKILL_SOURCE_BYTES} bytes")
                raw = path.read_bytes()
                source = raw.decode("utf-8")
                tree = ast.parse(source, filename=source_path)
            except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
                issues.append(
                    CatalogIssue(
                        code="source_parse_failed",
                        severity="error",
                        detail=f"{type(exc).__name__}: {exc}",
                        module_path=module_path,
                        source_path=source_path,
                        line=int(getattr(exc, "lineno", 0) or 0),
                    )
                )
                continue
            modules.append(
                _ModuleRecord(
                    path=path,
                    module_path=module_path,
                    source_kind=root.source_kind,
                    source_path=source_path,
                    source_sha256=hashlib.sha256(raw).hexdigest(),
                    tree=tree,
                )
            )
    return modules, issues


def _parse_classes(modules: list[_ModuleRecord]) -> dict[str, _ParsedClass]:
    parsed: dict[str, _ParsedClass] = {}
    for module in modules:
        module.local_classes = {
            node.name for node in module.tree.body if isinstance(node, ast.ClassDef)
        }
        for node in _top_level_import_statements(module.tree.body):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module.aliases[alias.asname or alias.name.partition(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                imported_module = _resolve_from_import(module.module_path, module.path, node)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    module.aliases[local_name] = (
                        f"{imported_module}.{alias.name}" if imported_module else alias.name
                    )

        for node in module.tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            metadata, metadata_errors, decorated = _class_metadata(node)
            bases = tuple(
                _resolve_symbol(_attribute_name(base), module)
                for base in node.bases
                if _attribute_name(base)
            )
            record = _ParsedClass(
                module_path=module.module_path,
                class_name=node.name,
                source_kind=module.source_kind,
                source_path=module.source_path,
                source_sha256=module.source_sha256,
                line=int(node.lineno),
                bases=bases,
                metadata=metadata,
                metadata_errors=metadata_errors,
                decorated=decorated,
                meaningful_body=_has_meaningful_body(node),
            )
            parsed[record.qualified_name] = record
    return parsed


def _top_level_import_statements(statements: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
    """Yield imports reachable at module load without descending into callables."""

    for statement in statements:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            yield statement
        elif isinstance(statement, ast.Try):
            yield from _top_level_import_statements(statement.body)
            for handler in statement.handlers:
                yield from _top_level_import_statements(handler.body)
            yield from _top_level_import_statements(statement.orelse)
            yield from _top_level_import_statements(statement.finalbody)
        elif isinstance(statement, ast.If):
            yield from _top_level_import_statements(statement.body)
            yield from _top_level_import_statements(statement.orelse)


def _is_skill_class(
    qualified_name: str,
    classes: dict[str, _ParsedClass],
    visiting: set[str] | None = None,
) -> bool:
    if qualified_name in _BASE_SKILL_NAMES:
        return True
    record = classes.get(qualified_name)
    if record is None:
        return False
    if record.decorated:
        return True
    visiting = set(visiting or ())
    if qualified_name in visiting:
        return False
    visiting.add(qualified_name)
    return any(
        base in _BASE_SKILL_NAMES or _is_skill_class(base, classes, visiting)
        for base in record.bases
    )


def _effective_metadata(
    qualified_name: str,
    field_name: str,
    classes: dict[str, _ParsedClass],
    visiting: set[str] | None = None,
) -> tuple[Any, bool]:
    record = classes.get(qualified_name)
    if record is None:
        return _MISSING, False
    if field_name in record.metadata:
        return record.metadata[field_name], False
    visiting = set(visiting or ())
    if qualified_name in visiting:
        return _MISSING, False
    visiting.add(qualified_name)
    for base in record.bases:
        if base in _BASE_SKILL_NAMES:
            continue
        value, _ = _effective_metadata(base, field_name, classes, visiting)
        if value is not _MISSING:
            return value, True
    return _MISSING, False


def _constructor_dependencies(value: Any) -> tuple[str, ...] | None:
    if value is _MISSING:
        return ()
    if not isinstance(value, (list, tuple)):
        return None
    dependencies = tuple(str(item).strip() for item in value)
    if any(not item or not item.replace("_", "").isalnum() for item in dependencies):
        return None
    if len(set(dependencies)) != len(dependencies):
        return None
    return dependencies


def _declarations_from_classes(
    classes: dict[str, _ParsedClass],
) -> tuple[list[SkillDeclaration], list[CatalogIssue]]:
    declarations: list[SkillDeclaration] = []
    issues: list[CatalogIssue] = []
    for qualified_name in sorted(classes):
        record = classes[qualified_name]
        if qualified_name in _BASE_SKILL_NAMES or not _is_skill_class(qualified_name, classes):
            continue
        if record.class_name.startswith("_"):
            continue
        if record.metadata.get("abstract") is True:
            continue
        if record.metadata_errors:
            issues.append(
                CatalogIssue(
                    code="dynamic_metadata",
                    severity="error",
                    detail="; ".join(record.metadata_errors),
                    module_path=record.module_path,
                    class_name=record.class_name,
                    source_path=record.source_path,
                    line=record.line,
                )
            )
            continue

        name, inherited_name = _effective_metadata(qualified_name, "name", classes)
        if name is _MISSING:
            if record.meaningful_body:
                issues.append(
                    CatalogIssue(
                        code="missing_static_name",
                        severity="error",
                        detail="concrete skill classes require a literal name or skill decorator",
                        module_path=record.module_path,
                        class_name=record.class_name,
                        source_path=record.source_path,
                        line=record.line,
                    )
                )
            continue
        if not isinstance(name, str) or not _SKILL_NAME_RE.fullmatch(name):
            issues.append(
                CatalogIssue(
                    code="invalid_skill_name",
                    severity="error",
                    detail=f"invalid literal skill name: {name!r}",
                    module_path=record.module_path,
                    class_name=record.class_name,
                    source_path=record.source_path,
                    line=record.line,
                )
            )
            continue

        description, inherited_description = _effective_metadata(
            qualified_name, "description", classes
        )
        if not isinstance(description, str) or not description.strip():
            issues.append(
                CatalogIssue(
                    code="missing_static_description",
                    severity="error",
                    detail=f"skill {name!r} requires a non-empty literal description",
                    module_path=record.module_path,
                    class_name=record.class_name,
                    source_path=record.source_path,
                    line=record.line,
                )
            )
            continue

        declared_scope, inherited_scope = _effective_metadata(
            qualified_name, "effect_scope", classes
        )
        policy = resolve_skill_policy(
            name,
            declared_scope if isinstance(declared_scope, str) else "",
        )
        if policy is None:
            issues.append(
                CatalogIssue(
                    code="unclassified_effect",
                    severity="error",
                    detail=f"skill {name!r} needs a recognized literal or catalog effect_scope",
                    module_path=record.module_path,
                    class_name=record.class_name,
                    source_path=record.source_path,
                    line=record.line,
                )
            )
            continue

        dependencies_raw, inherited_dependencies = _effective_metadata(
            qualified_name, "constructor_dependencies", classes
        )
        dependencies = _constructor_dependencies(dependencies_raw)
        if dependencies is None:
            issues.append(
                CatalogIssue(
                    code="invalid_constructor_dependencies",
                    severity="error",
                    detail="constructor_dependencies must be a unique literal list of identifiers",
                    module_path=record.module_path,
                    class_name=record.class_name,
                    source_path=record.source_path,
                    line=record.line,
                )
            )
            continue

        compatibility_wrapper = inherited_name and not record.meaningful_body
        reason = "compatibility_wrapper" if compatibility_wrapper else ""
        declarations.append(
            SkillDeclaration(
                name=name,
                description=description.strip(),
                module_path=record.module_path,
                class_name=record.class_name,
                source_kind=record.source_kind,
                source_path=record.source_path,
                source_sha256=record.source_sha256,
                line=record.line,
                effect_scope=policy.effect_scope,
                authority_class=policy.authority_class,
                constructor_dependencies=dependencies,
                decorated=record.decorated,
                inherited_metadata=bool(
                    inherited_name
                    or inherited_description
                    or inherited_scope
                    or inherited_dependencies
                ),
                exclusion_reason=reason,
            )
        )
    return declarations, issues


def parse_skill_sources(
    roots: Iterable[SkillSourceRoot],
) -> tuple[list[SkillDeclaration], list[CatalogIssue], int]:
    normalized_roots = tuple(
        SkillSourceRoot(Path(root.path).expanduser().resolve(), root.module_prefix, root.source_kind)
        for root in roots
    )
    modules, issues = _read_modules(normalized_roots)
    classes = _parse_classes(modules)
    declarations, declaration_issues = _declarations_from_classes(classes)
    issues.extend(declaration_issues)
    return declarations, issues, len(modules)


def canonicalize_skill_candidates(candidate_json: str) -> str:
    """Canonicalize eligible declarations and reject name collisions."""

    payload = json.loads(candidate_json)
    candidates = list(payload.get("candidates") or [])
    candidates.sort(
        key=lambda item: (
            str(item.get("name") or "").casefold(),
            str(item.get("name") or ""),
            str(item.get("module_path") or ""),
            str(item.get("class_name") or ""),
            str(item.get("source_path") or ""),
            int(item.get("line") or 0),
        )
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(str(candidate.get("name") or "").casefold(), []).append(candidate)

    accepted: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for name_key in sorted(groups):
        group = groups[name_key]
        if len(group) == 1:
            accepted.append(group[0])
        else:
            duplicates.append({"candidates": group, "name_key": name_key})
    return json.dumps(
        {"accepted": accepted, "duplicates": duplicates},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _candidate_payload_json(candidates: Iterable[SkillDeclaration]) -> str:
    return json.dumps(
        {"candidates": [candidate.to_dict() for candidate in candidates]},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def skill_index_candidates_json() -> str:
    declarations, _issues, _source_count = parse_skill_sources(default_skill_roots())
    eligible = [
        declaration
        for declaration in declarations
        if not declaration.exclusion_reason
        and declaration.name not in INTERNAL_ONLY_SKILLS
        and class_exclusion_reason(declaration.module_path, declaration.class_name) is None
    ]
    return _candidate_payload_json(eligible)


def _index_dict_from_canonical_json(canonical_json: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(canonical_json)
    return {
        str(item["name"]): {
            key: value
            for key, value in item.items()
            if key not in {"name", "catalog_id"}
        }
        for item in payload.get("accepted") or []
    }


def _load_rust_builder() -> Callable[[str], str] | None:
    try:
        from aura_m1_ext import build_skill_index

        return cast(Callable[[str], str], build_skill_index)
    # not a failure: the Rust extension is an acceleration path; the Python builder
    # below is what runs without it.
    except (ImportError, AttributeError):
        return None


def _load_rust_discoverer() -> Callable[[str], str] | None:
    try:
        from aura_m1_ext import discover_skill_candidates

        return cast(Callable[[str], str], discover_skill_candidates)
    # not a failure: the same, for the discoverer.
    except (ImportError, AttributeError):
        return None


def _rust_roots_json(roots: Iterable[SkillSourceRoot]) -> str:
    return json.dumps(
        [
            {
                "kind": root.source_kind,
                "package": root.module_prefix,
                "path": str(root.path.expanduser().resolve()),
            }
            for root in roots
        ],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _issue_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ("class_name", "code", "line", "module_path", "severity", "source_path")
    }


def _sorted_payloads(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(payload) for payload in payloads),
        key=lambda payload: json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _filesystem_parity_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": list(payload.get("accepted") or ()),
        "candidates": _sorted_payloads(payload.get("candidates") or ()),
        "duplicates": list(payload.get("duplicates") or ()),
        "excluded": list(payload.get("excluded") or ()),
        "issues": _sorted_payloads(
            _issue_identity(dict(item)) for item in payload.get("issues") or ()
        ),
        "source_file_count": payload.get("source_file_count"),
    }


def build_skill_catalog(
    roots: Iterable[SkillSourceRoot] | None = None,
    *,
    try_rust: bool = True,
    rust_builder: Callable[[str], str] | None = None,
    rust_discoverer: Callable[[str], str] | None = None,
) -> SkillCatalog:
    roots = tuple(roots or default_skill_roots())
    declarations, issues, source_file_count = parse_skill_sources(roots)

    eligible: list[SkillDeclaration] = []
    excluded: list[SkillDeclaration] = []
    for declaration in declarations:
        reason = declaration.exclusion_reason
        if not reason:
            reason = class_exclusion_reason(declaration.module_path, declaration.class_name) or ""
        if not reason and declaration.name in INTERNAL_ONLY_SKILLS:
            reason = "internal_only"
        if reason:
            excluded.append(replace(declaration, exclusion_reason=reason))
        else:
            eligible.append(declaration)

    candidate_json = _candidate_payload_json(eligible)
    python_json = canonicalize_skill_candidates(candidate_json)
    python_payload = json.loads(python_json)
    duplicates = list(python_payload.get("duplicates") or [])
    for duplicate in duplicates:
        candidates = list(duplicate.get("candidates") or [])
        locations = ", ".join(
            f"{item.get('module_path')}.{item.get('class_name')}" for item in candidates
        )
        issues.append(
            CatalogIssue(
                code="duplicate_skill_name",
                severity="error",
                detail=(
                    f"ambiguous case-insensitive name {duplicate.get('name_key')!r}: {locations}"
                ),
            )
        )

    excluded_tuple = tuple(
        sorted(excluded, key=lambda item: (item.name.casefold(), item.module_path, item.class_name))
    )
    python_filesystem_payload = {
        "accepted": list(python_payload.get("accepted") or ()),
        "candidates": json.loads(candidate_json).get("candidates") or [],
        "duplicates": duplicates,
        "excluded": [item.to_dict() for item in excluded_tuple],
        "issues": [item.to_dict() for item in issues],
        "source_file_count": source_file_count,
    }
    backend = "python"
    parity_status = "unavailable" if try_rust else "python_only"
    if try_rust:
        builder = rust_builder or _load_rust_builder()
        discoverer = rust_discoverer
        if discoverer is None and rust_builder is None:
            discoverer = _load_rust_discoverer()
        canonicalizer_state = "unavailable"
        filesystem_state = "unavailable"
        if builder is not None:
            try:
                rust_json = builder(candidate_json)
                rust_payload = json.loads(str(rust_json))
                if rust_payload != python_payload:
                    canonicalizer_state = "diverged"
                    issues.append(
                        CatalogIssue(
                            code="rust_python_catalog_divergence",
                            severity="error",
                            detail=(
                                "Rust and Python canonicalizers produced different skill catalogs"
                            ),
                        )
                    )
                else:
                    canonicalizer_state = "matched"
            except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                canonicalizer_state = "failed"
                issues.append(
                    CatalogIssue(
                        code="rust_catalog_failed",
                        severity="error",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
        if discoverer is not None:
            try:
                rust_discovery_json = discoverer(_rust_roots_json(roots))
                rust_discovery_payload = json.loads(str(rust_discovery_json))
                if _filesystem_parity_projection(
                    rust_discovery_payload
                ) != _filesystem_parity_projection(python_filesystem_payload):
                    filesystem_state = "diverged"
                    issues.append(
                        CatalogIssue(
                            code="rust_python_filesystem_catalog_divergence",
                            severity="error",
                            detail=(
                                "Independent Rust filesystem discovery and Python AST discovery "
                                "produced different skill catalogs"
                            ),
                        )
                    )
                else:
                    filesystem_state = "matched"
            except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                filesystem_state = "failed"
                issues.append(
                    CatalogIssue(
                        code="rust_filesystem_catalog_failed",
                        severity="error",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )

        states = {canonicalizer_state, filesystem_state}
        if "diverged" in states:
            parity_status = "diverged"
        elif "failed" in states:
            parity_status = "failed"
        elif filesystem_state == "matched":
            backend = "rust-filesystem+python-parity"
            parity_status = "matched"
        elif canonicalizer_state == "matched":
            backend = "rust-canonicalizer+python-discovery"
            parity_status = "canonicalizer_matched"

    accepted = tuple(
        SkillDeclaration.from_dict(item) for item in python_payload.get("accepted") or []
    )
    digest_payload = {
        "accepted": [item.to_dict() for item in accepted],
        "excluded": [item.to_dict() for item in excluded_tuple],
        "issues": [item.to_dict() for item in issues],
        "parity_status": parity_status,
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SkillCatalog(
        accepted=accepted,
        excluded=excluded_tuple,
        issues=tuple(issues),
        backend=backend,
        parity_status=parity_status,
        digest=digest,
        source_file_count=source_file_count,
        candidate_count=len(declarations),
        duplicate_count=len(duplicates),
    )


_VALIDATION_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def validate_skill_catalog(
    catalog: SkillCatalog,
    *,
    project_root: Path | None = None,
    timeout_s: float = 45.0,
    use_cache: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run the accepted catalog through the isolated import/contract probe."""

    if use_cache and catalog.digest in _VALIDATION_CACHE:
        return {key: dict(value) for key, value in _VALIDATION_CACHE[catalog.digest].items()}
    if project_root is None:
        from core.config import config

        project_root = Path(config.paths.base_dir)
    from core.runtime.skill_catalog_probe import run_skill_catalog_probe

    payload = {
        "catalog_digest": catalog.digest,
        "declarations": [item.to_dict() for item in catalog.accepted],
    }
    result = run_skill_catalog_probe(
        payload,
        project_root=Path(project_root).expanduser().resolve(),
        timeout_s=timeout_s,
    )
    validations = {
        str(item.get("catalog_id") or ""): dict(item)
        for item in result.get("validations") or []
        if item.get("catalog_id")
    }
    for declaration in catalog.accepted:
        validations.setdefault(
            declaration.catalog_id,
            {
                "catalog_id": declaration.catalog_id,
                "error": "isolated probe returned no result for declaration",
                "stage": "probe_protocol",
                "status": "quarantined",
            },
        )
    if use_cache:
        _VALIDATION_CACHE[catalog.digest] = {
            key: dict(value) for key, value in validations.items()
        }
    return validations
