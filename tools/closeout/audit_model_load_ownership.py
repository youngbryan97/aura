#!/usr/bin/env python3
"""Audit every direct MLX model-load reference against an ownership contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "config" / "model_load_ownership.json"
MODEL_MODULES = frozenset({"mlx_lm", "mlx_vlm"})
MODEL_CONSTRUCTORS = {
    ("sentence_transformers", "SentenceTransformer"): "sentence_transformers",
    ("faster_whisper", "WhisperModel"): "faster_whisper",
}
SOURCE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".claude",
        ".aura",
        ".aura_architect",
        ".aura_runtime",
        ".aura_snapshots",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "archive",
        "artifacts",
        "build",
        "dev_archive",
        "dist",
        "logs",
        "scratch",
        "site-packages",
        "tests",
    }
)


@dataclass(frozen=True)
class LoadReference:
    path: str
    line: int
    module: str


@dataclass(frozen=True)
class AuditFinding:
    code: str
    path: str
    detail: str


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


#: Libraries that put WEIGHTS IN A PROCESS when imported through the
#: serialized-import helper. huggingface_hub is deliberately absent:
#: snapshot_download writes files to disk and loads nothing, and counting it
#: reported two download helpers as unowned model loads.
_SERIALIZED_MODEL_LOADERS = frozenset(
    {
        "sentence_transformers",
        "transformers",
        "mlx_lm",
        "mlx_vlm",
        "faster_whisper",
        "coqui_tts",
        "piper",
        "TTS",
    }
)


#: Receivers whose ``from_pretrained`` reads a vocabulary or a config and puts
#: no weights anywhere. This audit exists to say who may put a MODEL in
#: memory; a tokenizer is a text file, takes no lane and contends with
#: nothing. Counting it required two offline tools that read only a tokenizer
#: to hold the standalone model lane, which would serialise them behind the
#: live lane for no reason.
_WEIGHTLESS_LOADER_HINTS = ("tokenizer", "processor", "featureextractor", "config")


def _loads_no_weights(receiver: ast.expr) -> bool:
    name = ""
    if isinstance(receiver, ast.Name):
        name = receiver.id
    elif isinstance(receiver, ast.Attribute):
        name = receiver.attr
    lowered = name.lower()
    return any(hint in lowered for hint in _WEIGHTLESS_LOADER_HINTS)


def _references_in_tree(tree: ast.Module) -> set[tuple[int, str]]:
    direct_aliases: dict[str, str] = {}
    constructor_aliases: dict[str, str] = {}
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_family = str(node.module or "").split(".", 1)[0]
            if module_family in MODEL_MODULES:
                for alias in node.names:
                    if alias.name == "load":
                        direct_aliases[alias.asname or alias.name] = module_family
            else:
                for alias in node.names:
                    module = MODEL_CONSTRUCTORS.get((str(node.module), alias.name))
                    if module:
                        constructor_aliases[alias.asname or alias.name] = module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_family = alias.name.split(".", 1)[0]
                if module_family in MODEL_MODULES:
                    module_aliases[alias.asname or alias.name] = module_family

    references: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id in direct_aliases:
            references.add((node.lineno, direct_aliases[function.id]))
        elif isinstance(function, ast.Name) and function.id in constructor_aliases:
            references.add((node.lineno, constructor_aliases[function.id]))
        elif (
            isinstance(function, ast.Attribute)
            and function.attr == "load"
            and isinstance(function.value, ast.Name)
            and function.value.id in module_aliases
        ):
            references.add((node.lineno, module_aliases[function.value.id]))
        elif (
            isinstance(function, ast.Attribute)
            and function.attr == "from_pretrained"
            and not _loads_no_weights(function.value)
        ):
            references.add((node.lineno, "from_pretrained"))
        elif (
            isinstance(function, ast.Name)
            and function.id == "import_attribute_serialized"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in _SERIALIZED_MODEL_LOADERS
        ):
            # A load reached through the serialized-import helper is still a
            # load. core/memory/embedding_model.py resolves SentenceTransformer
            # this way, so its inventory entry read as stale — and dropping the
            # entry would have retired the guard requirement with it.
            references.add((node.lineno, node.args[0].value))
        elif isinstance(function, ast.Name) and function.id == "whisper_model_cls":
            references.add((node.lineno, "faster_whisper"))
        elif isinstance(function, ast.Name) and function.id == "TTS":
            references.add((node.lineno, "coqui_tts"))
        elif (
            isinstance(function, ast.Attribute)
            and function.attr == "load"
            and isinstance(function.value, ast.Name)
            and function.value.id == "PiperVoice"
        ):
            references.add((node.lineno, "piper"))
    return references


def _load_references(path: Path, relative_path: str) -> list[LoadReference]:
    tree = _parse(path)
    references = _references_in_tree(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        source = node.value
        if not any(
            marker in source
            for marker in (
                "mlx_lm",
                "mlx_vlm",
                "from_pretrained",
                "WhisperModel",
            )
        ):
            continue
        try:
            inline_tree = ast.parse(source, mode="exec")
        except SyntaxError:
            continue
        for _inner_line, module in _references_in_tree(inline_tree):
            references.add((node.lineno, module))
    return [
        LoadReference(relative_path, line, module)
        for line, module in sorted(references)
    ]


def _symbol_sites(tree: ast.Module, symbol: str) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == symbol
        )
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol
        )
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr == symbol
        )
    )


def _called_symbol(expression: ast.expr) -> str:
    function = expression.func if isinstance(expression, ast.Call) else expression
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _enclosing_context_call_lines(tree: ast.Module, symbol: str) -> set[int]:
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        if not any(_called_symbol(item.context_expr) == symbol for item in node.items):
            continue
        for statement in node.body:
            guarded.update(
                child.lineno
                for child in ast.walk(statement)
                if isinstance(child, ast.Call)
            )
    return guarded


def _guarded_finally_load_lines(
    tree: ast.Module,
    guard_symbol: str,
    cleanup_symbol: str,
    load_lines: set[int],
) -> set[int]:
    protected: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        if not any(
            _called_symbol(item.context_expr) == guard_symbol for item in node.items
        ):
            continue
        for statement in node.body:
            for candidate in ast.walk(statement):
                if not isinstance(candidate, ast.Try):
                    continue
                cleanup_lines = {
                    child.lineno
                    for final_statement in candidate.finalbody
                    for child in ast.walk(final_statement)
                    if isinstance(child, ast.Call)
                    and _called_symbol(child) == cleanup_symbol
                }
                if not cleanup_lines:
                    continue
                protected.update(
                    child.lineno
                    for protected_statement in (
                        *candidate.body,
                        *(body_item for handler in candidate.handlers for body_item in handler.body),
                        *candidate.orelse,
                    )
                    for child in ast.walk(protected_statement)
                    if isinstance(child, ast.Call) and child.lineno in load_lines
                )
    return protected


def _module_name(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def _repository_source_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, child_dirs, filenames in os.walk(root):
        child_dirs[:] = [
            name for name in child_dirs if name not in SOURCE_EXCLUDED_PARTS
        ]
        base = Path(directory)
        paths.extend(
            base / name
            for name in filenames
            if name.endswith(".py") and (base / name).is_file()
        )
    return paths


def _production_importers(root: Path, target_path: str) -> set[str]:
    target_module = _module_name(target_path)
    importers: set[str] = set()
    for path in _repository_source_paths(root):
        relative = path.relative_to(root).as_posix()
        if relative == target_path:
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == target_module:
                importers.add(relative)
            elif isinstance(node, ast.Import):
                if any(alias.name == target_module for alias in node.names):
                    importers.add(relative)
    return importers


def run_audit(
    *,
    root: Path = ROOT,
    inventory_path: Path = DEFAULT_INVENTORY,
) -> dict[str, Any]:
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    entries = {
        str(entry["path"]): dict(entry)
        for entry in list(inventory.get("entries") or [])
    }
    references: list[LoadReference] = []
    parse_findings: list[AuditFinding] = []
    source_paths = _repository_source_paths(root)
    for source_path in source_paths:
        relative = source_path.relative_to(root).as_posix()
        try:
            references.extend(_load_references(source_path, relative))
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_findings.append(
                AuditFinding("source_unreadable", relative, f"{type(exc).__name__}:{exc}")
            )

    findings = list(parse_findings)
    by_path: dict[str, list[LoadReference]] = {}
    for reference in references:
        by_path.setdefault(reference.path, []).append(reference)
    for owned_path in sorted(set(by_path) - set(entries)):
        findings.append(
            AuditFinding(
                "unowned_model_load",
                owned_path,
                f"load references at lines {[item.line for item in by_path[owned_path]]}",
            )
        )
    for owned_path in sorted(set(entries) - set(by_path)):
        findings.append(
            AuditFinding(
                "stale_inventory_entry",
                owned_path,
                "no model-load reference",
            )
        )

    for owned_path in sorted(set(entries) & set(by_path)):
        entry = entries[owned_path]
        observed = by_path[owned_path]
        expected_count = int(entry.get("expected_load_references") or 0)
        if len(observed) != expected_count:
            findings.append(
                AuditFinding(
                    "load_reference_count_changed",
                    owned_path,
                    f"expected={expected_count} observed={len(observed)}",
                )
            )
        expected_modules = {str(item) for item in entry.get("modules") or []}
        observed_modules = {item.module for item in observed}
        if observed_modules != expected_modules:
            findings.append(
                AuditFinding(
                    "model_module_set_changed",
                    owned_path,
                    f"expected={sorted(expected_modules)} observed={sorted(observed_modules)}",
                )
            )
        guard_path = str(entry.get("guard_path") or owned_path)
        guard_file = root / guard_path
        guard_symbol = str(entry.get("guard_symbol") or "")
        guard_tree: ast.Module | None = None
        try:
            guard_tree = _parse(guard_file)
            guard_sites = _symbol_sites(guard_tree, guard_symbol)
        except (OSError, SyntaxError, UnicodeError) as exc:
            findings.append(
                AuditFinding("guard_unreadable", guard_path, f"{type(exc).__name__}:{exc}")
            )
            guard_sites = 0
        minimum = int(entry.get("min_guard_sites") or 1)
        if not guard_symbol or guard_sites < minimum:
            findings.append(
                AuditFinding(
                    "ownership_guard_missing",
                    owned_path,
                    f"guard={guard_path}:{guard_symbol} expected_sites>={minimum} observed={guard_sites}",
                )
            )
        guard_scope = str(entry.get("guard_scope") or "symbol_present")
        if guard_scope not in {"symbol_present", "enclosing_context"}:
            findings.append(
                AuditFinding(
                    "ownership_guard_scope_invalid",
                    owned_path,
                    guard_scope,
                )
            )
        elif guard_scope == "enclosing_context" and guard_tree is not None:
            if guard_path != owned_path:
                findings.append(
                    AuditFinding(
                        "ownership_guard_not_enclosing_load",
                        owned_path,
                        f"guard path differs: {guard_path}",
                    )
                )
            else:
                guarded_lines = _enclosing_context_call_lines(guard_tree, guard_symbol)
                unguarded = sorted(
                    item.line for item in observed if item.line not in guarded_lines
                )
                if unguarded:
                    findings.append(
                        AuditFinding(
                            "ownership_guard_not_enclosing_load",
                            owned_path,
                            f"guard={guard_symbol} unguarded_load_lines={unguarded}",
                        )
                    )
        cleanup_scope = str(entry.get("cleanup_scope") or "")
        cleanup_symbol = str(entry.get("cleanup_symbol") or "")
        if cleanup_scope and guard_tree is not None:
            if cleanup_scope != "guarded_finally" or not cleanup_symbol:
                findings.append(
                    AuditFinding(
                        "ownership_cleanup_scope_invalid",
                        owned_path,
                        f"scope={cleanup_scope} symbol={cleanup_symbol}",
                    )
                )
            else:
                load_lines = {item.line for item in observed}
                protected_load_lines = _guarded_finally_load_lines(
                    guard_tree,
                    guard_symbol,
                    cleanup_symbol,
                    load_lines,
                )
                unprotected = sorted(load_lines - protected_load_lines)
                if unprotected:
                    findings.append(
                        AuditFinding(
                            "ownership_cleanup_not_guarded_finally",
                            owned_path,
                            (
                                f"guard={guard_symbol} cleanup={cleanup_symbol} "
                                f"unprotected_load_lines={unprotected}"
                            ),
                        )
                    )
        worker_entrypoint = str(entry.get("worker_entrypoint") or "")
        if worker_entrypoint:
            source_tree = _parse(root / owned_path)
            if _symbol_sites(source_tree, worker_entrypoint) < 1:
                findings.append(
                    AuditFinding(
                        "worker_entrypoint_missing",
                        owned_path,
                        worker_entrypoint,
                    )
                )
        allowed_importers = {str(item) for item in entry.get("allowed_importers") or []}
        if allowed_importers:
            observed_importers = _production_importers(root, owned_path)
            if observed_importers != allowed_importers:
                findings.append(
                    AuditFinding(
                        "worker_component_importers_changed",
                        owned_path,
                        f"expected={sorted(allowed_importers)} observed={sorted(observed_importers)}",
                    )
                )

    report = {
        "schema": "aura.model_load_ownership.audit.v1",
        "passed": not findings,
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "inventory_entries": len(entries),
        "source_paths_scanned": len(source_paths),
        "load_references": len(references),
        "owned_paths": len(by_path),
        "references": [asdict(item) for item in sorted(references, key=lambda item: (item.path, item.line))],
        "findings": [asdict(item) for item in findings],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_audit(root=args.root.resolve(), inventory_path=args.inventory.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["passed"] else "FAIL"
        print(
            f"MODEL_LOAD_OWNERSHIP={status} paths={report['owned_paths']} "
            f"references={report['load_references']} findings={len(report['findings'])}"
        )
        for finding in report["findings"]:
            print(f"- {finding['code']}: {finding['path']}: {finding['detail']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
