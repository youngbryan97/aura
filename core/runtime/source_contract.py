"""Stable syntax contracts for load-bearing parts of changing source files."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_OPTIONAL_AST_FIELDS = frozenset({"type_params"})


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _symbol_node(tree: ast.Module, qualified_name: str) -> ast.AST:
    body: list[ast.stmt] = tree.body
    current: ast.AST | None = None
    for part in qualified_name.split("."):
        current = next(
            (
                node
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if current is None:
            raise RuntimeError(f"source contract symbol is missing: {qualified_name}")
        body = list(getattr(current, "body", ()))
    return current


def canonical_ast(value: Any) -> Any:
    """Serialize syntax without coupling a seal to one Python AST inventory."""

    if isinstance(value, ast.AST):
        fields = {}
        for name, child in ast.iter_fields(value):
            if name in _OPTIONAL_AST_FIELDS and not child:
                continue
            fields[name] = canonical_ast(child)
        return {"node": type(value).__name__, "fields": fields}
    if isinstance(value, list):
        return [canonical_ast(item) for item in value]
    if isinstance(value, tuple):
        return {"tuple": [canonical_ast(item) for item in value]}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if value is Ellipsis:
        return {"ellipsis": True}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RuntimeError(f"unsupported source contract value: {type(value).__name__}")


def source_contract_sha256(path: str | Path, selector: str) -> str:
    """Hash one named symbol or one kind of call without hashing its whole file."""

    resolved = Path(path).expanduser().resolve(strict=True)
    tree = ast.parse(resolved.read_text(encoding="utf-8"), filename=str(resolved))
    kind, separator, target = selector.partition(":")
    if not separator or not target:
        raise RuntimeError(f"source contract selector is invalid: {selector}")
    if kind == "symbol":
        payload: Any = canonical_ast(_symbol_node(tree, target))
    elif kind == "call":
        calls = sorted(
            _sha(canonical_ast(node))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node) == target
        )
        if not calls:
            raise RuntimeError(f"source contract call is missing: {target}")
        payload = calls
    else:
        raise RuntimeError(f"source contract selector kind is invalid: {kind}")
    return _sha(payload)


def source_contract_sha256s(
    root: str | Path,
    contracts: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Hash a declared inventory of load-bearing source contracts."""

    resolved_root = Path(root).expanduser().resolve(strict=True)
    return {
        f"{relative}::{selector}": source_contract_sha256(
            resolved_root / relative,
            selector,
        )
        for relative, selectors in contracts.items()
        for selector in selectors
    }


__all__ = [
    "canonical_ast",
    "source_contract_sha256",
    "source_contract_sha256s",
]
