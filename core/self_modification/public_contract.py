"""Compare a patch's declared Python interface without executing either source."""
from __future__ import annotations

import ast
from typing import Any

MAX_SOURCE_BYTES = 2 * 1024 * 1024


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[Any, ...]:
    args = node.args
    return (
        "async" if isinstance(node, ast.AsyncFunctionDef) else "sync",
        tuple(arg.arg for arg in args.posonlyargs),
        tuple(arg.arg for arg in args.args),
        len(args.defaults),
        args.vararg.arg if args.vararg else None,
        tuple((arg.arg, default is not None) for arg, default in zip(args.kwonlyargs, args.kw_defaults)),
        args.kwarg.arg if args.kwarg else None,
        tuple(ast.dump(decorator, include_attributes=False) for decorator in node.decorator_list),
    )


def _bound_names(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return {part.id for target in targets for part in ast.walk(target) if isinstance(part, ast.Name)}


def _surface(source: str) -> dict[str, tuple[Any, ...]]:
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds the static interface check budget")
    tree = ast.parse(source)
    exports = None
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, (list, tuple)) or not all(isinstance(name, str) for name in value):
                raise ValueError("__all__ must be a static sequence of names")
            exports = set(value)
    definitions = {name: node for node in tree.body for name in _bound_names(node)}
    if exports is None:
        exports = {name for name, node in definitions.items()
                   if not name.startswith("_") and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    if exports - definitions.keys():
        raise ValueError("declared public names cannot be resolved statically")
    surface = {}

    def keep(name: str, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            surface[name] = ("function", *_signature(node))
        elif isinstance(node, ast.ClassDef):
            surface[name] = ("class", tuple(ast.dump(base, include_attributes=False) for base in node.bases),
                             tuple(ast.dump(keyword, include_attributes=False) for keyword in node.keywords),
                             tuple(ast.dump(d, include_attributes=False) for d in node.decorator_list))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not child.name.startswith("_") or (child.name.startswith("__") and child.name.endswith("__")):
                        accessor = ""
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            for decorator in child.decorator_list:
                                if isinstance(decorator, ast.Attribute) and decorator.attr in {"setter", "deleter"}:
                                    accessor = f".{decorator.attr}"
                        keep(f"{name}.{child.name}{accessor}", child)
        else:
            surface[name] = ("binding",)

    for name in sorted(exports):
        keep(name, definitions[name])
    return surface


def public_contract_violations(before: str, after: str) -> list[str]:
    """Require existing names, parameter shapes and async flags to stay intact.

    Additional public names are allowed. Behavior, annotations and default
    values need separate tests; this contract checks the calling interface.
    A changed signature is quarantined for operator review even if its author
    believes it is backward compatible.
    """
    try:
        old, new = _surface(before), _surface(after)
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError) as exc:
        return [f"interface_unreadable:{type(exc).__name__}:{exc}"]
    return [f"interface_changed:{name}" for name, signature in old.items() if new.get(name) != signature]
