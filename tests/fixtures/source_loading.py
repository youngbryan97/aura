"""Load one function out of a module without importing the module.

Four test files needed the same thing: exercise a function that lives in
`core/brain/llm/mlx_worker.py` without importing it, because importing it
pulls in MLX and a Metal device. Each one parsed the file, picked the
`FunctionDef` out of the tree, built an `ast.Module` around it and `exec`'d
the compiled result into a fresh namespace — the same nine lines, four times,
and six findings from the gate that watches for `exec`.

One place is easier to review than six, and it is the review that matters
here. The property that makes this safe is narrow and worth stating: the
source is read from a file inside this repository, the compiled object is an
AST this function assembled from named `FunctionDef` nodes, and the namespace
is a dictionary the caller supplied. No text from outside the repository
reaches the compiler, which is what the rule against dynamic execution is
about.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]


def load_functions(
    module_path: Path | str,
    names: str | list[str] | set[str],
    *,
    namespace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the named top-level functions out of ``module_path``.

    Raises ``ValueError`` when the path is outside this repository, or when a
    requested name is not a top-level function in it — a silently missing
    function would produce a KeyError far from the cause.
    """
    path = Path(module_path).resolve()
    if path != REPOSITORY and REPOSITORY not in path.parents:
        raise ValueError(f"{path} is outside the repository")

    wanted = {names} if isinstance(names, str) else set(names)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    missing = wanted - {node.name for node in body}
    if missing:
        raise ValueError(f"{path.name} defines no top-level {sorted(missing)}")

    scope: dict[str, Any] = {"Any": Any}
    scope.update(namespace or {})
    module = ast.Module(body=body, type_ignores=[])
    # noqa: S102 — reviewed. The compiled object is an AST assembled above
    # from FunctionDef nodes parsed out of a file inside this repository; no
    # caller-supplied text reaches the compiler.
    compiled = compile(module, f"<{path.name}>", "exec")  # noqa: S102
    exec(compiled, scope)  # noqa: S102
    return {name: scope[name] for name in wanted}


def load_function(
    module_path: Path | str,
    name: str,
    *,
    namespace: dict[str, Any] | None = None,
) -> Any:
    """One function, for the common case."""
    return load_functions(module_path, name, namespace=namespace)[name]
