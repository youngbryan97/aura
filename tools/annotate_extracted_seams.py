#!/usr/bin/env python3
"""Annotate the helpers that came out of the God objects, from evidence.

`tools/extract_seam.py` lifts a block out of an oversized function and writes
a helper whose body is that block, token for token. What it does not write is
a signature: the block's free variables become parameters with no annotation
and the tuple it hands back has no return type. Twenty-five such lifts left
174 modules that `make typed-surface` now calls untyped, which is 724
definitions — a number nobody fixes by hand and nobody should.

The types are recoverable, because the code that lost them is still there:

* **The call site.** A helper is called from the method it came out of, with
  the same names. When an argument is an annotated parameter of that method,
  that annotation IS the parameter's type.
* **A binding just above the call.** `prompt = str(...)` a few lines before
  `_part_1(context, prompt)` says what `prompt` is.
* **The helper's own body.** A name only ever passed to `int()`, or compared
  with `is True`, or subscripted with string keys, narrows on its own.
* **The return.** A helper with no value-returning `return` is `-> None`. One
  that returns a tuple of names it bound itself is a tuple of those names'
  inferred types.

Where nothing says, the annotation is `Any` — which is the truth about a
local in an unannotated method, not a way of quieting the gate. The tool
reports how many it resolved and how many it could only call `Any`, so the
proportion is visible rather than assumed.

    python tools/annotate_extracted_seams.py --report
    python tools/annotate_extracted_seams.py --apply
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMPLICIT = {"self", "cls"}

#: What a call around a name says that name is.
_CALL_TYPES = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "list": "list[Any]",
    "dict": "dict[str, Any]",
    "set": "set[Any]",
    "tuple": "tuple[Any, ...]",
    "frozenset": "frozenset[Any]",
}


#: Decorators that READ a signature's annotations and act on them.
#:
#: An annotation is inert at runtime almost everywhere, which is why this
#: tool can add 2,000 of them and prove nothing changed. The exceptions are
#: the frameworks that build behaviour OUT of the signature: a FastAPI
#: endpoint's parameters become its request model, and annotating one that
#: was bare turns a passthrough into a validated field. Those are left alone,
#: matched on the decorator rather than on the directory, because a route can
#: be declared anywhere.
_READS_THE_SIGNATURE = (
    "get", "post", "put", "patch", "delete", "head", "options", "websocket",
    "route", "api_route", "middleware", "exception_handler", "on_event",
    "validator", "field_validator", "model_validator", "root_validator",
    "command", "callback", "task", "tool", "listen", "event",
)


def _built_from_its_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name in _READS_THE_SIGNATURE:
            return True
    return False


def _annotation_text(node: ast.AST | None) -> str:
    return "" if node is None else ast.unparse(node)


def _type_of_value(value: ast.AST | None) -> str:
    """What an expression says about itself, or "" when it says nothing."""
    if value is None:
        return ""
    if isinstance(value, ast.Constant):
        if value.value is None:
            return ""
        return type(value.value).__name__
    if isinstance(value, ast.JoinedStr):
        return "str"
    if isinstance(value, ast.Dict):
        return "dict[str, Any]"
    if isinstance(value, ast.List):
        return "list[Any]"
    if isinstance(value, ast.Set):
        return "set[Any]"
    if isinstance(value, ast.Tuple):
        return "tuple[Any, ...]"
    if isinstance(value, ast.ListComp):
        return "list[Any]"
    if isinstance(value, ast.DictComp):
        return "dict[str, Any]"
    if isinstance(value, ast.SetComp):
        return "set[Any]"
    if isinstance(value, ast.Compare):
        return "bool"
    if isinstance(value, ast.BoolOp):
        # `a or ""` is as specific as its most specific arm and no more.
        kinds = {_type_of_value(one) for one in value.values} - {""}
        return kinds.pop() if len(kinds) == 1 else ""
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
        return "bool"
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return _CALL_TYPES.get(value.func.id, "")
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        if value.func.attr in {"strip", "lower", "upper", "join", "format"}:
            return "str"
        if value.func.attr in {"keys", "values", "items"}:
            return ""
    return ""


def _bindings_in(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Every name this function binds, and what its binding says it is.

    A name bound twice with two answers is left unresolved: the first
    assignment is not authority over the second.
    """
    seen: dict[str, str | None] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            seen[node.target.id] = _annotation_text(node.annotation)
        elif isinstance(node, ast.Assign):
            kind = _type_of_value(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in seen and seen[target.id] != kind:
                        seen[target.id] = None
                    else:
                        seen[target.id] = kind
    return {name: kind for name, kind in seen.items() if kind}


def _parameters_of(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    args = function.args
    everything = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    return {
        one.arg: _annotation_text(one.annotation)
        for one in everything
        if one.annotation is not None
    }


def _return_annotation(function: ast.FunctionDef | ast.AsyncFunctionDef,
                       known: dict[str, str]) -> str:
    """What the helper hands back, read off its own returns."""
    returns = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    yields = [node for node in ast.walk(function) if isinstance(node, (ast.Yield, ast.YieldFrom))]
    if yields:
        return ""
    if not returns:
        return "None"
    shapes: set[str] = set()
    for node in returns:
        value = node.value
        if isinstance(value, ast.Tuple):
            if not value.elts:
                # `return ()` is the empty tuple, not a tuple of nothing.
                shapes.add("tuple[()]")
                continue
            if any(isinstance(element, ast.Starred) for element in value.elts):
                shapes.add("tuple[Any, ...]")
                continue
            parts = []
            for element in value.elts:
                if isinstance(element, ast.Name) and known.get(element.id):
                    parts.append(known[element.id])
                else:
                    parts.append(_type_of_value(element) or "Any")
            shapes.add(f"tuple[{', '.join(parts)}]")
        elif isinstance(value, ast.Name) and known.get(value.id):
            shapes.add(known[value.id])
        else:
            shapes.add(_type_of_value(value) or "Any")
    if len(shapes) == 1:
        return shapes.pop()
    if all(shape.startswith("tuple[") for shape in shapes):
        return "tuple[Any, ...]"
    return "Any"


def _call_sites(tree: ast.AST, name: str) -> list[ast.Call]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if called == name:
            found.append(node)
    return found


def _enclosing(tree: ast.AST, node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    best = None
    for candidate in ast.walk(tree):
        if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if candidate.lineno <= node.lineno <= (candidate.end_lineno or candidate.lineno):
            if best is None or candidate.lineno > best.lineno:
                best = candidate
    return best


def _callers_for(module_path: Path) -> list[Path]:
    """Where a lifted module's helpers are called from.

    An extracted mixin is imported by the module it came out of, and that is
    where the names still carry their meaning. The convention the lifts left
    behind is the filename: `inference_gate_turn_setup` came out of
    `inference_gate`.
    """
    stem = module_path.stem
    candidates: list[Path] = []
    parts = stem.split("_")
    for cut in range(len(parts) - 1, 0, -1):
        sibling = module_path.with_name("_".join(parts[:cut]) + ".py")
        if sibling.exists() and sibling != module_path:
            candidates.append(sibling)
            break
    return candidates


def _resolve_parameters(
    helper: ast.FunctionDef | ast.AsyncFunctionDef,
    own_bindings: dict[str, str],
    caller_trees: list[ast.AST],
) -> dict[str, str]:
    """One annotation per unannotated parameter, and where it came from."""
    resolved: dict[str, str] = {}
    positional = [*helper.args.posonlyargs, *helper.args.args]
    names = [one.arg for one in positional]

    for tree in caller_trees:
        for call in _call_sites(tree, helper.name):
            caller = _enclosing(tree, call)
            if caller is None:
                continue
            declared = _parameters_of(caller)
            bound = _bindings_in(caller)
            offset = 1 if names and names[0] in IMPLICIT else 0
            for index, argument in enumerate(call.args):
                position = index + offset
                if position >= len(names):
                    break
                name = names[position]
                if name in resolved or name in IMPLICIT:
                    continue
                if isinstance(argument, ast.Name):
                    kind = declared.get(argument.id) or bound.get(argument.id)
                    if kind:
                        resolved[name] = kind
                        continue
                kind = _type_of_value(argument)
                if kind:
                    resolved[name] = kind
            for keyword in call.keywords:
                if keyword.arg is None or keyword.arg in resolved:
                    continue
                value = keyword.value
                if isinstance(value, ast.Name):
                    kind = declared.get(value.id) or bound.get(value.id)
                    if kind:
                        resolved[keyword.arg] = kind
                        continue
                kind = _type_of_value(value)
                if kind:
                    resolved[keyword.arg] = kind

    # A default says what the parameter is when nobody passes one.
    defaults = helper.args.defaults
    tail = [*helper.args.posonlyargs, *helper.args.args][len(positional) - len(defaults):]
    for one, default in zip(tail, defaults, strict=False):
        if one.arg not in resolved:
            kind = _type_of_value(default)
            if kind:
                resolved[one.arg] = kind
    for one, default in zip(helper.args.kwonlyargs, helper.args.kw_defaults, strict=False):
        if one.arg not in resolved and default is not None:
            kind = _type_of_value(default)
            if kind:
                resolved[one.arg] = kind

    # And the body: a name rebound from something specific before first use.
    for name, kind in own_bindings.items():
        resolved.setdefault(name, kind)
    return resolved


def _signature_span(source: str, lines: list[str], node) -> tuple[int, int] | None:
    """The exact characters from `def` to the colon that ends the signature.

    Found by scanning and bracket-counting rather than by guessing a line, so
    a signature wrapped over six lines is one span like any other. Returns
    None when the scan cannot settle, and the function is then left alone —
    a signature this cannot read is one it must not rewrite.
    """
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    begin = starts[node.lineno - 1] + node.col_offset
    if node.decorator_list:
        # col_offset is the `def`, not the decorator; nothing to adjust.
        pass
    depth = 0
    index = begin
    seen_paren = False
    while index < len(source):
        char = source[index]
        if char in "([{":
            depth += 1
            seen_paren = True
        elif char in ")]}":
            depth -= 1
        elif char == ":" and depth == 0 and seen_paren:
            return begin, index
        elif char == "#" and depth == 0:
            index = source.find("\n", index)
            if index < 0:
                return None
            continue
        index += 1
    return None


def _rendered_signature(node, parameters: dict[str, str], returns: str) -> str:
    """The same signature with the annotations filled in."""
    import copy

    stub = copy.deepcopy(node)
    stub.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
    stub.decorator_list = []
    for one in [
        *stub.args.posonlyargs,
        *stub.args.args,
        *stub.args.kwonlyargs,
        stub.args.vararg,
        stub.args.kwarg,
    ]:
        if one is None or one.annotation is not None or one.arg in IMPLICIT:
            continue
        one.annotation = ast.Name(id=parameters.get(one.arg, "Any"), ctx=ast.Load())
    if stub.returns is None and returns:
        stub.returns = ast.Name(id=returns, ctx=ast.Load())
    ast.fix_missing_locations(stub)
    text = ast.unparse(stub)
    signature = text[: text.rindex(":")]
    indent = " " * node.col_offset
    if node.col_offset:
        signature = "\n".join(
            (indent + line if index else line)
            for index, line in enumerate(signature.splitlines())
        )
    if len(indent) + len(signature.splitlines()[-1]) <= 100:
        return signature
    # Too long for one line, so one parameter per line — the form a person
    # would have written it in.
    head, _, rest = signature.partition("(")
    body, _, tail = rest.rpartition(")")
    parts, depth, current = [], 0, ""
    for char in body:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())
    inner = indent + "    "
    lines = [f"{head}("]
    lines.extend(f"{inner}{one}," for one in parts)
    lines.append(f"{indent}){tail}")
    return "\n".join(lines)


def plan(module_path: Path) -> tuple[str, dict[str, int]]:
    """The rewritten module, and how many annotations came from evidence."""
    source = module_path.read_text("utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)

    caller_trees = []
    for caller in _callers_for(module_path):
        try:
            caller_trees.append(ast.parse(caller.read_text("utf-8")))
        except (OSError, SyntaxError):
            continue
    caller_trees.append(tree)

    counted = {"resolved": 0, "any": 0, "returns": 0, "skipped": 0}
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _built_from_its_signature(node):
            counted["skipped"] += 1
            continue
        missing = [
            one
            for one in [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
                node.args.vararg,
                node.args.kwarg,
            ]
            if one is not None and one.annotation is None and one.arg not in IMPLICIT
        ]
        own = _bindings_in(node)
        returns = "" if node.returns is not None else _return_annotation(node, own)
        if not missing and not returns:
            continue
        span = _signature_span(source, lines, node)
        if span is None:
            counted["skipped"] += 1
            continue
        resolved = _resolve_parameters(node, own, caller_trees) if missing else {}
        for one in missing:
            kind = "Any" if one in (node.args.vararg, node.args.kwarg) else resolved.get(one.arg, "Any")
            counted["resolved" if kind != "Any" else "any"] += 1
        if returns:
            counted["returns"] += 1
        spans.append((span[0], span[1], _rendered_signature(node, resolved, returns)))

    out = source
    for begin, end, text in sorted(spans, reverse=True):
        out = out[:begin] + text + out[end:]
    # The body is what must not change. Proving it here rather than trusting
    # the span arithmetic is the same rule extract_seam.py works under.
    try:
        rewritten = ast.parse(out)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"rewrite does not parse: {exc}") from exc
    if _bodies_of(rewritten) != _bodies_of(tree):
        raise ValueError("rewrite changed a function body")
    return out, counted


def _bodies_of(tree: ast.AST) -> list[str]:
    """Every function body, with every annotation in it erased.

    A nested `def` is part of its parent's body, so annotating the inner one
    changes the outer body's dump while changing nothing that runs. Erasing
    annotations everywhere first is what makes the comparison say what it is
    meant to say: the code is the same, only its types are written down.
    """
    import copy

    stripped = copy.deepcopy(tree)
    for node in ast.walk(stripped):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
        elif isinstance(node, ast.arg):
            node.annotation = None
    bodies = []
    for node in ast.walk(stripped):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies.append(
                ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
            )
    return sorted(bodies)


def _needs_any_import(source: str) -> bool:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            if any(alias.name == "Any" for alias in node.names):
                return False
    return True


def _with_any_import(source: str) -> str:
    """Add `from typing import Any` where the annotations now need it."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            end = node.end_lineno or node.lineno
            text = "".join(lines[node.lineno - 1 : end])
            names = sorted({alias.name for alias in node.names} | {"Any"})
            replacement = f"from typing import {', '.join(names)}\n"
            return "".join(lines[: node.lineno - 1]) + replacement + "".join(lines[end:])
    # No typing import at all: put one after the last __future__ or import.
    insert_at = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_at = node.end_lineno or node.lineno
    return (
        "".join(lines[:insert_at])
        + "from typing import Any\n"
        + "".join(lines[insert_at:])
    )


def main(argv: list[str] | None = None) -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("paths", nargs="*", help="modules to annotate; default is every new gap")
    ask.add_argument("--apply", action="store_true")
    ask.add_argument("--limit", type=int, default=0)
    args = ask.parse_args(argv)

    if args.paths:
        targets = [Path(one) for one in args.paths]
    else:
        sys.path.insert(0, str(ROOT / "tools"))
        import check_typed_surface as surface

        _, untyped = surface.scan()
        baseline = set(surface.load_baseline().get("untyped_modules", []))
        targets = [ROOT / rel for rel in sorted(untyped - baseline)]
    if args.limit:
        targets = targets[: args.limit]

    totals = {"resolved": 0, "any": 0, "returns": 0, "skipped": 0, "modules": 0}
    for module_path in targets:
        try:
            rewritten, counted = plan(module_path)
        except (OSError, SyntaxError, ValueError) as exc:
            print(f"   skipped {module_path}: {type(exc).__name__}: {exc}")
            continue
        if rewritten == module_path.read_text("utf-8"):
            continue
        if "Any" in rewritten and _needs_any_import(rewritten):
            rewritten = _with_any_import(rewritten)
        try:
            ast.parse(rewritten)
        except (SyntaxError, ValueError) as exc:
            print(f"   REFUSED {module_path}: rewrite does not parse ({exc})")
            continue
        if args.apply:
            module_path.write_text(rewritten)
        for key, value in counted.items():
            totals[key] += value
        totals["modules"] += 1

    print(
        f"{totals['modules']} module(s): {totals['resolved']} parameter(s) from "
        f"evidence, {totals['any']} left as Any, {totals['returns']} return "
        f"type(s), {totals['skipped']} signature(s) left alone"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
