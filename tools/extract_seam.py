#!/usr/bin/env python3
"""Move a block out of an oversized function, mechanically and checkably.

Twelve thousand lines of this repository sit inside ten functions, and the
reason none of it has moved is not that nobody knows where to cut. The seams
are measured — ``tools/find_extraction_seam.py`` prints them with their
inputs, their outputs and their blockers. The reason is that each cut is a
hundred lines of careful hand-editing on a live serving path, and hand-editing
is where a behaviour-preserving move stops being one.

So this does the edit. Given a function and a line range it:

* refuses the cut unless the block is separable — no ``return`` out of the
  middle, no ``yield``, no ``break``/``continue`` targeting a loop outside it,
  and no name that escapes while being bound only on some paths, because
  giving such a name a default turns a possible ``UnboundLocalError`` into a
  value and that is a behaviour CHANGE, not a move;
* computes the block's free variables in EVALUATION order, so a multi-line
  tuple assignment whose targets sit above the call that reads them is
  correctly seen as reading them;
* writes a helper whose body is the block, dedented and otherwise untouched;
* replaces the block with a call and an unpacking;
* re-parses both and asserts the helper's body is the original block, token
  for token.

The last step is the point. A refactoring tool that cannot prove it moved the
code is a refactoring tool that has rewritten it.

    python tools/extract_seam.py interface/routes/chat.py::_api_chat_turn \\
        --range 21328-21428 --name _apply_final_quality_gate --async
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import sys
import textwrap
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_seam_tools():
    spec = importlib.util.spec_from_file_location(
        "_aura_seam_tools", ROOT / "tools" / "find_extraction_seam.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_aura_seam_tools"] = module
    spec.loader.exec_module(module)
    return module


def _statements_in_range(fn: ast.AST, start: int, end: int) -> list[ast.stmt]:
    """The statements whose lines are exactly the requested range.

    A range that starts or ends inside a statement is not a block, and cutting
    one would produce a helper that compiles and means something else.
    """
    found: list[ast.stmt] = []

    def visit(body: list[ast.stmt]) -> bool:
        run: list[ast.stmt] = []
        for statement in body:
            first = statement.lineno
            last = getattr(statement, "end_lineno", first) or first
            if first >= start and last <= end:
                run.append(statement)
            elif first <= start and last >= end:
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(statement, field, None)
                    if isinstance(nested, list) and nested and visit(nested):
                        return True
                for handler in getattr(statement, "handlers", []) or []:
                    if visit(handler.body):
                        return True
        if run and run[0].lineno == start and (
            getattr(run[-1], "end_lineno", run[-1].lineno) or run[-1].lineno
        ) == end:
            found.extend(run)
            return True
        return False

    visit(list(getattr(fn, "body", [])))
    return found


def _normalised_tokens(source: str) -> list[tuple[int, str]]:
    """Tokens with layout removed, so two texts can be compared for meaning."""
    out: list[tuple[int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in {
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
            tokenize.ENDMARKER,
            tokenize.ENCODING,
        }:
            continue
        out.append((token.type, token.string))
    return out


def _imports_any(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            if any(alias.name == "Any" for alias in node.names):
                return True
    return False


def _add_any_import(source: str) -> str:
    """Put `from typing import Any` after the last top-level import."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, getattr(node, "end_lineno", node.lineno) or node.lineno)
    if last == 0:
        # No imports at all: after the module docstring, if there is one.
        first = tree.body[0] if tree.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            last = getattr(first, "end_lineno", first.lineno) or first.lineno
    lines.insert(last, "\nfrom typing import Any\n")
    return "".join(lines)


def extract(
    path: Path,
    function: str,
    start: int,
    end: int,
    name: str,
    *,
    is_async: bool,
    apply: bool,
    doc: str = "",
) -> int:
    tools = _load_seam_tools()
    source = path.read_text("utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == function.split(".")[-1]
        ),
        None,
    )
    if fn is None:
        print(f"error: no function named {function!r} in {path}", file=sys.stderr)
        return 2

    statements = _statements_in_range(fn, start, end)
    if not statements:
        print(
            f"error: lines {start}-{end} are not a whole run of statements inside "
            f"{function}",
            file=sys.stderr,
        )
        return 2

    refusals: list[str] = []
    returns = [n for s in statements for n in ast.walk(s) if isinstance(n, ast.Return)]
    if returns:
        refusals.append(f"{len(returns)} return(s) out of the middle of the block")
    if any(
        isinstance(n, (ast.Yield, ast.YieldFrom)) for s in statements for n in ast.walk(s)
    ):
        refusals.append("the block yields; it is a generator body")
    jumps = sum(tools._jump_count(s) for s in statements)
    if jumps:
        refusals.append(f"{jumps} break/continue targeting a loop outside the block")

    module_scope = tools._module_scope(tree)
    free, bound = tools.free_variables(statements)
    reads = sorted(free - module_scope - tools._BUILTINS)

    after: set[str] = set()
    seen_end = False
    for statement in ast.walk(fn):
        if not isinstance(statement, ast.stmt):
            continue
        if statement.lineno > end:
            after |= {
                n.id
                for n in ast.walk(statement)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            seen_end = True
    escapes = sorted(bound & after)

    # An escape is safe when the block binds it on EVERY path, or when it is
    # already an input, so the helper receives a value and hands it back.
    #
    # The first version of this asked whether the name was bound anywhere
    # earlier in the function. That is a different question and it produced a
    # live crash: `hard_final_quality_failed` was assigned earlier inside a
    # branch, so the tool called it safe, and the extracted helper reached its
    # `return` with the name unassigned on the path where the branch did not
    # fire — UnboundLocalError on the first turn that took it.
    guaranteed = tools.must_bind(statements) | set(reads)
    conditional = [n for n in escapes if n not in guaranteed]
    if conditional:
        refusals.append(
            f"{conditional} escape the block without being bound on every path "
            "through it; returning them would raise UnboundLocalError where the "
            "original kept an earlier value"
        )

    if refusals:
        print(f"refusing to extract {start}-{end}:")
        for refusal in refusals:
            print(f"   • {refusal}")
        return 1

    block = "".join(lines[start - 1 : end])
    indent = len(block) - len(block.lstrip(" "))
    body = textwrap.dedent(block)

    signature = "".join(f"    {n}: Any,\n" for n in reads)
    prefix = "async " if is_async else ""
    awaited = "await " if is_async else ""
    returns_one = len(escapes) == 1
    if not escapes:
        # A block that hands nothing back is the cleanest cut there is: it
        # runs for its effects and the caller needs no unpacking.
        annotation = "None"
        returned = "None"
    elif returns_one:
        annotation = "Any"
        returned = escapes[0]
    else:
        annotation = f"tuple[{', '.join('Any' for _ in escapes)}]"
        returned = ", ".join(escapes)
    summary = doc or f"Extracted from ``{function}``."
    helper = (
        f"{prefix}def {name}(\n"
        "    *,\n"
        f"{signature}"
        f") -> {annotation}:\n"
        f'    """{summary}\n'
        "\n"
        f"    Moved out of ``{function}`` by tools/extract_seam.py, which\n"
        "    checks the body against the original token for token before\n"
        f"    writing. It reads {len(reads)} name(s) from the turn and hands back\n"
        f"    {len(escapes)}.\n"
        '    """\n'
        + textwrap.indent(body, "    ")
        + ("" if not escapes else f"    return {returned}\n")
    )

    pad = " " * indent
    call_arguments = "".join(f"{pad}    {n}={n},\n" for n in reads)
    assignment = "" if not escapes else f"{', '.join(escapes)} = "
    call = (
        f"{pad}{assignment}{awaited}{name}(\n"
        f"{call_arguments}"
        f"{pad})\n"
    )

    # Proof: the helper's body is the block.
    helper_tree = ast.parse(helper)
    helper_fn = helper_tree.body[0]
    helper_body = helper_fn.body[1:] if not escapes else helper_fn.body[1:-1]
    original_tokens = _normalised_tokens(body)
    moved = ast.unparse(ast.Module(body=list(helper_body), type_ignores=[]))
    if _normalised_tokens(ast.unparse(ast.parse(body))) != _normalised_tokens(moved):
        print("error: the helper body is not the original block", file=sys.stderr)
        return 3
    if not original_tokens:
        print("error: the block is empty", file=sys.stderr)
        return 3

    print(f"seam {start}-{end} in {function}")
    print(f"  reads   ({len(reads)}): {', '.join(reads)}")
    print(f"  escapes ({len(escapes)}): {', '.join(escapes)}")
    print(f"  {end - start + 1} lines out, {call.count(chr(10))} back")

    if not apply:
        print("\n--- helper ---")
        print(helper)
        print("--- call site ---")
        print(call)
        print("(dry run: pass --apply to write)")
        return 0

    rewritten = "".join(lines[: start - 1]) + call + "".join(lines[end:])

    # The insertion point is found in the REWRITTEN text, not the original.
    # Computing it once against `source` and then adjusting by a character
    # offset using a LINE number produced an index in the middle of a token
    # and a file that began `asynasync def`.
    marker = f"{'async ' if isinstance(fn, ast.AsyncFunctionDef) else ''}def {fn.name}("
    insert_at = rewritten.index(marker)
    head = rewritten[:insert_at]
    decorator = head.rfind("\n@")
    if decorator != -1 and head[decorator:].count("\n") < 10:
        insert_at = decorator + 1
    rewritten = rewritten[:insert_at] + helper + "\n\n" + rewritten[insert_at:]

    # The generated signature annotates with `Any`. Most files this runs on
    # already import it; one that does not would gain a helper that raises
    # NameError at import, which a "behaviour-preserving move" must not do.
    if not _imports_any(rewritten):
        rewritten = _add_any_import(rewritten)

    ast.parse(rewritten)
    path.write_text(rewritten, encoding="utf-8")
    print(f"✅ wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="path/to/file.py::function_name")
    parser.add_argument("--range", required=True, help="START-END, inclusive")
    parser.add_argument("--name", required=True, help="the helper's name")
    parser.add_argument("--async", dest="is_async", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--doc", default="", help="the helper's one-line docstring")
    args = parser.parse_args(argv)

    rel, _, function = args.target.partition("::")
    start, _, end = args.range.partition("-")
    return extract(
        ROOT / rel,
        function,
        int(start),
        int(end),
        args.name,
        is_async=args.is_async,
        apply=args.apply,
        doc=args.doc,
    )


if __name__ == "__main__":
    raise SystemExit(main())
