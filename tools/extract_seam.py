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
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


#: The value a moved block's nested function returns when it fell through
#: rather than returning early. A module-level object, never a literal, so no
#: value the block could legitimately return can be mistaken for it.
_SENTINEL = "_SEAM_FELL_THROUGH"
_SENTINEL_DEFINITION = (
    "#: Returned by an extracted block that did NOT return early. A unique\n"
    "#: object, so no value a block legitimately returns can be mistaken for it.\n"
    f"{_SENTINEL} = object()\n"
)
_EARLY = "_seam_early_response"


def _render_nested_helper(
    *,
    name: str,
    prefix: str,
    signature: str,
    escapes: list[str],
    rebound: list[str],
    body: str,
    summary: str,
    function: str,
    reads: list[str],
) -> str:
    """A helper whose body is the block, inside a function that can return.

    The block returns out of its middle, so it cannot simply become the tail
    of a function — the caller has to learn the difference between "the block
    returned this" and "the block finished". Putting the block in a nested
    function makes its own ``return`` the signal, and leaves the block text
    untouched, which is what keeps the token-for-token check meaningful.

    ``nonlocal`` covers every name the block assigns that is also a parameter,
    not only the ones that escape: without it an assignment inside the nested
    function would shadow the parameter and the first read before that
    assignment would raise.
    """
    returned = ", ".join([_EARLY, *escapes]) if escapes else _EARLY
    annotation = (
        f"tuple[{', '.join('Any' for _ in range(len(escapes) + 1))}]"
        if escapes
        else "Any"
    )
    nonlocal_line = f"        nonlocal {', '.join(rebound)}\n" if rebound else ""
    return (
        f"{prefix}def {name}(\n"
        "    *,\n"
        f"{signature}"
        f") -> {annotation}:\n"
        f'    """{summary}\n'
        "\n"
        f"    Moved out of ``{function}`` by tools/extract_seam.py, which checks\n"
        "    the body against the original token for token before writing. The\n"
        f"    block returns early, so it sits in a nested function and {_SENTINEL}\n"
        f"    means it finished instead. It reads {len(reads)} name(s) and hands back\n"
        f"    {len(escapes)}.\n"
        '    """\n'
        f"    {prefix}def _block() -> Any:\n"
        f"{nonlocal_line}"
        + textwrap.indent(body, "        ")
        + f"        return {_SENTINEL}\n"
        "\n"
        f"    {_EARLY} = {'await ' if prefix else ''}_block()\n"
        f"    return {returned}\n"
    )


def _load_seam_tools() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_aura_seam_tools", ROOT / "tools" / "find_extraction_seam.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("tools/find_extraction_seam.py could not be loaded")
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
                for handler in getattr(statement, "handlers", None) or []:
                    if visit(list(handler.body)):
                        return True
        if run and run[0].lineno == start and (
            getattr(run[-1], "end_lineno", run[-1].lineno) or run[-1].lineno
        ) == end:
            found.extend(run)
            return True
        return False

    visit(list(getattr(fn, "body", [])))
    return found


def _body_chain(
    fn: ast.AST, start: int, end: int
) -> list[list[ast.stmt]]:
    """The bodies enclosing the block, outermost first.

    Used to work out what has certainly run by the time control reaches the
    call site: the statements before the block in its own body, and before
    each of its ancestors in theirs.
    """
    chain: list[list[ast.stmt]] = []

    def visit(body: list[ast.stmt]) -> bool:
        for statement in body:
            first = statement.lineno
            last = getattr(statement, "end_lineno", first) or first
            if first >= start and last <= end:
                chain.append(body)
                return True
            if first <= start and last >= end:
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(statement, field, None)
                    if isinstance(nested, list) and nested and visit(nested):
                        chain.append(body)
                        return True
                for handler in getattr(statement, "handlers", None) or []:
                    if visit(list(handler.body)):
                        chain.append(body)
                        return True
        return False

    visit(list(getattr(fn, "body", [])))
    chain.reverse()
    return chain


def _guaranteed_at_call_site(
    fn: ast.AST, start: int, end: int, tools: Any
) -> set[str]:
    """Names that certainly hold a value where the extracted call will sit.

    A parameter of the enclosing function always does. Beyond that, only what
    the statements before the block bind on every path — which is a different
    question from "assigned somewhere above", and the difference is the bug
    this tool shipped once.
    """
    names: set[str] = set()
    arguments = getattr(fn, "args", None)
    if arguments is not None:
        for group in (arguments.posonlyargs, arguments.args, arguments.kwonlyargs):
            names |= {a.arg for a in group}
        for special in (arguments.vararg, arguments.kwarg):
            if special is not None:
                names.add(special.arg)

    for body in _body_chain(fn, start, end):
        prefix: list[ast.stmt] = []
        for statement in body:
            first = statement.lineno
            last = getattr(statement, "end_lineno", first) or first
            if last >= start:
                break
            prefix.append(statement)
        names |= tools.must_bind(prefix)
    return names


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

    # An escape is safe under exactly one of two conditions: the block binds it
    # on EVERY path through itself, or the caller is certain to hold a value
    # for it where the call will sit, in which case it is passed in and handed
    # back unchanged on the paths that do not touch it.
    #
    # The first version of this asked whether the name was bound anywhere
    # earlier in the function. That is a third, wrong question, and it produced
    # a live crash: `hard_final_quality_failed` was assigned earlier inside a
    # branch, so the tool called it safe, and the extracted helper reached its
    # `return` with the name unassigned on the path where the branch did not
    # fire — UnboundLocalError on the first turn that took it.
    settled_inside = tools.must_bind(statements)
    held_by_caller = _guaranteed_at_call_site(fn, start, end, tools)
    certain = held_by_caller | set(reads)

    # Every input becomes an argument, and arguments are evaluated before the
    # call. A name the block reads only down one of its own branches was read
    # only when it had a value; as an argument it is read every time.
    #
    # That is how the second regression happened. `salvage_contract` was
    # assigned inside `if salvaged_no_reply:` and read inside the same guard
    # further down. Moving the guarded part into a helper hoisted the read into
    # the call, and the turn that took the other branch raised
    # UnboundLocalError before the helper ran at all.
    unsafe_reads = sorted(n for n in reads if n not in held_by_caller)
    if unsafe_reads:
        refusals.append(
            f"{unsafe_reads} are read by the block but are not certain to hold a "
            "value where the call will sit; passing them as arguments would read "
            "them on paths the original never did"
        )

    carried = [n for n in escapes if n not in settled_inside]
    unsafe = sorted(n for n in carried if n not in certain)
    if unsafe:
        refusals.append(
            f"{unsafe} escape the block without being bound on every path through "
            "it, and the caller is not certain to hold a value for them either; "
            "returning them would raise UnboundLocalError where the original kept "
            "an earlier value"
        )

    # A block that returns can still be moved, into a nested function whose
    # own `return` the helper reads. The block text is unchanged, so the
    # token-for-token proof still applies; what makes it safe is that every
    # name the block hands back is also a name it was given, so the early
    # path returns values that already existed rather than defaults.
    nested = bool(returns)
    if nested:
        # An early return leaves the block before its own assignments, so
        # every name it hands back has to have come in with it.
        early_unsafe = sorted(n for n in escapes if n not in certain)
        if early_unsafe:
            refusals.append(
                f"the block returns early and {early_unsafe} are not certain to "
                "hold a value where the call will sit, so the early path would "
                "hand back names that were never set"
            )
    if nested and any(
        isinstance(n, (ast.Yield, ast.YieldFrom)) for st in statements for n in ast.walk(st)
    ):
        refusals.append("a generator body cannot be moved into a nested function")

    if refusals:
        print(f"refusing to extract {start}-{end}:")
        for refusal in refusals:
            print(f"   • {refusal}")
        return 1

    block = "".join(lines[start - 1 : end])
    indent = len(block) - len(block.lstrip(" "))
    body = textwrap.dedent(block)

    # A carried escape is one the block may not set, so the helper takes it as
    # a parameter and returns it unchanged on those paths.
    parameters = (
        sorted(set(reads) | set(escapes))
        if nested
        else sorted(set(reads) | set(carried))
    )
    signature = "".join(f"    {n}: Any,\n" for n in parameters)
    prefix = "async " if is_async else ""
    awaited = "await " if is_async else ""
    rebound = sorted((set(parameters) & bound) | set(escapes))
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
    print(f"seam {start}-{end} in {function}")
    summary = doc or f"Extracted from ``{function}``."
    if nested:
        helper = _render_nested_helper(
            name=name,
            prefix=prefix,
            signature=signature,
            escapes=escapes,
            rebound=rebound,
            body=body,
            summary=summary,
            function=function,
            reads=parameters,
        )
        pad = " " * indent
        call_arguments = "".join(f"{pad}    {n}={n},\n" for n in parameters)
        unpack = ", ".join([_EARLY, *escapes]) if escapes else _EARLY
        call = (
            f"{pad}{unpack} = {awaited}{name}(\n"
            f"{call_arguments}"
            f"{pad})\n"
            f"{pad}if {_EARLY} is not {_SENTINEL}:\n"
            f"{pad}    return {_EARLY}\n"
        )
        return _finish(
            path=path,
            fn=fn,
            lines=lines,
            start=start,
            end=end,
            helper=helper,
            call=call,
            body=body,
            reads=reads,
            escapes=escapes,
            apply=apply,
            nested=True,
        )

    call_names = parameters
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
    call_arguments = "".join(f"{pad}    {n}={n},\n" for n in call_names)
    assignment = "" if not escapes else f"{', '.join(escapes)} = "
    call = (
        f"{pad}{assignment}{awaited}{name}(\n"
        f"{call_arguments}"
        f"{pad})\n"
    )

    return _finish(
        path=path,
        fn=fn,
        lines=lines,
        start=start,
        end=end,
        helper=helper,
        call=call,
        body=body,
        reads=parameters,
        escapes=escapes,
        apply=apply,
        nested=False,
    )


def _module_level_insertion_point(source: str, function: str, fn: ast.AST) -> int:
    """Where a module-level helper can go, above the thing it came out of.

    For a plain function that is the function itself. For a method it is the
    class that holds it, because a helper written at column zero cannot live
    inside a class body.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    def offset_of(line_number: int) -> int:
        return sum(len(line) for line in lines[: line_number - 1])

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == function
                ):
                    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    return offset_of(first)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function
        ):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return offset_of(first)

    marker = f"{'async ' if isinstance(fn, ast.AsyncFunctionDef) else ''}def {function}("
    return source.index(marker)


def _finish(
    *,
    path: Path,
    fn: ast.AST,
    lines: list[str],
    start: int,
    end: int,
    helper: str,
    call: str,
    body: str,
    reads: list[str],
    escapes: list[str],
    apply: bool,
    nested: bool,
) -> int:
    """Prove the move, then write it.

    The proof is the reason to use a tool instead of an editor: the helper's
    body has to be the original block, token for token, or this is a rewrite
    wearing a refactor's name.
    """
    helper_tree = ast.parse(helper)
    helper_fn = helper_tree.body[0]
    if not isinstance(helper_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        print("error: the rendered helper is not a function", file=sys.stderr)
        return 3
    if nested:
        # docstring, nested def, assignment, return — the block is the nested
        # function's body minus its appended sentinel return.
        inner = next(
            node
            for node in helper_fn.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        moved_body = [n for n in inner.body if not isinstance(n, ast.Nonlocal)][:-1]
    else:
        moved_body = helper_fn.body[1:] if not escapes else helper_fn.body[1:-1]

    original_tokens = _normalised_tokens(body)
    if not original_tokens:
        print("error: the block is empty", file=sys.stderr)
        return 3
    moved = ast.unparse(ast.Module(body=list(moved_body), type_ignores=[]))
    if _normalised_tokens(ast.unparse(ast.parse(body))) != _normalised_tokens(moved):
        print("error: the helper body is not the original block", file=sys.stderr)
        return 3

    print(f"  reads   ({len(reads)}): {', '.join(reads)}")
    print(f"  escapes ({len(escapes)}): {', '.join(escapes)}")
    print(f"  {end - start + 1} lines out, {call.count(chr(10))} back"
          + (" (early return, nested)" if nested else ""))

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
    #
    # A method's enclosing CLASS is the boundary, not the method: putting a
    # module-level helper immediately above `    def generate(` lands it inside
    # the class body at zero indentation, and the file stops parsing on the
    # helper's own docstring.
    name_of_fn = getattr(fn, "name", "")
    insert_at = _module_level_insertion_point(rewritten, name_of_fn, fn)
    rewritten = rewritten[:insert_at] + helper + "\n\n" + rewritten[insert_at:]

    if nested and f"\n{_SENTINEL} = object()" not in rewritten:
        rewritten = _add_sentinel(rewritten)

    # The generated signature annotates with `Any`. Most files this runs on
    # already import it; one that does not would gain a helper that raises
    # NameError at import, which a "behaviour-preserving move" must not do.
    if not _imports_any(rewritten):
        rewritten = _add_any_import(rewritten)

    ast.parse(rewritten)
    path.write_text(rewritten, encoding="utf-8")
    print(f"✅ wrote {path}")
    return 0


def _add_sentinel(source: str) -> str:
    """Define the fell-through marker once, after the imports."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, getattr(node, "end_lineno", node.lineno) or node.lineno)
    lines.insert(last, "\n" + _SENTINEL_DEFINITION)
    return "".join(lines)


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
