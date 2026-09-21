"""Lift a cluster of module-level functions into a sibling module.

The parent imports them straight back, so every caller and every patch that
names them on the parent still finds them. What the moved code takes FROM the
parent is imported at call time, for the same reason.

The module-size ratchet (`tools/lint_module_size.py`) holds one number, the
lines above the ceiling summed across the tree, and this is the move it exists
to allow. It lived in a session scratchpad until 2026-09-21.

    python tools/lift_module_functions.py \
        '["path/to/big.py", ["fn_a", "fn_b"], "path/to/big_part.py", "What the part is."]'
"""
import ast
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lift_methods import HEADER_BOUND, HEADER_IMPORTS, module_bound_names  # noqa: E402


def _annotate_what_moved(out: "pathlib.Path", source: "pathlib.Path") -> None:
    """Give the lifted module the annotations its new file now needs.

    A method untyped inside a grandfathered God object is untyped in a NEW
    module too, and `make typed-surface` refuses a new module that is —
    correctly, because the lift moved the debt rather than paying it. One
    lift of twenty-five clusters left 174 such modules and 724 definitions.

    The types are recoverable from the call sites the lift left behind, so
    they are recovered here rather than discovered by the gate afterwards.
    """
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    try:
        from annotate_extracted_seams import plan
    except ImportError:
        print("   (annotator unavailable; the lifted module keeps its bare signatures)")
        return
    try:
        rewritten, counted = plan(out)
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"   (annotation skipped: {type(exc).__name__}: {exc})")
        return
    if rewritten != out.read_text("utf-8"):
        out.write_text(rewritten, encoding="utf-8")
    print(
        f"   annotated {counted['resolved']} parameter(s) from the call sites, "
        f"{counted['any']} as Any, {counted['returns']} return type(s)"
    )


def run(src_path, names, out_path, doc):
    p = pathlib.Path(src_path)
    src = p.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [n for n in names if n not in top]
    if missing:
        print("no such function:", missing)
        return 1
    sel = [top[n] for n in names]
    bound = module_bound_names(tree)
    moved_names = set(names)

    for m in sel:
        if any(isinstance(n, ast.Global) for n in ast.walk(m)):
            print("REFUSING:", m.name, "mutates a module global")
            return 1
        for dec in m.decorator_list:
            for n in ast.walk(dec):
                if isinstance(n, ast.Name) and n.id in bound and n.id not in HEADER_BOUND:
                    print("REFUSING:", m.name, "decorated with the parent's", n.id)
                    return 1

    mod = "." + p.stem

    def _names_in(nodes):
        out = set()
        for node in nodes:
            if node is None:
                continue
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and n.id in bound and n.id not in HEADER_BOUND:
                    out.add(n.id)
        return out

    annotated = set()
    for m in sel:
        args = m.args
        every = [*args.args, *args.posonlyargs, *args.kwonlyargs, args.vararg, args.kwarg]
        annotated |= _names_in([a.annotation for a in every if a is not None])
        annotated |= _names_in([m.returns])
        defaults = _names_in([*args.defaults, *args.kw_defaults])
        if defaults:
            print("REFUSING:", m.name, "has a default built from", sorted(defaults))
            return 1

    blocks, moved = [], []
    for m in sorted(sel, key=lambda x: x.lineno):
        start = (m.decorator_list[0].lineno - 1) if m.decorator_list else (m.lineno - 1)
        end = m.end_lineno
        while end < len(lines) and not lines[end].strip():
            end += 1
        used = sorted({n.id for n in ast.walk(m) if isinstance(n, ast.Name) and n.id in bound}
                      - HEADER_BOUND - {m.name})
        body = "".join(lines[start:end])
        if used:
            doc0 = ast.get_docstring(m)
            first = m.body[0]
            if first.lineno == m.lineno and not doc0:
                print("REFUSING:", m.name, "body starts on the def line")
                return 1
            at = (first.end_lineno if doc0 else first.lineno - 1) - start
            imp = (f"    from {mod} import (\n"
                   + "".join(f"        {n},\n" for n in used)
                   + "    )\n\n")
            bl = body.splitlines(keepends=True)
            bl.insert(at, imp)
            body = "".join(bl)
        blocks.append((start, end))
        moved.append(body)

    out = pathlib.Path(out_path)
    header = (f'"""{doc}\n\nLifted whole out of `{p.stem}`, which imports them straight back: every\ncaller and every patch that names them there still finds them. What they\ntake from that module is imported at CALL time, for the same reason.\n"""\n'
              "from __future__ import annotations\n\n"
              + HEADER_IMPORTS
              + ("from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n"
                 + f"    from {mod} import (\n"
                 + "".join(f"        {n},\n" for n in sorted(annotated))
                 + "    )\n" if annotated else "")
              + "\n\n")
    out.write_text(header + "".join(moved), encoding="utf-8")
    _annotate_what_moved(out, p)

    for start, end in sorted(blocks, key=lambda b: -b[0]):
        del lines[start:end]
    s2 = "".join(lines)
    imp_line = (f"from .{out.stem} import (  # noqa: F401  (re-exported: they were defined here)\n"
                + "".join(f"    {n},\n" for n in sorted(moved_names))
                + ")\n")
    t2 = ast.parse(s2)
    anchor = None
    for n in t2.body:
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            continue
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            anchor = n
            break
    if anchor is None:
        print("no import anchor")
        return 1
    l2 = s2.splitlines(keepends=True)
    l2.insert(anchor.lineno - 1, imp_line)
    p.write_text("".join(l2), encoding="utf-8")
    print(f"moved {len(moved)} functions -> {out.name} "
          f"({(header + ''.join(moved)).count(chr(10))} lines); "
          f"{p.name} now {''.join(l2).count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    sys.exit(run(*json.loads(sys.argv[1])))


