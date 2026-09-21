"""Lift a cluster of methods out of a class into a mixin module beside it.

Every name the moved code takes from its old module is imported at CALL time,
from that module. A module-level import would bind the origin instead, and a
test that patches the name on the parent module would then patch something the
moved code never reads.

It refuses a method that assigns a module global, one decorated with a name
bound in the parent, one whose default is built from a parent name, and one
whose body starts on its `def` line. The class then inherits the mixin, so
callers and `patch.object` on the class find every method where they were.
This is the mechanism that took the module-size ratchet from 29 refusals to
none; it lived in a session scratchpad until 2026-09-21.

    python tools/lift_methods.py \
        '["path/to/big.py", "ClassName", {"names": ["_a", "_b"]}, \
          "path/to/big_part.py", "_PartMixin", "What the part does."]'
"""
import ast
import json
import pathlib
import sys

HEADER_IMPORTS = (
    "import asyncio\nimport contextlib\nimport json\nimport logging\nimport math\n"
    "import os\nimport random\nimport re\nimport time\n"
    "from collections.abc import Callable, Iterable, Mapping, Sequence\n"
    "from pathlib import Path\nfrom typing import Any\n"
)
HEADER_BOUND = {
    "asyncio", "contextlib", "json", "logging", "math", "os", "random", "re",
    "time", "Callable", "Iterable", "Mapping", "Sequence", "Path", "Any",
}


def module_bound_names(tree):
    bound = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            bound.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.Try):
            for sub in (*n.body, *(h for hh in n.handlers for h in hh.body), *n.orelse):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        bound.add((a.asname or a.name).split(".")[0])
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            bound.add(t.id)
    return bound


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



def _keep_the_strict_allowlist(out: "pathlib.Path", source: "pathlib.Path") -> None:
    """A module lifted out of a strictly-typed one is strictly typed too.

    `config/mypy_strict_files.txt` is the set mypy runs in strict mode over.
    A lift moves a class out of a module on that list into one that is not,
    and mypy then sees the parent subclassing `Any` — which is exactly what
    `core/container.py` did after `_SealsItsKeys` moved to
    `core/container_seal.py`: `make typecheck` red, and the class it was
    strictest about became untyped in one move.
    """
    listing = pathlib.Path("config/mypy_strict_files.txt")
    if not listing.exists():
        return
    try:
        root = pathlib.Path.cwd()
        source_rel = str(source.resolve().relative_to(root))
        out_rel = str(out.resolve().relative_to(root))
    except ValueError:
        return
    lines = listing.read_text(encoding="utf-8").splitlines()
    if source_rel not in lines or out_rel in lines:
        return
    lines.insert(lines.index(source_rel) + 1, out_rel)
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"   added {out_rel} to the mypy strict allowlist, beside {source_rel}")


def run(src_path, cls_name, selector, out_path, mixin_name, doc):
    p = pathlib.Path(src_path)
    src = p.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == cls_name)
    methods = [m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if isinstance(selector, dict) and "names" in selector:
        want = list(selector["names"])
        by_name = {m.name: m for m in methods}
        missing = [n for n in want if n not in by_name]
        if missing:
            print("no such method:", missing)
            return 1
        sel = [by_name[n] for n in want]
    else:
        prefixes = selector if isinstance(selector, list) else selector["prefixes"]
        sel = [m for m in methods if any(m.name.startswith(w) for w in prefixes)]
    if not sel:
        print("no methods matched")
        return 1

    bound = module_bound_names(tree)
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

    def block(m):
        start = (m.decorator_list[0].lineno - 1) if m.decorator_list else (m.lineno - 1)
        end = m.end_lineno
        while end < len(lines) and not lines[end].strip():
            end += 1
        return start, end

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
        start, end = block(m)
        names = sorted({n.id for n in ast.walk(m) if isinstance(n, ast.Name) and n.id in bound}
                       - HEADER_BOUND)
        body = "".join(lines[start:end])
        if names:
            doc0 = ast.get_docstring(m)
            first = m.body[0]
            if first.lineno == m.lineno and not doc0:
                print("REFUSING:", m.name, "body starts on the def line")
                return 1
            at = (first.end_lineno if doc0 else first.lineno - 1) - start
            imp = (f"        from {mod} import (\n"
                   + "".join(f"            {n},\n" for n in names)
                   + "        )\n\n")
            bl = body.splitlines(keepends=True)
            bl.insert(at, imp)
            body = "".join(bl)
        blocks.append((start, end))
        moved.append(body)

    header = (f'"""{doc}\n\nLifted whole out of `{p.stem}`. Every name taken from it is imported at\nCALL time: that module imports this one to build the class, and a test that\npatches a name on it has to reach the code that reads it.\n"""\n'
              "from __future__ import annotations\n\n"
              + HEADER_IMPORTS
              + ("from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n"
                 + f"    from {mod} import (\n"
                 + "".join(f"        {n},\n" for n in sorted(annotated))
                 + "    )\n" if annotated else "")
              + "\n\n"
              f"class {mixin_name}:\n"
              f'    """Lifted whole out of {cls_name}; see {p.name}."""\n\n')
    out = pathlib.Path(out_path)
    out.write_text(header + "".join(moved), encoding="utf-8")
    _annotate_what_moved(out, p)
    _keep_the_strict_allowlist(out, p)

    for start, end in sorted(blocks, key=lambda b: -b[0]):
        del lines[start:end]
    s2 = "".join(lines)
    i = s2.index(f"class {cls_name}")
    j = s2.index("\n", i)
    head = s2[i:j]
    if "(" in head and not head.startswith(f"class {cls_name}:"):
        head = head.replace("(", f"({mixin_name}, ", 1)
    else:
        head = head.replace(":", f"({mixin_name}):", 1)
    s2 = s2[:i] + head + s2[j:]

    imp_line = f"from {mod}_MIX import {mixin_name}\n".replace(
        f"{mod}_MIX", "." + out.stem)
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
    print(f"moved {len(moved)} methods -> {out.name} "
          f"({len(header.splitlines()) + sum(b.count(chr(10)) for b in moved)} lines); "
          f"{p.name} now {''.join(l2).count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    sys.exit(run(*json.loads(sys.argv[1])))


