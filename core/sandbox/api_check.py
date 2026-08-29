"""Check code against a library's real signatures before it runs.

A model writing code against an unfamiliar library invents. It reads a
reference, writes something plausible, and the plausible thing calls a method
that does not exist or passes an argument the function never took. On this
machine each attempt costs a full generation from a resident 27B, so every
invented name is paid for twice: once to write it and once to find out.

The library is right there. Its classes and functions are declared in its
source, and a call that cannot exist can be refused without running anything —
the part of "writing code" that needs no model at all and cannot hallucinate,
because it only ever reads what is written.

Read, never imported
--------------------
Importing the library to inspect it would run its top-level code inside this
process, which is what the sandbox exists to prevent — a check that executes
untrusted code to decide whether untrusted code is safe to execute has given
away the thing it was protecting. Everything here comes from the parse tree.
It also means a library with a dependency that is not installed still gets
checked, and a syntax error in one module costs only that module.

What this settles
-----------------
Names and shapes: an attribute that is not on the class, a name that is not in
the module, a keyword the function does not take, too many positional
arguments. Those are decidable from the source, and they are what invention
looks like.

What it does not settle: whether the code is correct. A ledger entry posted
with the debit and credit the wrong way round parses perfectly. This is the
floor, not the verdict — a class that builds its attributes at runtime, a
decorated function, a name reassigned in a branch, anything it cannot resolve,
it leaves alone rather than guessing. Silence here is "nothing decidable",
never "this is right".
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ApiFinding", "check_code_against_library"]

#: A class whose members are built at runtime cannot be checked by reading it.
_DYNAMIC_MEMBERS = {"__getattr__", "__getattribute__", "__setattr__"}


@dataclass(frozen=True)
class ApiFinding:
    """One thing the code says that the library does not support."""

    line: int
    said: str
    problem: str

    def describe(self) -> str:
        return f"line {self.line}: {self.said} — {self.problem}"


@dataclass(frozen=True)
class _Signature:
    """What a def will accept, as written."""

    positional: tuple[str, ...]
    keywords: frozenset[str]
    takes_varargs: bool
    takes_kwargs: bool
    decorated: bool


@dataclass
class _Definition:
    """A class or a function, as declared."""

    name: str
    signature: _Signature | None = None
    members: dict[str, _Signature | None] = field(default_factory=dict)
    is_class: bool = False
    dynamic: bool = False


def _signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> _Signature:
    args = node.args
    positional = [a.arg for a in (*args.posonlyargs, *args.args) if a.arg != "self"]
    keywords = {a.arg for a in (*args.args, *args.kwonlyargs) if a.arg != "self"}
    return _Signature(
        positional=tuple(positional),
        keywords=frozenset(keywords),
        takes_varargs=args.vararg is not None,
        takes_kwargs=args.kwarg is not None,
        # A decorator can accept anything the wrapped def does not.
        decorated=bool(node.decorator_list),
    )


def _read_module(source: str) -> dict[str, _Definition]:
    """Every public class and function a module declares."""

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    defined: dict[str, _Definition] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined[node.name] = _Definition(node.name, signature=_signature_of(node))
        elif isinstance(node, ast.ClassDef):
            definition = _Definition(node.name, is_class=True)
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definition.members[member.name] = _signature_of(member)
                    if member.name in _DYNAMIC_MEMBERS:
                        definition.dynamic = True
                    if member.name == "__init__":
                        definition.signature = _signature_of(member)
                elif isinstance(member, ast.Assign):
                    for target in member.targets:
                        if isinstance(target, ast.Name):
                            definition.members[target.id] = None
                elif isinstance(member, ast.AnnAssign) and isinstance(
                    member.target, ast.Name
                ):
                    definition.members[member.target.id] = None
            # A base this reader cannot see may carry anything.
            if any(not isinstance(b, ast.Name) or b.id != "object" for b in node.bases):
                definition.dynamic = True
            defined[node.name] = definition
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.setdefault(target.id, _Definition(target.id))
    return defined


def _read_library(library_root: str) -> dict[str, dict[str, _Definition]]:
    """Every module in the directory, by name, without importing one."""

    modules: dict[str, dict[str, _Definition]] = {}
    root = Path(library_root or "").expanduser()
    if not root.is_dir():
        return modules
    for entry in sorted(root.iterdir()):
        if entry.suffix != ".py" or entry.name.startswith("_"):
            continue
        try:
            source = entry.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        modules[entry.stem] = _read_module(source)
    return modules


def _offered(names: object) -> str:
    public = sorted(n for n in names if not str(n).startswith("_"))  # type: ignore[union-attr]
    if not public:
        return "nothing public"
    shown = ", ".join(public[:12])
    return shown + (", ..." if len(public) > 12 else "")


def _check_call_shape(
    signature: _Signature | None, node: ast.Call, said: str, findings: list[ApiFinding]
) -> None:
    """Whether these arguments could bind to the signature as written."""

    if signature is None or signature.decorated:
        return
    for keyword in node.keywords:
        if keyword.arg is None:
            return  # **kwargs at the call site: nothing decidable here
        if not signature.takes_kwargs and keyword.arg not in signature.keywords:
            findings.append(
                ApiFinding(
                    line=node.lineno,
                    said=f"{said}({keyword.arg}=...)",
                    problem=(
                        "takes "
                        + (
                            ", ".join(sorted(signature.keywords))
                            or "no keyword arguments"
                        )
                    ),
                )
            )
    if signature.takes_varargs or any(isinstance(a, ast.Starred) for a in node.args):
        return
    if len(node.args) > len(signature.positional):
        findings.append(
            ApiFinding(
                line=node.lineno,
                said=f"{said}() with {len(node.args)} positional arguments",
                problem=f"takes at most {len(signature.positional)}",
            )
        )


def check_code_against_library(code: str, library_root: str) -> list[ApiFinding]:
    """What this code says that the library does not support.

    Empty when nothing is decidably wrong, which includes the case where
    there is no library to check against. Never raises on the caller's code:
    a syntax error is somebody else's report to make.
    """

    text = str(code or "")
    if not text.strip():
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    library = _read_library(library_root)
    if not library:
        return []

    findings: list[ApiFinding] = []
    #: local name -> what the library says it is
    bound: dict[str, _Definition] = {}
    #: local name -> a whole module of the library
    modules: dict[str, dict[str, _Definition]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")[0]
            declared = library.get(head)
            if declared is None:
                continue  # not this library's business
            for alias in node.names:
                if alias.name == "*":
                    continue
                found = declared.get(alias.name)
                if found is None:
                    findings.append(
                        ApiFinding(
                            line=node.lineno,
                            said=f"from {node.module} import {alias.name}",
                            problem=f"{head} offers {_offered(declared)}",
                        )
                    )
                    continue
                bound[alias.asname or alias.name] = found
        elif isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                if head in library:
                    modules[alias.asname or head] = library[head]

    # What a local name was assigned from, one level deep: `L = Ledger(...)`
    # makes L an instance of Ledger, which is what a method call needs.
    # A name assigned more than once is left alone — the second assignment
    # may be anything, and a wrong guess here reads as an invented call.
    assigned_once: dict[str, ast.expr | None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assigned_once[target.id] = (
                    None if target.id in assigned_once else node.value
                )
        elif isinstance(node, (ast.For, ast.With, ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                    assigned_once[inner.id] = None

    instances: dict[str, _Definition] = {}
    for name, value in assigned_once.items():
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        made = None
        if isinstance(func, ast.Name):
            made = bound.get(func.id)
            said = func.id
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            declared = modules.get(func.value.id)
            made = declared.get(func.attr) if declared else None
            said = f"{func.value.id}.{func.attr}"
        if made is None or not made.is_class:
            continue
        instances[name] = made
        _check_call_shape(made.signature, value, said, findings)

    def _what_it_is(expr: ast.expr) -> tuple[str, _Definition | None]:
        """The library thing this expression evaluates to, if it is decidable.

        A name the code bound, a variable it assigned once, or a class called
        on the spot — ``Ledger('acme').post(...)`` names its own type, and a
        method call written that way is as checkable as one through a
        variable. Anything else is not decidable and comes back empty.
        """

        if isinstance(expr, ast.Name):
            return expr.id, instances.get(expr.id) or bound.get(expr.id)
        if isinstance(expr, ast.Call):
            inner = expr.func
            if isinstance(inner, ast.Name):
                made = bound.get(inner.id)
                if made is not None and made.is_class:
                    return f"{inner.id}()", made
            elif isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name):
                declared = modules.get(inner.value.id)
                made = declared.get(inner.attr) if declared else None
                if made is not None and made.is_class:
                    return f"{inner.value.id}.{inner.attr}()", made
        return "", None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            declared = bound.get(func.id)
            if declared is not None and not declared.is_class:
                _check_call_shape(declared.signature, node, func.id, findings)
            continue
        if not isinstance(func, ast.Attribute):
            continue
        if isinstance(func.value, ast.Name) and func.value.id in modules:
            owner = func.value.id
            said = f"{owner}.{func.attr}"
            declared = modules[owner]
            if func.attr not in declared:
                findings.append(
                    ApiFinding(
                        line=node.lineno,
                        said=said,
                        problem=f"{owner} offers {_offered(declared)}",
                    )
                )
                continue
            _check_call_shape(declared[func.attr].signature, node, said, findings)
            continue
        name, definition = _what_it_is(func.value)
        if definition is None or not definition.is_class or definition.dynamic:
            continue
        said = f"{name}.{func.attr}"
        if func.attr not in definition.members:
            findings.append(
                ApiFinding(
                    line=node.lineno,
                    said=said,
                    problem=f"{definition.name} offers {_offered(definition.members)}",
                )
            )
            continue
        _check_call_shape(definition.members[func.attr], node, said, findings)

    return findings
