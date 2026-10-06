"""Python drafts checked against independent examples in the existing OS sandbox."""
from __future__ import annotations

import ast
import asyncio
import json
import time
import io
import tokenize
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field


class FunctionExample(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    function: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    args: list[Any] = Field(default_factory=list, max_length=32)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    expected: Any


def examples_in(source: str) -> list[FunctionExample]:
    """Literal function-call examples in docstrings provide pinned inputs and expectations."""
    import doctest

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # A syntax fault outside an intact docstring must not erase its
        # original examples. Tokenisation reads literals without executing.
        docs = []
        try:
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.STRING and ('"""' in token.string[:6] or "'''" in token.string[:6]):
                    value = ast.literal_eval(token.string)
                    if isinstance(value, str):
                        docs.append(value)
        except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
            pass
    else:
        docs = [ast.get_docstring(node) or "" for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    examples = []
    for doc in docs:
        try:
            parsed = doctest.DocTestParser().get_examples(doc)
        except ValueError:
            continue
        for example in parsed:
            try:
                call = ast.parse(example.source.strip(), mode="eval").body
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                    continue
                if any(k.arg is None for k in call.keywords):
                    continue
                examples.append(FunctionExample(name=f"example {len(examples)+1}", function=call.func.id,
                    args=[ast.literal_eval(a) for a in call.args],
                    kwargs={k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg is not None},
                    expected=ast.literal_eval(example.want.strip())))
            except (SyntaxError, ValueError, TypeError):
                continue
    return examples[:32]


def same_result(actual: Any, expected: Any) -> bool:
    """JSON values retain their types; True is not the integer 1."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_result(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_result(actual[k], expected[k]) for k in expected)
    return actual == expected


async def check_python(source: str, examples: list[FunctionExample] | None = None, *, timeout_s: float = 10) -> dict[str, Any]:
    """Compile first; execute functions out of process and compare results outside the candidate."""
    from core.sandbox.untrusted_python import call_untrusted_function

    try:
        ast.parse(source)
    except SyntaxError as exc:
        return {"syntax": False, "verified": False, "error": f"line {exc.lineno}: {exc.msg}", "cases": []}
    examples = examples_in(source) if examples is None else examples
    if not examples:
        return {"syntax": True, "verified": False, "cases": [], "unmeasured": "no independent function examples"}
    if len(examples) > 32:
        raise ValueError("at most 32 independent examples may be checked")
    json.dumps([e.model_dump() for e in examples], allow_nan=False)
    groups: dict[str, list[FunctionExample]] = defaultdict(list)
    for example in examples:
        groups[example.function].append(example)
    cases = []
    deadline = time.monotonic() + min(30.0, max(0.1, float(timeout_s)) * len(groups))
    for function, group in groups.items():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cases.extend({"name": e.name, "function": function, "args": e.args,
                          "expected": e.expected, "actual": None, "verdict": "unmeasured",
                          "sandboxed": False, "boundary": "", "error": "function check budget exhausted"} for e in group)
            continue
        outcome = await asyncio.to_thread(call_untrusted_function, source, function,
            [{"args": e.args, "kwargs": e.kwargs} for e in group], timeout_s=min(timeout_s, remaining),
            require_boundary=True, source="checking_python")
        for index, example in enumerate(group):
            measured = outcome.ok and outcome.sandboxed and index < len(outcome.results)
            value = outcome.results[index] if measured else None
            cases.append({"name": example.name, "function": function, "args": example.args,
                          "expected": example.expected, "actual": value,
                          "verdict": "right" if measured and same_result(value, example.expected) else "wrong" if measured else "unmeasured",
                          "sandboxed": outcome.sandboxed, "boundary": outcome.boundary,
                          "error": outcome.error or outcome.stderr[-1000:]})
    return {"syntax": True, "verified": bool(cases) and all(c["verdict"] == "right" for c in cases), "cases": cases}
