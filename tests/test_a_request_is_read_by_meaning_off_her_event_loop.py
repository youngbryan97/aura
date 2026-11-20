"""Whether a request is screen work, or asks her to find something out, is read off her event loop.

Both are read by meaning on her encoder (core/language/search_request.py), so the first reading of a sentence is a
forward pass. LIVE 20 November 2026, 07:24, lockdep: the unitary phase read its objective inline and held her event
loop 164 ms, the thirteenth time from that line since 17 November; the kernel's dispatch did the same on 18 November.
The encoder keeps what it has read, so only the first reading of a string blocks, and lockdep saw only those two.
Fifteen more coroutines called these readers inline. Each now reads them on a thread, and this test fails on a new one.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
READERS = frozenset({"looks_like_desktop_objective", "_looks_like_desktop_objective", "read_search_request",
                     "asks_to_find_out"})


def _inline_calls_in_coroutines(source: str) -> list[int]:
    found: list[int] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.in_coroutine: list[bool] = []

        def _scope(self, node: ast.AST, coroutine: bool) -> None:
            self.in_coroutine.append(coroutine)
            self.generic_visit(node)
            self.in_coroutine.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._scope(node, False)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            self._scope(node, False)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._scope(node, True)

        def visit_Call(self, node: ast.Call) -> None:
            called = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if called in READERS and self.in_coroutine and self.in_coroutine[-1]:
                found.append(node.lineno)
            self.generic_visit(node)

    _Visitor().visit(ast.parse(source))
    return found


def test_the_scan_finds_an_inline_reading_and_passes_one_on_a_thread():
    inline = "async def turn(text):\n    return looks_like_desktop_objective(text)\n"
    threaded = "async def turn(text):\n    return await asyncio.to_thread(looks_like_desktop_objective, text)\n"
    nested = "async def turn(text):\n    def read():\n        return read_search_request(text)\n    return read\n"
    assert _inline_calls_in_coroutines(inline) == [2]
    assert _inline_calls_in_coroutines(threaded) == []
    assert _inline_calls_in_coroutines(nested) == [], "a sync function inside a coroutine is not on the loop by itself"


def test_no_coroutine_reads_a_request_by_meaning_on_her_event_loop():
    sources = {path: path.read_text(encoding="utf-8")
               for folder in ("core", "interface", "skills") for path in sorted((ROOT / folder).rglob("*.py"))}
    inline = [f"{path.relative_to(ROOT)}:{line}" for path, source in sources.items()
              if any(reader in source for reader in READERS) for line in _inline_calls_in_coroutines(source)]
    assert inline == []
