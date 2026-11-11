"""R06: a coroutine builds a response contract on a worker thread, never on the event loop.

LIVE 2026-11-10, 14:58: her event loop stopped for 67 seconds and her runtime lost its lease. The stall capture
(data/error_logs/stalls/stall_1794351476.txt) has the loop thread inside build_response_contract, called directly
from the unitary response phase's ``execute``: the contract asks whether the turn wants outside evidence, which
embeds the request, which waits on the encoder lock. Another thread held that lock, encoding the same request for
the routing phase, which had moved its own build off the loop on 21 September. The kernel timed the phase at 67,185
ms; on 9 November the same call held the loop 265,846 ms (lockdep, 252 ms of it on the CPU). Ten coroutines still
called the builder on the loop; none does now. The amplifier's check, is_amplifiable, reads with an entailment model
loaded on first use (lockdep: 562 ms on the CPU and 1,652 ms held, from her reasoning background); its five coroutine
callers hand it to a thread too.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]


#: Functions that reach a resident model (the encoder, an entailment reader): a coroutine hands them to a thread.
_REACH_A_MODEL = frozenset({"build_response_contract", "is_amplifiable"})


def _called_on_the_loop(tree: ast.AST) -> list[int]:
    """Lines where an ``async def`` calls one of them itself rather than handing it to a thread."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in _REACH_A_MODEL:
            continue
        scope = parents.get(node)
        while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            scope = parents.get(scope)
        if isinstance(scope, ast.AsyncFunctionDef):
            lines.append(node.lineno)
    return lines


def test_no_coroutine_builds_a_contract_on_the_loop():
    found = []
    for package in ("core", "interface", "skills"):
        for path in sorted((_ROOT / package).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            found += [f"{path.relative_to(_ROOT)}:{line}" for line in _called_on_the_loop(tree)]
    assert not found, "hand it to a thread: `await asyncio.to_thread(build_response_contract, ...)`: " + ", ".join(found)


def test_the_reader_finds_a_build_on_the_loop_and_passes_one_handed_to_a_thread():
    on_loop = ast.parse("async def f(s):\n    return build_response_contract(s, 'x')\n")
    off_loop = ast.parse("async def f(s):\n    return await asyncio.to_thread(build_response_contract, s, 'x')\n")
    in_sync = ast.parse("def f(s):\n    return build_response_contract(s, 'x')\n")
    amplifier = ast.parse("async def f(s):\n    return is_amplifiable(s)\n")
    assert _called_on_the_loop(on_loop) == [2]
    assert _called_on_the_loop(off_loop) == []
    assert _called_on_the_loop(in_sync) == []
    assert _called_on_the_loop(amplifier) == [2]

