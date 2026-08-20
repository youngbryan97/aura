"""A cancelled operation must not be reported as a completed one.

LIVE, 2026-08-20. A user asked for a value from an API. The fetch was
dispatched, ran, and was cancelled. The handler caught CancelledError
alongside RuntimeError and TimeoutError and returned
``{"status": "failed", "error": ""}``. The model was told its tool had failed
for no stated reason, produced nothing further, and the turn ended in "I
couldn't get to an answer I'd stand behind on that one."

Cancellation derives from BaseException precisely so ordinary handlers do not
catch it. Listing it beside a tuple of transient errors puts it back.
"""

from __future__ import annotations

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"

#: Handlers that convert a cancellation into a returned value. Each remaining
#: one is deliberate and documented at its site. The number only goes down.
BASELINE = 5


def _catches_cancellation(handler: ast.ExceptHandler) -> bool:
    node = handler.type
    items = node.elts if isinstance(node, ast.Tuple) else ([node] if node is not None else [])
    return any("CancelledError" in ast.unparse(item) for item in items)


def _reraises(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(node, ast.Raise) for node in ast.walk(handler))


def _returns_a_value(handler: ast.ExceptHandler) -> bool:
    for node in ast.walk(handler):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            continue
        return True
    return False


def offenders() -> list[str]:
    found: list[str] = []
    for path in sorted(CORE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and _catches_cancellation(node)
                and not _reraises(node)
                and _returns_a_value(node)
            ):
                found.append(f"{path.relative_to(CORE.parent)}:{node.lineno}")
    return found


def test_the_count_only_goes_down() -> None:
    found = offenders()
    assert len(found) <= BASELINE, (
        "a cancellation is being turned into a result in a new place:\n  "
        + "\n  ".join(found)
    )


def test_the_governor_that_broke_a_live_turn_propagates_cancellation() -> None:
    """Asserted as a property, not a spelling: every handler in that file that
    catches cancellation re-raises it, and none mixes it with ordinary errors."""
    path = CORE / "resilience" / "cognitive_governor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and _catches_cancellation(node)
    ]
    assert handlers, "the governor no longer handles cancellation at all"
    for handler in handlers:
        assert _reraises(handler), f"{path.name}:{handler.lineno} swallows cancellation"
        assert isinstance(handler.type, ast.Attribute | ast.Name), (
            f"{path.name}:{handler.lineno} mixes cancellation into an error tuple"
        )


def test_no_returned_error_can_be_empty() -> None:
    """describe_error names the class when the exception carries no message."""
    from core.runtime.errors import describe_error

    assert describe_error(RuntimeError()).startswith("RuntimeError")
    assert describe_error(TimeoutError()).startswith("TimeoutError")
    assert describe_error(ValueError("real")) == "ValueError: real"
