"""When readiness says no, it says which condition said no.

`/api/readyz` decides `ready` from five conditions and explained four of them.
The fifth, `healthy`, was a conjunct of the verdict and absent from the
fallback, so a runtime blocked on it answered 503 with an empty `issues` list
— a refusal that names nothing, which nobody can act on.

LIVE, 2026-09-07, mid-turn: `{"status":"not_ready","ready":false,"issues":[]}`.

The fix is not a fifth branch. The verdict and the explanation read one list,
so a condition added later appears in both or in neither.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = ROOT / "interface/routes/system.py"


def _readyz() -> ast.AsyncFunctionDef:
    tree = ast.parse(SYSTEM.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "readyz":
            return node
    raise AssertionError("readyz is gone; this gate now protects nothing")


def _source() -> str:
    return ast.get_source_segment(SYSTEM.read_text("utf-8"), _readyz()) or ""


def test_the_verdict_is_taken_from_the_condition_list() -> None:
    body = _source()
    assert "conditions" in body
    assert "all(satisfied for _name, satisfied in conditions)" in body


def test_the_explanation_is_taken_from_the_same_list() -> None:
    body = _source()
    assert "name for name, satisfied in conditions if not satisfied" in body


def test_no_condition_is_tested_twice_by_hand() -> None:
    """A second hand-written branch is how the two lists drifted apart."""

    body = _source()
    for stale in (
        'issues.append("system_not_ready")',
        'issues.append("conversation_lane_not_ready")',
        'issues.append("runtime_probe_unhealthy")',
        'issues.append("runtime_required_probes")',
    ):
        assert stale not in body, f"{stale} is a second copy of a condition"


def test_every_condition_carries_a_name() -> None:
    """A condition with no name cannot appear in an explanation."""

    body = _source()
    start = body.index("conditions: tuple[tuple[str, bool], ...] = (")
    end = body.index("\n        )", start)
    block = body[start:end]
    named = block.count('("')
    assert named >= 5, f"only {named} named conditions; the verdict had five"
    assert "healthy" in block, "the conjunct that was missing is still missing"
