"""Integration that one condition carries is not natural replication.

The v1 battery's answer to the natural-conditions question is three of eight:
the ten domains have to form one strongly connected graph in at least three
ordinary conditions, each condition's graph built from its own trials. v25 is a
separate campaign and does not move that bar. These tests hold the one
definition in one place and make sure no single workload, conversation, tool
use or stress, can satisfy it by itself.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from core.subject.battery import assemble

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _evidence(strongly_connected: set[str]) -> dict:
    from core.subject.driver import CONDITIONS

    per_condition = {
        condition.name: {
            "edges": 40 if condition.name in strongly_connected else 12,
            "one_component": condition.name in strongly_connected,
            "vertex_connectivity": 2 if condition.name in strongly_connected else 0,
        }
        for condition in CONDITIONS
    }
    return {
        "per_condition": {
            "per_condition": per_condition,
            "conditions_with_scc": len(strongly_connected),
        }
    }


def _replicates(strongly_connected: set[str]) -> bool:
    verdict = assemble(_evidence(strongly_connected))
    (criterion,) = [item for item in verdict.criteria if item.key == "natural_runtime_replication"]
    return bool(criterion.passed)


@pytest.mark.parametrize("alone", ["conversation", "tool_use", "stress"])
def test_one_workload_alone_does_not_replicate(alone: str) -> None:
    assert not _replicates({alone})


def test_two_workloads_do_not_replicate() -> None:
    assert not _replicates({"conversation", "tool_use"})


def test_three_ordinary_conditions_replicate() -> None:
    assert _replicates({"conversation", "idle", "memory"})


def test_there_are_more_than_three_conditions_to_replicate_across() -> None:
    from core.subject.driver import CONDITIONS

    assert len(CONDITIONS) > 3


def test_the_bar_says_three_conditions() -> None:
    verdict = assemble(_evidence({"conversation", "idle", "memory"}))
    (criterion,) = [item for item in verdict.criteria if item.key == "natural_runtime_replication"]
    assert "at least three conditions" in str(criterion.bar)


def test_the_count_is_compared_in_one_place() -> None:
    """A second comparison of the same count is a second definition of pass."""
    places = []
    for root in (ROOT / "core", ROOT / "tools"):
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "conditions_with_scc" not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Compare) and "conditions_with_scc" in ast.unparse(node):
                    places.append(str(path.relative_to(ROOT)))
    assert places == ["core/subject/battery.py"], places
