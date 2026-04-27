"""Tests for aura_remediation_planner.

These tests validate that the remediation planner aggregates issues from
multiple JSON reports correctly, deduplicates items, and respects category
boundaries.
"""

import json
import os
from typing import Dict, Any

import pytest


# We import relative to the package root; when the test runs in the Aura
# repository, the module will be available under scripts.
from scripts.aura_remediation_planner import aggregate_reports


def write_json(path: str, data: Dict[str, Any]) -> None:
    """Write JSON data to a file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_aggregate_reports_simple(tmp_path: pytest.TempPathFactory) -> None:
    """Basic aggregation across multiple categories and files."""
    # Create two sample reports
    r1 = tmp_path / "report1.json"
    r2 = tmp_path / "report2.json"
    write_json(
        r1,
        {
            "issues": ["error1", "error2"],
            "errors": ["errX"],
            "warnings": [],
        },
    )
    write_json(
        r2,
        {
            "tasks": ["task1", "task2"],
            "persistence": ["persist1"],
            "issues": ["error2", "error3"],
        },
    )
    aggregated = aggregate_reports(str(tmp_path))
    # Each category should contain all entries from the reports, deduplicated in order
    assert aggregated["issues"] == ["error1", "error2", "error3"]
    assert aggregated["errors"] == ["errX"]
    assert aggregated["tasks"] == ["task1", "task2"]
    assert aggregated["persistence"] == ["persist1"]


def test_aggregate_reports_nested_and_non_dict(tmp_path: pytest.TempPathFactory) -> None:
    """Ensure aggregator gracefully handles nested dicts and non-list values."""
    r1 = tmp_path / "report.json"
    write_json(
        r1,
        {
            "errors": [
                {"msg": "nested dict"},
                ["not a string", {"another": "dict"}],
                "plain string",
            ],
            "tasks": "not a list",  # Should be ignored
        },
    )
    aggregated = aggregate_reports(str(tmp_path))
    # The nested dicts should be JSON-stringified
    expected_errors = [
        json.dumps({"msg": "nested dict"}, sort_keys=True),
        repr(["not a string", {"another": "dict"}]),
        "plain string",
    ]
    assert aggregated["errors"] == expected_errors
    # 'tasks' key should not produce output since it is not a list
    assert "tasks" not in aggregated