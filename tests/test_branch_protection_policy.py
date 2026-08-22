from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "check_branch_protection.py"
POLICY = ROOT / "config" / "branch_protection_policy.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("branch_protection_tool", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_contexts_are_the_names_github_emits() -> None:
    tool = _load_tool()
    contexts = tool.workflow_jobs_on_pull_request()

    assert contexts["Static Analysis (CodeQL)"] == "security-gates.yml"
    assert contexts["Dependency Vulnerability Scan"] == "security-gates.yml"
    assert contexts["quality-gate"] == "quality-gate.yml"
    assert "sast" not in contexts
    assert "dependency-audit" not in contexts


def test_policy_is_coherent_with_emitted_contexts() -> None:
    tool = _load_tool()
    policy = json.loads(POLICY.read_text("utf-8"))

    assert tool.check_offline(policy) == []


def test_apply_payload_preserves_conversation_resolution_and_all_checks() -> None:
    tool = _load_tool()
    policy = json.loads(POLICY.read_text("utf-8"))

    body = tool._protection_body(policy)

    assert body["required_conversation_resolution"] is True
    assert set(body["required_status_checks"]["contexts"]) == set(
        policy["required_checks"]
    )
