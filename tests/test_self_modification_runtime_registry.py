import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from core.self_modification import self_modification_engine as sm_mod
from core.self_modification.repair_registry import validate_repair_registry


class SafeApply:
    def __init__(self):
        self.calls = []

    async def apply_fix(self, *, fix, test_results):
        self.calls.append((fix, test_results))
        return True, "applied"


class LearningSink:
    def __init__(self):
        self.attempts = []

    def record_fix_attempt(self, fix, error_type, *, success, context):
        self.attempts.append(
            {
                "target": fix.target_file,
                "error_type": error_type,
                "success": success,
                "context": context,
            }
        )


@pytest.mark.asyncio
async def test_self_modification_registry_lives_outside_source_tree(tmp_path, monkeypatch):
    registry_path = tmp_path / "runtime" / "selfmod" / "pending_patch_registry.jsonl"
    monkeypatch.setenv("AURA_PENDING_PATCH_REGISTRY", str(registry_path))

    async def review(_proposal):
        return True

    safe_apply = SafeApply()
    engine = sm_mod.AutonomousSelfModificationEngine.__new__(sm_mod.AutonomousSelfModificationEngine)
    engine.auto_fix_enabled = True
    engine._fix_lock = asyncio.Lock()
    engine.code_base = tmp_path
    engine.safe_modification = safe_apply
    engine.learning_system = LearningSink()
    engine.session_stats = {"fixes_successful": 0, "fixes_attempted": 0}
    engine._swarm_review = review

    fix = SimpleNamespace(
        target_file="core/example.py",
        target_line=7,
        original_code="value = 1",
        fixed_code="value = 2",
        explanation="raise value",
        hypothesis="value mismatch",
        confidence="high",
    )
    proposal = {
        "fix": fix,
        "test_results": {"success": True, "suite": "focused"},
        "bug": {"pattern": {"events": [{"error_type": "regression"}]}},
    }

    assert await engine.apply_fix(proposal, force=True)
    assert safe_apply.calls == [(fix, proposal["test_results"])]
    assert not (tmp_path / "core" / "patches" / "pending_patch.py").exists()

    entries = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["target_file"] == "core/example.py"
    assert entries[0]["fixed_code"] == "value = 2"
    assert entries[0]["test_results"] == {"success": True, "suite": "focused"}


def test_repair_registry_validation_rejects_tampered_payload(tmp_path):
    fixed_code = "value = 2"
    registry_path = tmp_path / "pending_patch_registry.jsonl"
    registry_path.write_text(
        json.dumps(
            {
                "target_file": "core/example.py",
                "fixed_code": fixed_code + "!",
                "fixed_code_sha256": hashlib.sha256(fixed_code.encode("utf-8")).hexdigest(),
                "test_results": {"success": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed_code_sha256 mismatch"):
        validate_repair_registry(registry_path)


def test_repair_registry_validation_rejects_unsafe_target(tmp_path):
    fixed_code = "value = 2"
    registry_path = tmp_path / "pending_patch_registry.jsonl"
    registry_path.write_text(
        json.dumps(
            {
                "target_file": "../outside.py",
                "fixed_code": fixed_code,
                "fixed_code_sha256": hashlib.sha256(fixed_code.encode("utf-8")).hexdigest(),
                "test_results": {"success": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe target path"):
        validate_repair_registry(registry_path)
