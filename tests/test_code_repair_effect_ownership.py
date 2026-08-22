"""Autonomous repair must never use the live tree as its mechanical sandbox."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.governance_context import get_active_governance
from core.self_modification import code_repair


@pytest.mark.asyncio
async def test_absolute_traceback_target_is_repaired_only_in_governed_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    target = root / "core" / "example.py"
    target.parent.mkdir(parents=True)
    original = "import os\n\nvalue = 1\n"
    fixed = "value = 1\n"
    target.write_text(original, encoding="utf-8")
    pinned_ruff = root / ".venv" / "bin" / "ruff"
    pinned_ruff.parent.mkdir(parents=True)
    pinned_ruff.write_bytes(b"test executable identity")

    calls: list[dict[str, object]] = []

    class _Gateway:
        async def run_async(self, command, **kwargs):
            token = get_active_governance()
            assert token is not None and token.authorizes
            assert token.domain == "self_modification"
            assert kwargs["accelerator_capability"] == "none"
            shadow = Path(command[-1])
            assert shadow != target
            assert root not in shadow.parents
            calls.append(
                {
                    "source": kwargs["source"],
                    "target": dict(token.constraints)["target"],
                    "command": list(command),
                }
            )
            if "--fix" in command:
                await asyncio.to_thread(shadow.write_text, fixed, encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    generator = SimpleNamespace(
        code_base=root,
        generate_fix=AsyncMock(return_value=None),
    )
    repair = object.__new__(code_repair.AutonomousCodeRepair)
    repair.generator = generator
    repair.validator = SimpleNamespace(
        validate_fix=lambda candidate, content: (content == fixed, "validated")
    )
    repair.tester = SimpleNamespace(
        test_fix=AsyncMock(return_value=(True, {"success": True}))
    )
    repair.harness = SimpleNamespace(
        evaluate_fix=AsyncMock(return_value=(True, "verified"))
    )
    repair._deep_repair_after_patch_failure = AsyncMock()
    monkeypatch.setattr(code_repair, "get_subprocess_gateway", lambda: _Gateway())

    success, candidate, result = await repair.repair_bug(
        str(target),
        1,
        {"hypotheses": []},
    )

    assert success is True
    assert result["success"] is True
    assert candidate is not None
    assert candidate.target_file == "core/example.py"
    assert candidate.original_code == original
    assert candidate.fixed_code == fixed
    assert target.read_text(encoding="utf-8") == original
    assert [call["source"] for call in calls] == [
        "core.self_modification.code_repair.ruff_shadow",
        "core.self_modification.code_repair.ruff_shadow.verify",
    ]
    assert {call["target"] for call in calls} == {"core/example.py"}
    generator.generate_fix.assert_not_awaited()


@pytest.mark.asyncio
async def test_repair_target_outside_code_base_is_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")

    generator = SimpleNamespace(
        code_base=root,
        generate_fix=AsyncMock(return_value=None),
    )
    repair = object.__new__(code_repair.AutonomousCodeRepair)
    repair.generator = generator
    gateway = SimpleNamespace(run_async=AsyncMock())
    monkeypatch.setattr(code_repair, "get_subprocess_gateway", lambda: gateway)

    success, candidate, result = await repair.repair_bug(
        str(outside),
        1,
        {"hypotheses": []},
    )

    assert success is False
    assert candidate is None
    assert "outside code base" in result["error"]
    gateway.run_async.assert_not_awaited()
    generator.generate_fix.assert_not_awaited()


def test_candidate_source_anchor_must_be_unique() -> None:
    candidate = code_repair.CodeFix(
        target_file="core/example.py",
        target_line=1,
        original_code="value = 1",
        fixed_code="value = 2",
        explanation="test",
        hypothesis="test",
        confidence="high",
    )

    with pytest.raises(ValueError, match="exactly once"):
        code_repair._apply_fix_once("value = 1\nvalue = 1\n", candidate)


def test_absolute_candidate_cannot_escape_sandbox(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be relative"):
        code_repair._sandbox_target(tmp_path, "/tmp/live.py")


def test_repair_uses_the_source_bound_ruff_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    pinned = root / ".venv" / "bin" / "ruff"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"binary")
    monkeypatch.setattr(code_repair.shutil, "which", lambda _name: "/other/ruff")

    assert code_repair._ruff_executable(root) == str(pinned)


@pytest.mark.asyncio
async def test_real_ruff_produces_a_candidate_without_mutating_source(
    tmp_path: Path,
) -> None:
    adjacent_ruff = Path(code_repair.sys.executable).parent / "ruff"
    if not adjacent_ruff.is_file():
        pytest.skip("the test interpreter has no adjacent Ruff executable")

    root = tmp_path / "repo"
    target = root / "core" / "example.py"
    target.parent.mkdir(parents=True)
    original = "import os\n\nvalue = 1\n"
    target.write_text(original, encoding="utf-8")
    generator = SimpleNamespace(code_base=root)
    repair = object.__new__(code_repair.AutonomousCodeRepair)
    repair.generator = generator

    candidate = await repair._generate_ruff_candidate("core/example.py", 1)

    assert candidate is not None
    assert candidate.target_file == "core/example.py"
    assert "import os" not in candidate.fixed_code
    assert candidate.fixed_code.strip() == "value = 1"
    assert target.read_text(encoding="utf-8") == original
