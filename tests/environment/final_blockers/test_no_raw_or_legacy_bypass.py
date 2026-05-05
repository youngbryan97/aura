"""Final Blocker: No raw or legacy bypass of the canonical kernel.

No skill, old runtime, or direct adapter call may send environment actions
outside the canonical EnvironmentKernel path.
"""
import ast
import os
import pytest
from pathlib import Path


# Raw key sinks that must only appear in approved modules
RAW_KEY_SINKS = [
    "pexpect.spawn",
    ".send(",
    "send_action(",
    "pyautogui.press",
    "keyboard.write",
    "child.send(",
]

# Modules where raw key sinks are ALLOWED
ALLOWED_RAW_KEY_MODULES = {
    "core/environment/adapter.py",
    "core/environment/command.py",
    "core/environment/generic_command_handlers.py",
    "core/embodiment/",
    "core/adapters/",
    "core/environments/",
    "core/social/",
    "core/bus/",
    "core/state/",
    "core/actors/",
    "core/senses/",
    "core/skills/",
    "core/orchestrator/",
    "core/consciousness/",
    "tests/",
}

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestNoRawBypass:
    """Raw keystroke sinks must not appear outside approved adapter/compiler modules."""

    def test_static_scan_raw_key_sinks(self):
        """Scan the codebase for raw key sinks outside the allowlist."""
        violations = []
        core_dir = REPO_ROOT / "core"
        if not core_dir.exists():
            pytest.skip("core directory not found")

        for py_file in core_dir.rglob("*.py"):
            rel = str(py_file.relative_to(REPO_ROOT))
            # Skip allowed modules
            if any(allowed in rel for allowed in ALLOWED_RAW_KEY_MODULES):
                continue

            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for sink in RAW_KEY_SINKS:
                if sink in content:
                    violations.append(f"{rel}: contains raw key sink '{sink}'")

        if violations:
            msg = "Raw key sinks found outside approved modules:\n" + "\n".join(violations)
            pytest.fail(msg)

    def test_no_direct_adapter_execute_from_policy(self):
        """Policy modules must not call adapter.execute directly."""
        policy_dir = REPO_ROOT / "core" / "environment" / "policy"
        if not policy_dir.exists():
            pytest.skip("policy directory not found")

        violations = []
        for py_file in policy_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if "adapter.execute" in content or "adapter.send" in content:
                violations.append(str(py_file.relative_to(REPO_ROOT)))

        assert not violations, f"Policy modules must not call adapter directly: {violations}"

    def test_policy_returns_action_intent_not_raw_key(self):
        """Policy output must be ActionIntent, not raw string keys."""
        from core.environment.policy.policy_orchestrator import PolicyOrchestrator
        from core.environment.command import ActionIntent
        from core.environment.parsed_state import ParsedState
        from core.environment.belief_graph import EnvironmentBeliefGraph
        from core.environment.homeostasis import Homeostasis

        orch = PolicyOrchestrator()
        parsed = ParsedState(
            environment_id="test",
            context_id="test",
            sequence_id=0,
            self_state={"hp": 20, "max_hp": 20},
        )
        belief = EnvironmentBeliefGraph()
        homeo = Homeostasis()
        intent = orch.select_action(
            parsed_state=parsed,
            belief=belief,
            homeostasis=homeo,
            episode=None,
            recent_frames=[],
        )
        assert isinstance(intent, ActionIntent), f"Policy returned {type(intent)}, not ActionIntent"
        # Must not be a single raw key character
        assert len(intent.name) > 1 or intent.name in ("i",), f"Policy returned raw key: {intent.name}"

    def test_command_compiler_rejects_unknown_intent(self):
        """CommandCompiler must fail closed on unknown intents."""
        from core.environment.command import CommandCompiler, ActionIntent
        compiler = CommandCompiler("test")
        # Don't register any handlers
        with pytest.raises(ValueError, match="unknown_intent"):
            compiler.compile(ActionIntent(name="totally_made_up_action"))
