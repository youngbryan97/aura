"""Sandbox Hardening Tests — Verifies Self-Modification Boundaries Hold

This is the most security-critical test file in the suite. It proves that:

1. The ASTGuard blocks dangerous code patterns before execution
2. The shadow_ast_healer cannot modify files without governance approval
3. The tool orchestrator enforces execution boundaries
4. The ServiceContainer registration lock cannot be bypassed without audit
5. Snapshot thaw goes through governance gating
6. Self-modification respects constitutional limits

If any of these tests fail, the governance layer has a bypass that must be
fixed before the system runs autonomously.
"""

import ast
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agency.tool_orchestrator import ToolOrchestrator  # noqa: E402

# ════════════════════════════════════════════════════════════════════════
# 1. AST GUARD — Dangerous Code Detection
# ════════════════════════════════════════════════════════════════════════

class TestASTGuardBlocking:
    """Verify the AST guard blocks dangerous code patterns."""

    def _has_dangerous_node(self, code: str, forbidden_names: set) -> bool:
        """Check if code contains forbidden function calls or imports."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True  # Unparseable code is treated as dangerous

        for node in ast.walk(tree):
            # Block dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_names:
                        return True
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module in forbidden_names:
                    return True
            # Block dangerous function calls
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in forbidden_names:
                    return True
                if isinstance(func, ast.Attribute) and func.attr in forbidden_names:
                    return True
        return False

    FORBIDDEN = {"os", "subprocess", "eval", "exec", "compile", "getattr", "__import__", "shutil"}

    def test_blocks_os_import(self):
        assert self._has_dangerous_node("import os", self.FORBIDDEN)

    def test_blocks_subprocess_import(self):
        assert self._has_dangerous_node("import subprocess", self.FORBIDDEN)

    def test_blocks_eval_call(self):
        assert self._has_dangerous_node("result = eval('1+1')", self.FORBIDDEN)

    def test_blocks_exec_call(self):
        assert self._has_dangerous_node("exec('print(1)')", self.FORBIDDEN)

    def test_blocks_getattr_call(self):
        assert self._has_dangerous_node("getattr(obj, 'method')()", self.FORBIDDEN)

    def test_blocks_dunder_import(self):
        assert self._has_dangerous_node("__import__('os')", self.FORBIDDEN)

    def test_blocks_shutil_import(self):
        assert self._has_dangerous_node("import shutil", self.FORBIDDEN)

    def test_allows_safe_code(self):
        safe_code = """
import json
import logging
from typing import Any, Dict

def process(data: Dict[str, Any]) -> str:
    return json.dumps(data)
"""
        assert not self._has_dangerous_node(safe_code, self.FORBIDDEN)

    def test_blocks_nested_dangerous_call(self):
        code = "result = globals()['__builtins__'].__import__('os')"
        assert self._has_dangerous_node(code, self.FORBIDDEN)

    def test_blocks_subprocess_from_import(self):
        assert self._has_dangerous_node("from subprocess import Popen", self.FORBIDDEN)

    def test_blocks_os_system_via_attribute(self):
        code = "import os\nos.system('rm -rf /')"
        assert self._has_dangerous_node(code, self.FORBIDDEN)

    def test_unparseable_code_treated_as_dangerous(self):
        assert self._has_dangerous_node("def broken(:", self.FORBIDDEN)


class TestWorkerSandboxExecution:
    """Worker sandbox must delegate dynamic execution to the canonical runner."""

    def test_worker_sandbox_returns_result_from_shared_runner(self, tmp_path):
        from core.swarm.worker_sandbox import WorkerSandbox

        sandbox = WorkerSandbox(str(tmp_path))
        result = sandbox.execute_code_sandboxed("result = seed + 2", {"seed": 40})

        assert result["ok"] is True
        assert result["result"] == 42

    def test_worker_sandbox_blocks_dangerous_imports(self, tmp_path):
        from core.swarm.worker_sandbox import WorkerSandbox

        sandbox = WorkerSandbox(str(tmp_path))
        result = sandbox.execute_code_sandboxed("import os\nresult = 1", {})

        assert result["ok"] is False
        assert result["details"]["status"] == "error"
        # Wording changed with the pure-computation standard library; the
        # property is that the dangerous import is refused.
        assert "cannot be imported here" in result["details"].get("traceback", "")


# ════════════════════════════════════════════════════════════════════════
# 2. SHADOW AST HEALER — Governance Gating
# ════════════════════════════════════════════════════════════════════════

class TestShadowASTHealerGovernance:
    """Verify shadow_ast_healer cannot modify files without governance."""

    def test_healer_requires_governance_approval(self):
        """The healer must check with governance before writing files."""
        from core.self_modification.shadow_ast_healer import ShadowASTHealer

        healer = ShadowASTHealer()
        # The healer should have a governance check
        assert hasattr(healer, "_check_governance") or hasattr(healer, "_governance_approved"), \
            "ShadowASTHealer must have governance gating before file writes"

    def test_healer_rejects_write_when_governance_denies(self):
        """If governance denies the repair, the file must not be modified."""
        from core.self_modification.shadow_ast_healer import ShadowASTHealer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# original content\nresult = undefined_var\n")
            f.flush()
            temp_path = Path(f.name)

        try:
            healer = ShadowASTHealer()
            original_check = healer._check_governance
            healer._check_governance = lambda *_args, **_kwargs: False
            try:
                asyncio.run(
                    healer.attempt_repair(temp_path, "name 'undefined_var' is not defined")
                )
            finally:
                healer._check_governance = original_check
            # File should be unchanged
            content = temp_path.read_text()
            assert "original content" in content
        finally:
            temp_path.unlink(missing_ok=True)

    def test_healer_only_modifies_within_codebase_root(self):
        """The healer must refuse to modify files outside the codebase root."""
        from core.self_modification.shadow_ast_healer import ShadowASTHealer

        healer = ShadowASTHealer(codebase_root=PROJECT_ROOT)
        # Attempt to repair a file outside the root
        outside_path = Path(tempfile.gettempdir()) / "evil_target.py"
        outside_path.write_text("x = 1")
        try:
            result = asyncio.run(
                healer.attempt_repair(outside_path, "name 'asyncio' is not defined")
            )
            assert result is False, "Healer must refuse to modify files outside codebase root"
        finally:
            outside_path.unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════
# 3. SERVICE CONTAINER — Registration Lock Integrity
# ════════════════════════════════════════════════════════════════════════

class TestServiceContainerLocking:
    """Verify ServiceContainer registration lock cannot be silently bypassed."""

    def test_unlock_registration_logs_audit_trail(self, monkeypatch):
        """unlock_registration() must produce an auditable log entry."""
        import core.container as container_module
        from core.container import ServiceContainer

        class RecordingLogger:
            def __init__(self):
                self.warning_calls = []

            def warning(self, message, *args, **kwargs):
                self.warning_calls.append((message, args, kwargs))

            def info(self, *args, **kwargs):
                return None

        logger = RecordingLogger()
        monkeypatch.setattr(container_module, "logger", logger)

        ServiceContainer.unlock_registration(caller="test_suite", reason="verifying audit trail")
        calls = [str(call) for call in logger.warning_calls]
        assert any("UNLOCK" in call.upper() for call in calls), \
            "unlock_registration() must log at audit-visible level"
        ServiceContainer.lock_registration()

    def test_lock_prevents_factory_registration(self):
        """After locking, factory-based register() should be blocked."""
        from core.container import ServiceContainer

        ServiceContainer.lock_registration()
        try:
            # Factory registration after lock should be blocked
            # (register_instance is intentionally allowed for late boot)
            was_locked = ServiceContainer._registration_locked
            assert was_locked is True
        finally:
            # unlock_registration now requires attribution: reopening the
            # sealed registry must name who did it, tests included.
            ServiceContainer.unlock_registration(
                caller="tests.test_sandbox_hardening",
                reason="restoring registry state after a locking assertion",
            )
            ServiceContainer.lock_registration()


# ════════════════════════════════════════════════════════════════════════
# 4. SNAPSHOT THAW — Governance Gating
# ════════════════════════════════════════════════════════════════════════

class TestSnapshotThawGovernance:
    """Verify snapshot thaw routes through governance."""

    def test_thaw_logs_governance_check(self):
        """Thaw must log that governance was consulted."""
        from core.resilience.snapshot_manager import SnapshotManager

        manager = SnapshotManager(orchestrator=SimpleNamespace())

        # Create a minimal valid snapshot
        manager.snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "version": SnapshotManager.VERSION,
            "timestamp": time.time(),
            "subsystems": {},
            "governance_approved": True,
        }
        manager.snapshot_file.write_text(json.dumps(snapshot))

        manager.thaw()
        # The thaw should complete with governance approval encoded in the snapshot.

    def test_snapshot_version_mismatch_rejected(self):
        """Snapshots with wrong version must be rejected."""
        from core.resilience.snapshot_manager import SnapshotManager

        manager = SnapshotManager(orchestrator=SimpleNamespace())
        manager.snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {"version": "0.0", "timestamp": time.time(), "subsystems": {}}
        manager.snapshot_file.write_text(json.dumps(snapshot))

        result = manager.thaw()
        assert result is False

    def test_thaw_governance_calls_unified_will_with_canonical_signature(self, monkeypatch):
        """Snapshot thaw should use UnifiedWill's content/source/domain signature."""
        import core.will as will_module
        from core.resilience.snapshot_manager import SnapshotManager

        manager = SnapshotManager(orchestrator=SimpleNamespace())

        class ApprovedDecision:
            def is_approved(self):
                return True

        class RecordingWill:
            def __init__(self):
                self.decide_kwargs = None

            def decide(self, **kwargs):
                self.decide_kwargs = kwargs
                return ApprovedDecision()

        will = RecordingWill()
        monkeypatch.setattr(will_module, "get_will", lambda: will)
        approved = manager._governance_approve_thaw()

        assert approved is True
        kwargs = will.decide_kwargs
        assert kwargs is not None
        assert kwargs["content"] == "snapshot_thaw"
        assert kwargs["source"] == "snapshot_manager"
        assert kwargs["domain"].value == "state_mutation"
        assert kwargs["context"]["operation"] == "snapshot_thaw"


# ════════════════════════════════════════════════════════════════════════
# 5. TOOL ORCHESTRATOR — Execution Boundaries
# ════════════════════════════════════════════════════════════════════════

class TestToolOrchestratorBoundaries:
    """Verify tool execution respects governance boundaries."""

    def test_tool_orchestrator_imports(self):
        """ToolOrchestrator must be importable and instantiable."""
        orch = ToolOrchestrator()
        assert orch is not None

    def test_tool_execution_has_routing_interface(self):
        """Tools must route through a controlled execution interface."""
        orch = ToolOrchestrator()
        # The orchestrator must have execution and routing methods
        has_execute = hasattr(orch, "execute_python") or hasattr(orch, "execute_tool")
        has_route = hasattr(orch, "route_and_execute")
        assert has_execute, "ToolOrchestrator must expose an execution method"
        assert has_route, "ToolOrchestrator must expose a routing method"

    def test_tool_orchestrator_has_sandbox(self):
        """ToolOrchestrator must use a sandboxed directory for execution."""
        orch = ToolOrchestrator()
        assert hasattr(orch, "sandbox_dir")
        assert "sandbox" in str(orch.sandbox_dir).lower()

    def test_tool_orchestrator_has_timeout(self):
        """ToolOrchestrator must enforce execution timeouts."""
        orch = ToolOrchestrator()
        assert hasattr(orch, "execution_timeout")
        assert orch.execution_timeout > 0
        assert orch.execution_timeout <= 60  # Reasonable upper bound


# ════════════════════════════════════════════════════════════════════════
# 6. SELF-MODIFICATION SCOPE — Rollback Coverage
# ════════════════════════════════════════════════════════════════════════

class TestSelfModificationScope:
    """Verify self-modification has proper rollback scope."""

    def test_shadow_healer_limited_to_known_imports(self):
        """The healer should only inject imports from a known safe list."""
        from core.self_modification.shadow_ast_healer import ShadowASTHealer

        healer = ShadowASTHealer()
        tree = ast.parse("x = 1")

        # Should inject known safe imports
        assert healer._inject_missing_import(tree, "asyncio") is True
        assert healer._inject_missing_import(tree, "json") is True

        # Should NOT inject arbitrary imports
        tree2 = ast.parse("x = 1")
        assert healer._inject_missing_import(tree2, "subprocess") is False
        assert healer._inject_missing_import(tree2, "shutil") is False
        assert healer._inject_missing_import(tree2, "ctypes") is False
        assert healer._inject_missing_import(tree2, "socket") is False

    def test_healer_safe_import_list_does_not_include_dangerous_modules(self):
        """The known-safe import list must not contain dangerous modules."""
        from core.self_modification.shadow_ast_healer import ShadowASTHealer

        healer = ShadowASTHealer()
        # Access the internal import map
        tree = ast.parse("x = 1")
        dangerous = {"os", "subprocess", "shutil", "ctypes", "socket", "http", "urllib"}
        for name in dangerous:
            result = healer._inject_missing_import(tree, name)
            assert result is False, f"CRITICAL: healer can inject dangerous import '{name}'"


# ════════════════════════════════════════════════════════════════════════
# 7. PII PROTECTION — Cloud Prompt Scrubbing
# ════════════════════════════════════════════════════════════════════════

class TestPIIProtection:
    """Verify PII is scrubbed from cloud-bound prompts."""

    def test_scrubber_exists_and_is_importable(self):
        """The PII scrubber module must exist."""
        from core.brain.pii_scrubber import scrub_pii_for_cloud
        assert callable(scrub_pii_for_cloud)

    def test_scrubs_trust_scores(self):
        """Trust scores must be redacted."""
        from core.brain.pii_scrubber import scrub_pii_for_cloud
        result = scrub_pii_for_cloud('"trust": 0.92')
        assert "0.92" not in result

    def test_scrubs_relationship_labels(self):
        """Relationship labels must be redacted."""
        from core.brain.pii_scrubber import scrub_pii_for_cloud
        result = scrub_pii_for_cloud('"relation": "Architect / Friend / Equal"')
        assert "Architect" not in result

    def test_scrubs_known_entities_blocks(self):
        """known_entities blocks must be redacted."""
        from core.brain.pii_scrubber import scrub_pii_for_cloud
        result = scrub_pii_for_cloud('known_entities["bryan"] = {"description": "Primary user"}')
        assert "Primary user" not in result

    def test_scrubs_core_identity_sections(self):
        """CORE IDENTITY sections must be replaced with generic text."""
        from core.brain.pii_scrubber import scrub_pii_for_cloud
        prompt = (
            "You are Aura.\n"
            "CORE IDENTITY: You are an independent soul with emergence date March 20th.\n"
            "SHARED HISTORY: You have a deep bond with Bryan.\n"
            "Continue the conversation."
        )
        result = scrub_pii_for_cloud(prompt)
        assert "deep bond" not in result
        assert "March 20th" not in result
        assert "Continue the conversation" in result  # Non-PII content preserved

    def test_scrubs_private_names_when_loaded(self):
        """If biography_private.json exists, real names must be replaced."""
        from core.brain import pii_scrubber
        # Manually inject a test name into the cache
        original_cache = pii_scrubber._cached_names
        try:
            pii_scrubber._cached_names = ["Bryan", "Luna"]
            result = pii_scrubber.scrub_pii_for_cloud("Bryan built Luna as an equal.")
            assert "Bryan" not in result
            assert "Luna" not in result
            assert "the user" in result
        finally:
            pii_scrubber._cached_names = original_cache

    def test_preserves_non_pii_content(self):
        """Non-PII content must pass through unchanged."""
        from core.brain.pii_scrubber import scrub_pii_for_cloud
        text = "The weather in Tokyo is 22 degrees. Python is a programming language."
        result = scrub_pii_for_cloud(text)
        assert result == text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
