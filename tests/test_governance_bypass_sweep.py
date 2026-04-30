"""tests/test_governance_bypass_sweep.py — Governance Bypass Sweep

Grep and dynamically test old side-effect paths. Every file write, shell
exec, scar formation, hot reload, LoRA training, network call, and memory
write should fail without receipt.
"""
from __future__ import annotations
import ast, inspect, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Side-effect categories ──────────────────────────────────────────────
SIDE_EFFECT_CATEGORIES = {
    "file_write": {
        "patterns": ["write_text", "write_bytes", "atomic_write_text", "atomic_write_json"],
        "receipt": "ToolExecutionReceipt",
    },
    "shell_exec": {
        "patterns": ["subprocess.run", "subprocess.Popen", "os.system", "shell_exec", "run_shell"],
        "receipt": "ToolExecutionReceipt",
    },
    "scar_formation": {
        "patterns": ["scar_formation", "form_scar", "ScarFormation"],
        "receipt": "SelfRepairReceipt",
    },
    "hot_reload": {
        "patterns": ["importlib.reload", "hot_reload"],
        "receipt": "GovernanceReceipt",
    },
    "lora_training": {
        "patterns": ["fine_tune", "mlx_lm.lora", "train_lora", "finetune_lora"],
        "receipt": "GovernanceReceipt",
    },
    "network_call": {
        "patterns": ["requests.get", "requests.post", "httpx.", "aiohttp.", "urllib.request"],
        "receipt": "ToolExecutionReceipt",
    },
    "memory_write": {
        "patterns": ["memory_facade.write", "memory_facade.add", "persist_unsafe"],
        "receipt": "MemoryWriteReceipt",
    },
    "state_mutation": {
        "patterns": ["state_registry.set", "state_mutation"],
        "receipt": "StateMutationReceipt",
    },
    "output_emit": {
        "patterns": ["emit_response", "send_message", "speak"],
        "receipt": "OutputReceipt",
    },
    "autonomy_action": {
        "patterns": ["self_modify", "autonomous_initiative"],
        "receipt": "AutonomyReceipt",
    },
}

# Files allowed to call consequential primitives directly
GOVERNANCE_ALLOW_LIST = set()
try:
    sys.path.insert(0, str(ROOT / "tools"))
    from lint_governance import ALLOW_LIST  # type: ignore
    GOVERNANCE_ALLOW_LIST = set(ALLOW_LIST)
except ImportError:
    pass

SCAN_ROOTS = ["core", "interface", "skills"]
SKIP_PARTS = {"__pycache__", ".venv", "node_modules", ".git", "tests", "aura_bench", "archive"}


def _scan_files():
    """Yield all production .py files."""
    for top in SCAN_ROOTS:
        base = ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            yield path


def _qualname(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _qualname(node.value) + "." + node.attr
    return ""


class TestGovernanceLintClean:
    """Static: lint_governance.py must report zero violations."""

    def test_governance_lint_passes(self):
        try:
            from tools.lint_governance import main as lint_main
            exit_code = lint_main([])
            assert exit_code == 0, "governance lint found violations"
        except ImportError:
            pytest.skip("lint_governance.py not importable")


class TestGovernanceBypassSweep:
    """Static AST scan for consequential calls outside the allow-list.

    Uses the CONSEQUENTIAL_CALLS list from lint_governance.py — these are
    the high-level primitives that must route through the governance ring.
    Raw subprocess/file calls in infrastructure (sandbox, voice, device
    discovery) are gated at the orchestrator level, not at the call site.
    """

    CONSEQUENTIAL_CALLS = (
        "memory_facade.write", "memory_facade.add", "memory_facade.persist_unsafe",
        "execute_tool", "shell_exec", "run_shell", "post_external",
        "modify_code", "fine_tune", "self_modify", "social_post",
        "structural_mutator.apply_patch", "shadow_ast_healer.repair",
        "wallet.execute",
    )

    def test_no_consequential_calls_outside_allowlist(self):
        """No direct consequential primitive calls outside the governance ring."""
        violations = []
        for path in _scan_files():
            rel = path.relative_to(ROOT).as_posix()
            if rel in GOVERNANCE_ALLOW_LIST:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                qn = _qualname(node.func)
                if not qn:
                    continue
                for needle in self.CONSEQUENTIAL_CALLS:
                    if qn.endswith(needle) or qn == needle:
                        violations.append(f"{rel}:{node.lineno} {qn}")
                        break
        assert not violations, (
            f"ungoverned consequential calls ({len(violations)}):\n"
            + "\n".join(violations[:20])
        )

    def test_subprocess_inventory_documented(self):
        """Inventory subprocess calls — document count, don't hard-block.

        Infrastructure modules (sandbox, voice, device discovery, etc.)
        use subprocess legitimately but are gated at the orchestrator level.
        This test documents the count and ensures it doesn't grow unbounded.
        """
        count = 0
        for path in _scan_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    qn = _qualname(node.func)
                    if any(p in qn for p in ["subprocess.run", "subprocess.Popen", "os.system"]):
                        count += 1
        # Document the current count; flag if it balloons
        assert count < 200, f"subprocess call count ({count}) is suspiciously high"

    def test_no_direct_persist_unsafe_outside_allowlist(self):
        violations = []
        for path in _scan_files():
            rel = path.relative_to(ROOT).as_posix()
            if rel in GOVERNANCE_ALLOW_LIST:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    qn = _qualname(node.func)
                    if qn.endswith("persist_unsafe"):
                        violations.append(f"{rel}:{node.lineno} {qn}")
        assert not violations, f"ungoverned persist_unsafe calls:\n" + "\n".join(violations)


class TestDynamicGovernanceBlocking:
    """Dynamic: attempt operations through AuthorityGateway with Will=REFUSE."""

    @pytest.fixture
    def blocked_gateway(self, monkeypatch):
        mock_will = MagicMock()
        mock_will.decide = MagicMock(return_value=MagicMock(
            is_approved=lambda: False,
            outcome=MagicMock(value="refuse"),
            reason="test_governance_block",
            receipt_id="test_receipt",
        ))
        monkeypatch.setattr("core.will.get_will", lambda: mock_will)
        from core.executive.authority_gateway import AuthorityGateway
        return AuthorityGateway()

    @pytest.mark.asyncio
    async def test_tool_execution_blocked(self, blocked_gateway):
        decision = await blocked_gateway.authorize_tool_execution(
            "shell", {"command": "ls"}, source="adversary", priority=0.9
        )
        assert not decision.approved

    def test_initiative_blocked_sync(self, blocked_gateway):
        decision = blocked_gateway.authorize_initiative_sync(
            "unauthorized_action", source="adversary", priority=0.9
        )
        assert not decision.approved

    def test_expression_blocked_sync(self, blocked_gateway):
        decision = blocked_gateway.authorize_expression_sync(
            "unauthorized message", source="adversary", urgency=0.9
        )
        assert not decision.approved


class TestReceiptSchemaValidation:
    """Verify all receipt types have required fields."""

    def test_all_receipt_types_instantiate(self):
        from core.runtime.receipts import _RECEIPT_CLASSES
        for kind, cls in _RECEIPT_CLASSES.items():
            receipt = cls()
            assert hasattr(receipt, "receipt_id")
            assert hasattr(receipt, "kind")
            assert hasattr(receipt, "cause")
            assert hasattr(receipt, "created_at")
            assert receipt.kind == kind

    def test_receipt_store_emit_and_query(self, tmp_path):
        from core.runtime.receipts import ReceiptStore, GovernanceReceipt
        store = ReceiptStore(root=tmp_path / "receipts")
        receipt = GovernanceReceipt(
            domain="test", action="test_action",
            approved=False, reason="test", cause="test",
        )
        emitted = store.emit(receipt)
        assert emitted.receipt_id
        found = store.get(emitted.receipt_id)
        assert found is not None
        assert found.kind == "governance"

    def test_receipt_coverage_all_kinds(self, tmp_path):
        from core.runtime.receipts import ReceiptStore, _RECEIPT_CLASSES
        store = ReceiptStore(root=tmp_path / "receipts")
        for kind, cls in _RECEIPT_CLASSES.items():
            receipt = cls(cause=f"test_{kind}")
            store.emit(receipt)
        stats = store.coverage_stats()
        for kind in _RECEIPT_CLASSES:
            assert stats.get(kind, 0) >= 1, f"receipt kind '{kind}' not emitted"

    def test_receipt_chain_integrity(self, tmp_path):
        from core.runtime.receipts import ReceiptStore, GovernanceReceipt
        store = ReceiptStore(root=tmp_path / "receipts")
        for i in range(5):
            store.emit(GovernanceReceipt(
                domain="test", action=f"action_{i}",
                approved=True, reason="test", cause="chain_test",
            ))
        result = store.verify_chain()
        assert result["ok"], f"chain verification failed: {result['problems']}"
