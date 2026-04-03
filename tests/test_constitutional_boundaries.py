from pathlib import Path
from types import SimpleNamespace

from core.middleware.capability_guard import CapabilityGuard
from core.self_modification.safe_modification import SafeSelfModification


def test_capability_guard_blocks_restricted_write_even_with_global_allow():
    guard = CapabilityGuard()
    protected = Path("core/security/trust_engine.py")
    allowed = Path("core/brain/llm/mlx_client.py")

    assert guard.can_write_path(str(protected)) is False
    assert guard.can_write_path(str(allowed)) is True


def test_safe_self_modification_blocks_constitutionally_protected_paths(tmp_path):
    code_base = tmp_path / "repo"
    (code_base / "core" / "security").mkdir(parents=True)
    (code_base / "core" / "brain").mkdir(parents=True)
    (code_base / "core" / "security" / "trust_engine.py").write_text("x = 1\n", encoding="utf-8")
    (code_base / "core" / "brain" / "module.py").write_text("x = 1\n", encoding="utf-8")

    modifier = SafeSelfModification(code_base_path=str(code_base))

    protected_fix = SimpleNamespace(
        target_file="core/security/trust_engine.py",
        risk_level=1,
        lines_changed=1,
        replacement_content="x = 2\n",
        content="x = 2\n",
    )
    allowed_fix = SimpleNamespace(
        target_file="core/brain/module.py",
        risk_level=1,
        lines_changed=1,
        replacement_content="x = 2\n",
        content="x = 2\n",
    )

    allowed, reason = modifier.validate_proposal(protected_fix)
    assert allowed is False
    assert "constitutionally protected" in reason.lower()

    allowed, reason = modifier.validate_proposal(allowed_fix)
    assert allowed is True
