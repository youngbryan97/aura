from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from core.container import ServiceContainer
from core.config import SecurityConfig
from core.executive.authority_gateway import AuthorityGateway
from core.skills.malware_analysis import MalwareAnalysisSkill
from core.version import VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_authority_gateway_fails_closed_when_will_unavailable(monkeypatch):
    import core.will as will_module

    def _boom():
        raise RuntimeError("will offline")

    monkeypatch.setattr(will_module, "get_will", _boom)

    gateway = AuthorityGateway()
    decision = gateway.authorize_expression_sync(
        "Send unauthorized message",
        source="audit_probe",
        urgency=0.9,
    )

    assert not decision.approved
    assert decision.outcome == "will_unavailable"
    assert "offline" in decision.reason


def test_version_contract_is_consistent():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    canonical_version = pyproject["project"]["version"]
    requires_python = pyproject["project"]["requires-python"]

    assert VERSION.split("-", 1)[0] == canonical_version
    assert requires_python == ">=3.12"

    shell_pkg = json.loads(
        (ROOT / "interface" / "static" / "shell" / "package.json").read_text(encoding="utf-8")
    )
    memory_pkg = json.loads(
        (ROOT / "interface" / "static" / "memory" / "package.json").read_text(encoding="utf-8")
    )

    assert shell_pkg["version"] == canonical_version
    assert memory_pkg["version"] == canonical_version

    mycelial = (ROOT / "interface" / "static" / "mycelial.html").read_text(encoding="utf-8")
    assert VERSION in mycelial

    requirements_header = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()[0]
    assert canonical_version in requirements_header

    server_py = (ROOT / "interface" / "server.py").read_text(encoding="utf-8")
    assert 'title="Aura Luna Agent"' in server_py
    assert 'description="Secure interface for the Aura Luna autonomous engine."' in server_py

    telemetry = (ROOT / "interface" / "static" / "telemetry.html").read_text(encoding="utf-8")
    assert "Aura Luna | Telemetry HUD" in telemetry
    assert "Aura Luna Engine Initialized" in telemetry


def test_runtime_python_guard_matches_pyproject_contract():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires_python = pyproject["project"]["requires-python"]
    aura_main = (ROOT / "aura_main.py").read_text(encoding="utf-8")

    assert requires_python == ">=3.12"
    assert "if sys.version_info < (3, 12):" in aura_main
    assert "Aura requires Python 3.12+" in aura_main


def test_default_security_posture_is_explicitly_operator_autonomous():
    security = SecurityConfig()

    assert security.security_profile == "owner_autonomous"
    assert security.internal_only_mode is False
    assert security.auto_fix_enabled is True
    assert security.aura_full_autonomy is True
    assert security.allow_network_access is True
    assert security.allowed_domains == ["*"]
    assert security.enable_stealth_mode is True


def test_ownership_paths_exist():
    ownership = (ROOT / "OWNERSHIP.md").read_text(encoding="utf-8").splitlines()
    paths = []
    for line in ownership:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().split("|")[1:-1]]
        if len(cells) < 3:
            continue
        candidate = cells[-1].strip("`")
        if candidate.startswith("core/"):
            paths.append(candidate)

    assert paths, "OWNERSHIP.md should enumerate canonical file owners."
    missing = [path for path in paths if not (ROOT / path).exists()]
    assert missing == []


@pytest.mark.parametrize(
    "relative_path",
    [
        "skills/clock.py",
        "skills/curiosity.py",
        "skills/environment_info.py",
        "skills/file_operation.py",
        "skills/inter_agent_comm.py",
        "skills/internal_sandbox.py",
        "skills/listen.py",
        "skills/malware_analysis.py",
        "skills/memory_ops.py",
        "skills/native_chat.py",
        "skills/personality_skill.py",
        "skills/self_evolution.py",
        "skills/self_improvement.py",
        "skills/self_repair.py",
        "skills/sleep.py",
        "skills/social_lurker.py",
        "skills/uplink_local.py",
    ],
)
def test_legacy_skill_modules_are_thin_wrappers(relative_path: str):
    content = (ROOT / relative_path).read_text(encoding="utf-8")
    assert "core.skills." in content
    assert len(content.splitlines()) <= 4


def test_legacy_train_self_wrapper_preserves_workspace_root(tmp_path):
    from skills.train_self import TrainSelfSkill

    skill = TrainSelfSkill(workspace_root=str(tmp_path))
    assert skill.dataset_path == tmp_path / "data" / "training" / "dataset.jsonl"


def test_legacy_speak_wrapper_preserves_voice_alias():
    from skills.speak import SpeakSkill, VoiceSkill

    assert VoiceSkill is SpeakSkill


def test_tracked_avatar_asset_is_self_contained():
    svg_path = ROOT / "interface" / "static" / "aura_avatar.svg"
    assert svg_path.exists()

    aura_css = (ROOT / "interface" / "static" / "aura.css").read_text(encoding="utf-8")
    shell_css = (
        ROOT / "interface" / "static" / "shell" / "src" / "shell.css"
    ).read_text(encoding="utf-8")

    assert "/static/aura_avatar.svg" in aura_css
    assert "aura_avatar.svg" in shell_css
    assert "/static/aura_avatar.png" not in aura_css
    assert "/static/aura_avatar.png" not in shell_css


def test_malware_analysis_records_learning_without_name_error(monkeypatch):
    recorded = []

    class Learning:
        def record_threat(self, payload):
            recorded.append(payload)

    ServiceContainer.register_instance("learning_system", Learning(), required=False)
    skill = MalwareAnalysisSkill()

    skill._record_threat_intelligence({"sha256": "abc"})

    assert recorded == [{"sha256": "abc"}]
