"""The active configuration plane cannot enable a remote model provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.llm_health_router import HealthAwareLLMRouter
from core.config import LLMConfig, SecurityConfig
from core.runtime.settings_schema import (
    DEFAULT_VALUES,
    SCHEMA_BY_KEY,
    SETTINGS_SCHEMA_VERSION,
    migrated_settings_snapshot,
    validate_settings_patch,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_models_have_no_remote_model_credentials_or_teacher_controls():
    assert {
        "api_key",
        "gemini_api_key",
        "teacher_model",
    }.isdisjoint(LLMConfig.model_fields)
    assert {
        "allow_cloud_teacher_distillation",
        "redact_personal_data_to_model_providers",
    }.isdisjoint(SecurityConfig.model_fields)


def test_config_fallbacks_name_runtime_roles_instead_of_retired_artifacts():
    llm = LLMConfig()

    assert llm.fast_model == "Aura-Cortex"
    assert llm.deep_model == "Aura-Cortex"
    assert llm.vision_model == "Aura-Cortex"
    assert llm.chat_model == "Qwen3.5-9B-4bit"


def test_boot_configuration_never_reads_the_retired_gemini_secret():
    config_source = (ROOT / "core/config.py").read_text(encoding="utf-8")
    baseline_source = (
        ROOT / "core/orchestrator/initializers/core_baseline.py"
    ).read_text(encoding="utf-8")

    assert "GEMINI_API_KEY" not in config_source
    assert "GEMINI_API_KEY" not in baseline_source
    assert "_gemini_key" not in baseline_source


def test_retired_cloud_setting_is_rejected_and_removed_during_migration():
    key = "model.cloud_fallback_enabled"

    assert SETTINGS_SCHEMA_VERSION == 3
    assert key not in SCHEMA_BY_KEY
    assert key not in DEFAULT_VALUES
    with pytest.raises(KeyError, match="unknown_setting:model.cloud_fallback_enabled"):
        validate_settings_patch({key: True})

    values, unknown = migrated_settings_snapshot({key: True})
    assert key not in values
    assert unknown == (key,)


def test_first_run_wizard_has_no_remote_model_control():
    source = (ROOT / "interface/static/first_run.js").read_text(encoding="utf-8")

    assert "cloud_fallback" not in source
    assert "cloud provider" not in source
    assert "available local model lanes" in source
    assert "Models/Aura-Cortex" in source
    assert "Qwen2.5-32B" not in source


def test_nethack_runner_defaults_to_the_runtime_cortex_role():
    source = (ROOT / "scripts/nethack_runner.sh").read_text(encoding="utf-8")

    assert '${AURA_MODEL:=Aura-Cortex}' in source
    assert "Qwen2.5-32B" not in source


def test_retired_provider_implementation_and_offload_router_are_absent():
    assert not (ROOT / "core/brain/llm/gemini_adapter.py").exists()
    assert not (ROOT / "core/brain/llm/cloud_errors.py").exists()
    assert not (ROOT / "core/brain/compute_router.py").exists()


@pytest.mark.parametrize(
    "relative_path",
    ("pyproject.toml", "requirements.txt", "requirements/core.txt", "requirements_lock.txt"),
)
def test_dependency_manifests_do_not_install_the_retired_provider(relative_path):
    content = (ROOT / relative_path).read_text(encoding="utf-8").lower()
    assert "google-genai" not in content
    assert "google.genai" not in content


def test_health_router_rejects_every_remote_model_endpoint():
    router = HealthAwareLLMRouter()

    with pytest.raises(ValueError, match="remote model providers are not supported"):
        router.register(
            name="external",
            url="https://model-provider.invalid/v1",
            model="external-model",
            is_local=False,
            tier="api_fast",
        )

    assert router.endpoints == {}


def test_active_model_provider_modules_have_no_retired_sdk_or_secret_surface():
    provider_paths = (
        "core/adapters/api_adapter.py",
        "core/brain/inference_gate.py",
        "core/brain/llm/autonomous_brain_integration.py",
        "core/brain/llm_health_router.py",
        "core/adaptation/distillation_pipe.py",
    )
    forbidden = ("google.genai", "google_genai", "gemini_api_key", "geminiadapter")

    for relative_path in provider_paths:
        content = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert all(token not in content for token in forbidden), relative_path


def test_system_api_has_no_remote_model_usage_endpoint():
    source = (ROOT / "interface/routes/system.py").read_text(encoding="utf-8")
    assert '"/gemini-usage"' not in source
    assert "get_gemini_usage" not in source


@pytest.mark.parametrize(
    "relative_path",
    ("MODEL_CARD.md", "AI_SYSTEM_CARD.md", "HARDWARE_PROFILES.md"),
)
def test_current_cards_make_no_gemini_or_cloud_model_claim(relative_path):
    content = (ROOT / relative_path).read_text(encoding="utf-8").lower()

    assert "gemini" not in content
    assert "cloud fallback" not in content
    assert "cloud / external model profile" not in content
