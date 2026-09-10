import json

import core.brain.llm.model_paths as model_paths
from core.brain.llm.model_registry import resolve_personality_adapter

ADAPTER_TEST_PAYLOAD = "adapter-test-payload"


def test_mlx_personality_adapter_requires_compatible_model(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "aura-personality"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_text(ADAPTER_TEST_PAYLOAD)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"model": "models/Qwen2.5-32B-Instruct-8bit"})
    )

    monkeypatch.setenv("AURA_LORA_PATH", str(adapter_dir))
    monkeypatch.delenv("AURA_LORA_TARGET_MODEL", raising=False)
    monkeypatch.delenv("AURA_DISABLE_PERSONALITY_LORA", raising=False)

    assert (
        resolve_personality_adapter("/models/Qwen2.5-32B-Instruct-8bit", backend="mlx")
        == str(adapter_dir)
    )
    assert resolve_personality_adapter("/models/Qwen2.5-7B-Instruct-4bit", backend="mlx") is None


def test_default_mlx_personality_adapter_is_opt_in(monkeypatch, tmp_path):
    default_dir = tmp_path / "training" / "adapters" / "aura-personality"
    default_dir.mkdir(parents=True)
    (default_dir / "adapters.safetensors").write_text(ADAPTER_TEST_PAYLOAD)
    (default_dir / "adapter_config.json").write_text(
        json.dumps({"model": "models/Qwen2.5-32B-Instruct-8bit"})
    )

    # model_paths owns the base directory; the name bound in model_registry is
    # only for the frozen path tables built at import. Patching that name left
    # the call reading the real repository, which is the exact lie its own
    # comment warns about ("a second name for one value is how a patched test
    # comes to lie about what it patched").
    monkeypatch.setattr(model_paths, "BASE_DIR", tmp_path)
    monkeypatch.delenv("AURA_LORA_PATH", raising=False)
    monkeypatch.delenv("AURA_ENABLE_PERSONALITY_LORA", raising=False)
    monkeypatch.delenv("AURA_ENABLE_MLX_LORA", raising=False)
    monkeypatch.delenv("AURA_DISABLE_PERSONALITY_LORA", raising=False)

    assert resolve_personality_adapter("/models/Qwen2.5-32B-Instruct-8bit", backend="mlx") is None

    monkeypatch.setenv("AURA_ENABLE_PERSONALITY_LORA", "1")

    assert (
        resolve_personality_adapter("/models/Qwen2.5-32B-Instruct-8bit", backend="mlx")
        == str(default_dir)
    )


def test_personality_adapter_disable_overrides_explicit_path(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "aura-personality"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_text(ADAPTER_TEST_PAYLOAD)

    monkeypatch.setenv("AURA_LORA_PATH", str(adapter_dir))
    monkeypatch.setenv("AURA_DISABLE_PERSONALITY_LORA", "1")

    assert resolve_personality_adapter("/models/Qwen2.5-32B-Instruct-8bit", backend="mlx") is None


def test_non_mlx_personality_adapter_backend_is_retired(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "aura-personality"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_text(ADAPTER_TEST_PAYLOAD)

    monkeypatch.setenv("AURA_LORA_PATH", str(adapter_dir))

    assert resolve_personality_adapter("/models/Qwen2.5-32B-Instruct-8bit", backend="retired") is None
