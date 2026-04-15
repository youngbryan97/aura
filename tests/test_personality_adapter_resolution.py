import json

from core.brain.llm.model_registry import resolve_personality_adapter


def test_mlx_personality_adapter_requires_compatible_model(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "aura-personality"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_text("stub")
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"model": "models/Qwen2.5-32B-Instruct-8bit"})
    )

    monkeypatch.setenv("AURA_LORA_PATH", str(adapter_dir))
    monkeypatch.delenv("AURA_LORA_TARGET_MODEL", raising=False)

    assert (
        resolve_personality_adapter("/models/Qwen2.5-32B-Instruct-8bit", backend="mlx")
        == str(adapter_dir)
    )
    assert resolve_personality_adapter("/models/Qwen2.5-7B-Instruct-4bit", backend="mlx") is None


def test_gguf_personality_adapter_can_be_hard_pinned_to_target_model(monkeypatch, tmp_path):
    adapter_file = tmp_path / "aura-personality-lora.gguf"
    adapter_file.write_text("stub")

    monkeypatch.setenv("AURA_GGUF_LORA_PATH", str(adapter_file))
    monkeypatch.setenv("AURA_GGUF_LORA_TARGET_MODEL", "Qwen2.5-32B-Instruct-8bit")

    assert (
        resolve_personality_adapter(
            "/models_gguf/qwen2.5-32b-instruct-q5_k_m.gguf",
            backend="gguf",
        )
        == str(adapter_file)
    )
    assert (
        resolve_personality_adapter(
            "/models_gguf/qwen2.5-7b-instruct-q4_k_m.gguf",
            backend="gguf",
        )
        is None
    )
