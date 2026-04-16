import pytest

import core.brain.llm.model_registry as model_registry


def test_default_deep_model_prefers_mlx_artifact_for_mlx_backend():
    assert model_registry._default_deep_model_name(backend="mlx") == "Qwen2.5-72B-Instruct-4bit"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("mlx", "Qwen2.5-72B-Instruct-4bit"),
        ("llama_cpp", "Qwen2.5-72B-Instruct-Q4"),
    ],
)
def test_normalize_runtime_model_name_respects_backend(backend, expected):
    assert (
        model_registry.normalize_runtime_model_name(
            "Qwen2.5-72B-Instruct-Q4",
            backend=backend,
        )
        == expected
    )


def test_get_model_path_maps_q4_alias_to_existing_mlx_model_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "models" / "Qwen2.5-72B-Instruct-4bit"
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(model_registry, "BASE_DIR", tmp_path)
    monkeypatch.setattr(model_registry, "LOCAL_BACKEND", "mlx")
    monkeypatch.setitem(
        model_registry.MODEL_PATHS,
        "Qwen2.5-72B-Instruct-4bit",
        model_dir,
    )
    monkeypatch.setitem(
        model_registry.MODEL_PATHS,
        "Qwen2.5-72B-Instruct-Q4",
        tmp_path / "models" / "Qwen2.5-72B-Instruct-Q4",
    )

    resolved = model_registry.get_model_path("Qwen2.5-72B-Instruct-Q4")

    assert resolved == str(model_dir.resolve())


def test_get_model_path_preserves_missing_absolute_paths(monkeypatch, tmp_path):
    missing = tmp_path / "missing-model"
    monkeypatch.setattr(model_registry, "LOCAL_BACKEND", "mlx")

    assert model_registry.get_model_path(str(missing)) == str(missing)
