"""What a model costs is a property of the checkpoint, not of its folder name.

Two CP126 findings in how mlx_client sizes a model.

48f80787 — solver/cortex classification, recurrent depth, minimum RAM,
overhead and projected size all came from substrings of the PATH: "72b",
"32b", "cortex", "zenith", "q4", "fused-model". A directory named for one
model and holding another got the other model's memory gate and recurrence
contract, and a rename was enough to change either.

50d8ed03 — the size estimate walked every descendant and counted every file,
so tokenizer caches, training logs, adapters, receipts and temporary artifacts
inflated the footprint RAM admission is computed from, with no depth or count
ceiling.
"""
from __future__ import annotations

import pytest

from core.brain.llm import mlx_client

pytestmark = pytest.mark.unit


def _artifact(tmp_path, name="aura-32b", *, weight_mb=8, junk_mb=0, depth=0):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(b"\0" * (weight_mb * 1024 * 1024))
    if junk_mb:
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        (logs / "train.log").write_bytes(b"\0" * (junk_mb * 1024 * 1024))
        (root / "tokenizer.json").write_bytes(b"\0" * (junk_mb * 1024 * 1024))
    node = root
    for level in range(depth):
        node = node / f"deep{level}"
        node.mkdir(exist_ok=True)
        (node / "buried.safetensors").write_bytes(b"\0" * (4 * 1024 * 1024))
    return root


@pytest.fixture(autouse=True)
def _clear_size_cache():
    mlx_client._PATH_SIZE_CACHE.clear()
    yield
    mlx_client._PATH_SIZE_CACHE.clear()


# --- the footprint counts weights, not everything (50d8ed03) ------------


def test_only_weight_files_count_toward_the_footprint(tmp_path):
    """Logs and tokenizer caches are not loaded into memory, so they are not
    part of the footprint the RAM gate is computed from."""
    root = _artifact(tmp_path, weight_mb=8, junk_mb=16)

    size_gb = mlx_client._path_size_gb(str(root))

    # 8MB of weights, not 40MB of directory.
    assert size_gb == pytest.approx(8 / 1024, rel=0.2)


def test_the_scan_does_not_follow_an_arbitrary_tree(tmp_path):
    root = _artifact(tmp_path, weight_mb=8, depth=6)

    size_gb = mlx_client._path_size_gb(str(root))

    # The top level plus one, not all six.
    assert size_gb < (8 + 4 + 4) / 1024


def test_a_weight_file_one_level_down_is_still_counted(tmp_path):
    root = _artifact(tmp_path, weight_mb=8, depth=1)

    assert mlx_client._path_size_gb(str(root)) == pytest.approx(12 / 1024, rel=0.2)


def test_the_scan_is_bounded_by_file_count():
    assert mlx_client._MAX_ARTIFACT_FILES_SCANNED <= 4096
    assert mlx_client._MAX_ARTIFACT_SCAN_DEPTH <= 3


def test_a_single_file_artifact_still_measures(tmp_path):
    blob = tmp_path / "model.gguf"
    blob.write_bytes(b"\0" * (4 * 1024 * 1024))

    assert mlx_client._path_size_gb(str(blob)) == pytest.approx(4 / 1024, rel=0.2)


def test_a_missing_artifact_is_zero_not_an_error(tmp_path):
    assert mlx_client._path_size_gb(str(tmp_path / "nope")) == 0.0


def test_the_result_is_cached_per_path_and_mtime(tmp_path):
    root = _artifact(tmp_path, weight_mb=8)

    first = mlx_client._path_size_gb(str(root))
    assert mlx_client._PATH_SIZE_CACHE
    assert mlx_client._path_size_gb(str(root)) == first


# --- classification asks the artifact first (48f80787) ------------------


def test_the_measured_class_wins_over_the_directory_name(tmp_path, monkeypatch):
    """A directory named 72b that measures as 32b must get the 32b contract."""
    root = _artifact(tmp_path, name="aura-72b-solver")
    monkeypatch.setattr(mlx_client, "_measured_size_class", lambda _p: "32b")

    assert mlx_client._model_matches_class(str(root), ("32b",)) is True
    assert mlx_client._model_matches_class(str(root), ("72b", "solver")) is False


def test_the_name_is_used_only_when_the_artifact_cannot_be_read(monkeypatch):
    monkeypatch.setattr(mlx_client, "_measured_size_class", lambda _p: None)

    assert mlx_client._model_matches_class("/models/aura-72b-solver", ("72b",)) is True


def test_an_unknown_measurement_falls_back_to_the_name(monkeypatch):
    monkeypatch.setattr(mlx_client, "_measured_size_class", lambda _p: "unknown")

    assert mlx_client._model_matches_class("/models/aura-32b", ("32b",)) is True


def test_recurrent_depth_follows_the_measured_class(tmp_path, monkeypatch):
    monkeypatch.delenv("AURA_RECURRENT_LOOPS", raising=False)
    monkeypatch.setattr(mlx_client, "_measured_size_class", lambda _p: "32b")

    loops = mlx_client._expected_recurrent_loops_from_model_path("/models/whatever")

    # One since 2026-09-12: the interactive default went back to the identity
    # depth on the evidence (see the note above MODEL_PROFILE_DEFAULTS in
    # recurrent_depth.py), and the parent's mirror follows the table. What this
    # test holds is that the measured class is what decides, not the name.
    assert loops == 1
    monkeypatch.setattr(mlx_client, "_measured_size_class", lambda _p: "7b")
    assert mlx_client._expected_recurrent_loops_from_model_path("/models/whatever") == 1


def test_the_ram_gate_follows_the_measured_class(monkeypatch):
    monkeypatch.setattr(mlx_client, "_measured_size_class", lambda _p: "72b")

    required = mlx_client._model_load_min_available_gb("/models/tiny-name")

    assert required >= 34.0


def test_quantization_comes_from_the_checkpoint(monkeypatch):
    """It changes the footprint by more than a third; a rename must not move
    the admission gate."""
    monkeypatch.setattr(
        "core.brain.llm.model_artifact_profile.get_model_artifact_profile",
        lambda _p: type("P", (), {"quantization_bits": 4, "measured": True, "size_class": "32b"})(),
    )

    assert mlx_client._model_is_quantized("/models/no-hint-in-the-name") is True


def test_quantization_falls_back_to_the_name_when_unreadable(monkeypatch):
    monkeypatch.setattr(
        "core.brain.llm.model_artifact_profile.get_model_artifact_profile",
        lambda _p: type("P", (), {"quantization_bits": 0, "measured": False, "size_class": "unknown"})(),
    )

    assert mlx_client._model_is_quantized("/models/aura-32b-q4") is True
    assert mlx_client._model_is_quantized("/models/aura-32b-bf16") is False
