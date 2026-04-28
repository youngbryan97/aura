import json
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from core.learning.live_learner import LiveLearner, TrainingPolicy


def _example(i: int, quality: float = 0.8) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are Aura."},
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        ],
        "_quality": quality,
        "_timestamp": 1_700_000_000 + i,
    }


def _bare_learner(tmp_path: Path, *, policy: TrainingPolicy | None = None) -> LiveLearner:
    learner = LiveLearner.__new__(LiveLearner)
    learner._policy = policy or TrainingPolicy(max_examples_per_run=40, replay_fraction=0.35)
    learner._buffer = deque(maxlen=5000)
    learner._lock = threading.Lock()
    learner._session_scores = []
    learner._active = False
    learner._training_task = None
    learner._buffer_path = tmp_path / "experience_buffer.jsonl"
    learner._buffer_path.parent.mkdir(parents=True, exist_ok=True)
    learner._data_dir = tmp_path
    learner._fused_dir = tmp_path / "fused-model"
    learner._active_model_manifest = learner._fused_dir / "active.json"
    learner._model_path = "/models/aura-base"
    return learner


def test_training_policy_refuses_full_weight_updates_without_explicit_unlock(monkeypatch):
    monkeypatch.setenv("AURA_SELF_TRAIN_FINE_TUNE_TYPE", "full")
    monkeypatch.delenv("AURA_SELF_TRAIN_ALLOW_FULL_WEIGHTS", raising=False)

    policy = TrainingPolicy.from_env()

    assert policy.fine_tune_type == "lora"
    assert policy.allow_full_weights is False


def test_training_policy_allows_full_weight_updates_when_unlocked(monkeypatch):
    monkeypatch.setenv("AURA_SELF_TRAIN_FINE_TUNE_TYPE", "full")
    monkeypatch.setenv("AURA_SELF_TRAIN_ALLOW_FULL_WEIGHTS", "1")

    policy = TrainingPolicy.from_env()

    assert policy.fine_tune_type == "full"
    assert policy.allow_full_weights is True


def test_training_dataset_writes_mlx_splits_and_strips_private_metadata(tmp_path):
    learner = _bare_learner(tmp_path)
    examples = [_example(i, quality=0.9 - (i * 0.001)) for i in range(30)]

    data_dir, counts = learner._write_training_dataset(examples, tmp_path / "adapter")

    assert counts == {"train": 27, "valid": 2, "test": 1}
    assert (data_dir / "train.jsonl").exists()
    assert (data_dir / "valid.jsonl").exists()
    assert (data_dir / "test.jsonl").exists()

    first = json.loads((data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "messages" in first
    assert "_quality" not in first
    assert first["messages"][1]["content"] == "question 0"


def test_replay_selection_preserves_old_high_quality_examples_under_pressure(tmp_path):
    policy = TrainingPolicy(max_examples_per_run=20, replay_fraction=0.4)
    learner = _bare_learner(tmp_path, policy=policy)
    with learner._lock:
        for i in range(80):
            learner._buffer.append(_example(i, quality=1.0 - i * 0.005))

    selected = learner._select_training_examples()
    selected_questions = {
        row["messages"][1]["content"]
        for row in selected
    }

    assert len(selected) == 20
    assert "question 0" in selected_questions
    assert any(int(q.split()[-1]) >= 12 for q in selected_questions)


def test_mlx_command_uses_supported_config_flags_and_no_removed_lora_flags(monkeypatch, tmp_path):
    policy = TrainingPolicy(
        fine_tune_type="lora",
        iters=3,
        batch_size=1,
        num_layers=4,
        rank=12,
        scale=24.0,
        save_every=3,
        timeout_seconds=90,
    )
    learner = _bare_learner(tmp_path, policy=policy)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(json.dumps(_example(1)) + "\n", encoding="utf-8")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, output = learner._run_lora_subprocess("/models/aura-base", data_dir, adapter_dir)

    assert ok is True
    assert output == "ok"
    cmd = captured["cmd"]
    assert cmd[:4] == [sys.executable, "-m", "mlx_lm", "lora"]
    assert "--fine-tune-type" in cmd
    assert "lora" in cmd
    assert "--lora-rank" not in cmd
    assert "--lora-alpha" not in cmd
    assert "-c" in cmd
    config_text = (adapter_dir / "lora_config.yaml").read_text(encoding="utf-8")
    assert "rank: 12" in config_text
    assert "scale: 24.0" in config_text


def test_record_tick_accepts_affect_payload_without_state_affect_object(tmp_path):
    learner = _bare_learner(tmp_path)
    state = SimpleNamespace(
        identity=SimpleNamespace(current_narrative="steady"),
        phi=0.7,
    )

    score = learner.record_tick(
        state,
        user_input="What changed?",
        response="I tracked the failure, corrected the training path, and kept the rollback surface intact.",
        affect={"valence": 0.6, "curiosity": 0.9},
    )

    assert score is not None
    assert score.worth_training is True
    assert len(learner._buffer) == 1
