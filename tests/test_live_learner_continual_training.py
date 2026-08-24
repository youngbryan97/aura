import json
import sys
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import core.learning.live_learner as live_learner_module
from core.learning.live_learner import AdapterRegistry, LiveLearner, TrainingPolicy
from core.tasks.managed_command import ManagedCommandResult


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

    def fake_run(cmd, *, timeout_s):
        captured["cmd"] = cmd
        captured["timeout_s"] = timeout_s
        return ManagedCommandResult(tuple(cmd), 0, "ok", "", 0.1)

    monkeypatch.setattr(live_learner_module, "run_project_command", fake_run)

    ok, output = learner._run_lora_subprocess("/models/aura-base", data_dir, adapter_dir)

    assert ok is True
    assert output == "ok"
    cmd = captured["cmd"]
    assert cmd[:4] == (sys.executable, "-m", "mlx_lm", "lora")
    assert "--fine-tune-type" in cmd
    assert "lora" in cmd
    assert "--lora-rank" not in cmd
    assert "--lora-alpha" not in cmd
    assert "-c" in cmd
    config_text = (adapter_dir / "lora_config.yaml").read_text(encoding="utf-8")
    assert "rank: 12" in config_text
    assert "scale: 24.0" in config_text


def test_mlx_command_reports_managed_timeout(monkeypatch, tmp_path):
    policy = TrainingPolicy(iters=3, timeout_seconds=90)
    learner = _bare_learner(tmp_path, policy=policy)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    def fake_run(cmd, *, timeout_s):
        return ManagedCommandResult(tuple(cmd), None, "", "", timeout_s, timed_out=True)

    monkeypatch.setattr(live_learner_module, "run_project_command", fake_run)

    ok, output = learner._run_lora_subprocess("/models/aura-base", data_dir, adapter_dir)

    assert ok is False
    assert output == "timeout after 90 seconds"


def test_adapter_registry_malformed_json_starts_empty(tmp_path):
    adapter_base = tmp_path / "adapters"
    adapter_base.mkdir()
    (adapter_base / "registry.json").write_text("{bad json", encoding="utf-8")

    registry = AdapterRegistry(adapter_base)

    assert registry.list_versions() == []


def test_adapter_registry_activation_is_exclusive_and_rollback_is_durable(tmp_path):
    first = tmp_path / "adapters" / "first"
    second = tmp_path / "adapters" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    registry = AdapterRegistry(tmp_path / "adapters")

    registry.register(str(first), 10, benchmark_passed=True, active=True)
    registry.register(str(second), 10, benchmark_passed=True, active=True)

    versions = list(reversed(registry.list_versions()))
    assert [v["active"] for v in versions] == [False, True]
    assert registry.rollback() == str(first)

    reloaded = AdapterRegistry(tmp_path / "adapters")
    versions = list(reversed(reloaded.list_versions()))
    assert [v["active"] for v in versions] == [True, False]


def test_live_learner_benchmark_scoring_rejects_banned_regressions():
    score, failures = LiveLearner._score_benchmark_response(
        "As an AI language model, I cannot answer.",
        must_contain=["aura", "i am", "i'm"],
        must_not_contain=["language model", "i cannot"],
    )

    assert score < 1.0
    assert any("language model" in failure for failure in failures)
    assert any("missing" in failure for failure in failures)


def test_fused_live_learning_output_is_candidate_not_active_pointer(tmp_path):
    learner = _bare_learner(tmp_path)
    base_model = tmp_path / "base-model"
    candidate = tmp_path / "candidate-model"
    for model, payload in ((base_model, b"base"), (candidate, b"candidate")):
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 8,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "vocab_size": 128,
                    "max_position_embeddings": 4096,
                    "quantization": {"bits": 4, "group_size": 64},
                }
            ),
            encoding="utf-8",
        )
        (model / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes(payload)
    learner._model_path = str(base_model)

    receipt = learner._record_fused_model_candidate(
        candidate,
        base_model=base_model,
        tag="live-learner",
        metadata={"benchmark": "passed"},
    )

    assert receipt["qualification_state"] == "awaiting_evaluation"
    assert receipt["active_pointer_unchanged"] is True
    assert not learner._active_model_manifest.exists()
    assert Path(receipt["candidate_receipt_path"]).exists()


async def test_training_cycle_never_hot_swaps_fused_candidate(monkeypatch, tmp_path):
    learner = _bare_learner(
        tmp_path,
        policy=TrainingPolicy(
            publish_fused_model=True,
            max_examples_per_run=40,
            replay_fraction=0.0,
        ),
    )
    learner._training_in_progress = False
    learner._last_train_time = 0.0
    learner._adapter_registry = AdapterRegistry(tmp_path / "registry")
    learner._buffer.extend(_example(i) for i in range(30))
    fused_candidate = tmp_path / "fused-candidate"
    fused_candidate.mkdir()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        learner,
        "_write_training_dataset",
        lambda _examples, adapter_dir: (
            adapter_dir / "data",
            {"train": 27, "valid": 2, "test": 1},
        ),
    )
    monkeypatch.setattr(learner, "_run_lora_subprocess", lambda *_args: (True, "ok"))
    monkeypatch.setattr(
        learner,
        "_run_fuse_subprocess",
        lambda *_args: (True, "ok", fused_candidate),
    )

    async def _benchmark(adapter_dir, *, promoted_model_path=None):
        observed["benchmark_adapter"] = adapter_dir
        observed["benchmark_fused"] = promoted_model_path
        return True, []

    def _record(model_path, **_kwargs):
        observed["candidate"] = model_path
        return {"candidate_receipt_path": str(tmp_path / "candidate.json")}

    async def _swap(path):
        observed["swapped"] = path
        return True

    monkeypatch.setattr(learner, "_run_benchmark", _benchmark)
    monkeypatch.setattr(learner, "_record_fused_model_candidate", _record)
    monkeypatch.setattr(learner, "_hot_swap_adapter", _swap)
    monkeypatch.setattr(learner, "_compute_quality_delta", lambda: 0.1)

    assert await learner._run_training_cycle() is True
    assert observed["candidate"] == fused_candidate
    assert observed["benchmark_fused"] == fused_candidate
    assert observed["swapped"] != str(fused_candidate)
    assert observed["swapped"] == str(observed["benchmark_adapter"])


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


def test_record_tick_persists_training_row_under_strict_governance(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "strict")
    learner = _bare_learner(tmp_path)
    state = SimpleNamespace(
        identity=SimpleNamespace(current_narrative="steady"),
        phi=0.7,
    )

    score = learner.record_tick(
        state,
        user_input="What did you repair?",
        response="I repaired the learner persistence path and kept the runtime evidence attached.",
        affect={"valence": 0.6, "curiosity": 0.9},
    )

    assert score is not None
    assert score.worth_training is True
    rows = learner._buffer_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["messages"][1]["content"] == "What did you repair?"


def test_record_tick_refuses_silent_repair_prompts_as_training_data(tmp_path):
    learner = _bare_learner(tmp_path)
    state = SimpleNamespace(identity=SimpleNamespace(current_narrative="steady"), phi=0.7)

    score = learner.record_tick(
        state,
        user_input="[SILENT AUTO-FIX] Investigate a timeout. Error: KernelInterface chat timed out. Handle this silently.",
        response="I'm processing that, but I haven't reached a verbal conclusion yet.",
        affect={"valence": 0.6, "curiosity": 0.9},
    )

    assert score is not None
    assert score.worth_training is False
    assert any("training_contamination" in reason for reason in score.reasons_negative)
    assert len(learner._buffer) == 0


def test_clean_training_example_rejects_traceback_and_assistant_regression(tmp_path):
    learner = _bare_learner(tmp_path)
    contaminated = {
        "messages": [
            {"role": "system", "content": "You are Aura."},
            {"role": "user", "content": "Traceback (most recent call last): ValueError"},
            {"role": "assistant", "content": "As an AI language model, I cannot answer."},
        ],
        "_quality": 0.9,
    }

    assert learner._clean_training_example(contaminated) is None


def test_live_learner_load_buffer_skips_contaminated_rows(tmp_path):
    learner = _bare_learner(tmp_path)
    good = _example(1)
    bad = {
        "messages": [
            {"role": "user", "content": "[SILENT AUTO-FIX] Fix a data access error. Handle this silently."},
            {"role": "assistant", "content": "I'm having trouble formulating a response."},
        ],
        "_quality": 0.9,
    }
    learner._buffer_path.write_text(
        json.dumps(good) + "\n" + json.dumps(bad) + "\n",
        encoding="utf-8",
    )

    learner._load_buffer()

    assert len(learner._buffer) == 1
    assert learner._buffer[0]["messages"][1]["content"] == "question 1"
    assert learner._buffer_path.read_text(encoding="utf-8") == json.dumps(good) + "\n"

    quarantine = tmp_path / "experience_buffer.quarantine.jsonl"
    quarantined = [json.loads(line) for line in quarantine.read_text(encoding="utf-8").splitlines()]
    assert len(quarantined) == 1
    assert "silent_autofix_prompt" in quarantined[0]["reasons"]

    learner._buffer.clear()
    learner._load_buffer()
    assert len(learner._buffer) == 1
    assert len(quarantine.read_text(encoding="utf-8").splitlines()) == 1
