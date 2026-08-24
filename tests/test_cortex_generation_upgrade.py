"""Contract tests: the cortex generation-upgrade pipeline.

The pipeline's promises, proven on real machinery:
- the capability battery runs real decodes and DISCRIMINATES (a sabotaged
  model scores measurably worse than its healthy twin);
- the comparison verdict demands a Pareto gain without either-axis regression;
- the memory guard refuses candidates the host cannot afford;
- staging writes a byte-exact rollback and changes nothing live;
- activation is impossible without operator authorization + PASS verdict;
- rollback restores the pointer byte-exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.model_artifact_profile import (  # noqa: E402
    build_model_artifact_descriptor,
    build_model_serving_profile,
)
from core.learning.cortex_generation_upgrade import (  # noqa: E402
    IDENTITY_BACKUP_POINTER_NAME,
    ROLLBACK_POINTER_NAME,
    STAGED_POINTER_NAME,
    MemoryGuard,
    _answer_matches,
    _greedy_decode,
    activate_upgrade,
    build_migration_contract,
    build_migration_plan,
    capability_battery,
    compare_batteries,
    normalize_active_pointer_identity,
    record_upgrade_candidate,
    rollback_upgrade,
    stage_upgrade,
)
from tests.support.cortex_migration_authority import (  # noqa: E402
    build_signed_migration_authorities,
)


@pytest.fixture(autouse=True)
def _isolated_state_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) % 127 + 1 for c in str(text)][:32] or [5]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class ThinkingTokenizer(TinyTokenizer):
    chat_template = "{{ enable_thinking }}"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        return "think" if kwargs.get("enable_thinking") else "direct"


def _model(seed=0):
    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    mx.random.seed(seed)
    model = Model(args)
    mx.eval(model.parameters())
    return model


# ── Battery ─────────────────────────────────────────────────────────────


def test_battery_runs_and_is_deterministic():
    model = _model()
    first = capability_battery(model, TinyTokenizer(), label="tiny")
    second = capability_battery(model, TinyTokenizer(), label="tiny")
    assert first["breadth_total"] == 24 and first["reasoning_total"] == 12
    assert first["breadth_accuracy"] == second["breadth_accuracy"]
    assert first["identity_digests"] == second["identity_digests"]
    assert len(first["identity_digests"]) >= 8


def test_battery_discriminates_a_sabotaged_model():
    healthy = _model(seed=0)
    healthy_receipt = capability_battery(healthy, TinyTokenizer(), label="healthy")

    wrecked = _model(seed=0)
    layer = wrecked.model.layers[4]
    layer.mlp.down_proj.weight = layer.mlp.down_proj.weight + mx.random.normal(
        layer.mlp.down_proj.weight.shape, key=mx.random.key(9)
    )
    wrecked_receipt = capability_battery(wrecked, TinyTokenizer(), label="wrecked")
    # Identity behavior MUST move when the weights are wrecked — the battery
    # sees real model behavior, not fixtures.
    assert healthy_receipt["identity_digests"] != wrecked_receipt["identity_digests"]


def test_battery_decode_uses_typed_native_thinking_mode():
    tokenizer = ThinkingTokenizer()
    model = _model()

    _greedy_decode(model, tokenizer, "simple", max_tokens=1, cognitive_mode="reactive")
    assert tokenizer.calls[-1]["enable_thinking"] is False

    _greedy_decode(model, tokenizer, "hard", max_tokens=1, cognitive_mode="deliberate")
    assert tokenizer.calls[-1]["enable_thinking"] is True


@pytest.mark.parametrize(
    ("answer", "accepted", "expected"),
    [
        ("The formula is **$H_2O$**.", ("h2o",), True),
        ("It was Charles Darwin.", ("darwin",), True),
        ("No relevant fact here.", ("darwin",), False),
    ],
)
def test_breadth_matching_ignores_display_markup(answer, accepted, expected):
    assert _answer_matches(answer, accepted) is expected


def test_breadth_battery_does_not_reintroduce_ten_token_clipping(monkeypatch):
    import core.learning.cortex_generation_upgrade as upgrade

    calls = []

    def decode(
        _model,
        _tokenizer,
        prompt,
        *,
        max_tokens,
        cognitive_mode,
        stop_strings=(),
    ):
        calls.append((prompt, max_tokens, cognitive_mode))
        if cognitive_mode == "reactive":
            assert stop_strings
        return "no matched answer"

    monkeypatch.setattr(upgrade, "_greedy_decode", decode)
    receipt = capability_battery(object(), TinyTokenizer(), label="budget-contract")

    breadth_calls = calls[: len(upgrade.BREADTH_PROBES)]
    assert all(max_tokens >= 32 for _prompt, max_tokens, _mode in breadth_calls)
    assert all(mode == "reactive" for _prompt, _max_tokens, mode in breadth_calls)
    assert all(row["max_tokens"] >= 256 for row in receipt["breadth_rows"])


def test_battery_resumes_exact_durable_cells_without_repeating_decode(monkeypatch):
    import core.learning.cortex_generation_upgrade as upgrade
    import core.learning.interference_battery as interference

    calls = []
    events = []

    def decode(
        _model,
        _tokenizer,
        prompt,
        *,
        max_tokens,
        cognitive_mode,
        stop_strings=(),
    ):
        calls.append((prompt, max_tokens, cognitive_mode, stop_strings))
        return "12 true false au darwin tokyo"

    monkeypatch.setattr(upgrade, "_greedy_decode", decode)
    monkeypatch.setattr(interference, "natural_stability_probes", lambda _tokenizer: [[1]])
    monkeypatch.setattr(
        interference,
        "snapshot_probe_behavior",
        lambda _model, _probes: [{"digest": "a" * 16}],
    )

    first = capability_battery(
        object(),
        TinyTokenizer(),
        label="resume",
        progress_callback=events.append,
    )
    assert len(calls) == 36
    assert len(events) == 37

    calls.clear()
    second = capability_battery(
        object(),
        TinyTokenizer(),
        label="resume",
        resume_events=events,
    )

    assert calls == []
    assert second["breadth_rows"] == first["breadth_rows"]
    assert second["reasoning_rows"] == first["reasoning_rows"]
    assert second["identity_digests"] == first["identity_digests"]


def test_battery_recomputes_a_resume_cell_when_its_prompt_binding_changed(monkeypatch):
    import core.learning.cortex_generation_upgrade as upgrade
    import core.learning.interference_battery as interference

    events = []

    def decode(
        _model,
        _tokenizer,
        _prompt,
        *,
        max_tokens,
        cognitive_mode,
        stop_strings=(),
    ):
        return "answer"

    monkeypatch.setattr(upgrade, "_greedy_decode", decode)
    monkeypatch.setattr(interference, "natural_stability_probes", lambda _tokenizer: [[1]])
    monkeypatch.setattr(
        interference,
        "snapshot_probe_behavior",
        lambda _model, _probes: [{"digest": "b" * 16}],
    )
    capability_battery(
        object(),
        TinyTokenizer(),
        label="resume",
        progress_callback=events.append,
    )
    events[0]["row"]["prompt"] = "tampered"

    repeated = []
    monkeypatch.setattr(
        upgrade,
        "_greedy_decode",
        lambda *args, **kwargs: repeated.append(args[2]) or "answer",
    )
    capability_battery(
        object(),
        TinyTokenizer(),
        label="resume",
        resume_events=events,
    )

    assert repeated == [upgrade.BREADTH_PROBES[0][0]]


def test_comparison_verdict_requires_pareto_gain_without_regression():
    current = {"label": "cur", "breadth_accuracy": 0.5, "reasoning_accuracy": 0.5,
               "identity_digests": ["a"]}
    better = {"label": "cand", "breadth_accuracy": 0.7, "reasoning_accuracy": 0.5,
              "identity_digests": ["b"]}
    worse_reasoning = {"label": "cand", "breadth_accuracy": 0.7,
                       "reasoning_accuracy": 0.3, "identity_digests": ["b"]}
    reasoning_gain = {"label": "cand", "breadth_accuracy": 0.5,
                      "reasoning_accuracy": 0.9, "identity_digests": ["b"]}
    unchanged = {"label": "cand", "breadth_accuracy": 0.5,
                 "reasoning_accuracy": 0.5, "identity_digests": ["b"]}
    breadth_regression = {"label": "cand", "breadth_accuracy": 0.4,
                          "reasoning_accuracy": 0.9, "identity_digests": ["b"]}
    assert compare_batteries(current, better)["verdict"] == "PASS"
    assert compare_batteries(current, worse_reasoning)["verdict"] == "FAIL"
    assert compare_batteries(current, reasoning_gain)["verdict"] == "PASS"
    assert compare_batteries(current, unchanged)["verdict"] == "FAIL"
    assert compare_batteries(current, breadth_regression)["verdict"] == "FAIL"
    assert compare_batteries(current, better)["identity_behavior_changed"] is True


# ── Memory guard ────────────────────────────────────────────────────────


def test_memory_guard_refuses_empty_and_oversized(tmp_path, monkeypatch):
    guard = MemoryGuard()
    empty = guard.admit(tmp_path)
    assert empty["admitted"] is False and "no weight files" in empty["refusal_reason"]

    # Availability is injected: the REAL host may legitimately be under
    # model-scale pressure (a 32B training run), and the guard must report
    # the injected world, not the test machine's mood.
    monkeypatch.setattr(MemoryGuard, "_available_gb", staticmethod(lambda: 32.0))
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)
    admitted = guard.admit(tmp_path)
    assert admitted["admitted"] is True

    giant_guard = MemoryGuard(free_margin_gb=10**6)  # impossible margin
    refused = giant_guard.admit(tmp_path)
    assert refused["admitted"] is False
    assert "headroom" in refused["refusal_reason"]

    monkeypatch.setattr(MemoryGuard, "_available_gb", staticmethod(lambda: 2.0))
    pressured = MemoryGuard().admit(tmp_path)
    assert pressured["admitted"] is False, "a strained host must refuse"


# ── Migration plan ──────────────────────────────────────────────────────


def test_migration_plan_names_real_artifacts_and_lanes(tmp_path):
    plan = build_migration_plan(
        fused_model_dir=tmp_path / "fused", data_dir=tmp_path / "data"
    )
    names = {step["name"] for step in plan["steps"]}
    assert {"activation_pointer", "persona_crsm_delta", "caa_steering_vectors",
            "expert_adapters", "recurrence_native_adapter"} <= names
    assert plan["automatic_steps"] == ["activation_pointer"]
    assert len(plan["operator_steps"]) == 4
    # Honest existence flags for a bare tmp dir.
    pointer = next(s for s in plan["steps"] if s["name"] == "activation_pointer")
    assert pointer["exists"] is False


# ── Stage / activate / rollback ─────────────────────────────────────────


def _fused_dir(tmp_path):
    fused = tmp_path / "fused-model"
    fused.mkdir()
    current = {
        "active_model_path": str(tmp_path / "current-model"),
        "base_model": "Qwen2.5-32B",
        "fused_at": 1000,
        "schema_version": 2,
        "size": "32B",
        "tag": "current",
    }
    current_model = tmp_path / "current-model"
    _write_model_artifact(current_model, b"current-weights", model_type="qwen2")
    (fused / "active.json").write_text(json.dumps(current, indent=2) + "\n")
    candidate = tmp_path / "candidate-model"
    _write_model_artifact(candidate, b"candidate-weights", model_type="qwen3_5")
    return fused, candidate


def _write_model_artifact(path, weights, *, model_type):
    path.mkdir()
    config = {
        "architectures": [
            "Qwen3_5ForConditionalGeneration"
            if model_type == "qwen3_5"
            else "Qwen2ForCausalLM"
        ],
        "model_type": model_type,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "max_position_embeddings": 4096,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(weights)
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_parameters": 27_000_000_000,
                    "total_size": len(weights),
                }
            }
        ),
        encoding="utf-8",
    )


def _digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_candidate_record_is_exact_and_does_not_change_active_pointer(tmp_path):
    fused, candidate = _fused_dir(tmp_path)
    active_before = (fused / "active.json").read_bytes()
    descriptor = build_model_artifact_descriptor(candidate)

    receipt = record_upgrade_candidate(
        candidate_model_path=candidate,
        base_model_path=tmp_path / "current-model",
        tag="candidate",
        fused_model_dir=fused,
        artifact_descriptor=descriptor,
        source="test",
        metadata={"load_verified": True},
    )

    candidate_receipt = Path(receipt["candidate_receipt_path"])
    persisted = json.loads(candidate_receipt.read_text(encoding="utf-8"))
    assert receipt["active_pointer_unchanged"] is True
    assert (fused / "active.json").read_bytes() == active_before
    assert persisted["schema"] == "aura.cortex_upgrade.candidate.v1"
    assert persisted["model_descriptor_sha256"] == descriptor["descriptor_sha256"]
    assert persisted["incumbent_pointer_sha256"] == hashlib.sha256(
        active_before
    ).hexdigest()
    assert persisted["qualification_state"] == "awaiting_evaluation"
    assert persisted["required_next_step"] == "evaluate_plan_stage_activate"


def _upgrade_contracts(candidate, *, repository_id="", revision=""):
    descriptor = build_model_artifact_descriptor(
        candidate,
        repository_id=repository_id,
        revision=revision,
    )
    current = {
        "label": "current",
        "breadth_accuracy": 0.5,
        "reasoning_accuracy": 0.5,
        "identity_digests": ["old"],
    }
    proposed = {
        "label": "candidate",
        "breadth_accuracy": 0.75,
        "reasoning_accuracy": 0.75,
        "identity_digests": ["new"],
    }
    evaluation = compare_batteries(
        current,
        proposed,
        candidate_descriptor=descriptor,
        critical_gates={
            "template": True,
            "complete_answer": True,
            "tool_contract": True,
            "code_contract": True,
            "context": True,
            "identity_migration": True,
            "latency": True,
            "memory": True,
        },
    )
    serving = build_model_serving_profile(
        descriptor,
        served_context_tokens=4096,
        prefill_chunk_tokens=512,
        lane_limits={
            "foreground_simple": {"max_input_tokens": 2048, "max_output_tokens": 256},
            "foreground_standard": {"max_input_tokens": 2048, "max_output_tokens": 512},
            "foreground_extended": {"max_input_tokens": 2048, "max_output_tokens": 1024},
            "deep_reasoning": {"max_input_tokens": 2048, "max_output_tokens": 1024},
            "tool_execution": {"max_input_tokens": 2048, "max_output_tokens": 512},
            "code": {"max_input_tokens": 2048, "max_output_tokens": 1024},
            "document": {"max_input_tokens": 2048, "max_output_tokens": 1024},
        },
        qualification={
            "schema": "aura.model_serving_qualification.v2",
            "verdict": "PASS",
            "model_descriptor_sha256": descriptor["descriptor_sha256"],
            "template_pass": True,
            "complete_answer_pass": True,
            "tool_contract_pass": True,
            "code_contract_pass": True,
            "context_pass": True,
            "latency_pass": True,
            "memory_pass": True,
            "served_context_tokens": 4096,
            "requested_context_tokens": 4096,
            "prefill_chunk_tokens": 512,
            "evidence_sha256": _digest("serving"),
        },
    )
    components = build_signed_migration_authorities(
        candidate.parent,
        descriptor_sha256=descriptor["descriptor_sha256"],
        state_root=Path(os.environ["AURA_STATE_ROOT"]),
    )
    migration = build_migration_contract(
        descriptor,
        components=components,
    )
    return descriptor, evaluation, serving, migration


def test_migration_contract_reopens_component_authority_instead_of_trusting_a_digest(
    tmp_path,
):
    candidate = tmp_path / "candidate-model"
    _write_model_artifact(candidate, b"candidate-weights", model_type="qwen3_5")
    descriptor = build_model_artifact_descriptor(candidate)
    components = build_signed_migration_authorities(
        tmp_path,
        descriptor_sha256=descriptor["descriptor_sha256"],
        state_root=Path(os.environ["AURA_STATE_ROOT"]),
    )
    components["persona_crsm"]["authority_sha256"] = "0" * 64

    with pytest.raises(
        ValueError, match="migration_component_authority_signature_invalid:persona_crsm"
    ):
        build_migration_contract(descriptor, components=components)


def test_active_identity_normalization_is_exact_idempotent_and_model_preserving(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, _candidate = _fused_dir(tmp_path)
    original = (fused / "active.json").read_bytes()
    active = Path(json.loads(original)["active_model_path"])
    descriptor = build_model_artifact_descriptor(active)

    first = normalize_active_pointer_identity(
        artifact_descriptor=descriptor,
        fused_model_dir=fused,
    )
    normalized = json.loads((fused / "active.json").read_text())

    assert first["changed"] is True
    assert normalized["schema_version"] == 3
    assert normalized["active_model_path"] == str(active)
    assert normalized["artifact_descriptor"] == descriptor
    assert normalized["identity_transition"] == {
        "schema": "aura.cortex_upgrade.identity_transition.v1",
        "kind": "model_identity_normalization",
        "previous_pointer_sha256": hashlib.sha256(original).hexdigest(),
        "active_model_path": str(active),
        "model_descriptor_sha256": descriptor["descriptor_sha256"],
        "transition_sha256": first["identity_transition_sha256"],
    }
    assert (fused / IDENTITY_BACKUP_POINTER_NAME).read_bytes() == original

    second = normalize_active_pointer_identity(
        artifact_descriptor=descriptor,
        fused_model_dir=fused,
    )
    assert second["changed"] is False
    assert second["after_sha256"] == first["after_sha256"]


def test_stage_writes_rollback_and_changes_nothing_live(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    descriptor, evaluation, serving, migration = _upgrade_contracts(candidate)
    before = (fused / "active.json").read_bytes()
    receipt = stage_upgrade(
        candidate_model_path=candidate,
        base_model_path="Qwen3-32B",
        tag="qwen3-gen",
        fused_model_dir=fused,
        evaluation=evaluation,
        serving_profile=serving,
        migration_contract=migration,
    )
    assert (fused / "active.json").read_bytes() == before, "staging must not touch live"
    assert (fused / ROLLBACK_POINTER_NAME).read_bytes() == before
    staged = json.loads((fused / STAGED_POINTER_NAME).read_text())
    assert staged["active_model_path"] == str(candidate)
    assert staged["base_model"] == "Qwen3-32B"
    assert staged["schema_version"] == 3
    assert staged["artifact_descriptor"]["descriptor_sha256"] == descriptor["descriptor_sha256"]
    assert staged["serving_profile"]["model_descriptor_sha256"] == descriptor["descriptor_sha256"]
    assert receipt["staged_active_model"] == str(candidate)


def test_activation_gates_and_flip(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    _, evaluation, serving, migration = _upgrade_contracts(candidate)
    stage_upgrade(
        candidate_model_path=candidate, base_model_path="Qwen3-32B",
        tag="qwen3-gen", fused_model_dir=fused, evaluation=evaluation,
        serving_profile=serving, migration_contract=migration,
    )
    with pytest.raises(PermissionError, match="authorization"):
        activate_upgrade(fused_model_dir=fused, authorized_by="", 
                         evaluation=evaluation)
    with pytest.raises(PermissionError, match="PASS"):
        activate_upgrade(fused_model_dir=fused, authorized_by="bryan",
                         evaluation={"verdict": "FAIL"})
    receipt = activate_upgrade(
        fused_model_dir=fused, authorized_by="bryan",
        evaluation=evaluation,
    )
    active = json.loads((fused / "active.json").read_text())
    assert active["active_model_path"] == str(candidate)
    assert receipt["effective"] == "next_boot"
    assert receipt["authorized_by"] == "bryan"


def test_rollback_is_byte_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    _, evaluation, serving, migration = _upgrade_contracts(candidate)
    original = (fused / "active.json").read_bytes()
    stage_upgrade(candidate_model_path=candidate, base_model_path="Qwen3-32B",
                  tag="qwen3-gen", fused_model_dir=fused, evaluation=evaluation,
                  serving_profile=serving, migration_contract=migration)
    activate_upgrade(fused_model_dir=fused, authorized_by="bryan",
                     evaluation=evaluation)
    assert (fused / "active.json").read_bytes() != original
    receipt = rollback_upgrade(fused_model_dir=fused)
    assert receipt["byte_exact"] is True
    assert (fused / "active.json").read_bytes() == original


def test_activation_without_staging_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, _ = _fused_dir(tmp_path)
    with pytest.raises(ValueError, match="nothing staged"):
        activate_upgrade(fused_model_dir=fused, authorized_by="bryan",
                         evaluation={"verdict": "PASS"})
    with pytest.raises(ValueError, match="no rollback"):
        rollback_upgrade(fused_model_dir=fused)


def test_stage_rejects_a_steering_receipt_from_same_width_old_model(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    _fused, candidate = _fused_dir(tmp_path)
    descriptor, _evaluation, _serving, migration = _upgrade_contracts(candidate)
    old_model = tmp_path / "same-width-old-model"
    _write_model_artifact(old_model, b"old-model-weights", model_type="qwen3_5")
    old_descriptor = build_model_artifact_descriptor(old_model)
    old_authorities = build_signed_migration_authorities(
        tmp_path / "old-authority",
        descriptor_sha256=old_descriptor["descriptor_sha256"],
        state_root=Path(os.environ["AURA_STATE_ROOT"]),
    )
    components = dict(migration["components"])
    components["steering"] = old_authorities["steering"]

    with pytest.raises(
        ValueError, match="migration_component_authority_invalid:steering"
    ):
        build_migration_contract(
            descriptor,
            components=components,
        )


def test_activation_reopens_staged_migration_authority_and_refuses_drift(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    _, evaluation, serving, migration = _upgrade_contracts(candidate)
    stage_upgrade(
        candidate_model_path=candidate,
        base_model_path="Qwen3.8-27B",
        tag="qwen3.8-gen",
        fused_model_dir=fused,
        evaluation=evaluation,
        serving_profile=serving,
        migration_contract=migration,
    )
    Path(
        migration["components"]["persona_crsm"]["evidence"]["fusion_plan"]["path"]
    ).write_text(
        "substituted-persona-authority",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="migration_component:persona_crsm:fusion_plan_binding_drift",
    ):
        activate_upgrade(
            fused_model_dir=fused,
            authorized_by="bryan",
            evaluation=evaluation,
        )


def test_activation_rehashes_the_staged_candidate_and_refuses_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    _, evaluation, serving, migration = _upgrade_contracts(candidate)
    stage_upgrade(
        candidate_model_path=candidate,
        base_model_path="Qwen3-32B",
        tag="qwen3-gen",
        fused_model_dir=fused,
        evaluation=evaluation,
        serving_profile=serving,
        migration_contract=migration,
    )
    (candidate / "model.safetensors").write_bytes(b"candidate-weightS")

    with pytest.raises(ValueError, match="descriptor_mismatch"):
        activate_upgrade(
            fused_model_dir=fused,
            authorized_by="bryan",
            evaluation=evaluation,
        )


def test_activation_rejects_a_different_pass_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    _, evaluation, serving, migration = _upgrade_contracts(candidate)
    stage_upgrade(
        candidate_model_path=candidate,
        base_model_path="Qwen3-32B",
        tag="qwen3-gen",
        fused_model_dir=fused,
        evaluation=evaluation,
        serving_profile=serving,
        migration_contract=migration,
    )
    substituted = dict(evaluation)
    substituted["compared_at"] = float(evaluation["compared_at"]) + 1.0
    substituted.pop("evaluation_sha256")

    with pytest.raises(PermissionError, match="staged evaluation"):
        activate_upgrade(
            fused_model_dir=fused,
            authorized_by="bryan",
            evaluation=substituted,
        )


def test_stage_preserves_revision_pinned_artifact_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    fused, candidate = _fused_dir(tmp_path)
    descriptor, evaluation, serving, migration = _upgrade_contracts(
        candidate,
        repository_id="mlx-community/Qwen3.8-27B-4bit",
        revision="3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
    )

    stage_upgrade(
        candidate_model_path=candidate,
        base_model_path="mlx-community/Qwen3.8-27B-4bit",
        tag="qwen3.8-gen",
        fused_model_dir=fused,
        artifact_descriptor=descriptor,
        evaluation=evaluation,
        serving_profile=serving,
        migration_contract=migration,
    )

    staged = json.loads((fused / STAGED_POINTER_NAME).read_text())
    assert staged["artifact_descriptor"]["repository_id"] == descriptor["repository_id"]
    assert staged["artifact_descriptor"]["revision"] == descriptor["revision"]
