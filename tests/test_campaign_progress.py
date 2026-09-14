"""A crash preserves a prefix, never a fabricated completed experiment."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.evaluation.campaign_progress import CampaignProgress, _digest
from tools import run_caa_steering_campaign as campaign


def progress(path, **kwargs):
    return CampaignProgress(path, identity={"model": "one", "seed": 7},
                            conditions=("base", "lesion"), samples_per_condition=2,
                            write=lambda payload: campaign.write_campaign_json(path, payload), **kwargs)


def test_restart_preserves_exact_outputs_and_continuation(tmp_path):
    path = tmp_path / "progress.json"
    first = progress(path)
    first.record("base", "one", {"composite": [0.25, -0.125]}, seconds=2)
    second = progress(path, resume=True)
    assert second.outputs["base"] == ["one"]
    assert second.state_for("base") == {"composite": [0.25, -0.125]}
    assert second.state_for("lesion") is None
    assert not second.complete
    second.record("base", "two", {}, seconds=3)
    second.record("lesion", "three", {}, seconds=4)
    second.record("lesion", "four", {}, seconds=5)
    final = progress(path, resume=True)
    assert final.complete and final.decode_seconds == 14
    with pytest.raises(ValueError, match="out_of_order"):
        final.record("lesion", "extra", {}, seconds=1)


def test_failed_durable_write_does_not_advance_memory(tmp_path, monkeypatch):
    run = progress(tmp_path / "progress.json")
    monkeypatch.setattr(run, "_write",
                        lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")))
    with pytest.raises(OSError):
        run.record("base", "one", {}, seconds=1)
    assert run.outputs["base"] == [] and run.decode_seconds == 0


@pytest.mark.parametrize("change", ["identity", "conditions", "samples_per_condition"])
def test_resume_refuses_experiment_drift(tmp_path, change):
    path = tmp_path / "progress.json"
    progress(path).record("base", "one", {}, seconds=0)
    kwargs = dict(identity={"model": "one", "seed": 7}, conditions=("base", "lesion"),
                  samples_per_condition=2, resume=True)
    kwargs[change] = {"identity": {"model": "two"}, "conditions": ("lesion", "base"),
                      "samples_per_condition": 3}[change]
    with pytest.raises(ValueError, match="identity_mismatch"):
        CampaignProgress(path, write=lambda payload: campaign.write_campaign_json(path, payload), **kwargs)


def test_resume_requires_explicit_existing_progress(tmp_path):
    path = tmp_path / "progress.json"
    with pytest.raises(ValueError, match="missing"):
        progress(path, resume=True)
    progress(path).record("base", "one", {}, seconds=0)
    with pytest.raises(ValueError, match="use_resume"):
        progress(path)


def test_integrity_and_prefix_both_checked(tmp_path):
    path = tmp_path / "progress.json"
    progress(path).record("base", "one", {}, seconds=0)
    payload = json.loads(path.read_text())
    payload["outputs"]["lesion"] = ["impossible"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="corrupt"):
        progress(path, resume=True)
    payload.pop("payload_sha256")
    payload["payload_sha256"] = _digest(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not_a_prefix"):
        progress(path, resume=True)


def test_record_enforces_order_and_finite_state(tmp_path):
    run = progress(tmp_path / "progress.json")
    with pytest.raises(ValueError, match="out_of_order"):
        run.record("lesion", "early", {}, seconds=1)
    with pytest.raises(ValueError):
        run.record("base", "bad", {"state": float("nan")}, seconds=1)
    with pytest.raises(ValueError, match="time_invalid"):
        run.record("base", "bad", {}, seconds=float("inf"))


def test_real_hook_composite_continues_bit_exactly(tmp_path):
    pytest.importorskip("mlx.core")
    from core.consciousness.affective_steering import AffectiveSteeringHook, SteeringVector
    from core.consciousness.fusion_probe import STATE_HIGH, STATE_LOW
    rng = np.random.default_rng(77)
    values = rng.normal(size=64).astype(np.float32)
    values /= np.linalg.norm(values)
    vector = SteeringVector(key="valence", layer_idx=1, d_model=64,
                            v=values, substrate_idx=0, substrate_fn="linear")
    one = AffectiveSteeringHook(object(), 1, {"valence": vector})
    for _ in range(40):
        one.update_substrate(STATE_LOW)
    for _ in range(42):
        one.update_substrate(STATE_HIGH)
    assert one.current_composite_vector() is not None
    run = progress(tmp_path / "progress.json")
    run.record("base", "first", campaign.capture_continuation([one]), seconds=1)
    restored = AffectiveSteeringHook(object(), 1, {"valence": vector})
    campaign.restore_continuation([restored], progress(run.path, resume=True).state_for("base"), 64)
    for _ in range(2):
        one.update_substrate(STATE_HIGH)
        restored.update_substrate(STATE_HIGH)
    np.testing.assert_array_equal(one.current_composite_vector(), restored.current_composite_vector())


def test_input_identity_changes_with_vector_bytes_and_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.importlib.metadata, "version", lambda name: "test")
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    (vectors / "metadata.json").write_text("{}")
    vector = vectors / "vector.npz"
    vector.write_bytes(b"one")
    arguments = SimpleNamespace(vectors=vectors, trials=4, max_tokens=256, temperature=0.7)
    first = campaign.campaign_identity(arguments, {"hidden_size": 64}, "checkpoint", 0.2)
    vector.write_bytes(b"two")
    assert first != campaign.campaign_identity(arguments, {"hidden_size": 64}, "checkpoint", 0.2)
    vector.write_bytes(b"one")
    arguments.temperature = 0.8
    assert first != campaign.campaign_identity(arguments, {"hidden_size": 64}, "checkpoint", 0.2)


def test_campaign_main_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import contextlib
    import importlib

    import mlx.core as mx
    import mlx_lm

    from core.brain.llm import decoder_topology, model_registry
    from core.consciousness import affective_steering
    from core.evaluation import caa_causal_evaluation
    from core.runtime import model_lane_control

    plan_path = tmp_path / "plan.json"
    plan = {"model_path": str(tmp_path / "model"), "hidden_size": 2,
            "target_layers": [{"index": 0, "kind": "full_attention"}]}
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(model_registry, "get_active_cortex_spec", lambda **kw:
                        SimpleNamespace(model_path=plan["model_path"], descriptor_sha256="test"))
    monkeypatch.setattr(model_lane_control, "standalone_model_lane", lambda **kw: contextlib.nullcontext())
    monkeypatch.setattr(campaign, "campaign_identity", lambda *args: {"test": "fixed"})
    class Hook:
        def __init__(self, block, index, vectors):
            self._vectors = vectors
            self._layer_idx = index
            self._alpha = 0.0
            self.state = None
            active[:] = [self]
        def install(self):
            pass
        def update_substrate(self, moods):
            value = sum(v.v for v in self._vectors.values())
            self.state = value if self.state is None else 0.85 * self.state + 0.15 * value
        def current_composite_vector(self):
            return self.state
        def override_composite_vector(self, value):
            self.state = value
    class Library:
        def __init__(self, **kwargs):
            pass
        def load_or_derive(self, *args):
            return {0: {"a": SimpleNamespace(v=np.array([0.3, -0.7], dtype=np.float32))}}
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[0]["content"])
    monkeypatch.setattr(mlx_lm, "load", lambda path: (object(), tokenizer))
    monkeypatch.setattr(decoder_topology, "resolve_language_model", lambda model: SimpleNamespace(layers=[object()]))
    monkeypatch.setattr(affective_steering, "AffectiveSteeringHook", Hook)
    monkeypatch.setattr(affective_steering, "SteeringVectorLibrary", Library)
    monkeypatch.setattr(caa_causal_evaluation, "replay_campaign", lambda result: dict(
        treatment_successes=0, matched_control_successes=0, lesion_successes=0,
        no_regression=True, causal_effect_positive=False, unmet_requirements=["test only"]))
    active = []
    calls = []
    failure = [None]
    def generate(*args, prompt, **kwargs):
        if failure[0] == len(calls):
            raise RuntimeError("injected interruption")
        calls.append(prompt)
        hook = active[0]
        return json.dumps([prompt, hook._alpha, None if hook.state is None else hook.state.tolist(),
                           float(mx.random.uniform())])
    monkeypatch.setattr(importlib.import_module("mlx_lm.generate"), "generate", generate)
    args = ["--plan", str(plan_path), "--vectors", str(tmp_path), "--trials", "1", "--alpha", "0.2"]
    full = tmp_path / "full.json"
    assert campaign.main([*args, "--out", str(full)]) == 2
    calls.clear()
    interrupted = tmp_path / "resumed.json"
    failure[0] = 45  # inside the random-vector arm, after the zero lesion
    with pytest.raises(RuntimeError, match="interruption"):
        campaign.main([*args, "--out", str(interrupted)])
    assert not interrupted.exists()
    failure[0] = None
    assert campaign.main([*args, "--out", str(interrupted), "--resume"]) == 2
    assert len(calls) == 54
    assert json.loads(full.read_text())["condition_outputs"] == json.loads(interrupted.read_text())["condition_outputs"]
