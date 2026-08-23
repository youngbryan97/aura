"""Contract tests: consolidation pipeline + anti-interference battery.

End-to-end on real machinery: a tiny-model episode with export enabled
lands a candidate in the queue; the consumer validates evidence and refuses
corrupt/unproven candidates; domains only propose after enough independent
wins; and the interference battery passes an identity change while failing
a genuinely disruptive one — proving accumulated learning is gated on NOT
trashing prior behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.heldout_battery import BatterySpec, generate_battery
from core.learning.latent_consolidation import (
    build_proposals,
    run_consolidation_cycle,
    scan_queue,
    validate_candidate,
)


def _heldout_contract(*, regress_after: bool = False):
    spec = BatterySpec(seed=9182, size=8)
    answers = {
        task.task_id: f"Answer: {task.answer}" for task in generate_battery(spec)
    }
    calls = 0

    def evaluate(_model, prompts):
        nonlocal calls
        calls += 1
        assert {task_id for task_id, _prompt in prompts} == set(answers)
        if regress_after and calls == 2:
            return {task_id: "Answer: definitely-wrong" for task_id in answers}
        return dict(answers)

    return {
        "heldout_solver": evaluate,
        "heldout_spec": spec,
        "heldout_evaluator_id": "test.sealed_exact.v1",
    }


def _make_candidate(root, episode_id, *, domain="math", erase=True, steps=2,
                    trail=(3.0, 2.0), flags=(), fingerprint="fp-abc"):
    d = root / episode_id
    d.mkdir(parents=True)
    (d / "delta_weights.npz").write_bytes(b"npz-bytes")
    (d / "evidence.json").write_text(json.dumps({
        "episode_id": episode_id,
        "created_at": 1000.0,
        "lifecycle": {"erase_proven": erase, "optimized_steps": steps},
        "evidence": {
            "domain": domain,
            "loss_trail": list(trail),
            "honest_flags": list(flags),
            "checkpoint_fingerprint": fingerprint,
        },
    }))
    return d


# ── Candidate validation ────────────────────────────────────────────────


def test_clean_candidate_validates(tmp_path):
    d = _make_candidate(tmp_path, "ep-1")
    record = validate_candidate(d)
    assert record.valid and record.domain == "math"
    assert record.loss_improvement == pytest.approx(1.0)


def test_unproven_erase_is_rejected(tmp_path):
    record = validate_candidate(_make_candidate(tmp_path, "ep-2", erase=False))
    assert not record.valid and "erase_unproven" in record.rejection_reasons


def test_flat_loss_and_no_steps_rejected(tmp_path):
    flat = validate_candidate(_make_candidate(tmp_path, "ep-3", trail=(2.0, 2.0)))
    assert "loss_not_descending" in flat.rejection_reasons
    lazy = validate_candidate(_make_candidate(tmp_path, "ep-4", steps=0))
    assert "no_accepted_optimization" in lazy.rejection_reasons


def test_honest_flags_block_consolidation(tmp_path):
    record = validate_candidate(
        _make_candidate(tmp_path, "ep-5", flags=("fallback_vanilla:RuntimeError",))
    )
    assert not record.valid
    assert any("honest_flags_present" in r for r in record.rejection_reasons)


def test_corrupt_evidence_rejected(tmp_path):
    d = tmp_path / "ep-6"
    d.mkdir()
    (d / "evidence.json").write_text("{not json")
    record = validate_candidate(d)
    assert not record.valid
    assert {"evidence_unreadable", "delta_weights_missing"} <= set(record.rejection_reasons)


# ── Aggregation ─────────────────────────────────────────────────────────


def test_domain_needs_enough_independent_wins(tmp_path):
    for i in range(2):
        _make_candidate(tmp_path, f"math-{i}", domain="math")
    for i in range(3):
        _make_candidate(tmp_path, f"code-{i}", domain="code")
    _make_candidate(tmp_path, "bad-1", domain="code", erase=False)

    records = scan_queue(tmp_path)
    proposals = build_proposals(records, min_candidates=3)
    assert [p["domain"] for p in proposals] == ["code"]
    assert proposals[0]["candidate_count"] == 3  # the invalid one never counts
    assert proposals[0]["mean_loss_improvement"] == pytest.approx(1.0)
    assert "interference battery verdict PASS before activation" in (
        proposals[0]["activation_requirements"]
    )


def test_cycle_writes_proposals_and_reports_rejections(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    queue = tmp_path / "queue"
    for i in range(3):
        _make_candidate(queue, f"ep-{i}", domain="planning")
    _make_candidate(queue, "ep-bad", domain="planning", erase=False)

    receipt = run_consolidation_cycle(queue, tmp_path / "proposals")
    assert receipt["scanned"] == 4 and receipt["valid"] == 3
    assert receipt["proposals"] == ["planning"]
    assert receipt["rejections"]["ep-bad"] == ["erase_unproven"]
    assert len(receipt["written"]) == 1
    written = json.loads((tmp_path / "proposals" / receipt["written"][0]).read_text())
    assert written["domain"] == "planning" and written["status"] == "proposed"


# ── Interference battery ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_model():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_identity_change_passes_battery(tiny_model):
    from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights
    from core.brain.llm.latent_cortex.types import FastWeightsConfig
    from core.learning.interference_battery import run_interference_battery

    fw = EpisodicFastWeights(FastWeightsConfig(enabled=True, rank=2, target="o_proj"))
    receipt = run_interference_battery(
        tiny_model,
        # V=0 attach is EXACT identity — the battery must agree.
        apply_change=lambda: fw.attach(
            tiny_model.model, (2, 6), seed_stat=0.4, episode_id="battery-identity"
        ),
        revert_change=fw.detach,
    )
    assert receipt["verdict"] == "PASS"
    assert receipt["stable_fraction"] == 1.0


def test_disruptive_change_fails_battery_and_reverts(tiny_model):
    import mlx.core as mx

    from core.learning.interference_battery import (
        run_interference_battery,
        snapshot_probe_behavior,
    )

    layer = tiny_model.model.layers[3]
    original = layer.mlp.down_proj.weight
    baseline = snapshot_probe_behavior(tiny_model)

    def wreck():
        layer.mlp.down_proj.weight = original + mx.random.normal(
            original.shape, key=mx.random.key(1)
        ) * 0.5

    def restore():
        layer.mlp.down_proj.weight = original

    receipt = run_interference_battery(tiny_model, wreck, restore)
    assert receipt["verdict"] == "FAIL"
    assert receipt["stable_fraction"] < 0.9
    # Revert restored protected behavior exactly.
    after = snapshot_probe_behavior(tiny_model)
    assert [r["digest"] for r in after] == [r["digest"] for r in baseline]


def test_interference_probe_uses_qwen35_public_mixed_attention_forward():
    """Identity evidence must work for the Qwen3.8 architecture family."""
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    from core.learning.interference_battery import snapshot_probe_behavior

    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 2,
        "full_attention_interval": 2,
        "head_dim": 16,
        "tie_word_embeddings": False,
    }
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text_config))
    mx.eval(model.parameters())

    rows = snapshot_probe_behavior(model, probes=[[3, 5, 7, 9]])

    assert len(rows) == 1
    assert len(rows[0]["top8_ids"]) == 8
    assert len(rows[0]["digest"]) == 16


def test_engine_export_lands_valid_candidate_in_queue(tiny_model, tmp_path, monkeypatch):
    """Full loop: episode with export enabled → queue → consumer validates."""
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    import core.config as config_mod

    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "data", raising=False)

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        FastWeightsConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    engine = LatentCortexEngine(
        tiny_model,
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=3),
            recurrence=RecurrenceConfig(max_steps=3, min_steps=2),
            branches=BranchConfig(n_branches=1),
            latent_opt=LatentOptConfig(enabled=False),
            fast_weights=FastWeightsConfig(
                enabled=True, rank=2, target="o_proj", opt_steps=3, lr=0.05,
                export_candidates=True,
            ),
            decode_max_tokens=6,
        ),
    )
    result = engine.reason(token_ids=[5, 9, 17, 3, 42], domain="unit-loop")
    assert result.ok
    queue = tmp_path / "data" / "latent_cortex" / "consolidation_queue"
    if "fast_weight_candidate_exported" in result.receipt.honest_flags:
        records = scan_queue(queue)
        assert len(records) == 1
        assert records[0].valid, records[0].rejection_reasons
        assert records[0].domain == "unit-loop"
    else:
        # A tiny random model may reject every optimizer step — then the
        # export must NOT have happened and the queue must be empty.
        assert not queue.exists() or not any(queue.iterdir())


def test_natural_probes_measure_protected_behavior_regions(tiny_model):
    from core.learning.interference_battery import (
        NATURAL_STABILITY_PROBE_TEXTS,
        natural_stability_probes,
        stability_probes_for,
    )

    class Vocab8Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 8 for ch in text][:16]

    tokenizer = Vocab8Tokenizer()
    probes = natural_stability_probes(tokenizer)
    assert len(probes) == len(NATURAL_STABILITY_PROBE_TEXTS)
    assert all(probe for probe in probes)
    # With a tokenizer the selector must prefer the natural battery.
    assert stability_probes_for(tiny_model, tokenizer) == probes


def test_probe_selector_falls_back_without_tokenizer(tiny_model):
    from core.learning.interference_battery import (
        default_stability_probes,
        stability_probes_for,
    )

    assert stability_probes_for(tiny_model, None) == default_stability_probes()


def test_battery_runs_on_natural_probes(tiny_model):
    from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights
    from core.brain.llm.latent_cortex.types import FastWeightsConfig
    from core.learning.interference_battery import run_interference_battery

    class Vocab8Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 8 for ch in text][:16]

    fw = EpisodicFastWeights(FastWeightsConfig(enabled=True, rank=2, target="o_proj"))
    receipt = run_interference_battery(
        tiny_model,
        lambda: fw.attach(
            tiny_model.model, (2, 6), seed_stat=0.4, episode_id="battery-natural"
        ),
        fw.detach,
        tokenizer=Vocab8Tokenizer(),
    )
    assert receipt["verdict"] == "PASS"  # identity attach must not move behavior
    assert receipt["probes"] == 11


# ── The complete durable-learning train ──────────────────────────────────


def _make_weighted_candidate(
    root, episode_id, *, domain="math", layers=(2, 3), rank=2, hidden=64,
    v_scale=0.0, fingerprint="fp-abc",
):
    """A candidate with REAL delta arrays in the export wire format."""
    import io as _io

    import numpy as np

    d = root / episode_id
    d.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(episode_id)) % (2**32))
    arrays = {}
    for layer in layers:
        arrays[f"layer{layer}_U"] = rng.normal(size=(hidden, rank)).astype("float32")
        arrays[f"layer{layer}_V"] = (
            rng.normal(size=(rank, hidden)).astype("float32") * v_scale
        )
    buffer = _io.BytesIO()
    np.savez(buffer, **arrays)
    (d / "delta_weights.npz").write_bytes(buffer.getvalue())
    (d / "evidence.json").write_text(json.dumps({
        "episode_id": episode_id,
        "created_at": 1000.0,
        "target": "o_proj",
        "rank": rank,
        "layers": list(layers),
        "scale": 1.0,
        "lifecycle": {"erase_proven": True, "optimized_steps": 2},
        "evidence": {
            "domain": domain,
            "loss_trail": [3.0, 2.0],
            "honest_flags": [],
            "checkpoint_fingerprint": fingerprint,
        },
    }))
    return d


def _proposal_for(queue_dir, domain="math"):
    records = scan_queue(queue_dir)
    proposals = build_proposals(records, min_candidates=3)
    assert proposals and proposals[0]["domain"] == domain
    return proposals[0]


def test_distillation_merge_is_the_exact_mean_delta(tmp_path):
    import io as _io

    import numpy as np

    from core.learning.latent_adapter_distillation import distill_proposal_to_adapter

    queue = tmp_path / "queue"
    dirs = [
        _make_weighted_candidate(queue, f"ep-{i}", v_scale=0.1) for i in range(3)
    ]
    proposal = _proposal_for(queue)
    result = distill_proposal_to_adapter(proposal, adapter_dir=tmp_path / "adapters")
    assert result["ok"], result

    with np.load(
        _io.BytesIO(
            (Path(result["adapter_dir"]) / "delta_weights.npz").read_bytes()
        )
    ) as merged:
        merged_delta = merged["layer2_U"] @ merged["layer2_V"]
    expected = np.zeros_like(merged_delta)
    for d in dirs:
        with np.load(_io.BytesIO((d / "delta_weights.npz").read_bytes())) as c:
            expected += c["layer2_U"] @ c["layer2_V"]
    expected /= 3.0
    assert np.allclose(merged_delta, expected, atol=1e-5)
    assert result["manifest"]["candidate_count"] == 3
    assert result["manifest"]["checkpoint_fingerprint"] == "fp-abc"


def test_mixed_fingerprints_refuse_distillation(tmp_path):
    """Aggregation is fingerprint-scoped, so a mixed proposal can no longer
    be BUILT — but the distiller's own refusal stays as belt and braces
    against hand-assembled or legacy proposals."""
    from core.learning.latent_adapter_distillation import distill_proposal_to_adapter

    queue = tmp_path / "queue"
    dirs = [
        _make_weighted_candidate(queue, "ep-0", fingerprint="fp-a"),
        _make_weighted_candidate(queue, "ep-1", fingerprint="fp-a"),
        _make_weighted_candidate(queue, "ep-2", fingerprint="fp-DIFFERENT"),
    ]
    # Fingerprint-scoped aggregation refuses to build the mixed proposal.
    assert build_proposals(scan_queue(queue), min_candidates=3) == []
    # A hand-assembled mixed proposal still refuses at the distiller.
    proposal = {
        "domain": "math",
        "candidates": [validate_candidate(d).to_dict() for d in dirs],
    }
    result = distill_proposal_to_adapter(proposal, adapter_dir=tmp_path / "adapters")
    assert not result["ok"]
    assert result["reason"] == "mixed_checkpoint_fingerprints"


def test_full_train_activates_identity_adapter_and_rolls_back(tiny_model, tmp_path):
    from core.learning.latent_adapter_distillation import (
        rollback_adapter,
        run_consolidation_train,
    )

    queue = tmp_path / "queue"
    for i in range(3):
        _make_weighted_candidate(queue, f"ep-{i}", v_scale=0.0)  # V=0 ⇒ identity
    proposal = _proposal_for(queue)

    receipt = run_consolidation_train(
        proposal,
        tiny_model,
        adapter_dir=tmp_path / "adapters",
        **_heldout_contract(),
    )
    assert receipt["activated"] is True, receipt
    assert receipt["interference_battery"]["verdict"] == "PASS"
    assert receipt["heldout"]["gate"]["verdict"] == "PASS"
    assert receipt["heldout"]["before"]["result"]["accuracy"] == 1.0
    assert receipt["heldout"]["after"]["result"]["accuracy"] == 1.0
    assert receipt["heldout"]["before"]["manifest"] == receipt["heldout"]["battery"]
    active = receipt["active_adapter"]
    # The adapter is genuinely attached: the target module is the wrapper.
    layer = tiny_model.model.layers[active.handles[0].layer_index]
    assert layer.self_attn.o_proj is active.handles[0].wrapper

    rollback = rollback_adapter(tiny_model, active)
    assert rollback["rollback_proven"] is True
    assert rollback["restored_layers"] >= 1
    assert not active.active
    layer = tiny_model.model.layers[2]
    assert not hasattr(layer.self_attn.o_proj, "wrapper")


def test_missing_heldout_contract_refuses_before_distillation(tiny_model, tmp_path):
    from core.learning.latent_adapter_distillation import run_consolidation_train

    queue = tmp_path / "queue"
    for i in range(3):
        _make_weighted_candidate(queue, f"ep-{i}", v_scale=0.0)
    adapter_dir = tmp_path / "adapters"

    receipt = run_consolidation_train(
        _proposal_for(queue),
        tiny_model,
        adapter_dir=adapter_dir,
    )

    assert receipt["activated"] is False
    assert receipt["refusal_reason"] == "heldout_promotion_contract_missing"
    assert not adapter_dir.exists()


def test_incomplete_heldout_response_set_refuses_activation(tiny_model, tmp_path):
    from core.learning.latent_adapter_distillation import run_consolidation_train

    queue = tmp_path / "queue"
    for i in range(3):
        _make_weighted_candidate(queue, f"ep-{i}", v_scale=0.0)
    contract = _heldout_contract()
    valid_solver = contract["heldout_solver"]

    def incomplete_solver(model, prompts):
        responses = valid_solver(model, prompts)
        responses.pop(next(iter(responses)))
        return responses

    contract["heldout_solver"] = incomplete_solver
    receipt = run_consolidation_train(
        _proposal_for(queue),
        tiny_model,
        adapter_dir=tmp_path / "adapters",
        **contract,
    )

    assert receipt["activated"] is False
    assert receipt["refusal_reason"].startswith(
        "heldout_evaluation_failed:heldout_response_coverage_mismatch"
    )


def test_disruptive_adapter_is_refused_and_model_untouched(tiny_model, tmp_path):
    from core.learning.interference_battery import snapshot_probe_behavior
    from core.learning.latent_adapter_distillation import run_consolidation_train

    queue = tmp_path / "queue"
    for i in range(3):
        _make_weighted_candidate(queue, f"ep-{i}", v_scale=25.0)  # wrecking ball
    proposal = _proposal_for(queue)

    before = snapshot_probe_behavior(tiny_model)
    receipt = run_consolidation_train(
        proposal,
        tiny_model,
        adapter_dir=tmp_path / "adapters",
        **_heldout_contract(),
    )
    assert receipt["activated"] is False
    assert receipt["refusal_reason"] == "interference_battery_failed"
    after = snapshot_probe_behavior(tiny_model)
    assert [r["digest"] for r in before] == [r["digest"] for r in after]


def test_heldout_regression_blocks_activation(tiny_model, tmp_path):
    from core.learning.latent_adapter_distillation import run_consolidation_train

    queue = tmp_path / "queue"
    for i in range(3):
        _make_weighted_candidate(queue, f"ep-{i}", v_scale=0.0)
    proposal = _proposal_for(queue)

    receipt = run_consolidation_train(
        proposal,
        tiny_model,
        adapter_dir=tmp_path / "adapters",
        **_heldout_contract(regress_after=True),
    )
    assert receipt["activated"] is False
    assert receipt["refusal_reason"] == "heldout_regression"


# ── Provenance discipline (the Jul-2026 leaked-candidate incident) ───────


def test_missing_fingerprint_candidate_is_rejected(tmp_path):
    record = validate_candidate(_make_candidate(tmp_path, "ep-anon", fingerprint=""))
    assert not record.valid
    assert "checkpoint_fingerprint_missing" in record.rejection_reasons


def test_proposals_are_scoped_per_checkpoint_fingerprint(tmp_path):
    for i in range(3):
        _make_candidate(tmp_path, f"a-{i}", domain="chat", fingerprint="fp-32b")
    for i in range(2):
        _make_candidate(tmp_path, f"b-{i}", domain="chat", fingerprint="fp-tiny")
    proposals = build_proposals(scan_queue(tmp_path), min_candidates=3)
    assert len(proposals) == 1
    assert proposals[0]["checkpoint_fingerprint"] == "fp-32b"
    assert proposals[0]["candidate_count"] == 3


def test_cross_dimension_candidates_refuse_distillation(tmp_path):
    from core.learning.latent_adapter_distillation import distill_proposal_to_adapter

    queue = tmp_path / "queue"
    _make_weighted_candidate(queue, "big-0", hidden=128)
    _make_weighted_candidate(queue, "big-1", hidden=128)
    _make_weighted_candidate(queue, "tiny-0", hidden=64)
    proposal = _proposal_for(queue)
    result = distill_proposal_to_adapter(proposal, adapter_dir=tmp_path / "adapters")
    assert not result["ok"]
    assert result["reason"].startswith("candidate_dimension_mismatch")


def test_attach_dimension_mismatch_is_refused_not_crashed(tiny_model, tmp_path):
    """An adapter distilled for a different model must refuse cleanly at the
    train, leave the model untouched, and never crash the run."""
    from core.learning.interference_battery import snapshot_probe_behavior
    from core.learning.latent_adapter_distillation import run_consolidation_train

    queue = tmp_path / "queue"
    for i in range(3):
        # hidden=128 adapters against the hidden=64 tiny model.
        _make_weighted_candidate(queue, f"ep-{i}", hidden=128, layers=(2, 3))
    proposal = _proposal_for(queue)
    before = snapshot_probe_behavior(tiny_model)
    receipt = run_consolidation_train(
        proposal,
        tiny_model,
        adapter_dir=tmp_path / "adapters",
        **_heldout_contract(),
    )
    assert receipt["activated"] is False
    assert receipt["refusal_reason"].startswith("attach_failed")
    after = snapshot_probe_behavior(tiny_model)
    assert [r["digest"] for r in before] == [r["digest"] for r in after]


def test_anonymous_episode_never_exports(tiny_model):
    """An engine without model_path (no checkpoint fingerprint) must refuse
    candidate export even when the episode is mechanically clean."""
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        FastWeightsConfig,
        LatentOptConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    engine = LatentCortexEngine(
        tiny_model,
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=3),
            recurrence=RecurrenceConfig(max_steps=3, min_steps=2),
            branches=BranchConfig(n_branches=1),
            latent_opt=LatentOptConfig(enabled=False),
            fast_weights=FastWeightsConfig(
                enabled=True, rank=2, target="o_proj", opt_steps=3, lr=0.05,
                export_candidates=True,
            ),
            decode_max_tokens=6,
        ),
    )
    result = engine.reason(token_ids=[5, 9, 17, 3, 42], domain="anon-test")
    assert result.ok
    assert "fast_weight_candidate_exported" not in result.receipt.honest_flags
