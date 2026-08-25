"""`ok` meant the machinery ran, and everything downstream read it as more.

The task verifier is optional. Branch selection falls back to the ensemble's
own score. Latent optimization can descend on a proxy. After all of that the
method returned ok=True, and a consumer wanting to know whether the episode
improved anything had only that boolean to read.

The consolidation queue was the sharpest case. Candidate export ran inside
fast-weight finalization — while an exception could still be propagating, and
before the post-episode checkpoint invariant — and its predicate could see the
optimizer and the erase proof and nothing else. Failed, cancelled and
fallback-answered episodes entered the learning queue carrying a falling loss
curve and none of the flags that said so.

Three more claims that outran their evidence:

**The erase proof was one narrow probe.** Byte equality on a fixed eight-token
input cannot see a residual delta that this input does not excite, a wrapper
still installed on a layer the probe does not reach, or a handle nobody
released.

**Optimization ran a different function from the answer.** The teacher-
trajectory forward passed mask=None over a multi-token sequence, which is full
attention, while the answer is produced causally through a cache.

**Structural layer access was read as compatibility.** Any decoder exposing
`model.layers` was admitted and its middle window reapplied, which establishes
that the call succeeds and nothing else.

And a caller-supplied branch verifier could be dropped for budget and replaced
by the internal score with the episode still returning ok — so a caller who
supplied it as a correctness gate silently lost the gate.
"""
from __future__ import annotations

import ast
import inspect

import pytest

import core.brain.llm.latent_cortex.engine as engine_mod
from core.brain.llm.latent_cortex.recurrence_support import (
    CERTIFIED,
    EXPERIMENTAL,
    STRUCTURAL,
    classify_recurrence_support,
    load_certified_architectures,
)


# ─────────────────────────── ok is not a quality claim


def test_the_receipt_separates_ran_from_checked_from_improved():
    from core.brain.llm.latent_cortex.types import EpisodeReceipt

    receipt = EpisodeReceipt(episode_id="e")

    assert receipt.quality_verified is False
    assert receipt.gain_established is False
    assert receipt.verifier_identity == ""
    published = receipt.to_dict()
    assert published["quality_verified"] is False
    assert published["gain_established"] is False


def test_an_episode_without_a_verifier_is_not_quality_verified():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "quality_verified"
                for target in node.targets
            )
        ):
            continue
        rendered = ast.get_source_segment(source, node.value) or ""
        assert "verifier is not None" in rendered
        assert "branch_selection_admitted" in rendered
        assert "branch_verifier_skipped_budget" in rendered
        return
    raise AssertionError("quality_verified is never assigned")


def test_gain_needs_the_verifier_to_have_accepted_something():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "gain_established"
                for target in node.targets
            )
        ):
            continue
        rendered = ast.get_source_segment(source, node.value) or ""
        assert "receipt.quality_verified" in rendered
        assert "accepted_causal_improvement" in rendered
        return
    raise AssertionError("gain_established is never assigned")


def test_a_flag_family_is_matched_by_prefix():
    from core.brain.llm.latent_cortex.types import EpisodeReceipt

    receipt = EpisodeReceipt(episode_id="e")
    receipt.flag("admission_failed:ValueError")

    assert receipt.has_flag("admission_failed") is True
    assert receipt.has_flag("admission_failed:ValueError") is True
    assert receipt.has_flag("admission") is False


# ─────────────────────────── the queue waits for the outcome


def _engine():
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    engine = LatentCortexEngine.__new__(LatentCortexEngine)
    engine._staged_consolidation_export = None
    return engine


def _receipt(**kwargs):
    from core.brain.llm.latent_cortex.types import EpisodeReceipt

    receipt = EpisodeReceipt(episode_id="e")
    for key, value in kwargs.items():
        setattr(receipt, key, value)
    return receipt


class _Exporter:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.last_export_error = ""

    def export_candidate(self, queue_dir, *, episode_id, evidence):
        self.calls.append({"queue_dir": queue_dir, "episode_id": episode_id, **evidence})
        return queue_dir


def test_nothing_staged_means_nothing_written():
    engine = _engine()
    receipt = _receipt(params_unchanged=True)

    engine._flush_consolidation_export(receipt, failure_reason="")

    assert not receipt.honest_flags


def test_a_failed_episode_withholds_its_candidate():
    engine = _engine()
    exporter = _Exporter()
    engine._staged_consolidation_export = (exporter, {"schema": "x"})
    receipt = _receipt(params_unchanged=True)

    engine._flush_consolidation_export(
        receipt, failure_reason="decode_incomplete:no_tokens_generated"
    )

    assert exporter.calls == []
    assert any(
        flag.startswith("fast_weight_candidate_withheld:")
        for flag in receipt.honest_flags
    )


def test_an_unproven_checkpoint_invariant_withholds_the_candidate():
    engine = _engine()
    exporter = _Exporter()
    engine._staged_consolidation_export = (exporter, {"schema": "x"})
    receipt = _receipt(params_unchanged=False)

    engine._flush_consolidation_export(receipt, failure_reason="")

    assert exporter.calls == []
    assert any(
        "checkpoint_invariant_unproven" in flag for flag in receipt.honest_flags
    )


def test_a_landed_episode_writes_its_candidate_with_the_outcome():
    engine = _engine()
    exporter = _Exporter()
    engine._staged_consolidation_export = (exporter, {"schema": "x"})
    receipt = _receipt(
        params_unchanged=True,
        decode_termination="eos",
        quality_verified=True,
        gain_established=True,
    )
    receipt.flag("fast_weight_site:o_proj")

    engine._flush_consolidation_export(receipt, failure_reason="")

    assert len(exporter.calls) == 1
    written = exporter.calls[0]
    assert written["episode_ok"] is True
    assert written["decode_termination"] == "eos"
    assert written["params_unchanged"] is True
    assert "fast_weight_site:o_proj" in written["honest_flags"]
    assert "fast_weight_candidate_exported" in receipt.honest_flags


def test_a_proxy_only_candidate_is_labelled_unvalidated():
    """A falling loss curve is descent on the optimizer's own proxy. A
    consumer reading only the curve will believe the task got better."""
    engine = _engine()
    exporter = _Exporter()
    engine._staged_consolidation_export = (exporter, {"schema": "x"})
    receipt = _receipt(params_unchanged=True, gain_established=False)

    engine._flush_consolidation_export(receipt, failure_reason="")

    assert exporter.calls[0]["gain_status"] == "unvalidated_optimization_candidate"


def test_a_verifier_confirmed_candidate_says_so():
    engine = _engine()
    exporter = _Exporter()
    engine._staged_consolidation_export = (exporter, {"schema": "x"})
    receipt = _receipt(params_unchanged=True, gain_established=True)

    engine._flush_consolidation_export(receipt, failure_reason="")

    assert exporter.calls[0]["gain_status"] == "verifier_confirmed"


def test_the_candidate_carries_the_verifier_it_was_judged_by():
    engine = _engine()
    exporter = _Exporter()
    engine._staged_consolidation_export = (exporter, {"schema": "x"})
    receipt = _receipt(
        params_unchanged=True,
        verifier_identity="tests.ScriptedVerifier",
        fast_weight_verifier={"decision": "accepted_causal_improvement"},
    )

    engine._flush_consolidation_export(receipt, failure_reason="")

    written = exporter.calls[0]
    assert written["verifier_identity"] == "tests.ScriptedVerifier"
    assert written["verifier"]["decision"] == "accepted_causal_improvement"


def test_the_stage_is_cleared_so_it_cannot_be_written_twice():
    engine = _engine()
    exporter = _Exporter()
    engine._staged_consolidation_export = (exporter, {"schema": "x"})
    receipt = _receipt(params_unchanged=True)

    engine._flush_consolidation_export(receipt, failure_reason="")
    engine._flush_consolidation_export(receipt, failure_reason="")

    assert len(exporter.calls) == 1


def test_finalization_only_stages_and_never_writes():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_finalize_fast_weights"
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "self._staged_consolidation_export = (" in rendered
        assert "export_candidate(" not in rendered
        return
    raise AssertionError("_finalize_fast_weights was not found")


def test_the_write_happens_after_the_checkpoint_invariant():
    """Inside the episode, not merely somewhere in the module: ast.walk is
    not source order, so the comparison has to be scoped."""
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    episode = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_reason_episode"
    )
    flush = None
    post_episode = None
    for node in ast.walk(episode):
        if not isinstance(node, ast.Call):
            continue
        attr = getattr(node.func, "attr", "")
        if attr == "_flush_consolidation_export":
            flush = node.lineno
        if attr == "post_episode" and post_episode is None:
            post_episode = node.lineno

    assert flush is not None, "the episode never flushes the staged candidate"
    assert post_episode is not None, "the episode never closes the invariant"
    assert post_episode < flush


# ─────────────────────────── the erase proof checks structure


def test_the_proof_requires_structural_restoration():
    import core.brain.llm.latent_cortex.fast_weights as fw_mod

    source = inspect.getsource(fw_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "prove_erase"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "self.structural_erase_report()" in rendered
        assert 'structural["structurally_restored"] is True' in rendered
        return
    raise AssertionError("prove_erase was not found")


def test_the_structural_report_names_what_it_checked():
    import core.brain.llm.latent_cortex.fast_weights as fw_mod

    source = inspect.getsource(fw_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "structural_erase_report"
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        for key in (
            "restored_layers",
            "wrapped_layers_remaining",
            "handles_remaining",
            "parameters_before_sha256",
            "parameters_after_sha256",
        ):
            assert key in rendered, key
        return
    raise AssertionError("structural_erase_report was not found")


def test_the_structural_evidence_reaches_the_lifecycle_receipt():
    from core.brain.llm.latent_cortex.fast_weights import FastWeightsLifecycle

    lifecycle = FastWeightsLifecycle()

    assert lifecycle.structural_erase == {}
    assert "structural_erase" in lifecycle.to_receipt()


def test_an_episode_that_touched_nothing_is_not_structurally_restored():
    """Vacuous truth is the failure mode this guards: an empty site list must
    not read as "everything was put back"."""
    from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights

    weights = EpisodicFastWeights.__new__(EpisodicFastWeights)
    weights.handles = []

    report = weights.structural_erase_report()

    assert report["touched_layers"] == []
    assert report["structurally_restored"] is False


# ─────────────────────────── the optimizer sees the answer's function


def test_the_teacher_trajectory_forward_is_causal():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "forward"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "embed_tokens(mx.array([context_tokens]))" not in rendered:
            continue
        # The property is causality, not one spelling of it. The forward built
        # its own create_attention_mask once; it goes through
        # decoder_layer_masks now, which also keeps the state-space layers of
        # a hybrid decoder on their own mask contract. What must never come
        # back is a bare None mask over a multi-token sequence — that is FULL
        # attention, and it optimises the delta against a function that can
        # see its own future.
        assert "decoder_layer_masks(" in rendered
        assert "layer(h, masks[index], None)" in rendered
        assert "layer(h, None, None)" not in rendered
        return
    raise AssertionError("the teacher-trajectory forward was not found")


def test_the_mask_the_teacher_forward_builds_is_actually_causal():
    """Reading the call is not reading the mask. This runs it."""
    import mlx.core as mx

    from core.brain.llm.decoder_topology import decoder_layer_masks

    class _Layer:
        pass

    class _Dense:
        layers = [_Layer(), _Layer()]

    hidden = mx.zeros((1, 6, 8))
    masks = decoder_layer_masks(_Dense(), hidden, None)
    assert len(masks) == 2
    for mask in masks:
        # A dense decoder over six tokens must be given a mask. None here is
        # the full-attention case the forward exists to avoid.
        assert mask is not None


def test_the_identity_probe_stays_deliberately_mask_free():
    """It is only ever compared against itself, so sameness is the property
    that matters — but the reason has to be written down."""
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_fw_probe"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "layer(h, None, None)" in rendered
        assert "compared against ITSELF" in rendered
        return
    raise AssertionError("_fw_probe was not found")


# ─────────────────────────── the cache and the weights are named


def test_no_plasticity_target_claims_cache_independence():
    from core.brain.llm.latent_cortex.fast_weights import CACHE_INDEPENDENT_TARGETS

    assert CACHE_INDEPENDENT_TARGETS == frozenset()


@pytest.mark.parametrize("target", ["o_proj", "down_proj"])
def test_both_supported_targets_attest_a_base_weight_prefix(target):
    from core.brain.llm.latent_cortex.fast_weights import target_cache_attestation

    attestation = target_cache_attestation(target)

    assert attestation["cache_independent"] is False
    assert attestation["prefix_kv_under_base_weights"] is True
    assert "residual stream" in attestation["reason"]


def test_the_episode_flags_the_mixture_when_it_attaches():
    source = inspect.getsource(engine_mod)

    assert "receipt.fast_weight_cache_attestation = target_cache_attestation(" in source
    assert 'receipt.flag("fast_weight_prefix_kv_under_base_weights")' in source


def test_the_attestation_is_published():
    from core.brain.llm.latent_cortex.types import EpisodeReceipt

    receipt = EpisodeReceipt(episode_id="e")

    assert receipt.fast_weight_cache_attestation == {}
    assert "fast_weight_cache_attestation" in receipt.to_dict()


# ─────────────────────────── recurrence support is stated, not assumed


class _Args:
    def __init__(self, model_type="", positions=0):
        if model_type:
            self.model_type = model_type
        if positions:
            self.max_position_embeddings = positions


class _Model:
    def __init__(self, model_type="", positions=0):
        self.args = _Args(model_type, positions)


def test_an_unregistered_architecture_is_structural_only():
    verdict = classify_recurrence_support(
        _Model("qwen2", 32768), layer_count=32, window=(8, 24), registry={}
    )

    assert verdict["level"] == STRUCTURAL
    assert "architecture_not_registered" in verdict["reasons"]


def test_a_model_that_declares_no_position_limit_is_experimental():
    verdict = classify_recurrence_support(
        _Model("qwen2"), layer_count=32, window=(8, 24), registry={}
    )

    assert verdict["level"] == EXPERIMENTAL
    assert "position_limit_undeclared" in verdict["reasons"]


def test_a_registered_architecture_with_a_declared_window_is_certified():
    registry = {"qwen2": {"evidence_path": "artifacts/closeout/paired_rlc.json"}}

    verdict = classify_recurrence_support(
        _Model("qwen2", 32768), layer_count=32, window=(8, 24), registry=registry
    )

    assert verdict["level"] == CERTIFIED
    assert verdict["evidence"]["evidence_path"].endswith("paired_rlc.json")


def test_an_invalid_window_cannot_be_certified():
    registry = {"qwen2": {"evidence_path": "x"}}

    verdict = classify_recurrence_support(
        _Model("qwen2", 32768), layer_count=32, window=(24, 8), registry=registry
    )

    assert verdict["level"] == EXPERIMENTAL
    assert "recurrent_window_invalid" in verdict["reasons"]


def test_a_missing_registry_certifies_nothing():
    from pathlib import Path

    assert load_certified_architectures(Path("/nonexistent/registry.json")) == {}


def test_a_registry_row_without_evidence_is_a_claim_and_is_dropped(tmp_path):
    import json
    from pathlib import Path

    path = Path(tmp_path) / "registry.json"
    path.write_text(
        json.dumps(
            {
                "architectures": {
                    "with_evidence": {"evidence_path": "artifacts/x.json"},
                    "bare_claim": {"note": "it works"},
                }
            }
        )
    )

    loaded = load_certified_architectures(path)

    assert set(loaded) == {"with_evidence"}


def test_the_engine_publishes_the_verdict_on_every_episode():
    source = inspect.getsource(engine_mod)

    assert "self.recurrence_support = classify_recurrence_support(" in source
    assert "receipt.recurrence_support = dict(self.recurrence_support)" in source


# ─────────────────────────── a required gate is not optional


def test_the_verifier_mode_is_declared():
    from core.brain.llm.latent_cortex.types import CortexConfig

    assert CortexConfig().branch_verifier_mode == "advisory"


def test_an_unknown_verifier_mode_is_rejected():
    from core.brain.llm.latent_cortex.types import CortexConfig

    problems = CortexConfig(branch_verifier_mode="whenever").validate()

    assert any("branch_verifier_mode" in problem for problem in problems)


def test_required_mode_fails_rather_than_substituting_the_internal_score():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if 'self.config.branch_verifier_mode == "required"' not in rendered:
            continue
        if "raise RuntimeError" not in rendered:
            continue
        assert "branch verification is required" in rendered
        return
    raise AssertionError("the required-mode refusal was not found")


def test_advisory_mode_still_records_the_skip():
    source = inspect.getsource(engine_mod)

    assert 'receipt.flag("branch_verifier_skipped_budget")' in source


def test_required_mode_reserves_the_gate_at_admission():
    """A gate admission never priced could only ever be skipped for budget."""
    source = inspect.getsource(engine_mod)

    assert "required_branch_verification_cost" in source
    assert "+ required_branch_verification_cost" in source
