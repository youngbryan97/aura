"""The frozen bundle has to be the whole launch decision.

Anything the campaign could still choose after preparation is something that
can be chosen once the numbers are visible. These tests hold the properties
that make the bundle worth freezing: one model residency, gates fixed in
advance, source pinned by digest, and no authority inherited from the 32B
result the campaign exists to re-earn.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.hybrid_recurrence_geometry import LayerGeometry
from tools import prepare_27b_recurrence_campaign as campaign

INSTALL = Path("/Users/bryan/.aura/live-source")


@pytest.fixture(scope="module")
def bundle():
    manifest = INSTALL / "training/fused-model/active.json"
    if not manifest.exists():
        pytest.skip("no active model manifest in this environment")
    model = Path(json.loads(manifest.read_text())["active_model_path"])
    if not (model / "config.json").exists():
        pytest.skip("active checkpoint is not installed")
    return campaign.build(model, INSTALL)


def test_the_model_loads_exactly_once(bundle):
    stages = bundle["stages"]
    active = [i for i, s in enumerate(stages) if s["model_active"]]
    assert active, "a campaign with no model-active stage trains nothing"
    # Contiguous means one load and one unload. A gap is a second residency.
    assert active == list(range(active[0], active[-1] + 1))


def test_verification_happens_after_the_unload(bundle):
    names = [s["name"] for s in bundle["stages"]]
    assert names.index("unload") < names.index("independent_verification")
    assert names.index("independent_verification") < names.index("adjudication")
    verification = next(
        s for s in bundle["stages"] if s["name"] == "independent_verification"
    )
    assert verification["model_active"] is False


def test_activation_is_the_last_stage_and_needs_no_model(bundle):
    last = bundle["stages"][-1]
    assert last["name"] == "activation_materialization"
    assert last["model_active"] is False


def test_every_futility_gate_names_a_measure_and_a_threshold(bundle):
    assert bundle["futility_gates"]
    stage_names = {s["name"] for s in bundle["stages"]}
    for gate in bundle["futility_gates"]:
        assert gate["after"] in stage_names
        assert gate["measure"] and gate["stops_when"] and gate["reason"]


def test_the_lesion_gate_exists_because_a_gain_without_it_is_a_budget_effect(bundle):
    measures = {gate["measure"] for gate in bundle["futility_gates"]}
    assert "families_separating_under_lesion" in measures


def test_the_32b_result_carries_no_authority(bundle):
    legacy = bundle["legacy_claim"]
    assert legacy["verdict"] == "BOUNDED_WOW_SIGNAL"
    assert legacy["authority_over_this_campaign"] == "none"
    assert "different checkpoint" in legacy["why"]


def test_training_completion_never_authorizes_serving(bundle):
    never = bundle["experiments"]["never_authorized_by_training_completion"]
    assert "ordinary_chat_authorized" in never
    assert "arbitrary_reasoning_authorized" in never
    assert "global runtime promotion" in never


def test_generalization_cannot_run_before_recovery_is_adjudicated(bundle):
    generalization = bundle["experiments"]["generalization"]
    assert generalization["precondition"] == "recovery adjudicated positive"


def test_the_recovery_arms_match_the_claim_being_re_earned(bundle):
    arms = bundle["experiments"]["recovery"]["arms"]
    # The four controls are what made CP566 a mechanism claim rather than a
    # score. Dropping any one of them changes what a positive result means.
    for arm in (
        "treatment",
        "ordinary_base",
        "matched_wire_base",
        "coefficient_lesion",
        "matched_wrong_state",
    ):
        assert arm in arms


def test_source_is_pinned_by_digest_and_matches_the_tree(bundle):
    assert len(bundle["source_freeze"]) >= 15
    for relative, digest in bundle["source_freeze"].items():
        path = campaign.REPO_ROOT / relative
        assert path.exists(), relative
        assert campaign._sha_file(path) == digest


def test_the_bundle_pins_this_checkpoint_not_a_class_of_them(bundle):
    descriptor = bundle["target_checkpoint"]
    assert descriptor["config_sha256"]
    assert descriptor["weights_index_sha256"]
    assert descriptor["tokenizer_sha256"]
    assert descriptor["weight_file_count"] > 0


def test_the_window_is_aligned_to_the_hybrid_layout(bundle):
    geometry = bundle["geometry"]
    assert geometry["alignment_errors"] == []
    mapping = bundle["recurrence_layer_mapping"]
    if mapping["full_attention_interval"]:
        interval = mapping["full_attention_interval"]
        start, end = mapping["window"]
        assert start % interval == 0
        assert end % interval == 0


def test_adding_a_feed_forward_target_recovers_the_thinned_window(bundle):
    # Attention targets alone reach one layer in four on a hybrid checkpoint.
    # The bundle names down_proj too, which is why the site count is not 16.
    geometry = bundle["geometry"]
    if not geometry["is_hybrid"]:
        pytest.skip("dense checkpoint; nothing thins out")
    assert "down_proj" in geometry["targets"]
    assert geometry["expected_adapter_site_count"] > len(
        geometry["attention_layers_in_window"]
    ) * 2


def test_portable_tissue_is_pinned_and_present(bundle):
    assert bundle["portable_tissue"]
    for relative, digest in bundle["portable_tissue"].items():
        path = campaign.REPO_ROOT / relative
        assert path.exists(), relative
        assert campaign._sha_file(path) == digest


def test_grounding_dispositions_come_from_the_bound_measurement(bundle):
    grounding = bundle["grounding_portability"]
    assert "recurrent_literal_grounding.py" in grounding["portable"]
    assert "recurrent_opcode_grounding.py" in grounding["must_regenerate"]
    stage = next(stage for stage in bundle["stages"] if stage["name"] == "regrounding")
    assert "every recorded token id is stale" not in stage["note"]
    assert "recurrent_literal_grounding.py" in stage["note"]


def test_grounding_measurement_for_another_checkpoint_is_refused(bundle):
    report = json.loads(
        (campaign.REPO_ROOT / "artifacts/migration/27b/grounding_portability.json").read_text()
    )
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    body["target_checkpoint_identity"] = {
        **body["target_checkpoint_identity"],
        "tokenizer_sha256": "0" * 64,
    }
    damaged = {**body, "report_sha256": campaign._sha(body)}
    model = Path(bundle["target_checkpoint"]["path"])
    with pytest.raises(SystemExit, match="another checkpoint"):
        campaign.build(model, INSTALL, damaged)


def test_the_bundle_digest_covers_the_bundle(bundle):
    body = {k: v for k, v in bundle.items() if k != "campaign_sha256"}
    assert campaign._sha(body) == bundle["campaign_sha256"]


def test_a_misaligned_window_refuses_to_build(monkeypatch, tmp_path):
    # A checkpoint whose layer count puts the quarter-fraction off a group
    # boundary must not produce a bundle at all.
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 66,
            "full_attention_interval": 4,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
        },
    }
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="misaligned"):
        campaign.build(model, INSTALL)


def test_a_dense_checkpoint_still_prepares(tmp_path):
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 64,
        "vocab_size": 152064,
        "max_position_embeddings": 32768,
    }
    model = tmp_path / "dense"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config))
    (model / "tokenizer.json").write_text("{}")
    grounding_body = {
        "schema": "aura.rlc.27b_grounding_portability.v1",
        "target_checkpoint_identity": {
            "path": str(model.resolve()),
            "tokenizer_sha256": campaign._sha_file(model / "tokenizer.json"),
        },
        "portable": ["recurrent_literal_grounding.py"],
        "must_regenerate": ["recurrent_opcode_grounding.py"],
    }
    grounding = {
        **grounding_body,
        "report_sha256": campaign._sha(grounding_body),
    }
    built = campaign.build(model, INSTALL, grounding)
    assert built["geometry"]["is_hybrid"] is False
    assert LayerGeometry.from_config(config).attention_layers() == tuple(range(64))
