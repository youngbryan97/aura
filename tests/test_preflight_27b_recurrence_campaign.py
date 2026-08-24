"""The preflight has to refuse each way a frozen bundle can stop being true.

A gate that only ever passes is a gate nobody can trust, so every finding kind
gets a test that produces it. The bundle under test is built once from the real
checkpoint and then damaged in one specific way per case.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import prepare_27b_recurrence_campaign as prepare
from tools import preflight_27b_recurrence_campaign as preflight

INSTALL = Path("/Users/bryan/.aura/live-source")


@pytest.fixture(scope="module")
def bundle():
    manifest = INSTALL / "training/fused-model/active.json"
    if not manifest.exists():
        pytest.skip("no active model manifest in this environment")
    model = Path(json.loads(manifest.read_text())["active_model_path"])
    if not (model / "config.json").exists():
        pytest.skip("active checkpoint is not installed")
    return prepare.build(model, INSTALL)


def _kinds(bundle):
    return {finding["kind"] for finding in preflight.check(bundle)}


def _reseal(bundle):
    body = {k: v for k, v in bundle.items() if k != "campaign_sha256"}
    return {**body, "campaign_sha256": prepare._sha(body)}


def test_an_untouched_bundle_may_launch(bundle):
    assert preflight.check(bundle) == []


def test_a_tampered_bundle_is_refused_before_anything_else(bundle):
    damaged = dict(bundle)
    damaged["futility_gates"] = []
    # Digest not recomputed: this is what editing a frozen bundle looks like.
    assert _kinds(damaged) == {"bundle_tampered"}


def test_an_unknown_schema_is_refused(bundle):
    damaged = _reseal({**bundle, "schema": "something.else.v1"})
    assert "schema_unrecognised" in _kinds(damaged)


def test_source_drift_names_the_file(bundle):
    relative = sorted(bundle["source_freeze"])[0]
    damaged = _reseal(
        {**bundle, "source_freeze": {**bundle["source_freeze"], relative: "0" * 64}}
    )
    findings = preflight.check(damaged)
    assert any(
        f["kind"] == "source_drifted" and f["detail"] == relative for f in findings
    )


def test_a_missing_source_file_is_refused(bundle):
    damaged = _reseal(
        {
            **bundle,
            "source_freeze": {**bundle["source_freeze"], "core/not_a_file.py": "0" * 64},
        }
    )
    assert "source_missing" in _kinds(damaged)


def test_portable_tissue_drift_is_refused(bundle):
    relative = sorted(bundle["portable_tissue"])[0]
    damaged = _reseal(
        {
            **bundle,
            "portable_tissue": {**bundle["portable_tissue"], relative: "0" * 64},
        }
    )
    assert "tissue_drifted" in _kinds(damaged)


def test_a_swapped_checkpoint_config_is_refused(bundle):
    damaged = _reseal(
        {
            **bundle,
            "target_checkpoint": {
                **bundle["target_checkpoint"],
                "config_sha256": "0" * 64,
            },
        }
    )
    assert "checkpoint_config_drifted" in _kinds(damaged)


def test_a_swapped_tokenizer_is_refused(bundle):
    damaged = _reseal(
        {
            **bundle,
            "target_checkpoint": {
                **bundle["target_checkpoint"],
                "tokenizer_sha256": "0" * 64,
            },
        }
    )
    assert "checkpoint_file_drifted" in _kinds(damaged)


def test_an_absent_checkpoint_is_refused(bundle):
    damaged = _reseal(
        {
            **bundle,
            "target_checkpoint": {**bundle["target_checkpoint"], "path": "/nope"},
        }
    )
    assert "checkpoint_absent" in _kinds(damaged)


def test_the_runtime_pointing_elsewhere_is_refused(bundle, tmp_path):
    # The campaign is defined against one checkpoint. If the runtime has been
    # repointed, the run would measure one model and receipt another.
    legacy = INSTALL / "training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118"
    if not legacy.exists():
        pytest.skip("legacy checkpoint is not installed")
    damaged = prepare.build(legacy, INSTALL)
    assert "active_model_moved" in _kinds(damaged)


def test_a_changed_attention_layout_is_refused(bundle):
    mapping = dict(bundle["recurrence_layer_mapping"])
    mapping["attention_layer_indices"] = [0, 1, 2, 3]
    damaged = _reseal({**bundle, "recurrence_layer_mapping": mapping})
    assert "attention_layout_changed" in _kinds(damaged)


def test_a_misaligned_window_is_refused(bundle):
    mapping = dict(bundle["recurrence_layer_mapping"])
    if not mapping.get("full_attention_interval"):
        pytest.skip("dense checkpoint; alignment is unconstrained")
    mapping["window"] = [15, 48]
    damaged = _reseal({**bundle, "recurrence_layer_mapping": mapping})
    assert "window_misaligned" in _kinds(damaged)
