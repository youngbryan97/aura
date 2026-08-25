"""The preflight has to refuse each way a frozen bundle can stop being true.

A gate that only ever passes is a gate nobody can trust, so every finding kind
gets a test that produces it. The bundle under test is built once from the real
checkpoint and then damaged in one specific way per case.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import preflight_27b_recurrence_campaign as preflight
from tools import prepare_27b_recurrence_campaign as prepare
from tools import verify_27b_grounding_portability as grounding

#: The checkout this test is running from. Derived rather than written
#: down: a literal install path reads another checkout's artifacts when
#: the suite runs in a worktree, and names one machine's account.
INSTALL = Path(__file__).resolve().parents[1]


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


@pytest.fixture
def quiet_host(monkeypatch):
    """Neutralise the environmental guards.

    Host memory and lane ownership are real launch blockers and are tested on
    their own below. A bundle-integrity test that also depended on how much RAM
    happened to be free would fail for reasons that say nothing about the
    bundle.
    """
    monkeypatch.setattr(preflight, "_resource_findings", lambda b, m: [])
    monkeypatch.setattr(preflight, "_ownership_findings", list)


def test_an_untouched_bundle_may_launch(bundle, quiet_host):
    assert preflight.check(bundle) == []


def test_a_tampered_bundle_is_refused_before_anything_else(bundle, quiet_host):
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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(legacy))
    legacy_grounding = grounding.build(
        {"legacy": tokenizer, "target": tokenizer},
        legacy_model=legacy,
        target_model=legacy,
    )
    damaged = prepare.build(legacy, INSTALL, legacy_grounding)
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


# ── Host and ownership guards ───────────────────────────────────────────
# An out-of-memory kill mid-training loses the residency and leaves a partial
# journal that a resume then has to adjudicate. Refusing up front is cheaper
# than deciding whether half a campaign is scientifically resumable.


def test_a_host_without_room_is_refused(bundle, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "_resource_findings",
        lambda b, m: [{"kind": "insufficient_ram", "detail": "not enough"}],
    )
    assert "insufficient_ram" in _kinds(bundle)


def test_unmeasured_memory_is_refused_rather_than_assumed_fine(monkeypatch):
    import sys

    module = type(sys)("core.runtime.mlx_memory_guard")
    module.host_pressure = lambda: {}
    monkeypatch.setitem(sys.modules, "core.runtime.mlx_memory_guard", module)
    findings = preflight._resource_findings(
        {"target_checkpoint": {"weight_bytes": 1}}, Path("/")
    )
    assert any(f["kind"] == "ram_unmeasured" for f in findings)


def test_free_and_reclaimable_are_both_counted(monkeypatch):
    import sys

    module = type(sys)("core.runtime.mlx_memory_guard")
    # macOS "Pages free" excludes pages the kernel hands back on demand, so
    # counting only free_gb would refuse a host that is actually fine.
    module.host_pressure = lambda: {
        "free_gb": 1.0,
        "reclaimable_gb": 60.0,
        "under_pressure": False,
    }
    monkeypatch.setitem(sys.modules, "core.runtime.mlx_memory_guard", module)
    findings = preflight._resource_findings(
        {"target_checkpoint": {"weight_bytes": 15 * 1024**3}}, Path("/")
    )
    assert not any(f["kind"] == "insufficient_ram" for f in findings)


def test_a_host_under_pressure_is_flagged_even_with_room(monkeypatch):
    import sys

    module = type(sys)("core.runtime.mlx_memory_guard")
    module.host_pressure = lambda: {
        "free_gb": 1.0,
        "reclaimable_gb": 60.0,
        "under_pressure": True,
        "pressure_reasons": ["compressor_high"],
    }
    monkeypatch.setitem(sys.modules, "core.runtime.mlx_memory_guard", module)
    findings = preflight._resource_findings(
        {"target_checkpoint": {"weight_bytes": 15 * 1024**3}}, Path("/")
    )
    assert any(f["kind"] == "host_under_memory_pressure" for f in findings)


def test_an_owned_model_lane_is_refused(bundle, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "_ownership_findings",
        lambda: [{"kind": "model_lane_already_owned", "detail": "trainer-7"}],
    )
    # Two campaigns on one 64 GB host is the failure that takes the machine down.
    assert "model_lane_already_owned" in _kinds(bundle)


def test_evidence_aimed_at_the_legacy_namespace_is_refused(bundle):
    damaged = _reseal(
        {
            **bundle,
            "evidence_paths": [
                "artifacts/closeout/latent_cortex/cp566_resident_mixed_multidomain_replication/result.json"
            ],
        }
    )
    assert "stale_evidence_root" in _kinds(damaged)


def test_evidence_in_the_recovery_namespace_is_accepted(bundle, monkeypatch):
    monkeypatch.setattr(preflight, "_resource_findings", lambda b, m: [])
    monkeypatch.setattr(preflight, "_ownership_findings", list)
    damaged = _reseal(
        {**bundle, "evidence_paths": ["artifacts/migration/27b/recovery/canary.json"]}
    )
    assert "stale_evidence_root" not in _kinds(damaged)
