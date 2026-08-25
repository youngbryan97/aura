"""A 27B resident must say 27B, and a 32B artifact must stay 32B.

`size_class` is a resource bucket whose token list maps 27b onto "32b" on
purpose: the two cost the host the same class of memory and the eviction policy
keys on that. Reading it as a display name is how a 27B resident came to
announce itself as a 32B, and the fix is not to change the bucket.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm import model_registry

#: The checkout this test is running from. Derived rather than written
#: down: a literal install path reads another checkout's artifacts when
#: the suite runs in a worktree, and names one machine's account.
INSTALL = Path(__file__).resolve().parents[1]


def _spec(parameters: int | None, *, tag: str = "cp954-27b-resident",
          size_class: str = "32B"):
    profile = {"model_type": "qwen3_5_text", "native_context_window": 262144}
    if parameters is not None:
        profile["total_parameters"] = parameters
    return SimpleNamespace(
        tag=tag,
        size_class=size_class,
        descriptor_sha256="d" * 64,
        artifact_descriptor=lambda: {"artifact_profile": profile},
    )


def test_a_27b_resident_reports_27b(monkeypatch):
    monkeypatch.setattr(
        model_registry, "get_active_cortex_spec", lambda **k: _spec(26_895_993_856)
    )
    assert model_registry.resident_model_label() == "27B"


def test_the_resource_bucket_is_not_the_label(monkeypatch):
    # The bucket still says 32B, deliberately, and the label does not.
    monkeypatch.setattr(
        model_registry, "get_active_cortex_spec", lambda **k: _spec(26_895_993_856)
    )
    identity = model_registry.resident_model_identity()
    assert identity["label"] == "27B"
    assert identity["resource_class"] == "32B"


def test_a_32b_resident_still_reports_32b(monkeypatch):
    monkeypatch.setattr(
        model_registry, "get_active_cortex_spec", lambda **k: _spec(32_000_000_000)
    )
    assert model_registry.resident_model_label() == "32B"


def test_the_lane_label_carries_the_derived_name(monkeypatch):
    monkeypatch.setattr(
        model_registry, "get_active_cortex_spec", lambda **k: _spec(26_895_993_856)
    )
    assert model_registry.resident_model_identity()["lane"] == "Cortex (27B)"


def test_a_descriptor_without_a_parameter_count_falls_back_to_the_tag(monkeypatch):
    monkeypatch.setattr(
        model_registry, "get_active_cortex_spec", lambda **k: _spec(None, tag="my-tag")
    )
    assert model_registry.resident_model_label() == "my-tag"


def test_no_active_pointer_yields_the_caller_default(monkeypatch):
    monkeypatch.setattr(model_registry, "get_active_cortex_spec", lambda **k: None)
    assert model_registry.resident_model_label() == "Cortex"
    assert model_registry.resident_model_label(default="") == ""


def test_an_unreadable_pointer_does_not_raise(monkeypatch):
    def _boom(**kwargs):
        raise OSError("pointer unreadable")

    monkeypatch.setattr(model_registry, "get_active_cortex_spec", _boom)
    assert model_registry.resident_model_label() == "Cortex"
    assert model_registry.resident_model_identity()["label"] == "Cortex"


def test_the_live_resident_reports_its_actual_size():
    manifest = INSTALL / "training/fused-model/active.json"
    if not manifest.exists():
        pytest.skip("no active model manifest")
    identity = model_registry.resident_model_identity()
    if not identity["tag"]:
        # An isolated state root cannot read the signed pointer, and falling
        # back to the caller default is the correct answer there.
        pytest.skip("no readable active cortex pointer in this environment")
    profile = json.loads(manifest.read_text())["artifact_descriptor"]["artifact_profile"]
    parameters = profile.get("total_parameters")
    if not parameters:
        pytest.skip("descriptor carries no parameter count")
    assert identity["label"] == f"{parameters / 1e9:.0f}B"
    assert identity["model_type"] == profile.get("model_type")


# ── Historical artifacts keep their own identity ────────────────────────


def test_a_historical_32b_certificate_is_not_relabelled():
    """CP566/CP568 were measured on the 32B and must still say so."""
    adjudication = (
        INSTALL
        / "artifacts/closeout/latent_cortex"
        / "cp566_resident_mixed_multidomain_replication/adjudication.json"
    )
    if not adjudication.exists():
        pytest.skip("CP566 evidence is not installed")
    payload = json.loads(adjudication.read_text())
    assert payload["verdict"] == "BOUNDED_WOW_SIGNAL"
    assert "resident-32B" in payload["claim"]


def test_the_cp568_package_still_names_the_32b_checkpoint():
    receipt = (
        INSTALL
        / "artifacts/closeout/latent_cortex/cp568_semantic_neural_active_r1"
        / "runtime_verification.json"
    )
    if not receipt.exists():
        pytest.skip("CP568 evidence is not installed")
    identity = json.loads(receipt.read_text())["activation_receipt"]["model_identity"]
    assert "Aura-32B-crsm-closeout-jul1" in identity["path"]


# ── The context window is served, not named ─────────────────────────────


def test_a_measured_window_wins_over_the_table():
    from core.context import context_manager

    # The registry knows what the promoted checkpoint was qualified for;
    # layering forbids either module from reaching for the other, so the
    # caller holding the limits passes the number down.
    assert context_manager.resolved_context_limit("Cortex", served_tokens=131072) == 131072


def test_no_measured_window_falls_back_to_the_table():
    from core.context import context_manager

    assert context_manager.resolved_context_limit("Cortex") == 32_000
    assert context_manager.resolved_context_limit("Cortex", served_tokens=None) == 32_000


def test_a_nonsense_window_is_ignored_rather_than_trusted():
    from core.context import context_manager

    for bogus in (0, -1, True, "32768", 3.5):
        assert context_manager.resolved_context_limit("Cortex", served_tokens=bogus) == 32_000


def test_the_registry_can_report_the_served_window():
    identity = model_registry.resident_model_identity()
    served = identity["served_context_tokens"]
    assert served is None or (isinstance(served, int) and served > 0)


def test_other_lanes_still_read_from_the_table():
    from core.context import context_manager

    assert context_manager.resolved_context_limit("Brainstem") == 8_000
    assert context_manager.resolved_context_limit("unknown-lane") == 16_000


def test_no_runtime_label_surface_still_hardcodes_a_parameter_count():
    """The UI fallback and the idle-unload log named a model size."""
    ui = INSTALL / "interface/static/aura.js"
    monitor = INSTALL / "core/autonomic/core_monitor.py"
    for path in (ui, monitor):
        if not path.exists():
            continue
        # Read from the worktree copy, which is what this branch changes.
        local = Path(__file__).resolve().parents[1] / path.relative_to(INSTALL)
        text = local.read_text()
        assert "Cortex (32B)" not in text
        assert "Unloading 32B cortex" not in text
