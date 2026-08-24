"""Every label surface names the checkpoint that is loaded.

Chat built its lane label by lowercasing every lane field it had and looking
for the substring "32b". That cannot match a 27B, and the "cortex" token it
fell back on was wired to the literal "Cortex (32B)" — so the surface named a
checkpoint that had been replaced while its signed descriptor sat one call
away. The inference gate's protected-lane log line did the same thing.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from core.brain.llm import model_registry

INSTALL = Path("/Users/bryan/.aura/live-source")
HERE = Path(__file__).resolve().parents[1]


# ── The lane label is derived ───────────────────────────────────────────


def test_the_cortex_lane_reads_from_the_signed_descriptor():
    label = model_registry.lane_display_label("Cortex")
    identity = model_registry.resident_model_identity()
    assert label == identity["lane"] or label == "Cortex"


def test_a_lane_pointed_at_the_cortex_gets_the_cortex_label():
    # The deep lane is served by the cortex when no separate solver is
    # resident; showing it with no size, or a stale 72B, would both be wrong.
    if model_registry.get_lane_model_name("Solver") != model_registry.CORTEX_LOGICAL_NAME:
        pytest.skip("solver lane has its own model in this environment")
    label = model_registry.resident_model_label(default="")
    if not label:
        # An isolated state root cannot read the signed pointer; rendering the
        # bare endpoint is the correct answer there, not "Solver ()".
        assert model_registry.lane_display_label("Solver") == "Solver"
        return
    assert model_registry.lane_display_label("Solver") == f"Solver ({label})"


def test_other_lanes_read_the_size_the_registry_assigned_them():
    # A registry-declared name is a declaration, not a guess.
    assert model_registry.lane_display_label("Reflex").endswith("(1.5B)")


def test_a_lane_with_no_declared_size_renders_as_its_own_name(monkeypatch):
    monkeypatch.setattr(model_registry, "get_lane_model_name", lambda e: "mystery-model")
    assert model_registry.lane_display_label("Brainstem") == "Brainstem"


def test_a_missing_descriptor_does_not_invent_a_cortex_size(monkeypatch):
    monkeypatch.setattr(model_registry, "get_active_cortex_spec", lambda **k: None)
    assert model_registry.lane_display_label("Cortex") == "Cortex"


# ── No runtime surface still hardcodes a parameter count ────────────────


def test_chat_no_longer_matches_a_parameter_count_to_pick_a_label():
    production_surfaces = (
        HERE / "interface/routes/chat.py",
        HERE / "interface/routes/chat_preflight.py",
        HERE / "interface/routes/system.py",
        HERE / "core/brain/inference_gate.py",
        HERE / "core/health/boot_status.py",
    )
    for path in production_surfaces:
        text = path.read_text()
        literals = {
            node.value
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for literal in ('"Cortex (32B)"', '"Solver (72B)"', '"Brainstem (7B)"'):
            assert literal[1:-1] not in literals, path
    assert "lane_display_label" in (HERE / "interface/routes/chat.py").read_text()


def test_the_inference_gate_log_line_carries_the_signed_label():
    text = (HERE / "core/brain/inference_gate.py").read_text()
    assert "primary cortex lane (32B)" not in text
    assert "_primary_lane_label" in text


def test_the_gate_label_helper_degrades_rather_than_raising(monkeypatch):
    from core.brain import inference_gate

    monkeypatch.setattr(
        model_registry,
        "resident_model_label",
        lambda **k: (_ for _ in ()).throw(OSError("pointer gone")),
    )
    # A log line is never worth an exception.
    assert inference_gate._primary_lane_label() == "Cortex"


def test_the_ui_fallback_names_no_parameter_count():
    text = (HERE / "interface/static/aura.js").read_text()
    assert "Cortex (32B)" not in text


# ── Historical evidence stays 32B ───────────────────────────────────────


def test_cp566_remains_a_32b_claim():
    path = (
        INSTALL
        / "artifacts/closeout/latent_cortex"
        / "cp566_resident_mixed_multidomain_replication/adjudication.json"
    )
    if not path.exists():
        pytest.skip("CP566 evidence is not installed")
    payload = json.loads(path.read_text())
    assert "resident-32B" in payload["claim"]
    assert payload["verdict"] == "BOUNDED_WOW_SIGNAL"


def test_cp568_still_pins_the_32b_checkpoint():
    path = (
        INSTALL
        / "artifacts/closeout/latent_cortex/cp568_semantic_neural_active_r1"
        / "runtime_verification.json"
    )
    if not path.exists():
        pytest.skip("CP568 evidence is not installed")
    identity = json.loads(path.read_text())["activation_receipt"]["model_identity"]
    assert "Aura-32B-crsm-closeout-jul1" in identity["path"]


def test_the_resource_bucket_stays_deliberately_coarse():
    # 27b maps onto the 32b memory class on purpose; the eviction policy keys
    # on cost, and renaming the bucket would change behaviour, not labels.
    from core.brain.llm.model_artifact_profile import _32B_PATH_TOKENS

    assert "27b" in _32B_PATH_TOKENS
    identity = model_registry.resident_model_identity()
    if identity["tag"]:
        assert identity["resource_class"].lower() == "32b"
        assert identity["label"] != identity["resource_class"]
