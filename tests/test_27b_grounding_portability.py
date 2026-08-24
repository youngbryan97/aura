"""Grounding portability is measured per binding, never inferred from a count.

The first migration pass reasoned that a vocabulary growing from 152,064 to
248,320 entries must retire every token id bound to it. Re-deriving the
bindings disagrees in both directions, and both directions are expensive:
calling a portable contract dead discards working tissue, and calling a changed
one portable serves answers assembled from ids that now mean something else.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import verify_27b_grounding_portability as portability

INSTALL = Path("/Users/bryan/.aura/live-source")


@pytest.fixture(scope="module")
def report():
    if not portability.LEGACY_MODEL.exists():
        pytest.skip("legacy checkpoint is not installed")
    manifest = INSTALL / "training/fused-model/active.json"
    if not manifest.exists():
        pytest.skip("no active model manifest")
    return portability.build()


def test_the_digit_ids_did_not_move(report):
    # Literal grounding is the contract the typed answer path leans on most.
    finding = next(
        f for f in report["findings"] if f["binding"] == "digit_token_ids"
    )
    assert finding["portable"] is True
    assert report["legacy_bindings"]["digit_token_ids"] == list(range(15, 25))
    assert report["target_bindings"]["digit_token_ids"] == list(range(15, 25))


def test_every_opcode_marker_moved(report):
    finding = next(
        f for f in report["findings"] if f["binding"] == "opcode_marker_patterns"
    )
    assert finding["portable"] is False
    # Seven frontier families, all of them multi-token English phrases.
    assert len(report["changed_opcode_markers"]) == 7
    for pair in report["changed_opcode_markers"].values():
        assert pair["legacy"] != pair["target"]


def test_the_typed_schemas_were_never_at_risk(report):
    typed = report["typed_only_contracts"]
    assert "core/learning/recurrent_action_schema.py" in typed
    assert "core/learning/recurrent_state_schema.py" in typed


def test_portable_and_regenerate_lists_partition_the_findings(report):
    assert set(report["portable"]) & set(report["must_regenerate"]) == set()
    assert len(report["portable"]) + len(report["must_regenerate"]) == len(
        report["findings"]
    )


def test_the_report_digest_covers_the_report(report):
    body = {k: v for k, v in report.items() if k != "report_sha256"}
    assert portability._digest(body) == report["report_sha256"]


def test_the_report_is_bound_to_the_exact_target_tokenizer(report):
    manifest = json.loads(
        (INSTALL / "training/fused-model/active.json").read_text()
    )
    model = Path(manifest["active_model_path"]).resolve(strict=True)
    identity = report["target_checkpoint_identity"]
    assert identity == {
        "path": str(model),
        "tokenizer_sha256": portability._file_digest(model / "tokenizer.json"),
    }


def test_identical_tokenizers_make_everything_portable():
    """A checkpoint compared against itself must show no drift at all."""
    manifest = INSTALL / "training/fused-model/active.json"
    if not manifest.exists():
        pytest.skip("no active model manifest")
    from transformers import AutoTokenizer

    active = json.loads(manifest.read_text())["active_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(str(active))
    built = portability.build({"legacy": tokenizer, "target": tokenizer})
    assert built["must_regenerate"] == []
    assert built["changed_opcode_markers"] == {}


def test_the_inventory_reads_the_measurement_rather_than_assuming(report):
    """The inventory must not report a measured-portable contract as bound."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "inventory_27b", portability.REPO_ROOT / "tools/inventory_27b_migration.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["inventory_27b"] = module
    spec.loader.exec_module(module)

    verdicts = module._grounding_verdicts()
    # An empty mapping means the probe failed silently, which is the defect
    # that made the inventory call a portable contract bound.
    assert verdicts, "grounding verdicts could not be measured"
    assert (
        verdicts["core/learning/recurrent_literal_grounding.py"] == "portable"
    )
    assert (
        verdicts["core/learning/recurrent_opcode_grounding.py"] == "must_regenerate"
    )


def test_an_unmeasured_contract_is_never_called_portable(monkeypatch):
    """Fail closed: no measurement means bound, not fine.

    The probe failing silently is exactly how the inventory first reported a
    portable contract as dead, so the fallback is asserted rather than trusted.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "inventory_27b_b", portability.REPO_ROOT / "tools/inventory_27b_migration.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["inventory_27b_b"] = module
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_grounding_verdicts", dict)
    artifacts = module.collect(
        {"num_hidden_layers": 64, "full_attention_interval": 4, "vocab_size": 248320},
        {"num_hidden_layers": 64, "hidden_size": 5120, "vocab_size": 152064},
    )
    grounding = [a for a in artifacts if a.kind == "grounding_contract"]
    assert grounding, "no grounding contracts were classified"
    for artifact in grounding:
        assert artifact.verdict == "token_id_bound"
        assert artifact.detail["measured"] is False
        assert artifact.needs_fresh_training is True


def test_a_measured_portable_contract_is_not_marked_for_retraining(monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "inventory_27b_c", portability.REPO_ROOT / "tools/inventory_27b_migration.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["inventory_27b_c"] = module
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "_grounding_verdicts",
        lambda: {
            "core/learning/recurrent_literal_grounding.py": "portable",
            "core/learning/recurrent_opcode_grounding.py": "must_regenerate",
            "core/learning/recurrent_answer_emission.py": "must_regenerate",
        },
    )
    artifacts = module.collect(
        {"num_hidden_layers": 64, "full_attention_interval": 4, "vocab_size": 248320},
        {"num_hidden_layers": 64, "hidden_size": 5120, "vocab_size": 152064},
    )
    by_path = {a.path: a for a in artifacts if a.kind == "grounding_contract"}
    literal = by_path["core/learning/recurrent_literal_grounding.py"]
    assert literal.verdict == "model_independent"
    assert literal.needs_fresh_training is False
    opcode = by_path["core/learning/recurrent_opcode_grounding.py"]
    assert opcode.needs_fresh_training is True
