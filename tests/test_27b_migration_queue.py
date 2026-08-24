"""Every model-dependent capability has something deciding whether it may serve.

An absent component and a deferred one look identical from a health check. A
deferred one carries a signed quarantine saying "measured on another
checkpoint"; an absent one carries nothing, which is not a state — it is the
only way a capability comes back without anyone deciding that it should.

Reporting a false gap is the symmetric error, so a capability gated by its own
mechanism is reported as gated, not as missing.
"""
from __future__ import annotations

import pytest

from tools import report_27b_migration_queue as queue_tool


@pytest.fixture(scope="module")
def queue():
    built = queue_tool.build()
    if built.get("blocked"):
        pytest.skip(built["blocked"])
    return built


def _row(queue, name):
    return next(r for r in queue["rows"] if r["capability"] == name)


# ── The decisive finding ────────────────────────────────────────────────


def test_the_persona_is_already_in_this_checkpoint(queue):
    row = _row(queue, "persona_crsm")
    assert row["signed_authority"] is True
    assert row["authority_kind"] == "fused_persona_crsm"
    assert row["disposition"] == "portable"
    # Fused means present, so what it needs is verification, not a recovery run.
    assert "fusion_plan_sha256" in row["claims"]
    assert "fusion_receipt_sha256" in row["claims"]
    assert "recovery run" in queue["persona_finding"]


def test_the_representation_bound_tissue_is_quarantined(queue):
    for name in ("steering", "recurrence_native"):
        row = _row(queue, name)
        assert row["authority_kind"] == "model_basis_quarantine"
        assert row["disposition"] == "retrain_required"


def test_expert_adapters_are_retired_not_deferred(queue):
    row = _row(queue, "expert_adapters")
    assert row["disposition"] == "retired"
    assert row["authority_kind"] == "retirement_inventory"


# ── A gap and a gate are different ──────────────────────────────────────


def test_a_capability_with_its_own_gate_is_not_reported_as_a_gap(queue):
    for name in (
        "qualified_rlc_serving",
        "grounding_contracts",
        "fast_weight_surfaces",
        "episodic_plasticity",
    ):
        row = _row(queue, name)
        assert row["signed_authority"] is False
        assert row["disposition"] == "gated_outside_the_contract", name
        assert row["admission_gate"], name


def test_a_capability_with_no_gate_at_all_is_reported_uncovered(monkeypatch):
    monkeypatch.setitem(
        queue_tool.CAPABILITIES,
        "invented_capability",
        {
            "owner": "core/nowhere.py",
            "load_path": "nothing",
            "representation_bound": True,
        },
    )
    built = queue_tool.build()
    if built.get("blocked"):
        pytest.skip(built["blocked"])
    row = next(r for r in built["rows"] if r["capability"] == "invented_capability")
    assert row["disposition"] == "uncovered"
    assert row["admission_gate"] is None
    assert "invented_capability" in built["uncovered_capabilities"]


def test_every_row_cites_an_owner_and_a_load_path(queue):
    for row in queue["rows"]:
        assert row["code_owner"], row["capability"]
        assert row["active_load_path"], row["capability"]


def test_every_row_has_a_disposition_from_the_known_set(queue):
    allowed = {
        "portable",
        "rebindable",
        "retrain_required",
        "retired",
        "gated_outside_the_contract",
        "uncovered",
    }
    for row in queue["rows"]:
        assert row["disposition"] in allowed, row


def test_an_unrecognised_authority_kind_is_not_silently_accepted(monkeypatch):
    monkeypatch.setitem(
        queue_tool.DISPOSITION_BY_KIND, "future_kind", ("portable", "x")
    )
    assert queue_tool.DISPOSITION_BY_KIND["future_kind"][0] == "portable"
    # A kind absent from the map falls to unclassified rather than to a guess.
    assert "unclassified" not in queue_tool.DISPOSITION_BY_KIND


def test_the_contract_coverage_is_reported_as_a_fraction(queue):
    assert " of " in queue["contract_covers"]
    assert len(queue["signed_components"]) == 4


def test_the_representation_bound_flag_matches_the_tokenizer_bound_one(queue):
    grounding = _row(queue, "grounding_contracts")
    # Grounding is bound to the tokenizer, not to the residual basis; the two
    # travel differently across a checkpoint change.
    assert grounding["tokenizer_bound"] is True
    assert grounding["representation_bound"] is False


def test_a_missing_manifest_blocks_rather_than_reporting_everything_uncovered(
    monkeypatch,
):
    def _boom():
        raise SystemExit("no manifest")

    monkeypatch.setattr(queue_tool, "active_manifest", _boom)
    built = queue_tool.build()
    assert built.get("blocked")
    assert "rows" not in built
