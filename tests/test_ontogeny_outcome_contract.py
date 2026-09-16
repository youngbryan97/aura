"""Changed outcome semantics cannot train or restore the previous policy."""

from dataclasses import replace

import pytest

from core.ontogeny.authority import AuthorityLedger, AuthorityStage, compare
from core.ontogeny.experience import Episode, ExperienceSpine, Outcome, OutcomeKind, Provenance
from core.ontogeny.features import EXECUTIVE_ADMISSION, FeatureSchema
from core.ontogeny.heads import PredictionHead
from core.ontogeny.calibration import CalibrationMonitor
from core.ontogeny.service import ControlPoint, OntogenyCore
from core.ontogeny.trainer import Trainer, design_width


def test_outcome_contract_changes_evidence_identity_without_changing_geometry():
    old = FeatureSchema("cp", ("signal",))
    current = replace(old, outcome_contract="direct-consequence.v2")
    assert old.schema_id != current.schema_id
    assert old.width == current.width
    assert old.names == current.names
    assert current.describe()["outcome_contract"] == "direct-consequence.v2"
    assert EXECUTIVE_ADMISSION.outcome_contract


def test_episode_deduplication_keeps_different_contracts_separate():
    old = Episode(control_point="cp", features={"x": 1.0}, decision="a", feature_schema="old")
    current = replace(old, feature_schema="current")
    assert old.dedup_key != current.dedup_key
    assert replace(current).dedup_key == current.dedup_key


def test_revoked_evidence_cannot_retrain_under_a_new_outcome_contract(tmp_path):
    old = FeatureSchema("cp", ("signal",))
    current = replace(old, outcome_contract="direct-consequence.v2")
    spine = ExperienceSpine(tmp_path / "experience.db", autoflush=False)
    try:
        for index in range(160):
            episode = Episode(
                control_point="cp", features={"signal": float(index)},
                decision="approved", feature_schema=old.schema_id,
                provenance=Provenance.TEST, decided_at=float(index),
            )
            spine.record(episode)
            spine.resolve(episode.episode_id, Outcome(
                kind=OutcomeKind.SUCCESS, utility=1.0, resolver="old-grader",
            ))
        spine.flush()
        monitor = CalibrationMonitor()
        authority = AuthorityLedger(tmp_path / "authority.json", calibration=monitor)
        trainer = Trainer(spine, authority, monitor, units=4, seed=7)
        heads = {"approved": PredictionHead(
            "cp.approved", ("failure", "success"), design_width(current, 4),
        )}
        result = trainer.train("cp", current, heads, ("approved",))
        assert not result.fitted
        assert result.samples == 0
        assert len(spine.episodes("cp", feature_schema=old.schema_id)) == 160
    finally:
        spine.close()


def test_schema_binding_retires_grant_once_and_survives_reload(tmp_path):
    path = tmp_path / "authority.json"
    ledger = AuthorityLedger(path)
    ledger.bind_evidence_contract("cp", "old")
    ledger.set_stage("cp", AuthorityStage.AUTHORITY, reason="old evidence")
    ledger.bind_evidence_contract("cp", "new")
    assert ledger.stage("cp") is AuthorityStage.OBSERVE
    assert ledger.grant_of("cp").evidence["previous_contract"] == "old"
    ledger.set_stage("cp", AuthorityStage.AUTHORITY, reason="new evidence")
    restored = AuthorityLedger(path)
    restored.bind_evidence_contract("cp", "new")
    assert restored.stage("cp") is AuthorityStage.AUTHORITY


def test_registering_changed_contract_does_not_restore_old_heads(tmp_path):
    old = FeatureSchema("cp", ("signal",))
    current = replace(old, outcome_contract="direct-consequence.v2")
    db = tmp_path / "experience.db"
    authority_path = tmp_path / "authority.json"
    first = OntogenyCore(
        spine=ExperienceSpine(db, autoflush=False),
        authority=AuthorityLedger(authority_path), units=4, autostart=False,
    )
    cp = first.register(ControlPoint("cp", old, ("approved", "deferred")))
    for head in cp.heads.values():
        head.version = 17
    first._save_head(cp)
    first.authority.set_stage("cp", AuthorityStage.AUTHORITY, reason="old evidence")
    first.stop()

    second = OntogenyCore(
        spine=ExperienceSpine(db, autoflush=False),
        authority=AuthorityLedger(authority_path), units=4, autostart=False,
    )
    try:
        cp = second.register(ControlPoint("cp", current, ("approved", "deferred")))
        second._load_heads()
        assert all(head.version == 0 for head in cp.heads.values())
        assert second.authority.stage("cp") is AuthorityStage.OBSERVE
    finally:
        second.stop()


def test_a_deciding_grant_cannot_outlive_its_head(tmp_path):
    ledger = AuthorityLedger(tmp_path / "authority.json")
    ledger.set_stage("cp", AuthorityStage.AUTHORITY, reason="qualified")
    result = ledger.evaluate("cp", (), head_ready=False)
    assert result["action"] == "demoted_unready"
    assert ledger.stage("cp") is AuthorityStage.OBSERVE


def test_repetitions_cannot_manufacture_independent_authority_trials():
    challenger = Episode(
        control_point="cp", features={}, decision="a", decider="ontogeny:cp@1",
        repeat_count=10_000, outcome=Outcome(OutcomeKind.SUCCESS),
    )
    incumbent = replace(
        challenger, episode_id="incumbent", decider="incumbent",
        outcome=Outcome(OutcomeKind.FAILURE),
    )
    result = compare("cp", [challenger, incumbent] * 100)
    assert result.challenger_total == result.incumbent_total == 1
    assert not result.sufficient
    assert not result.challenger_wins


def test_comparison_counts_distinct_decisions_only_for_its_control_point():
    rows = [Episode(
        control_point="cp", features={}, decision="a", decider="ontogeny:cp@1",
        outcome=Outcome(OutcomeKind.SUCCESS),
    ) for _ in range(50)]
    foreign = replace(rows[0], control_point="other", episode_id="foreign")
    result = compare("cp", [*rows, foreign])
    assert result.challenger_total == result.challenger_successes == 50
    assert result.incumbent_total == 0


def test_comparison_refuses_conflicting_outcomes_for_one_decision():
    row = Episode(
        control_point="cp", features={}, decision="a",
        outcome=Outcome(OutcomeKind.SUCCESS),
    )
    conflicting = replace(row, outcome=Outcome(OutcomeKind.FAILURE))
    with pytest.raises(ValueError, match="conflicting.*episode"):
        compare("cp", [row, conflicting])


@pytest.mark.parametrize("legacy", [False, True])
def test_restored_comparison_grant_requires_current_evidence_units(tmp_path, legacy):
    path = tmp_path / "authority.json"
    ledger = AuthorityLedger(path)
    comparison = compare("cp", []).as_dict()
    if legacy:
        del comparison["evidence_unit"]
    ledger.set_stage(
        "cp", AuthorityStage.AUTHORITY, reason="comparison",
        evidence={"comparison": comparison},
    )
    restored = AuthorityLedger(path)
    expected = AuthorityStage.ADVISORY if legacy else AuthorityStage.AUTHORITY
    assert restored.stage("cp") is expected
    assert AuthorityLedger(path).stage("cp") is expected


def test_authority_comparison_cannot_reuse_old_contract_rows(tmp_path):
    ledger = AuthorityLedger(tmp_path / "authority.json")
    ledger.bind_evidence_contract("cp", "current")
    ledger.set_stage("cp", AuthorityStage.ADVISORY, reason="current candidate")
    old = [Episode(
        control_point="cp", features={}, decision="approved",
        feature_schema="old", decider="ontogeny:cp@3",
        outcome=Outcome(kind=OutcomeKind.SUCCESS, utility=1.0, resolver="old-grader"),
    ) for _ in range(200)]
    result = ledger.evaluate("cp", old, head_ready=True)
    assert result["comparison"]["challenger"]["total"] == 0
    assert result["action"] == "hold"


def test_unchanged_unversioned_control_point_is_not_revoked(tmp_path):
    ledger = AuthorityLedger(tmp_path / "authority.json")
    ledger.set_stage("unrelated", AuthorityStage.AUTHORITY, reason="existing proof")
    core = OntogenyCore(
        spine=ExperienceSpine(tmp_path / "experience.db", autoflush=False),
        authority=ledger, units=4, autostart=False,
    )
    try:
        core.register(ControlPoint("unrelated", FeatureSchema("unrelated", ("x",)), ("a",)))
        assert ledger.stage("unrelated") is AuthorityStage.AUTHORITY
    finally:
        core.stop()


@pytest.mark.parametrize("contract", [None, 2, {}])
def test_outcome_contract_rejects_nonstring_identity(contract):
    with pytest.raises(TypeError):
        FeatureSchema("cp", ("x",), outcome_contract=contract)


def test_runtime_invariant_detects_mismatched_deciding_contract(tmp_path, monkeypatch):
    from core.ontogeny import invariants

    core = OntogenyCore(
        spine=ExperienceSpine(tmp_path / "experience.db", autoflush=False),
        authority=AuthorityLedger(tmp_path / "authority.json"), units=4, autostart=False,
    )
    monkeypatch.setattr(invariants, "_organ", lambda: core)
    try:
        name = EXECUTIVE_ADMISSION.control_point
        core.authority.set_stage(name, AuthorityStage.AUTHORITY, reason="test")
        assert list(invariants._authority_matches_outcome_contract()) == []
        core.authority.note(f"evidence_contract:{name}", "old")
        failures = list(invariants._authority_matches_outcome_contract())
        assert len(failures) == 1
        assert failures[0].subject == name
    finally:
        core.stop()
