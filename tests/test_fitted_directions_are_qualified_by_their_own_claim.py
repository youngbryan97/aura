"""Directions fitted to her replies are qualified by their own claim, and installed alone.

The September generation was qualified by whether its directions made her use
more affect words than a prompt asking for them. Directions fitted to keep her
voice claim something narrower: steered by the state a reply was said in, the
reply is more likely than steered by another of her states (prereg addendum,
23 October, midday). These tests carry a record of that claim through the
replay, the evaluation, the signed authority and the active pointer, and check
that each refuses what it should.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.evaluation.caa_causal_evaluation import canonical_sha256
from core.evaluation.fitted_steering_evaluation import (
    ARMS,
    LESIONS,
    MATCHED_CONTROL,
    RESULT_SCHEMA,
    TREATMENT,
    PAIRED_PROTOCOL,
    FittedEvaluationError,
    build_causal_evaluation,
    build_independent_verifier_evidence,
    derangement,
    fitted_generation_metadata,
    paired_qualification_plan,
    replay,
    sign_test_p,
)
from core.learning.cortex_migration_authority import CortexMigrationAuthorityError, validate_component_authority
from core.learning.cortex_migration_authority_issuer import issue_steering_authority

HERS = "a" * 64
PLAN = {"model_path": "/models/hers", "hidden_size": 8,
        "target_layers": [{"index": 3, "kind": "decoder"}, {"index": 7, "kind": "decoder"}]}


@pytest.fixture(autouse=True)
def _isolated_state_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))


def _generation(root: Path, descriptor: str = HERS) -> dict:
    root.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for layer in (3, 7):
        for key in ("valence_positive", "arousal"):
            v = rng.normal(size=8).astype(np.float32)
            np.savez(root / f"{key}_layer{layer}.npz", v=v / np.linalg.norm(v), model_descriptor_sha256=descriptor,
                     reference=0.1, spread=0.2, push=0.01, source="fitted_to_her_replies", extracted=True)
    (root / "held_out_replies.jsonl").write_text('{"text": "a reply", "s": [0.1]}\n')
    (root / "basis.json").write_text(json.dumps({"model_descriptor_sha256": descriptor, "fit_rounds": 1}))
    metadata = fitted_generation_metadata(root, descriptor=descriptor, plan=PLAN)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _record(metadata: dict, *, wins: dict[str, int], replies: int = 40, voice: float = 0.1) -> dict:
    """A record in which each arm's own state beats another state on `wins[arm]` replies."""
    rows = []
    for index in range(replies):
        loglik = {"plain": -100.0}
        for arm in ARMS:
            if arm == LESIONS[0]:
                loglik[f"{arm}:own"] = loglik[f"{arm}:other"] = -100.0
                continue
            own_wins = index < wins[arm]
            loglik[f"{arm}:own"] = -100.0 + voice * 10 + (0.5 if own_wins else -0.5)
            loglik[f"{arm}:other"] = -100.0 + voice * 10
        rows.append({"index": index, "tokens": 10, "loglik": loglik})
    return {"schema": RESULT_SCHEMA, "model_descriptor_sha256": HERS,
            "generation_sha256": metadata["generation_sha256"],
            "held_out_replies_sha256": metadata["extraction_contract"]["held_out_replies_sha256"],
            "alpha": 0.5, "seed": 1, "replies": rows}


WINS = {TREATMENT: 30, MATCHED_CONTROL: 21, LESIONS[1]: 18}


def test_a_derangement_moves_every_reply_and_is_fixed_by_its_seed():
    order = derangement(110, 20260908)
    assert sorted(order) == list(range(110)) and all(i != j for i, j in enumerate(order))
    assert order == derangement(110, 20260908)


def test_the_sign_test_is_the_binomial_tail():
    assert sign_test_p(10, 10) == pytest.approx(1 / 1024)
    assert sign_test_p(0, 10) == pytest.approx(1.0)


def test_the_claim_holds_only_when_every_condition_does(tmp_path):
    metadata = _generation(tmp_path / "gen")
    assert replay(_record(metadata, wins=WINS))["causal_effect_positive"] is True
    as_well = replay(_record(metadata, wins={**WINS, MATCHED_CONTROL: 30}))
    assert "random_directions_carry_the_state_as_well" in as_well["unmet_requirements"]
    lesion = replay(_record(metadata, wins={**WINS, LESIONS[1]: 31}))
    assert "a_lesion_keeps_the_effect" in lesion["unmet_requirements"]
    weak = replay(_record(metadata, wins={TREATMENT: 23, MATCHED_CONTROL: 10, LESIONS[1]: 10}))
    assert "own_state_not_more_likely_than_another_by_sign_test" in weak["unmet_requirements"]
    costly = replay(_record(metadata, wins=WINS, voice=-0.1))
    assert costly["no_regression"] is False and costly["causal_effect_positive"] is False


@pytest.mark.parametrize("paired", [False, True])
def test_a_qualified_record_is_signed_into_a_steering_authority(tmp_path, paired):
    root = tmp_path / "gen"
    metadata = _generation(root)
    result = _record(metadata, wins=WINS)
    if paired:
        result["qualification_protocol"] = PAIRED_PROTOCOL
        result["analysis_plan"] = paired_qualification_plan(result)
        result["analysis_plan_sha256"] = canonical_sha256(result["analysis_plan"])
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    evidence = build_independent_verifier_evidence(
        result=result, result_sha256="b" * 64, metadata=metadata, metadata_sha256="c" * 64, generation_dir=root)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    evaluation = build_causal_evaluation(result=result, metadata=metadata, verifier_evidence=evidence,
                                         verifier_evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest())
    if paired:
        assert evaluation["qualification_protocol"] == PAIRED_PROTOCOL
        assert evaluation["analysis_plan_sha256"] == result["analysis_plan_sha256"]
        assert all(row["rejects"] for row in evaluation["paired_controls"].values())
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation))
    authority = issue_steering_authority(metadata_path=root / "metadata.json", causal_evaluation_path=evaluation_path,
                                         independent_evidence_path=evidence_path, descriptor_sha256=HERS,
                                         custody_base=tmp_path / "issued", issued_at=2.0)
    assert validate_component_authority(authority, component="steering", descriptor_sha256=HERS) == authority
    assert authority["claims"]["generation_sha256"] == metadata["generation_sha256"]


def test_a_record_on_other_replies_or_an_edited_count_is_refused(tmp_path):
    root = tmp_path / "gen"
    metadata = _generation(root)
    result = _record(metadata, wins=WINS)
    with pytest.raises(FittedEvaluationError, match="other_replies"):
        build_independent_verifier_evidence(
            result={**result, "held_out_replies_sha256": "d" * 64}, result_sha256="b" * 64,
            metadata=metadata, metadata_sha256="c" * 64, generation_dir=root)
    evidence = build_independent_verifier_evidence(
        result=result, result_sha256="b" * 64, metadata=metadata, metadata_sha256="c" * 64, generation_dir=root)
    edited = json.loads(json.dumps(evidence))
    edited["replay"]["matched_control_successes"] = 0
    edited["verification_sha256"] = canonical_sha256({k: v for k, v in edited.items() if k != "verification_sha256"})
    with pytest.raises(FittedEvaluationError, match="verifier_invalid"):
        build_causal_evaluation(result=result, metadata=metadata, verifier_evidence=edited,
                                verifier_evidence_sha256="e" * 64)


@pytest.mark.parametrize("tamper", ["paired_tail", "verification_status", "verification_digest"])
def test_authority_rejects_invalid_paired_verification_evidence(tmp_path, tamper):
    root = tmp_path / "gen"
    metadata = _generation(root)
    result = _record(metadata, wins=WINS)
    result["qualification_protocol"] = PAIRED_PROTOCOL
    result["analysis_plan"] = paired_qualification_plan(result)
    result["analysis_plan_sha256"] = canonical_sha256(result["analysis_plan"])
    evidence = build_independent_verifier_evidence(
        result=result, result_sha256="b" * 64, metadata=metadata, metadata_sha256="c" * 64, generation_dir=root)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    evaluation = build_causal_evaluation(
        result=result, metadata=metadata, verifier_evidence=evidence,
        verifier_evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest())
    # Keep all marginals and both summaries consistent, but leave the claimed
    # small tail on discordances whose actual corrected tail does not pass.
    if tamper == "paired_tail":
        for contrasts in (evaluation["paired_controls"], evidence["replay"]["paired_controls"]):
            contrasts[MATCHED_CONTROL].update(treatment_only=16, control_only=7, both=14, neither=3)
    elif tamper == "verification_status":
        evidence["verified"] = False
    evidence["verification_sha256"] = canonical_sha256({k: v for k, v in evidence.items() if k != "verification_sha256"})
    if tamper == "verification_digest":
        evidence["verification_sha256"] = "f" * 64
    evidence_path.write_text(json.dumps(evidence))
    evaluation["independent_verifier"]["evidence_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    evaluation["evaluation_sha256"] = canonical_sha256({k: v for k, v in evaluation.items() if k != "evaluation_sha256"})
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation))
    with pytest.raises(CortexMigrationAuthorityError, match="paired_control_evidence_invalid"):
        issue_steering_authority(metadata_path=root / "metadata.json", causal_evaluation_path=evaluation_path,
                                 independent_evidence_path=evidence_path, descriptor_sha256=HERS,
                                 custody_base=tmp_path / "issued", issued_at=2.0)


def test_a_new_steering_generation_replaces_only_the_steering_component(tmp_path, monkeypatch):
    from core.learning import serving_requalification
    from core.learning.cortex_generation_upgrade import activate_upgrade, rollback_upgrade, stage_upgrade
    from core.learning.steering_requalification import apply_steering_requalification
    from tests.test_cortex_generation_upgrade import _fused_dir, _upgrade_contracts

    fused, candidate = _fused_dir(tmp_path)
    descriptor, evaluation, serving, migration = _upgrade_contracts(candidate)
    stage_upgrade(candidate_model_path=candidate, base_model_path="Qwen3-32B", tag="qwen3-gen",
                  fused_model_dir=fused, evaluation=evaluation, serving_profile=serving,
                  migration_contract=migration)
    activate_upgrade(fused_model_dir=fused, authorized_by="bryan", evaluation=evaluation)
    before_bytes = (fused / "active.json").read_bytes()
    before = json.loads(before_bytes)
    digest = descriptor["descriptor_sha256"]

    root = tmp_path / "gen"
    metadata = _generation(root, descriptor=digest)
    result = {**_record(metadata, wins=WINS), "model_descriptor_sha256": digest}
    evidence = build_independent_verifier_evidence(
        result=result, result_sha256="b" * 64, metadata=metadata, metadata_sha256="c" * 64, generation_dir=root)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(build_causal_evaluation(
        result=result, metadata=metadata, verifier_evidence=evidence,
        verifier_evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest())))
    authority = issue_steering_authority(metadata_path=root / "metadata.json", causal_evaluation_path=evaluation_path,
                                         independent_evidence_path=evidence_path, descriptor_sha256=digest,
                                         custody_base=Path(os.environ["AURA_STATE_ROOT"]) / "issued")

    with pytest.raises(PermissionError):
        apply_steering_requalification(authority=authority, authorized_by="", fused_model_dir=fused)
    assert (fused / "active.json").read_bytes() == before_bytes

    surfaces = iter([{"surface"}, set()])
    monkeypatch.setattr(serving_requalification, "_valid_serving_surfaces", lambda: next(surfaces))
    with pytest.raises(ValueError, match="switch off a qualified serving surface"):
        apply_steering_requalification(authority=authority, authorized_by="bryan", fused_model_dir=fused)
    assert (fused / "active.json").read_bytes() == before_bytes
    monkeypatch.setattr(serving_requalification, "_valid_serving_surfaces", lambda: set())

    receipt = apply_steering_requalification(authority=authority, authorized_by="bryan", fused_model_dir=fused)
    after = json.loads((fused / "active.json").read_text())
    assert receipt["changed"] and receipt["vectors"] == 4 and receipt["effective"] == "next_boot"
    assert after["migration_contract"]["components"]["steering"] == authority
    for name in ("persona_crsm", "expert_adapters", "recurrence_native"):
        assert after["migration_contract"]["components"][name] == before["migration_contract"]["components"][name]
    assert {k: v for k, v in after.items() if k != "migration_contract"} == {
        k: v for k, v in before.items() if k != "migration_contract"}
    assert apply_steering_requalification(authority=authority, authorized_by="bryan",
                                          fused_model_dir=fused)["changed"] is False
    assert rollback_upgrade(fused_model_dir=fused)["byte_exact"] is True
    assert (fused / "active.json").read_bytes() == before_bytes
