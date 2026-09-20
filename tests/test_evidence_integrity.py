"""tests/test_evidence_integrity.py — a claim may not outrank its evidence.

Claim 14 (AGI-Candidate) was classified `locally demonstrated` while the same
table cell explained that its primary evidence was an unfair comparison. Both
statements were committed, both were true about the repository, and the
classification is the one that travelled. Nothing checked, because the
retraction was prose: `BASELINES.json` still read `"status": "RUN"` with clean
pass rates, so every automated reader saw a passing artifact.

These tests pin the two properties that make that impossible to repeat, and the
last one reconstructs the original state and asserts the gate rejects it.
"""

from __future__ import annotations

import json

import pytest

from tools.check_evidence_integrity import (
    ASSERTING_FLOOR,
    CLASSIFICATION_RANK,
    RETRACTION_SCHEMA,
    IntegrityFailure,
    ROOT,
    _covers,
    check,
    cited_artifacts,
    load_retraction,
    parse_claims,
    retracted_paths,
)

RETRACTION_SCHEMA = "aura.evidence_retraction.v1"


def _write_matrix(root, rows: str) -> None:
    (root / "CLAIMS_MATRIX.md").write_text(
        "| Claim | Classification | Evidence |\n| :--- | :--- | :--- |\n" + rows,
        encoding="utf-8",
    )


def _write_retraction(root, relative: str, **overrides) -> None:
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": RETRACTION_SCHEMA,
        "status": "retracted",
        "reason": "the baselines ran at 160 tokens against an unbounded treatment",
    }
    payload.update(overrides)
    (directory / "RETRACTION.json").write_text(json.dumps(payload), encoding="utf-8")


def test_the_live_matrix_has_no_claim_outranking_its_evidence():
    """The gate's actual job, run against the actual repository."""
    report = check(ROOT)
    assert report["passed"], report["violations"]
    assert report["claims_checked"] > 0


def test_a_matrix_the_gate_cannot_parse_fails_rather_than_passes(tmp_path):
    """A table-shape change must not silently disable the gate.

    A checker that parses zero rows and reports success is worse than no
    checker: it is a green light with nothing behind it, and it looks exactly
    like a green light with everything behind it.
    """
    (tmp_path / "CLAIMS_MATRIX.md").write_text(
        "# Aura Claims Matrix\n\nThe table moved somewhere else.\n", encoding="utf-8"
    )

    with pytest.raises(IntegrityFailure, match="no claim rows parsed"):
        check(tmp_path)


def test_a_missing_matrix_fails_rather_than_passes(tmp_path):
    """Deleting the file must not be a way to pass the gate."""
    with pytest.raises(IntegrityFailure, match="missing"):
        check(tmp_path)


def test_an_asserting_claim_citing_retracted_evidence_is_rejected(tmp_path):
    """The original defect, reconstructed.

    This is claim 14 exactly as it was committed: `locally demonstrated`,
    citing a bundle whose comparison had been withdrawn.
    """
    _write_retraction(tmp_path, "artifacts/current/agi_live")
    _write_matrix(
        tmp_path,
        "| **14. AGI-Candidate** | `locally demonstrated` | "
        "`artifacts/current/agi_live/`, receipt coverage, ablations, baselines |\n",
    )

    report = check(tmp_path)

    assert report["passed"] is False
    (violation,) = report["violations"]
    assert violation["problem"] == "asserting_claim_cites_retracted_evidence"
    assert violation["claim"] == 14
    assert violation["artifact"] == "artifacts/current/agi_live"


def test_a_not_proven_claim_may_cite_retracted_evidence(tmp_path):
    """Demotion is the fix, so the demoted row must be allowed to explain itself.

    A row that says "this is not proven, and here is the retracted bundle that
    is why" is the gate working, not a violation.
    """
    _write_retraction(tmp_path, "artifacts/current/agi_live")
    _write_matrix(
        tmp_path,
        "| **14. AGI-Candidate** | `not proven` | Retracted: "
        "`artifacts/current/agi_live/` — the baselines were handicapped |\n",
    )

    assert check(tmp_path)["passed"] is True


def test_citing_a_parent_directory_still_reaches_the_retraction(tmp_path):
    """`artifacts/current/` cites everything under it, including what is withdrawn.

    Claim 14 cited `artifacts/current/` bundles collectively. A checker that
    only matched exact paths would have let the broad citation through — which
    is the more common way a claim leans on evidence it never names directly.
    """
    _write_retraction(tmp_path, "artifacts/current/agi_live")
    _write_matrix(
        tmp_path,
        "| **14. AGI-Candidate** | `locally demonstrated` | `artifacts/current/` bundles |\n",
    )

    report = check(tmp_path)
    assert report["passed"] is False
    assert report["violations"][0]["artifact"] == "artifacts/current/agi_live"


def test_an_unreadable_retraction_counts_as_retracted(tmp_path):
    """Fail closed. An unreadable validity statement is not a clean bill of health."""
    directory = tmp_path / "artifacts" / "current" / "broken"
    directory.mkdir(parents=True)
    (directory / "RETRACTION.json").write_text("{ not json", encoding="utf-8")

    record = load_retraction(directory)
    assert record is not None
    assert record["status"] == "retracted"
    assert record["malformed"] is True


def test_a_sidecar_without_the_schema_counts_as_retracted(tmp_path):
    """The same argument: an unrecognised statement is not an absent one."""
    directory = tmp_path / "artifacts" / "current" / "unschemad"
    directory.mkdir(parents=True)
    (directory / "RETRACTION.json").write_text(
        json.dumps({"status": "retracted"}), encoding="utf-8"
    )

    record = load_retraction(directory)
    assert record is not None
    assert record["malformed"] is True


def test_a_lifted_retraction_restores_the_evidence(tmp_path):
    """Retraction must be reversible, or nobody will ever record one.

    A replacement run that meets the requirements should be able to clear the
    flag by setting status, not by deleting the history of why it was set.
    """
    _write_retraction(tmp_path, "artifacts/current/agi_live", status="superseded")
    _write_matrix(
        tmp_path,
        "| **14. AGI-Candidate** | `locally demonstrated` | `artifacts/current/agi_live/` |\n",
    )

    assert check(tmp_path)["passed"] is True


def test_an_unrecognised_classification_is_reported_not_ignored(tmp_path):
    """A label the gate cannot rank is a label the gate cannot check.

    Silently skipping it is how a claim escapes by inventing a new word for
    how true it is.
    """
    _write_matrix(
        tmp_path,
        "| **99. Something** | `basically proven` | `artifacts/current/whatever/` |\n",
    )

    report = check(tmp_path)
    assert report["passed"] is False
    assert report["violations"][0]["problem"] == "unknown_classification"


def test_classification_ranks_put_every_asserting_label_above_the_floor():
    """The ranking is the gate's whole judgement; state it explicitly."""
    for label in ("locally demonstrated", "causally demonstrated", "externally validated"):
        assert CLASSIFICATION_RANK[label] >= ASSERTING_FLOOR
    for label in ("not proven", "deprecated", "retired", "blocked"):
        assert CLASSIFICATION_RANK[label] < ASSERTING_FLOOR


def test_evidence_paths_are_extracted_from_prose():
    """Evidence cells are prose; the paths inside them are what matters."""
    found = cited_artifacts(
        "`artifacts/current/agi_live/`, `artifacts/current/external_live_validation/`, "
        "receipt coverage and Aletheia Tier 5 evidence"
    )
    assert found == [
        "artifacts/current/agi_live",
        "artifacts/current/external_live_validation",
    ]


def test_the_committed_agi_live_bundle_is_actually_marked_retracted():
    """The retraction must be machine-readable, which was the original gap.

    The audit existed in prose for a month while BASELINES.json read
    "status": "RUN". Prose is not a marker anything can act on.
    """
    record = load_retraction(ROOT / "artifacts" / "current" / "agi_live")
    assert record is not None, (
        "artifacts/current/agi_live has no RETRACTION.json — its baseline "
        "comparison is withdrawn and nothing machine-readable says so"
    )
    assert record["status"] == "retracted"
    assert record.get("replacement_requirements"), (
        "a retraction that does not say what would replace it is a dead end"
    )


def test_every_retraction_is_tracked_beside_the_evidence_it_retracts():
    """A retraction that gitignore drops is a retraction that does not exist.

    `artifacts/current/` is gitignored while the bundles inside it are tracked
    from before the ignore was added. So RETRACTION.json was silently skipped
    by `git add -A`, and on a fresh clone the evidence would arrive without the
    statement withdrawing it — the gate would see a clean bundle and pass. The
    asymmetry is the whole bug: whatever is visible to a reviewer must carry
    its own retraction.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "ls-files", "artifacts"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    tracked_files = set(tracked.stdout.split())

    for retraction in (ROOT / "artifacts").rglob("RETRACTION.json"):
        relative = str(retraction.relative_to(ROOT))
        directory = retraction.parent
        siblings = {
            str(path.relative_to(ROOT))
            for path in directory.iterdir()
            if path.is_file() and path.name != "RETRACTION.json"
        }
        if not siblings & tracked_files:
            continue  # nothing tracked here, so nothing to be inconsistent with
        assert relative in tracked_files, (
            f"{relative} is untracked while the evidence it retracts is tracked. "
            "A fresh clone would get the bundle without the withdrawal. "
            "Force-add it: gitignore does not get a vote on retractions."
        )


def test_the_retracted_measurements_were_left_intact():
    """Never edit the measurement. The record of what was measured is a fact.

    Rewriting BASELINES.json to look correct would also erase the evidence
    that the handicap was ever there.
    """
    baselines = json.loads(
        (ROOT / "artifacts" / "current" / "agi_live" / "BASELINES.json").read_text(
            encoding="utf-8"
        )
    )
    assert baselines["raw_llm"]["pass_rate"] == 0.1667
    assert baselines["llm_with_tools"]["pass_rate"] == 0.1667
    assert baselines["react_agent"]["pass_rate"] == 0.1667


def test_claim_fourteen_is_not_asserting_in_the_live_matrix():
    """The specific demotion this work exists to make, pinned.

    Not a style preference: the claim's own evidence bundle is withdrawn, so
    any classification above `not proven` is an assertion with nothing behind
    it.
    """
    claims = {
        claim["number"]: claim
        for claim in parse_claims((ROOT / "CLAIMS_MATRIX.md").read_text(encoding="utf-8"))
    }
    assert 14 in claims, "claim 14 vanished from the matrix rather than being demoted"
    assert CLASSIFICATION_RANK[claims[14]["classification"]] < ASSERTING_FLOOR, (
        f"claim 14 is classified {claims[14]['classification']!r} while its primary "
        "evidence carries a retraction"
    )


# ---------------------------------------------------------------------------
# Per-file retraction
#
# `retracted_files` had been in the schema since the agi_live withdrawal and
# nothing read it, so a retraction was all-or-nothing on a directory. That is
# unusable for `artifacts/proof_bundle/latest`, which holds nineteen unrelated
# artifacts and needs exactly one of them withdrawn.
# ---------------------------------------------------------------------------


def test_named_files_narrow_a_retraction_to_those_files():
    record = {"schema": RETRACTION_SCHEMA, "status": "retracted",
              "retracted_files": ["UNDENIABLE_RSI.json"]}
    withdrawn = retracted_paths("artifacts/proof_bundle/latest", record)
    assert withdrawn == ["artifacts/proof_bundle/latest/UNDENIABLE_RSI.json"]
    assert _covers("artifacts/proof_bundle/latest/UNDENIABLE_RSI.json", withdrawn[0])
    assert not _covers("artifacts/proof_bundle/latest/BOOT_HEALTH.json", withdrawn[0])


def test_an_empty_file_list_still_retracts_the_whole_directory():
    """The agi_live behaviour, and the fail-closed default."""
    for record in (
        {"schema": RETRACTION_SCHEMA, "status": "retracted"},
        {"schema": RETRACTION_SCHEMA, "status": "retracted", "retracted_files": []},
        {"schema": RETRACTION_SCHEMA, "status": "retracted", "retracted_files": "oops"},
    ):
        assert retracted_paths("artifacts/current/agi_live", record) == [
            "artifacts/current/agi_live"
        ]


def test_citing_a_bundle_still_reaches_a_retracted_file_inside_it():
    """A claim resting on the bundle rests on the withdrawn part of it."""
    withdrawn = retracted_paths(
        "artifacts/proof_bundle/latest",
        {"schema": RETRACTION_SCHEMA, "status": "retracted",
         "retracted_files": ["UNDENIABLE_RSI.json"]},
    )
    assert _covers("artifacts/proof_bundle/latest", withdrawn[0])


def test_the_rsi_bundle_carries_its_withdrawal():
    path = ROOT / "artifacts" / "proof_bundle" / "latest" / "RETRACTION.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema"] == RETRACTION_SCHEMA
    assert record["status"] == "retracted"
    assert record["retracted_files"] == ["UNDENIABLE_RSI.retracted.json"]
    assert record["current_verdict"] == "BOUNDED_SELF_OPTIMIZATION"


def test_the_retracted_rsi_measurement_was_left_intact():
    """The published improver curve stays on the record, exactly as it was.

    Regenerating the bundle overwrites UNDENIABLE_RSI.json, so the withdrawn
    one is preserved beside it. Rewriting the measurement to look correct
    would also erase the evidence that the counter was ever in the formula.
    """
    bundle = json.loads(
        (
            ROOT / "artifacts" / "proof_bundle" / "latest" / "UNDENIABLE_RSI.retracted.json"
        ).read_text(encoding="utf-8")
    )
    assert bundle["lineage_verdict"]["improver_curve"] == [0.578, 0.8696, 0.9536, 1.0]
    assert bundle["lineage_verdict"]["verdict"] == "UNDENIABLE_RSI"


def test_the_regenerated_bundle_is_honest_about_what_it_shows():
    """The live bundle carries the measured verdict, not the authored one."""
    bundle = json.loads(
        (ROOT / "artifacts" / "proof_bundle" / "latest" / "UNDENIABLE_RSI.json").read_text(
            encoding="utf-8"
        )
    )
    assert bundle["lineage_verdict"]["verdict"] == "BOUNDED_SELF_OPTIMIZATION"
    assert bundle["l3_rsi_claim"] is False
    assert bundle["status"] == "not_l3_evidence"
    assert set(bundle["improver_provenance"]) == {"measured"}
    assert bundle["improver_order_invariance_violation"] == ""
    # Capability is the custodian's hidden-eval score, unblended.
    assert bundle["lineage_verdict"]["capability_curve"][0] == 0.5
    assert bundle["lineage_verdict"]["capability_curve"][-1] == 1.0
    # A run that booted no model may not report one.
    assert bundle["runtime"]["started_runtime"] is False
    assert bundle["runtime"]["llm_codegen_enabled"] is False
