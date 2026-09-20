"""Strict evidence-ledger and overlay tests (SCOPE-001)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from reqproof_testkit import make_registry_dict, make_requirement, write_evidence_receipt

from tools.reqproof.evidence import (
    LEDGER_SCHEMA_VERSION,
    EvidenceLedger,
    EvidenceLedgerEntry,
    EvidenceLedgerError,
    add_entry,
    load_evidence_ledger,
    resolve_evidence_target,
    verify_ledger_binding,
    write_evidence_ledger_atomic,
)
from tools.reqproof.schema import EvidenceRef, Registry
from tools.reqproof.validate import validate_registry

COMMIT = "a" * 40


def registry(*requirements: dict) -> Registry:
    return Registry.from_dict(make_registry_dict(list(requirements)))


def evidence_entry(requirement_id: str = "TEST-001") -> EvidenceLedgerEntry:
    return EvidenceLedgerEntry(
        requirement_id=requirement_id,
        acceptance_ids=("A1",),
        evidence=EvidenceRef(
            evidence_class="implementation",
            ref="artifacts/proof.json",
            sha256="b" * 64,
            commit=COMMIT,
            recorded_at="2026-07-20",
        ),
    )


class TestLedgerSchema:
    def test_empty_round_trip_is_canonical(self):
        current = EvidenceLedger.empty_for(registry(make_requirement()))
        reparsed = EvidenceLedger.from_dict(json.loads(current.to_canonical_json()))
        assert reparsed.to_canonical_json() == current.to_canonical_json()

    def test_content_edit_without_rehash_is_rejected(self):
        current = EvidenceLedger.empty_for(registry(make_requirement()))
        data = current.to_dict()
        data["entries"] = [evidence_entry().to_dict()]
        with pytest.raises(EvidenceLedgerError, match="does not match"):
            EvidenceLedger.from_dict(data)

    def test_unknown_fields_unsorted_and_duplicates_are_rejected(self):
        current = EvidenceLedger.empty_for(registry(make_requirement()))
        unknown = current.to_dict()
        unknown["percent"] = 100
        with pytest.raises(EvidenceLedgerError, match="unknown fields"):
            EvidenceLedger.from_dict(unknown)

        first = evidence_entry("BETA-001")
        second = evidence_entry("ALPHA-001")
        unsorted = EvidenceLedger(
            schema_version=LEDGER_SCHEMA_VERSION,
            registry_declaration_sha256=current.registry_declaration_sha256,
            entries=(first, second),
        ).to_dict()
        with pytest.raises(EvidenceLedgerError, match="canonically sorted"):
            EvidenceLedger.from_dict(unsorted)

        duplicate = EvidenceLedger(
            schema_version=LEDGER_SCHEMA_VERSION,
            registry_declaration_sha256=current.registry_declaration_sha256,
            entries=(first, first),
        ).to_dict()
        with pytest.raises(EvidenceLedgerError, match="duplicate"):
            EvidenceLedger.from_dict(duplicate)

    def test_malformed_requirement_id_is_rejected(self):
        current = EvidenceLedger.empty_for(registry(make_requirement()))
        data = EvidenceLedger(
            schema_version=LEDGER_SCHEMA_VERSION,
            registry_declaration_sha256=current.registry_declaration_sha256,
            entries=(evidence_entry("not-an-id"),),
        ).to_dict()
        with pytest.raises(EvidenceLedgerError, match="requirement ID pattern"):
            EvidenceLedger.from_dict(data)


class TestEvidenceLocation:
    @pytest.mark.parametrize(
        "ref",
        (
            "/tmp/proof.json",
            "../proof.json",
            "artifacts/../proof.json",
            "artifacts\\proof.json",
            "./artifacts/proof.json",
        ),
    )
    def test_unsafe_reference_is_rejected(self, tmp_path: Path, ref: str):
        with pytest.raises(EvidenceLedgerError):
            resolve_evidence_target(tmp_path, ref)

    def test_symlink_target_is_rejected(self, tmp_path: Path):
        outside = tmp_path.parent / "outside-proof.json"
        outside.write_text("outside", encoding="utf-8")
        link = tmp_path / "proof.json"
        link.symlink_to(outside)
        with pytest.raises(EvidenceLedgerError, match="symlink"):
            resolve_evidence_target(tmp_path, "proof.json")


class TestLedgerOperations:
    def test_new_evidence_supersedes_only_the_cells_it_reproves(self, tmp_path: Path):
        write_evidence_receipt(
            tmp_path,
            "old.json",
            targets=[("TEST-001", "implementation", ["A1", "A2"])],
            commit=COMMIT,
            source_ref="old-source.py",
        )
        write_evidence_receipt(
            tmp_path,
            "new.json",
            targets=[("TEST-001", "implementation", ["A1"])],
            commit="b" * 40,
            source_ref="new-source.py",
        )
        current_registry = registry(
            make_requirement(
                acceptance=["first", "second"],
                evidence_required=["implementation"],
            )
        )
        ledger = add_entry(
            EvidenceLedger.empty_for(current_registry),
            current_registry,
            requirement_id="TEST-001",
            evidence_class="implementation",
            acceptance_ids=("A1", "A2"),
            ref="old.json",
            commit=COMMIT,
            recorded_at="2026-08-05",
            root=tmp_path,
        )
        ledger = add_entry(
            ledger,
            current_registry,
            requirement_id="TEST-001",
            evidence_class="implementation",
            acceptance_ids=("A1",),
            ref="new.json",
            commit="b" * 40,
            recorded_at="2026-08-06",
            root=tmp_path,
        )

        assert [
            (entry.acceptance_ids, entry.evidence.ref) for entry in ledger.entries
        ] == [(('A1',), 'new.json'), (('A2',), 'old.json')]

    def test_globally_required_class_cannot_cover_wrong_acceptance(self, tmp_path: Path):
        write_evidence_receipt(
            tmp_path,
            "live-proof.json",
            targets=[("TEST-001", "live", ["A1"])],
            commit=COMMIT,
        )
        current_registry = registry(
            make_requirement(
                acceptance=["static", "live"],
                acceptance_evidence_required=[
                    ["implementation", "test"],
                    ["implementation", "test", "live"],
                ],
                evidence_required=["implementation", "test", "live"],
            )
        )
        with pytest.raises(EvidenceLedgerError, match="for acceptance IDs"):
            add_entry(
                EvidenceLedger.empty_for(current_registry),
                current_registry,
                requirement_id="TEST-001",
                evidence_class="live",
                acceptance_ids=("A1",),
                ref="live-proof.json",
                commit=COMMIT,
                recorded_at="2026-08-05",
                root=tmp_path,
            )

    def test_add_entry_hashes_exact_bytes_and_supplies_overlay(self, tmp_path: Path):
        write_evidence_receipt(
            tmp_path,
            "artifacts/proof.json",
            targets=[("TEST-001", "implementation", ["A1"])],
            commit=COMMIT,
        )
        current_registry = registry(
            make_requirement(
                state="complete",
                evidence_required=["implementation"],
            )
        )
        current = EvidenceLedger.empty_for(current_registry)
        updated = add_entry(
            current,
            current_registry,
            requirement_id="TEST-001",
            evidence_class="implementation",
            acceptance_ids=("A1",),
            ref="artifacts/proof.json",
            commit=COMMIT,
            recorded_at="2026-07-20",
            root=tmp_path,
        )
        assert len(updated.entries) == 1
        assert updated.entries[0].evidence.sha256 != "b" * 64
        assert validate_registry(
            current_registry,
            root=tmp_path,
            commit_exists=lambda commit: True,
            evidence_entries_by_requirement=updated.entries_by_requirement(),
        ) == []

    def test_manifested_source_change_revokes_ledger_coverage(self, tmp_path: Path):
        write_evidence_receipt(
            tmp_path,
            "proof.json",
            targets=[("TEST-001", "implementation", ["A1"])],
            commit=COMMIT,
        )
        current_registry = registry(
            make_requirement(state="complete", evidence_required=["implementation"])
        )
        ledger = add_entry(
            EvidenceLedger.empty_for(current_registry),
            current_registry,
            requirement_id="TEST-001",
            evidence_class="implementation",
            acceptance_ids=("A1",),
            ref="proof.json",
            commit=COMMIT,
            recorded_at="2026-07-20",
            root=tmp_path,
        )

        (tmp_path / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
        defects = validate_registry(
            current_registry,
            root=tmp_path,
            commit_exists=lambda commit: True,
            evidence_entries_by_requirement=ledger.entries_by_requirement(),
        )

        assert {defect.defect_class for defect in defects} == {
            "stale-evidence",
            "unproven-closure",
        }

    def test_external_ledger_rejects_unstructured_artifact(self, tmp_path: Path):
        artifact = tmp_path / "proof.json"
        artifact.write_text('{"proved": true}\n', encoding="utf-8")
        current_registry = registry(
            make_requirement(evidence_required=["implementation"])
        )

        with pytest.raises(EvidenceLedgerError, match="receipt schema"):
            add_entry(
                EvidenceLedger.empty_for(current_registry),
                current_registry,
                requirement_id="TEST-001",
                evidence_class="implementation",
                acceptance_ids=("A1",),
                ref="proof.json",
                commit=COMMIT,
                recorded_at="2026-07-20",
                root=tmp_path,
            )

    def test_every_acceptance_class_cell_is_required_for_closure(self, tmp_path: Path):
        write_evidence_receipt(
            tmp_path,
            "proof.json",
            targets=[
                ("TEST-001", "implementation", ["A1", "A2"]),
                ("TEST-001", "test", ["A1", "A2"]),
            ],
            commit=COMMIT,
        )
        current_registry = registry(
            make_requirement(
                state="complete",
                acceptance=["first", "second"],
                evidence_required=["implementation", "test"],
            )
        )
        ledger = EvidenceLedger.empty_for(current_registry)
        ledger = add_entry(
            ledger,
            current_registry,
            requirement_id="TEST-001",
            evidence_class="implementation",
            acceptance_ids=("A1", "A2"),
            ref="proof.json",
            commit=COMMIT,
            recorded_at="2026-07-20",
            root=tmp_path,
        )
        ledger = add_entry(
            ledger,
            current_registry,
            requirement_id="TEST-001",
            evidence_class="test",
            acceptance_ids=("A1",),
            ref="proof.json",
            commit=COMMIT,
            recorded_at="2026-07-20",
            root=tmp_path,
        )
        defects = validate_registry(
            current_registry,
            root=tmp_path,
            commit_exists=lambda commit: True,
            evidence_entries_by_requirement=ledger.entries_by_requirement(),
        )
        assert len(defects) == 1
        assert "test[A2]" in defects[0].detail

        ledger = add_entry(
            ledger,
            current_registry,
            requirement_id="TEST-001",
            evidence_class="test",
            acceptance_ids=("A2",),
            ref="proof.json",
            commit=COMMIT,
            recorded_at="2026-07-20",
            root=tmp_path,
        )
        assert validate_registry(
            current_registry,
            root=tmp_path,
            commit_exists=lambda commit: True,
            evidence_entries_by_requirement=ledger.entries_by_requirement(),
        ) == []

    def test_duplicate_and_unrequired_class_are_rejected(self, tmp_path: Path):
        write_evidence_receipt(
            tmp_path,
            "proof.json",
            targets=[("TEST-001", "implementation", ["A1", "A2"])],
            commit=COMMIT,
        )
        current_registry = registry(
            make_requirement(evidence_required=["implementation"])
        )
        current = EvidenceLedger.empty_for(current_registry)
        updated = add_entry(
            current,
            current_registry,
            requirement_id="TEST-001",
            evidence_class="implementation",
            acceptance_ids=("A1",),
            ref="proof.json",
            commit=COMMIT,
            recorded_at="2026-07-20",
            root=tmp_path,
        )
        with pytest.raises(EvidenceLedgerError, match="already exists"):
            add_entry(
                updated,
                current_registry,
                requirement_id="TEST-001",
                evidence_class="implementation",
                acceptance_ids=("A1",),
                ref="proof.json",
                commit=COMMIT,
                recorded_at="2026-07-20",
                root=tmp_path,
            )
        with pytest.raises(EvidenceLedgerError, match="does not require"):
            add_entry(
                current,
                current_registry,
                requirement_id="TEST-001",
                evidence_class="live",
                acceptance_ids=("A1",),
                ref="proof.json",
                commit=COMMIT,
                recorded_at="2026-07-20",
                root=tmp_path,
            )
        with pytest.raises(EvidenceLedgerError, match="unknown acceptance IDs"):
            add_entry(
                current,
                current_registry,
                requirement_id="TEST-001",
                evidence_class="implementation",
                acceptance_ids=("A2",),
                ref="proof.json",
                commit=COMMIT,
                recorded_at="2026-07-20",
                root=tmp_path,
            )

    def test_atomic_write_survives_replace_failure(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "ledger.json"
        first = EvidenceLedger.empty_for(registry(make_requirement()))
        write_evidence_ledger_atomic(first, path)
        before = path.read_bytes()

        second = EvidenceLedger(
            schema_version=LEDGER_SCHEMA_VERSION,
            registry_declaration_sha256=first.registry_declaration_sha256,
            entries=(evidence_entry(),),
        )
        original_replace = Path.replace

        def fail_replace(self, target):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated replace failure"):
            write_evidence_ledger_atomic(second, path)
        monkeypatch.setattr(Path, "replace", original_replace)
        assert path.read_bytes() == before
        assert load_evidence_ledger(path).entries == ()


class TestLedgerBinding:
    """What the ledger is bound to, and what it is not bound to.

    The binding used to be the registry FILE's digest, which also covers
    `registry_revision` and the tracker extraction hash. Rewording a sentence
    in the tracker regenerated the registry, moved that digest, and broke the
    ledger — and the only repair was copying the new hash across, which proves
    nothing about the evidence. It binds to what the registry declares now.
    """

    def test_a_provenance_bump_leaves_the_binding_alone(self):
        current_registry = registry(make_requirement())
        ledger = EvidenceLedger.empty_for(current_registry)

        data = current_registry.to_dict()
        data["registry_revision"] += 1
        data["generated_from"]["tracker_extraction_sha256"] = "c" * 64
        data.pop("content_sha256")
        regenerated = Registry.from_dict(data, verify_hash=False)

        assert regenerated.compute_content_sha256() != current_registry.compute_content_sha256()
        verify_ledger_binding(ledger, regenerated)

    def test_a_changed_acceptance_cell_breaks_the_binding(self):
        before = make_requirement()
        current_registry = registry(before)
        ledger = EvidenceLedger.empty_for(current_registry)

        after = json.loads(json.dumps(before))
        after["acceptance"] = ["Do a different thing and prove that instead."]
        with pytest.raises(EvidenceLedgerError, match="different set of requirements"):
            verify_ledger_binding(ledger, registry(after))

    def test_a_changed_evidence_class_breaks_the_binding(self):
        before = make_requirement()
        current_registry = registry(before)
        ledger = EvidenceLedger.empty_for(current_registry)

        after = json.loads(json.dumps(before))
        after["evidence_required"] = ["implementation"]
        after["acceptance_evidence_required"] = [["implementation"]]
        with pytest.raises(EvidenceLedgerError, match="different set of requirements"):
            verify_ledger_binding(ledger, registry(after))

    def test_the_shipped_ledger_is_bound_to_the_shipped_registry(self):
        """The one that was stale: a tracker reword, and nothing else."""
        from tools.reqproof.schema import load_registry

        root = Path(__file__).resolve().parents[1]
        shipped = load_registry(root / "config/requirement_registry.json")
        verify_ledger_binding(
            load_evidence_ledger(root / "config/requirement_evidence_ledger.json"), shipped
        )
