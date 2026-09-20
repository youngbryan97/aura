"""Anti-gaming battery for the requirement-to-proof gate.

Acceptance standard under test: a senior engineer must be unable to mark a
requirement complete by editing prose, adding a shallow test, pointing at an
old artifact, or omitting a child task — and every closure/remaining number
must be reproducible from exact registry and evidence state.
"""
from __future__ import annotations

import json
from pathlib import Path

from reqproof_testkit import mini_tracker

from tools.reqproof.coverage import _sha256_text, range_text
from tools.reqproof.evidence import (
    EvidenceLedger,
    load_evidence_ledger,
    write_evidence_ledger_atomic,
)
from tools.reqproof.gate import (
    _is_self_refreshable_stale_defect,
    run_gate,
)
from tools.reqproof.migrate import migrate
from tools.reqproof.progress import ScopeBaseline, write_scope_baseline_atomic
from tools.reqproof.schema import Registry, load_registry

MINI_CORPUS = """Build alpha end to end.
Prove beta with evidence.
"""


def build_mini_repo(root: Path, **tracker_kwargs) -> dict[str, Path]:
    """A complete miniature control-plane installation under ``root``."""
    paths = {
        "tracker": root / "docs" / "AURA_EXECUTION_TRACKER.md",
        "registry": root / "config" / "requirement_registry.json",
        "allowlist": root / "config" / "reqproof_prose_token_allowlist.json",
        "evidence_ledger": root / "config" / "requirement_evidence_ledger.json",
        "scope_baseline": root / "config" / "reqproof_scope_baseline.json",
        "baseline": root / "config" / "reqproof_defect_baseline.json",
        "report": root / "artifacts" / "reqproof" / "GATE_REPORT.json",
        "corpus": root / "config" / "requirement_sources" / "MINI.txt",
        "manifest": root / "config" / "requirement_sources" / "MANIFEST.json",
        "coverage_map": root / "config" / "requirement_coverage_map.json",
    }
    paths["tracker"].parent.mkdir(parents=True, exist_ok=True)
    paths["corpus"].parent.mkdir(parents=True, exist_ok=True)
    paths["tracker"].write_text(mini_tracker(**tracker_kwargs), encoding="utf-8")
    paths["corpus"].write_text(MINI_CORPUS, encoding="utf-8")
    paths["manifest"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpora": {
                    "mini": {
                        "snapshot": "config/requirement_sources/MINI.txt",
                        "original_path": "/nowhere/mini.txt",
                        "original_sha256": "0" * 64,
                        "description": "test corpus",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    corpus_lines = MINI_CORPUS.splitlines()
    paths["coverage_map"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "corpus": "mini",
                        "lines": "1-1",
                        "sha256": _sha256_text(range_text(corpus_lines, 1, 1)),
                        "class": "normative",
                        "requirements": ["ALPHA-001"],
                    },
                    {
                        "corpus": "mini",
                        "lines": "2-2",
                        "sha256": _sha256_text(range_text(corpus_lines, 2, 2)),
                        "class": "normative",
                        "requirements": ["BETA-001"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    migrate(
        tracker_path=paths["tracker"],
        registry_path=paths["registry"],
        allowlist_path=paths["allowlist"],
        write=True,
    )
    write_evidence_ledger_atomic(
        EvidenceLedger.empty_for(load_registry(paths["registry"])),
        paths["evidence_ledger"],
    )
    write_scope_baseline_atomic(
        ScopeBaseline.from_registry(load_registry(paths["registry"])),
        paths["scope_baseline"],
    )
    return paths


def gate(root: Path, paths: dict[str, Path], **kwargs):
    return run_gate(
        root=root,
        mode=kwargs.pop("mode", "structural"),
        registry_path=paths["registry"],
        tracker_path=paths["tracker"],
        allowlist_path=paths["allowlist"],
        evidence_ledger_path=paths["evidence_ledger"],
        scope_baseline_path=paths["scope_baseline"],
        baseline_path=paths["baseline"],
        report_path=paths["report"],
        **kwargs,
    )


def rebind_ledger(paths: dict[str, Path]) -> None:
    current = load_evidence_ledger(paths["evidence_ledger"])
    registry = load_registry(paths["registry"])
    write_evidence_ledger_atomic(
        EvidenceLedger(
            schema_version=current.schema_version,
            registry_declaration_sha256=registry.declaration_sha256(),
            entries=current.entries,
        ),
        paths["evidence_ledger"],
    )
    write_scope_baseline_atomic(
        ScopeBaseline.from_registry(registry), paths["scope_baseline"]
    )


class TestBaselinePass:
    def test_clean_mini_repo_passes_structurally(self, tmp_path):
        paths = build_mini_repo(tmp_path)
        code, report = gate(tmp_path, paths)
        assert code == 0, report["failures"]
        assert report["verdict"] == "pass"
        assert report["summary"]["requirements"] == 10
        assert report["coverage"]["unmapped_lines"] == 0

    def test_report_is_deterministic(self, tmp_path):
        paths = build_mini_repo(tmp_path)
        gate(tmp_path, paths)
        first = paths["report"].read_bytes()
        gate(tmp_path, paths)
        assert paths["report"].read_bytes() == first

    def test_missing_evidence_ledger_fails_structurally(self, tmp_path):
        paths = build_mini_repo(tmp_path)
        paths["evidence_ledger"].unlink()
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("evidence ledger not found" in failure for failure in report["failures"])

    def test_ledger_bound_to_other_requirements_fails_structurally(self, tmp_path):
        paths = build_mini_repo(tmp_path)
        data = json.loads(paths["evidence_ledger"].read_text(encoding="utf-8"))
        data["registry_declaration_sha256"] = "f" * 64
        unhashed = EvidenceLedger.from_dict(data, verify_hash=False)
        write_evidence_ledger_atomic(unhashed, paths["evidence_ledger"])
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any(
            "different set of requirements" in failure for failure in report["failures"]
        )

    def test_a_registry_regenerated_from_reworded_prose_still_binds(self, tmp_path):
        """The tracker moves often and declares nothing by moving."""
        paths = build_mini_repo(tmp_path)
        registry_data = json.loads(paths["registry"].read_text(encoding="utf-8"))
        registry_data["registry_revision"] += 1
        registry_data["generated_from"]["tracker_extraction_sha256"] = "c" * 64
        registry_data.pop("content_sha256")
        regenerated = Registry.from_dict(registry_data, verify_hash=False)
        paths["registry"].write_text(regenerated.to_canonical_json(), encoding="utf-8")

        code, report = gate(tmp_path, paths)
        assert not any(
            "different set of requirements" in failure for failure in report["failures"]
        ), report["failures"]
        # The extraction hash is fabricated here, so the gate must still say the
        # registry was generated from a tracker this one is not. That refusal is
        # the one that belongs to a reworded tracker; the binding is not.
        assert any("stale-migration" in failure for failure in report["failures"])
        assert code == 1

    def test_scope_denominator_growth_or_shrink_fails_structurally(self, tmp_path):
        paths = build_mini_repo(tmp_path)
        current = ScopeBaseline.from_registry(load_registry(paths["registry"]))
        write_scope_baseline_atomic(
            ScopeBaseline(fingerprints=current.fingerprints[:-1]),
            paths["scope_baseline"],
        )
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("baseline is stale" in failure for failure in report["failures"])

        write_scope_baseline_atomic(
            ScopeBaseline(
                fingerprints=tuple(sorted((*current.fingerprints, "ZZZ-999::A1::test")))
            ),
            paths["scope_baseline"],
        )
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("denominator shrank" in failure for failure in report["failures"])


class TestGamingAttempts:
    def test_prose_edit_cannot_change_anything(self, tmp_path):
        """Rewriting narrative prose neither closes work nor breaks the gate."""
        paths = build_mini_repo(tmp_path)
        code_before, report_before = gate(tmp_path, paths)
        tracker_text = paths["tracker"].read_text(encoding="utf-8")
        paths["tracker"].write_text(
            tracker_text.replace(
                "Narrative: Nothing happened.",
                "Narrative: Everything is finished, 100% complete, ship it.",
            ),
            encoding="utf-8",
        )
        code_after, report_after = gate(tmp_path, paths)
        assert (code_before, code_after) == (0, 0)
        assert (
            report_before["summary"]["mandatory_not_closed"]
            == report_after["summary"]["mandatory_not_closed"]
        )

    def test_flipping_status_to_complete_without_evidence_fails(self, tmp_path):
        """Editing a status cell to COMPLETE must surface, not close."""
        paths = build_mini_repo(tmp_path)
        assert gate(tmp_path, paths)[0] == 0

        # The gamer edits the tracker table status and even re-runs migration.
        paths["tracker"].write_text(
            mini_tracker(master_status="COMPLETE 2026-07-16"), encoding="utf-8"
        )
        migrate(
            tracker_path=paths["tracker"],
            registry_path=paths["registry"],
            allowlist_path=paths["allowlist"],
            write=True,
        )
        rebind_ledger(paths)
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("unproven-closure" in failure for failure in report["failures"])
        assert any("false-closure" in failure for failure in report["failures"])

    def test_stale_registry_after_tracker_status_edit_fails(self, tmp_path):
        """Editing the tracker without re-migration is a stale-migration failure."""
        paths = build_mini_repo(tmp_path)
        paths["tracker"].write_text(
            mini_tracker(master_status="IN PROGRESS 2026-07-16"), encoding="utf-8"
        )
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("stale-migration" in failure for failure in report["failures"])

    def test_hand_editing_registry_state_fails_hash_check(self, tmp_path):
        """Direct registry edits are tamper-evident."""
        paths = build_mini_repo(tmp_path)
        data = json.loads(paths["registry"].read_text(encoding="utf-8"))
        for requirement in data["requirements"]:
            requirement["state"] = "complete"
        paths["registry"].write_text(json.dumps(data), encoding="utf-8")
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any(
            "edited without regeneration" in failure for failure in report["failures"]
        )

    def test_deleting_an_obligation_from_the_corpus_map_fails(self, tmp_path):
        """Dropping a passage from the map reopens zero-unmapped."""
        paths = build_mini_repo(tmp_path)
        data = json.loads(paths["coverage_map"].read_text(encoding="utf-8"))
        data["entries"] = data["entries"][:1]
        paths["coverage_map"].write_text(json.dumps(data), encoding="utf-8")
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("unmapped-passage" in failure for failure in report["failures"])

    def test_pointing_map_at_nonexistent_requirement_fails(self, tmp_path):
        paths = build_mini_repo(tmp_path)
        data = json.loads(paths["coverage_map"].read_text(encoding="utf-8"))
        data["entries"][0]["requirements"] = ["INVENTED-999"]
        paths["coverage_map"].write_text(json.dumps(data), encoding="utf-8")
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("coverage-orphan-ref" in failure for failure in report["failures"])

    def test_new_prose_only_id_fails_until_registered(self, tmp_path):
        """Referencing a new ID in prose cannot silently drop the obligation."""
        paths = build_mini_repo(tmp_path)
        tracker_text = paths["tracker"].read_text(encoding="utf-8")
        paths["tracker"].write_text(
            tracker_text + "\nNew work `OMEGA-777` is mentioned only here.\n",
            encoding="utf-8",
        )
        code, report = gate(tmp_path, paths)
        assert code == 1
        # Both drift signals fire: the extraction changed and the token is
        # neither a requirement nor allowlisted.
        assert any("prose-only-token" in failure for failure in report["failures"])
        # Re-migration mints the obligation and the gate recovers.
        migrate(
            tracker_path=paths["tracker"],
            registry_path=paths["registry"],
            allowlist_path=paths["allowlist"],
            write=True,
        )
        rebind_ledger(paths)
        code_after, report_after = gate(tmp_path, paths, refresh_baseline=True)
        assert code_after == 0, report_after["failures"]
        assert "OMEGA-777" in {
            d["subject"]
            for d in report_after["defects"]
            if d["defect_class"] == "prose-minted"
        }


class TestRatchet:
    def test_new_ratcheted_defect_fails_even_with_baseline(self, tmp_path):
        paths = build_mini_repo(tmp_path, child_status="COMPLETE 2026-07-10")
        # Seed the baseline over the pre-existing debt (child completes
        # without evidence).
        code, report = gate(tmp_path, paths, refresh_baseline=True)
        assert code == 0, report["failures"]
        assert report["ratchet"]["baseline_fingerprints"] > 0

        # A NEW unproven closure appears: the gate must fail despite baseline.
        paths["tracker"].write_text(
            mini_tracker(
                master_status="COMPLETE 2026-07-16",
                child_status="COMPLETE 2026-07-10",
            ),
            encoding="utf-8",
        )
        migrate(
            tracker_path=paths["tracker"],
            registry_path=paths["registry"],
            allowlist_path=paths["allowlist"],
            write=True,
        )
        rebind_ledger(paths)
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("NEW ratcheted defect" in failure for failure in report["failures"])

    def test_baseline_refresh_refuses_growth(self, tmp_path):
        paths = build_mini_repo(tmp_path, child_status="COMPLETE 2026-07-10")
        assert gate(tmp_path, paths, refresh_baseline=True)[0] == 0
        paths["tracker"].write_text(
            mini_tracker(
                master_status="COMPLETE 2026-07-16",
                child_status="COMPLETE 2026-07-10",
            ),
            encoding="utf-8",
        )
        migrate(
            tracker_path=paths["tracker"],
            registry_path=paths["registry"],
            allowlist_path=paths["allowlist"],
            write=True,
        )
        rebind_ledger(paths)
        code, report = gate(tmp_path, paths, refresh_baseline=True)
        assert code == 1
        assert any("shrink-only" in failure for failure in report["failures"])

    def test_fixed_defect_makes_baseline_stale_then_shrinks(self, tmp_path):
        paths = build_mini_repo(tmp_path, child_status="COMPLETE 2026-07-10")
        assert gate(tmp_path, paths, refresh_baseline=True)[0] == 0

        # The debt is repaired: child honestly reopened.
        paths["tracker"].write_text(mini_tracker(), encoding="utf-8")
        migrate(
            tracker_path=paths["tracker"],
            registry_path=paths["registry"],
            allowlist_path=paths["allowlist"],
            write=True,
        )
        rebind_ledger(paths)
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("STALE" in failure for failure in report["failures"])
        code, report = gate(tmp_path, paths, refresh_baseline=True)
        assert code == 0
        assert report["ratchet"]["baseline_fingerprints"] == 0


class TestReleaseMode:
    def test_release_blocks_on_open_mandatory_work(self, tmp_path):
        paths = build_mini_repo(tmp_path)
        code, report = gate(tmp_path, paths, mode="release")
        assert code == 1
        blocked = [f for f in report["failures"] if "release blocked" in f]
        assert blocked
        assert str(report["summary"]["mandatory_not_closed"]) in blocked[0]

    def test_remaining_counts_are_reproducible_from_registry_state(self, tmp_path):
        """The report's numbers must be recomputable exactly from the registry."""
        from tools.reqproof.schema import CLOSED_STATES, load_registry

        paths = build_mini_repo(tmp_path)
        _, report = gate(tmp_path, paths)
        registry = load_registry(paths["registry"])
        expected = sum(
            1
            for requirement in registry.requirements
            if requirement.mandatory and requirement.state not in CLOSED_STATES
        )
        assert report["summary"]["mandatory_not_closed"] == expected
        assert report["summary"]["requirements"] == len(registry.requirements)


class TestFaultInjection:
    def test_crash_during_report_write_preserves_previous_report(
        self, tmp_path, monkeypatch
    ):
        paths = build_mini_repo(tmp_path)
        gate(tmp_path, paths)
        before = paths["report"].read_bytes()

        import pytest

        original_replace = Path.replace

        def exploding_replace(self, target):
            if self.name.startswith("GATE_REPORT"):
                raise OSError("simulated crash before rename")
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", exploding_replace)
        with pytest.raises(OSError, match="simulated crash"):
            gate(tmp_path, paths)
        monkeypatch.setattr(Path, "replace", original_replace)
        assert paths["report"].read_bytes() == before

    def test_truncated_registry_is_a_loud_failure(self, tmp_path):
        """A torn/partial registry file must fail parse, never half-load."""
        paths = build_mini_repo(tmp_path)
        full = paths["registry"].read_text(encoding="utf-8")
        paths["registry"].write_text(full[: len(full) // 2], encoding="utf-8")
        code, report = gate(tmp_path, paths)
        assert code == 1
        assert any("registry" in failure for failure in report["failures"])


class TestStructuralEvidenceRefresh:
    def test_only_the_structural_proofs_own_stale_receipt_is_eligible(self, tmp_path):
        from tools.reqproof.validate import Defect

        ref = (
            "artifacts/reqproof/evidence/reqproof-structural-gate-audit/"
            + "a" * 40
            + ".json"
        )
        receipt = tmp_path / ref
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps(
                {
                    "proof_id": "reqproof-structural-gate-audit",
                    "verdict": "pass",
                }
            ),
            encoding="utf-8",
        )
        own_stale = Defect("stale-evidence", f"SCOPE-001::{ref}", "changed")
        unrelated = Defect(
            "stale-evidence",
            "SCOPE-001::artifacts/reqproof/evidence/other-proof/x.json",
            "changed",
        )
        wrong_class = Defect("unproven-closure", f"SCOPE-001::{ref}", "missing")

        assert _is_self_refreshable_stale_defect(tmp_path, own_stale) is True
        assert _is_self_refreshable_stale_defect(tmp_path, unrelated) is False
        assert _is_self_refreshable_stale_defect(tmp_path, wrong_class) is False

    def test_self_refresh_keeps_every_other_defect_blocking(self, tmp_path, monkeypatch):
        import tools.reqproof.gate as gate_module
        from tools.reqproof.validate import Defect

        paths = build_mini_repo(tmp_path)
        ref = (
            "artifacts/reqproof/evidence/reqproof-structural-gate-audit/"
            + "b" * 40
            + ".json"
        )
        receipt = tmp_path / ref
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps(
                {
                    "proof_id": "reqproof-structural-gate-audit",
                    "verdict": "pass",
                }
            ),
            encoding="utf-8",
        )
        own = Defect("stale-evidence", f"SCOPE-001::{ref}", "changed")
        other = Defect(
            "stale-evidence",
            "ALPHA-001::artifacts/reqproof/evidence/another-proof/x.json",
            "changed",
        )
        monkeypatch.setattr(
            gate_module,
            "validate_registry",
            lambda *_args, **_kwargs: [own, other],
        )

        code, report = gate(
            tmp_path,
            paths,
            refresh_self_evidence=True,
        )

        assert code == 1
        assert report["self_evidence_refresh"]["ignored_stale_receipts"] == [
            own.fingerprint
        ]
        assert [row["subject"] for row in report["defects"]] == [other.subject]
        assert any(other.fingerprint in failure for failure in report["failures"])

    def test_self_refresh_is_structural_only(self, tmp_path):
        paths = build_mini_repo(tmp_path)
        code, report = gate(
            tmp_path,
            paths,
            mode="release",
            refresh_self_evidence=True,
        )
        assert code == 1
        assert "self-evidence refresh is structural-only" in report["failures"]


class TestRealRepositoryGate:
    """The checked-in registry/coverage/baseline must pass structurally.

    The tracker is compared at its COMMITTED (HEAD) content, not the working
    tree: a parallel agent's in-flight tracker edits must not fail the suite
    mid-session — the enforced boundary is that every COMMIT keeps the
    tracker and registry reconciled (run tools/reqproof/migrate.py --write
    in the same commit as any normative tracker change).
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _committed_tracker(self, tmp_path: Path) -> Path:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        result = get_subprocess_gateway().run(
            ["git", "-c", "core.fsmonitor=false", "show", "HEAD:docs/AURA_EXECUTION_TRACKER.md"],
            cwd=self.ROOT,
            timeout=60,
            read_only=True,
            source="reqproof_gate_test_tracker_at_head",
            accelerator_capability="none",
        )
        if result.returncode != 0:
            # Not a git checkout (e.g. exported tree): fall back to disk.
            return self.ROOT / "docs" / "AURA_EXECUTION_TRACKER.md"
        snapshot = tmp_path / "TRACKER_AT_HEAD.md"
        snapshot.write_text(result.stdout, encoding="utf-8")
        return snapshot

    def _run(self, tmp_path: Path, mode: str):
        return run_gate(
            root=self.ROOT,
            mode=mode,
            registry_path=self.ROOT / "config" / "requirement_registry.json",
            tracker_path=self._committed_tracker(tmp_path),
            allowlist_path=self.ROOT
            / "config"
            / "reqproof_prose_token_allowlist.json",
            evidence_ledger_path=self.ROOT
            / "config"
            / "requirement_evidence_ledger.json",
            scope_baseline_path=self.ROOT
            / "config"
            / "reqproof_scope_baseline.json",
            baseline_path=self.ROOT / "config" / "reqproof_defect_baseline.json",
            report_path=tmp_path / "GATE_REPORT.json",
        )

    def test_structural_gate_passes_on_the_repository(self, tmp_path):
        code, report = self._run(tmp_path, "structural")
        assert code == 0, report["failures"][:10]
        assert report["coverage"]["unmapped_lines"] == 0

    def test_release_gate_honestly_blocks_today(self, tmp_path):
        code, report = self._run(tmp_path, "release")
        assert code == 1
        assert any("release blocked" in failure for failure in report["failures"])
