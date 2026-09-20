"""Command-capture contracts for the requirement proof control plane."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from reqproof_testkit import make_registry_dict, make_requirement

from tools.reqproof.capture import (
    ProofCaptureError,
    ProofSpecRegistry,
    assert_pushed_clean_source,
    capture_proof,
    capture_proofs,
    load_proof_specs,
    validate_spec_targets,
)
from tools.reqproof.evidence import (
    EvidenceLedger,
    EvidenceLedgerError,
    load_evidence_ledger,
    write_evidence_ledger_atomic,
)
from tools.reqproof.schema import Registry
from tools.reqproof.validate import validate_registry


def _hashed_specs(specs: list[dict[str, Any]]) -> dict[str, Any]:
    body = {"schema_version": 1, "specs": specs}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {
        **body,
        "content_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _spec(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "bounded-proof",
        "command": ["{python}", "-m", "pytest", "-q", "tests/test_one.py"],
        "cwd": ".",
        "timeout_seconds": 30,
        "max_output_bytes": 4096,
        "source_globs": [],
        "source_paths": ["core/a.py", "tests/test_one.py"],
        "evidence_targets": [
            {
                "requirement_id": "TEST-001",
                "evidence_class": "test",
                "acceptance_ids": ["A1"],
            }
        ],
    }
    base.update(overrides)
    return base


class FakeGateway:
    def __init__(
        self,
        *,
        head: str = "a" * 40,
        remote: str | None = None,
        status: str = "",
    ):
        self.head = head
        self.remote = remote or head
        self.status = status

    def run(
        self, argv: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        command = tuple(argv)
        if command[-2:] == ("rev-parse", "HEAD"):
            stdout = self.head + "\n"
        elif command[-2:] == ("rev-parse", "origin/main"):
            stdout = self.remote + "\n"
        elif "status" in command:
            stdout = self.status
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


class CaptureGateway(FakeGateway):
    def __init__(
        self,
        *,
        command_returncode: int = 0,
        stdout: str | None = None,
        stderr: str | None = None,
        timeout: bool = False,
    ):
        super().__init__()
        self.command_returncode = command_returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timeout = timeout
        self.command_kwargs: dict[str, Any] | None = None

    def run(
        self, argv: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if tuple(argv[:2]) == ("git", "-c", "core.fsmonitor=false", "rev-parse") or "status" in argv:
            return super().run(argv, **kwargs)
        self.command_kwargs = kwargs
        if self.timeout:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(
            tuple(argv),
            self.command_returncode,
            stdout=(
                self.stdout
                if self.stdout is not None
                else "2 passed in 0.01s\n" if self.command_returncode == 0 else ""
            ),
            stderr=(
                self.stderr
                if self.stderr is not None
                else "proof failed\n" if self.command_returncode else ""
            ),
        )


def _capture_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    (tmp_path / "core").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "core" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_one.py").write_text(
        "def test_one(): assert True\n", encoding="utf-8"
    )
    spec_path = tmp_path / "config" / "specs.json"
    spec_path.write_text(json.dumps(_hashed_specs([_spec()])), encoding="utf-8")
    registry = Registry.from_dict(
        make_registry_dict(
            [make_requirement(evidence_required=["implementation", "test"])]
        )
    )
    registry_path = tmp_path / "config" / "registry.json"
    registry_path.write_text(registry.to_canonical_json(), encoding="utf-8")
    ledger_path = tmp_path / "config" / "ledger.json"
    write_evidence_ledger_atomic(EvidenceLedger.empty_for(registry), ledger_path)
    return spec_path, registry_path, ledger_path


def test_checked_registry_round_trips_and_targets_real_acceptance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "specs.json"
    path.write_text(json.dumps(_hashed_specs([_spec()])), encoding="utf-8")
    specs = load_proof_specs(path)
    registry = Registry.from_dict(
        make_registry_dict(
            [make_requirement(evidence_required=["implementation", "test"])]
        )
    )

    validate_spec_targets(specs, registry)
    assert specs.by_id()["bounded-proof"].timeout_seconds == 30


def test_spec_hash_tampering_and_shell_placeholders_fail_closed() -> None:
    data = _hashed_specs([_spec()])
    data["specs"][0]["timeout_seconds"] = 31
    with pytest.raises(ProofCaptureError, match="content hash mismatch"):
        ProofSpecRegistry.from_dict(data)

    with pytest.raises(ProofCaptureError, match="unsupported placeholder"):
        ProofSpecRegistry.from_dict(
            _hashed_specs([_spec(command=["sh", "-c", "{payload}"])])
        )
    with pytest.raises(ProofCaptureError, match="may not invoke a shell"):
        ProofSpecRegistry.from_dict(
            _hashed_specs([_spec(command=["sh", "-c", "pytest -q"])])
        )


def test_specs_reject_unsorted_sources_targets_and_unknown_cells() -> None:
    with pytest.raises(ProofCaptureError, match="paths must be sorted"):
        ProofSpecRegistry.from_dict(
            _hashed_specs([_spec(source_paths=["tests/z.py", "core/a.py"])])
        )

    specs = ProofSpecRegistry.from_dict(_hashed_specs([_spec()]))
    registry = Registry.from_dict(
        make_registry_dict([make_requirement(evidence_required=["implementation"])])
    )
    with pytest.raises(ProofCaptureError, match="unrequired class"):
        validate_spec_targets(specs, registry)


def test_source_must_equal_pushed_main_and_be_clean(tmp_path: Path) -> None:
    assert assert_pushed_clean_source(FakeGateway(), tmp_path) == "a" * 40
    with pytest.raises(ProofCaptureError, match="not exact pushed main"):
        assert_pushed_clean_source(FakeGateway(remote="b" * 40), tmp_path)
    with pytest.raises(ProofCaptureError, match="dirty"):
        assert_pushed_clean_source(FakeGateway(status=" M core/a.py\n"), tmp_path)


def test_capture_writes_hash_bound_receipt_and_ledger_cell(tmp_path: Path) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)
    gateway = CaptureGateway()

    receipt_path = capture_proof(
        root=tmp_path,
        spec_registry_path=spec_path,
        registry_path=registry_path,
        ledger_path=ledger_path,
        artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
        proof_id="bounded-proof",
        record=True,
        gateway=gateway,
    )

    receipt = json.loads(receipt_path.read_text())
    ledger = load_evidence_ledger(ledger_path)
    assert receipt["verdict"] == "pass"
    assert receipt["source_commit"] == "a" * 40
    assert receipt["schema"] == "aura.reqproof.command_receipt.v2"
    assert receipt["source_selectors"] == {
        "paths": ["core/a.py", "tests/test_one.py"],
        "globs": [],
    }
    assert receipt["stdout"] == "2 passed in 0.01s\n"
    assert [item["path"] for item in receipt["source_manifest"]] == [
        "core/a.py",
        "tests/test_one.py",
    ]
    assert ledger.entries[0].evidence.ref == receipt_path.relative_to(tmp_path).as_posix()
    assert ledger.entries[0].acceptance_ids == ("A1",)
    assert gateway.command_kwargs is not None
    assert gateway.command_kwargs["offline_tooling"] is True
    assert gateway.command_kwargs["accelerator_capability"] == "none"
    assert gateway.command_kwargs["stdin_devnull"] is True


def test_batch_capture_records_all_receipts_only_after_every_proof_passes(
    tmp_path: Path,
) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)
    specs = [
        _spec(),
        _spec(
            id="second-proof",
            evidence_targets=[
                {
                    "requirement_id": "TEST-001",
                    "evidence_class": "implementation",
                    "acceptance_ids": ["A1"],
                }
            ],
        ),
    ]
    spec_path.write_text(json.dumps(_hashed_specs(specs)), encoding="utf-8")

    receipts = capture_proofs(
        root=tmp_path,
        spec_registry_path=spec_path,
        registry_path=registry_path,
        ledger_path=ledger_path,
        artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
        proof_ids=("bounded-proof", "second-proof"),
        record=True,
        gateway=CaptureGateway(),
    )

    assert [path.parent.name for path in receipts] == [
        "bounded-proof",
        "second-proof",
    ]
    assert all(path.is_file() for path in receipts)
    ledger = load_evidence_ledger(ledger_path)
    assert {entry.evidence.ref for entry in ledger.entries} == {
        path.relative_to(tmp_path).as_posix() for path in receipts
    }


def test_batch_capture_failure_leaves_no_receipts_or_ledger_changes(
    tmp_path: Path,
) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)
    specs = [_spec(), _spec(id="missing-proof", source_paths=["core/missing.py"])]
    spec_path.write_text(json.dumps(_hashed_specs(specs)), encoding="utf-8")
    before = ledger_path.read_bytes()

    with pytest.raises(EvidenceLedgerError, match="does not name a regular file"):
        capture_proofs(
            root=tmp_path,
            spec_registry_path=spec_path,
            registry_path=registry_path,
            ledger_path=ledger_path,
            artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
            proof_ids=("bounded-proof", "missing-proof"),
            record=True,
            gateway=CaptureGateway(),
        )

    assert ledger_path.read_bytes() == before
    assert not list((tmp_path / "artifacts").rglob("*.json"))


def test_batch_capture_rejects_duplicate_proof_ids(tmp_path: Path) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)

    with pytest.raises(ProofCaptureError, match="must be unique"):
        capture_proofs(
            root=tmp_path,
            spec_registry_path=spec_path,
            registry_path=registry_path,
            ledger_path=ledger_path,
            artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
            proof_ids=("bounded-proof", "bounded-proof"),
            record=True,
            gateway=CaptureGateway(),
        )


def test_failed_command_reports_both_streams_and_leaves_no_evidence(
    tmp_path: Path,
) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)
    before = ledger_path.read_bytes()

    with pytest.raises(ProofCaptureError, match="failed with exit 1") as raised:
        capture_proof(
            root=tmp_path,
            spec_registry_path=spec_path,
            registry_path=registry_path,
            ledger_path=ledger_path,
            artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
            proof_id="bounded-proof",
            record=True,
            gateway=CaptureGateway(
                command_returncode=1,
                stdout="assertion details from pytest\n",
                stderr="runner diagnostics\n",
            ),
        )

    assert "stdout tail:\nassertion details from pytest" in str(raised.value)
    assert "stderr tail:\nrunner diagnostics" in str(raised.value)
    assert ledger_path.read_bytes() == before
    assert not (tmp_path / "artifacts").exists()


def test_failed_command_diagnostics_are_bounded_and_redact_common_secrets(
    tmp_path: Path,
) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)

    with pytest.raises(ProofCaptureError) as raised:
        capture_proof(
            root=tmp_path,
            spec_registry_path=spec_path,
            registry_path=registry_path,
            ledger_path=ledger_path,
            artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
            proof_id="bounded-proof",
            record=True,
            gateway=CaptureGateway(
                command_returncode=1,
                stdout="x" * 6000 + "\napi_key=do-not-print\nLATEST_STDOUT\n",
                stderr=(
                    "y" * 6000
                    + "\nAuthorization: Bearer do-not-print-either\nLATEST_STDERR\n"
                ),
            ),
        )

    message = str(raised.value)
    assert len(message.encode("utf-8")) <= 4300
    assert "LATEST_STDOUT" in message
    assert "LATEST_STDERR" in message
    assert "do-not-print" not in message
    assert "api_key=<redacted>" in message
    assert "Authorization: Bearer <redacted>" in message
    assert not (tmp_path / "artifacts").exists()


def test_new_file_matching_source_glob_revokes_captured_evidence(tmp_path: Path) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "first.py").write_text("FIRST = 1\n", encoding="utf-8")
    spec_path.write_text(
        json.dumps(_hashed_specs([_spec(source_globs=["plugins/**/*.py"])])),
        encoding="utf-8",
    )
    capture_proof(
        root=tmp_path,
        spec_registry_path=spec_path,
        registry_path=registry_path,
        ledger_path=ledger_path,
        artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
        proof_id="bounded-proof",
        record=True,
        gateway=CaptureGateway(),
    )
    registry = Registry.from_dict(json.loads(registry_path.read_text()))
    ledger = load_evidence_ledger(ledger_path)
    (tmp_path / "plugins" / "second.py").write_text("SECOND = 2\n", encoding="utf-8")

    defects = validate_registry(
        registry,
        root=tmp_path,
        commit_exists=lambda commit: True,
        evidence_entries_by_requirement=ledger.entries_by_requirement(),
    )

    assert [defect.defect_class for defect in defects] == ["stale-evidence"]
    assert "added=['plugins/second.py']" in defects[0].detail


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("gateway", "message"),
    [
        (CaptureGateway(timeout=True), "exceeded 30s"),
        (CaptureGateway(stdout="x" * 5000), "exceeds 4096-byte contract"),
    ],
)
def test_timeout_and_output_overflow_leave_no_evidence(
    tmp_path: Path,
    gateway: CaptureGateway,
    message: str,
) -> None:
    spec_path, registry_path, ledger_path = _capture_fixture(tmp_path)
    before = ledger_path.read_bytes()

    with pytest.raises(ProofCaptureError, match=message):
        capture_proof(
            root=tmp_path,
            spec_registry_path=spec_path,
            registry_path=registry_path,
            ledger_path=ledger_path,
            artifact_root=tmp_path / "artifacts" / "reqproof" / "evidence",
            proof_id="bounded-proof",
            record=True,
            gateway=gateway,
        )

    assert ledger_path.read_bytes() == before
    assert not (tmp_path / "artifacts").exists()


def test_checked_repo_spec_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    specs = load_proof_specs(root / "config" / "requirement_proof_specs.json")
    registry = Registry.from_dict(
        json.loads((root / "config" / "requirement_registry.json").read_text())
    )
    validate_spec_targets(specs, registry)
    assert specs.by_id()["model-lane-contract-tests"].command[0] == "{python}"
    assert "core/**/*.py" in specs.by_id()["model-load-ownership-audit"].source_globs
    assert specs.by_id()["resource-observation-contract-tests"].evidence_targets
    assert (
        "core/**/*.py"
        in specs.by_id()["resource-observation-ownership-audit"].source_globs
    )
    progress_target = specs.by_id()[
        "reqproof-progress-engine-audit"
    ].evidence_targets[0]
    assert (
        progress_target.requirement_id,
        progress_target.evidence_class,
        progress_target.acceptance_ids,
    ) == ("PROGRESS-CONTROL-001", "implementation", ("A1",))
    assert specs.by_id()["reqproof-progress-engine-audit"].command[-2:] == (
        "--markdown",
        "/tmp/aura-reqproof-progress.md",
    )
    assert specs.by_id()["reqproof-structural-gate-audit"].command[-2:] == (
        "--report",
        "/tmp/aura-reqproof-structural-gate.json",
    )
    for proof_id in (
        "reqproof-progress-engine-audit",
        "reqproof-structural-gate-audit",
    ):
        assert (
            "config/requirement_evidence_ledger.json"
            not in specs.by_id()[proof_id].source_paths
        ), "recording a receipt must not stale its own implementation proof"
    assert {
        (target.requirement_id, target.evidence_class)
        for target in specs.by_id()[
            "reqproof-control-plane-contract-tests"
        ].evidence_targets
    } == {
        ("PROGRESS-CONTROL-001", "test"),
        ("SCOPE-001", "test"),
    }
