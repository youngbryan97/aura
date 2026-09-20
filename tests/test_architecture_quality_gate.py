import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from core.architecture_quality import scorer as scorer_module
from core.architecture_quality.attestation import attest_payload
from core.architecture_quality.gate import ArchitectureQualityGate, ArchitectureQualityPolicy
from core.architecture_quality.scorer import _strongly_connected_components, score_codebase
from tools.closeout.migrate_architecture_quality_baseline import (
    _load_legacy_baseline,
    verify_migration_receipt,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_scorer_detects_project_local_import_cycle(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "import core.b\nVALUE = 1\n")
    _write(tmp_path / "core" / "b.py", "from core import a\nVALUE = a.VALUE\n")

    report = score_codebase(tmp_path, include_roots=("core",))

    assert report.metrics.module_count == 3
    assert report.metrics.cycle_count == 1
    assert ("core.a", "core.b") in report.cycles
    assert any(finding.code == "import_cycle" for finding in report.findings)


def test_gate_rejects_overlay_that_introduces_cycle(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")
    _write(tmp_path / "core" / "b.py", "import core.a\nVALUE = core.a.VALUE\n")

    gate = ArchitectureQualityGate(tmp_path, include_roots=("core",))
    result = gate.evaluate_overlay(
        {"core/a.py": "import core.b\nVALUE = core.b.VALUE\n"},
        changed_paths=("core/a.py",),
    )

    assert not result.passed
    assert any("new import cycle" in reason for reason in result.reasons)


def test_gate_allows_cycle_shrinkage_even_when_membership_changes(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "import core.b\nVALUE = 1\n")
    _write(tmp_path / "core" / "b.py", "import core.c\nVALUE = 2\n")
    _write(tmp_path / "core" / "c.py", "import core.a\nVALUE = 3\n")

    gate = ArchitectureQualityGate(tmp_path, include_roots=("core",))
    result = gate.evaluate_overlay(
        {"core/c.py": "import core.b\nVALUE = 3\n"},
        changed_paths=("core/c.py",),
    )

    assert result.passed
    assert result.after.metrics.largest_cycle_size < result.before.metrics.largest_cycle_size


def test_gate_allows_large_cycle_decomposition_into_smaller_cycles(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    for idx in range(12):
        current = chr(ord("a") + idx)
        nxt = chr(ord("a") + ((idx + 1) % 12))
        _write(tmp_path / "core" / f"{current}.py", f"import core.{nxt}\n")

    overlay = {}
    for idx in range(12):
        current = chr(ord("a") + idx)
        if idx < 3:
            nxt = chr(ord("a") + ((idx + 1) % 3))
        elif idx < 6:
            nxt = chr(ord("a") + 3 + ((idx - 3 + 1) % 3))
        else:
            nxt = None
        overlay[f"core/{current}.py"] = f"import core.{nxt}\n" if nxt else "VALUE = 1\n"

    gate = ArchitectureQualityGate(tmp_path, include_roots=("core",))
    result = gate.evaluate_overlay(overlay, changed_paths=overlay.keys())

    assert result.passed
    assert result.before.metrics.cycle_count == 1
    assert result.after.metrics.cycle_count == 2
    assert result.after.metrics.largest_cycle_size == 3


def test_gate_rejects_large_file_growth(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    base_content = "\n".join(f"BASE_{idx} = {idx}" for idx in range(25)) + "\n"
    grown_content = "\n".join(f"NEXT_{idx} = {idx}" for idx in range(60)) + "\n"
    _write(tmp_path / "core" / "big.py", base_content)

    gate = ArchitectureQualityGate(
        tmp_path,
        include_roots=("core",),
        god_file_threshold=20,
        policy=ArchitectureQualityPolicy(max_line_growth_for_large_file=10),
    )
    result = gate.evaluate_overlay(
        {"core/big.py": grown_content},
        changed_paths=("core/big.py",),
    )

    assert not result.passed
    assert any("grew to 60 lines" in reason for reason in result.reasons)


def test_gate_allows_neutral_overlay(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")

    gate = ArchitectureQualityGate(tmp_path, include_roots=("core",))
    result = gate.evaluate_overlay(
        {"core/a.py": "VALUE = 2\n"},
        changed_paths=("core/a.py",),
    )

    assert result.passed
    assert "architecture quality passed" in result.summary()


def test_gate_rejects_incomparable_report_schema(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    report = score_codebase(tmp_path, include_roots=("core",))
    legacy = replace(report, schema_version=1)

    result = ArchitectureQualityGate(tmp_path, include_roots=("core",)).evaluate_reports(
        legacy,
        report,
    )

    assert not result.passed
    assert any("schema mismatch" in reason for reason in result.reasons)


def test_cli_compares_against_written_baseline(tmp_path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    repo = tmp_path / "repo"
    _write(repo / "core" / "__init__.py", "")
    _write(repo / "core" / "a.py", "VALUE = 1\n")
    baseline = tmp_path / "baseline.json"
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_key = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    tool = Path("tools/closeout/architecture_quality_gate.py").resolve()

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--root",
            str(repo),
            "--include",
            "core",
            "--write-baseline",
            str(baseline),
            "--signing-key",
            str(private_path),
            "--trust-root",
            str(public_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["metrics"]["module_count"] == 2

    result = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--root",
            str(repo),
            "--include",
            "core",
            "--baseline",
            str(baseline),
            "--trust-root",
            str(public_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["passed"] is True

    unsigned = json.loads(baseline.read_text(encoding="utf-8"))
    unsigned.pop("signature")
    baseline.write_text(json.dumps(unsigned), encoding="utf-8")
    missing_signature = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--root",
            str(repo),
            "--include",
            "core",
            "--baseline",
            str(baseline),
            "--trust-root",
            str(public_path),
        ],
        capture_output=True,
        text=True,
    )
    assert missing_signature.returncode != 0
    assert "missing its detached signature" in missing_signature.stdout

    refused_replacement = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--root",
            str(repo),
            "--include",
            "core",
            "--write-baseline",
            str(baseline),
            "--signing-key",
            str(private_path),
        ],
        capture_output=True,
        text=True,
    )
    assert refused_replacement.returncode != 0
    assert "requires --migration-receipt" in refused_replacement.stdout

    baseline.unlink()
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--root",
            str(repo),
            "--include",
            "core",
            "--write-baseline",
            str(baseline),
            "--signing-key",
            str(private_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tampered = json.loads(baseline.read_text(encoding="utf-8"))
    tampered["score"] = 99.0
    baseline.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--root",
            str(repo),
            "--include",
            "core",
            "--baseline",
            str(baseline),
            "--trust-root",
            str(public_path),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "evidence attestation mismatch" in rejected.stdout


def test_migration_receipt_is_pinned_and_tamper_evident(tmp_path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    payload = {
        "schema": "aura.architecture_quality_baseline_migration.v1",
        "historical_claim_reproduction": {"all_checks_passed": True},
        "phases": [
            {"name": "legacy_to_observation", "integrity": {"all_checks_passed": True}},
            {"name": "observation_to_target", "integrity": {"all_checks_passed": True}},
        ],
        "migration_decision": {
            "legacy_debt_preserved": True,
            "future_regressions_waived": False,
        },
    }
    receipt = attest_payload(
        payload,
        digest_field="migration_sha256",
        signing_key_path=private_path,
    )

    verify_migration_receipt(receipt, trusted_public_key_pem=public_pem)
    receipt["phases"][0]["integrity"]["all_checks_passed"] = False
    with pytest.raises(ValueError, match="attestation mismatch"):
        verify_migration_receipt(receipt, trusted_public_key_pem=public_pem)


def test_migration_reloads_legacy_baseline_from_its_commit(tmp_path: Path):
    repo = tmp_path / "repo"
    baseline = repo / "config" / "aura_architecture_quality_baseline.json"
    _write(baseline, json.dumps({"score": 46.38, "metrics": {}, "include_roots": ["core"]}))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "add", "config/aura_architecture_quality_baseline.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "legacy"], cwd=repo, check=True)
    legacy_commit = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _write(baseline, json.dumps({"schema_version": 2, "score": 1.0}))

    loaded = _load_legacy_baseline(
        repo,
        legacy_commit=legacy_commit,
        baseline_path=baseline,
    )

    assert loaded["score"] == 46.38
    assert "schema_version" not in loaded


def test_self_modification_hook_rejects_architecture_regression(tmp_path: Path):
    from core.self_modification.safe_modification import SafeSelfModification

    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")
    _write(tmp_path / "core" / "b.py", "import core.a\nVALUE = core.a.VALUE\n")

    system = object.__new__(SafeSelfModification)
    system.code_base = tmp_path

    ok, message = asyncio.run(
        SafeSelfModification._run_architecture_quality_gate(
            system,
            "core/a.py",
            "import core.b\nVALUE = core.b.VALUE\n",
        )
    )

    assert not ok
    assert "new import cycle" in message


def test_syntax_invalid_overlay_fails_closed_instead_of_improving_graph(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "import core.b\n")
    _write(tmp_path / "core" / "b.py", "import core.a\n")

    result = ArchitectureQualityGate(tmp_path, include_roots=("core",)).evaluate_overlay(
        {"core/a.py": "def broken(:\n"},
        changed_paths=("core/a.py",),
    )

    assert not result.passed
    assert result.after.score == 0.0
    assert result.after.metrics.parse_error_count == 1
    assert any("unparseable source" in reason for reason in result.reasons)


def test_cycle_metrics_and_report_are_not_truncated(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    for index in range(45):
        _write(tmp_path / "core" / f"a{index}.py", f"import core.b{index}\n")
        _write(tmp_path / "core" / f"b{index}.py", f"import core.a{index}\n")

    report = score_codebase(
        tmp_path,
        include_roots=("core",),
        max_cycles_reported=3,
    )

    assert report.metrics.cycle_count == 45
    assert len(report.cycles) == 45
    assert sum(finding.code == "import_cycle" for finding in report.findings) == 45
    assert report.findings_complete
    assert report.findings_omitted == 0


def test_package_initializer_relative_import_resolves_inside_package(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "pkg" / "__init__.py", "from . import child\n")
    _write(tmp_path / "core" / "pkg" / "child.py", "VALUE = 1\n")

    report = score_codebase(tmp_path, include_roots=("core",))

    assert report.graph["core.pkg"] == ("core.pkg.child",)


def test_shadowed_module_file_keeps_its_lexical_relative_import_base(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "pkg.py", "from . import sibling\n")
    _write(tmp_path / "core" / "pkg" / "__init__.py", "VALUE = 1\n")
    _write(tmp_path / "core" / "sibling.py", "VALUE = 1\n")

    report = score_codebase(tmp_path, include_roots=("core",))

    assert report.graph["core.pkg.__file__"] == ("core", "core.sibling")
    assert any(
        finding.code == "ambiguous_module_owner"
        and finding.modules == ("core.pkg",)
        for finding in report.findings
    )


@pytest.mark.parametrize(
    "path",
    ("../core/a.py", "/core/a.py", "./core/a.py", "core//a.py", "core\\a.py"),
)
def test_overlay_paths_must_be_canonical_and_repository_relative(tmp_path: Path, path: str):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")

    with pytest.raises(ValueError):
        score_codebase(tmp_path, include_roots=("core",), overlay_content={path: "VALUE = 2\n"})


def test_overlay_path_cannot_escape_through_existing_symlink(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "core").symlink_to(outside, target_is_directory=True)

    try:
        with pytest.raises(ValueError, match="symlink"):
            score_codebase(
                tmp_path,
                include_roots=("core",),
                overlay_content={"core/new.py": "VALUE = 1\n"},
            )
    finally:
        outside.rmdir()


@pytest.mark.parametrize("include_root", ("../sibling", "/tmp", "./core", "core/../other"))
def test_include_roots_cannot_escape_or_alias_repository(tmp_path: Path, include_root: str):
    _write(tmp_path / "core" / "__init__.py", "")
    with pytest.raises(ValueError):
        score_codebase(tmp_path, include_roots=(include_root,))


def test_iterative_scc_handles_dependency_chains_beyond_recursion_limit():
    graph = {f"m{index}": (f"m{index + 1}",) for index in range(2500)}
    graph["m2500"] = ()

    assert _strongly_connected_components(graph) == ()


def test_report_collections_are_immutable_and_attested(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")
    report = score_codebase(tmp_path, include_roots=("core",))
    digest = report.attestation_sha256

    with pytest.raises(TypeError):
        report.line_counts["core/a.py"] = 999  # type: ignore[index]
    with pytest.raises(TypeError):
        report.graph["core.a"] = ("core",)  # type: ignore[index]

    assert report.attestation_sha256 == digest
    assert report.to_dict()["attestation_sha256"] == digest


def test_report_attestation_is_reproducible_across_checkout_paths(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        _write(root / "core" / "__init__.py", "")
        _write(root / "core" / "a.py", "VALUE = 1\n")

    first_report = score_codebase(first, include_roots=("core",))
    second_report = score_codebase(second, include_roots=("core",))

    assert first_report.root != second_report.root
    assert first_report.attestation_sha256 == second_report.attestation_sha256


def test_uncapped_debt_detects_regression_after_legacy_penalties_saturated(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    for index in range(125):
        _write(tmp_path / "core" / f"m{index}.py", "VALUE = 1\n")
    imports_100 = "".join(f"import core.m{index}\n" for index in range(100))
    imports_120 = "".join(f"import core.m{index}\n" for index in range(120))
    _write(tmp_path / "core" / "hub.py", imports_100)

    before = score_codebase(tmp_path, include_roots=("core",))
    after = score_codebase(
        tmp_path,
        include_roots=("core",),
        overlay_content={"core/hub.py": imports_120},
    )

    assert before.metrics.max_out_degree == 100
    assert after.metrics.max_out_degree == 120
    assert after.metrics.architecture_debt > before.metrics.architecture_debt
    assert after.score < before.score


def test_dependency_model_distinguishes_runtime_and_guarded_edges(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    for name in ("runtime", "types", "optional", "platform", "deferred", "dynamic"):
        _write(tmp_path / "core" / f"{name}.py", "VALUE = 1\n")
    _write(
        tmp_path / "core" / "subject.py",
        """import importlib
import sys
import core.runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import core.types
try:
    import core.optional
except ImportError:
    pass
if sys.platform == "darwin":
    import core.platform
importlib.import_module("core.dynamic")
name = "core.runtime"
importlib.import_module(name)
def later():
    import core.deferred
""",
    )

    report = score_codebase(tmp_path, include_roots=("core",))

    assert report.graph["core.subject"] == ("core.dynamic", "core.runtime")
    assert report.type_checking_graph["core.subject"] == ("core.types",)
    assert report.optional_graph["core.subject"] == ("core.optional",)
    assert report.conditional_graph["core.subject"] == ("core.platform",)
    assert report.deferred_graph["core.subject"] == ("core.deferred",)
    assert report.dynamic_graph["core.subject"] == ("core.dynamic",)
    assert report.metrics.unresolved_dynamic_imports == 1
    assert any(finding.code == "unresolved_dynamic_import" for finding in report.findings)


def test_overlay_tombstone_removes_module_and_incoming_edges(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "import core.b\n")
    _write(tmp_path / "core" / "b.py", "VALUE = 1\n")

    report = score_codebase(
        tmp_path,
        include_roots=("core",),
        overlay_content={"core/b.py": None},
    )

    assert "core.b" not in report.module_to_path
    assert "core.b" not in report.graph["core.a"]
    assert "core/b.py" not in report.line_counts
    assert report.metrics.unresolved_local_imports == 1

    result = ArchitectureQualityGate(tmp_path, include_roots=("core",)).evaluate_overlay(
        {"core/b.py": None},
        changed_paths=("core/b.py",),
    )
    assert not result.passed
    assert any("unresolved local import" in reason for reason in result.reasons)


def test_structural_metrics_do_not_treat_comments_as_executable_logic(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(
        tmp_path / "core" / "commentary.py",
        "".join(f"# documentation {index}\n" for index in range(200)) + "VALUE = 1\n",
    )
    _write(
        tmp_path / "core" / "logic.py",
        "def decide(a, b):\n    if a and b:\n        return 1\n    return 0\n",
    )

    report = score_codebase(tmp_path, include_roots=("core",), god_file_threshold=500)
    commentary = report.module_structures["core/commentary.py"]
    logic = report.module_structures["core/logic.py"]

    assert commentary.source_lines == 201
    assert commentary.code_lines == 1
    assert commentary.comment_lines == 200
    assert logic.branch_points >= 2
    assert report.metrics.max_code_lines < report.metrics.max_file_lines


def test_findings_materialize_all_detected_oversized_modules(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    for index in range(25):
        _write(
            tmp_path / "core" / f"big{index}.py",
            "\n".join(f"VALUE_{line} = {line}" for line in range(12)) + "\n",
        )

    report = score_codebase(tmp_path, include_roots=("core",), god_file_threshold=10)

    oversized = [f for f in report.findings if f.code == "structurally_oversized_module"]
    assert report.metrics.god_file_count == 25
    assert len(oversized) == 25


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("include_roots", ("core", "interface"), "include roots differ"),
        ("exclude_parts", ("__pycache__",), "exclusion scope differs"),
        ("god_file_threshold", 99, "threshold differs"),
    ),
)
def test_gate_rejects_incomparable_analysis_scope(
    tmp_path: Path,
    field: str,
    value: object,
    reason: str,
):
    _write(tmp_path / "core" / "__init__.py", "")
    report = score_codebase(tmp_path, include_roots=("core",))
    incomparable = replace(report, **{field: value})

    result = ArchitectureQualityGate(tmp_path, include_roots=("core",)).evaluate_reports(
        report,
        incomparable,
    )

    assert not result.passed
    assert any(reason in item for item in result.reasons)


def test_gate_rejects_cycle_replacement_even_when_aggregate_count_is_unchanged(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "import core.b\n")
    _write(tmp_path / "core" / "b.py", "import core.a\n")
    _write(tmp_path / "core" / "c.py", "VALUE = 1\n")
    _write(tmp_path / "core" / "d.py", "VALUE = 1\n")
    overlay = {
        "core/a.py": "VALUE = 1\n",
        "core/b.py": "VALUE = 1\n",
        "core/c.py": "import core.d\n",
        "core/d.py": "import core.c\n",
    }

    result = ArchitectureQualityGate(tmp_path, include_roots=("core",)).evaluate_overlay(
        overlay,
        changed_paths=overlay,
    )

    assert result.before.metrics.cycle_count == result.after.metrics.cycle_count == 1
    assert not result.passed
    assert any("new import cycle" in reason for reason in result.reasons)


def test_gate_binds_changed_paths_to_overlay_keys(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")

    with pytest.raises(ValueError, match="exactly match"):
        ArchitectureQualityGate(tmp_path, include_roots=("core",)).evaluate_overlay(
            {"core/a.py": "VALUE = 2\n"},
            changed_paths=(),
        )


def test_overlay_gate_reparses_only_changed_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _write(tmp_path / "core" / "a.py", "VALUE = 1\n")
    _write(tmp_path / "core" / "b.py", "from core.a import VALUE\n")
    scorer_module._clear_analysis_cache()
    original_parse = scorer_module.ast.parse
    parsed_paths: list[str] = []

    def counting_parse(source, filename="<unknown>", mode="exec", **kwargs):
        parsed_paths.append(str(filename))
        return original_parse(source, filename=filename, mode=mode, **kwargs)

    monkeypatch.setattr(scorer_module.ast, "parse", counting_parse)
    result = ArchitectureQualityGate(tmp_path, include_roots=("core",)).evaluate_overlay(
        {"core/a.py": "VALUE = 2\n"},
        changed_paths=("core/a.py",),
    )

    assert result.passed
    assert parsed_paths.count("core/a.py") == 2
    assert parsed_paths.count("core/b.py") == 1
    assert result.before.graph == result.after.graph


def test_deferred_cycle_is_measured_and_rejected_as_executable_coupling(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "def run():\n    return 1\n")
    _write(tmp_path / "core" / "b.py", "def run():\n    return 1\n")
    overlay = {
        "core/a.py": "def run():\n    import core.b\n    return core.b\n",
        "core/b.py": "def run():\n    import core.a\n    return core.a\n",
    }

    result = ArchitectureQualityGate(tmp_path, include_roots=("core",)).evaluate_overlay(
        overlay,
        changed_paths=overlay,
    )

    assert result.after.metrics.cycle_count == 0
    assert result.after.metrics.executable_cycle_count == 1
    assert not result.passed
    assert any("executable dependency cycle" in reason for reason in result.reasons)


def test_missing_from_import_and_invalid_relative_import_are_visible(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "a.py", "from core import missing\nfrom ...outside import value\n")

    report = score_codebase(tmp_path, include_roots=("core",))

    assert report.metrics.unresolved_local_imports == 1
    assert report.metrics.invalid_relative_imports == 1
    assert any(f.code == "unresolved_local_import" for f in report.findings)
    assert any(f.code == "invalid_relative_import" for f in report.findings)


def test_dynamic_import_requires_a_real_import_binding(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    _write(tmp_path / "core" / "target.py", "VALUE = 1\n")
    _write(
        tmp_path / "core" / "subject.py",
        "def import_module(name):\n    return name\nVALUE = import_module('core.target')\n",
    )

    report = score_codebase(tmp_path, include_roots=("core",))

    assert report.dynamic_graph["core.subject"] == ()
    assert report.graph["core.subject"] == ()


def test_python_encoding_cookie_is_honored(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    encoded = "# -*- coding: latin-1 -*-\nNAME = 'café'\n".encode("latin-1")
    (tmp_path / "core" / "latin.py").write_bytes(encoded)

    report = score_codebase(tmp_path, include_roots=("core",))

    assert report.metrics.parse_error_count == 0
    assert not any(f.path == "core/latin.py" for f in report.findings)


def test_report_normalizes_mutable_sequence_inputs(tmp_path: Path):
    _write(tmp_path / "core" / "__init__.py", "")
    report = score_codebase(tmp_path, include_roots=("core",))
    roots = ["core"]
    cycles = [["core.a", "core.b"]]
    findings = list(report.findings)
    normalized = replace(report, include_roots=roots, cycles=cycles, findings=findings)

    roots.append("interface")
    cycles[0].append("core.c")
    findings.clear()

    assert normalized.include_roots == ("core",)
    assert normalized.cycles == (("core.a", "core.b"),)
    assert isinstance(normalized.findings, tuple)
