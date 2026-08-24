"""The RLC figures gate has to fail on the defect it was built for.

A gate that cannot match reports green forever. `make rlc-figures` exists
because docs/INTRINSIC_RECURRENCE.md carried a "maximum 318 ms" that no receipt
in the repository contained, so the first test here reinstates exactly that
sentence and requires a finding.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "verify_rlc_figures.py"


def _gate():
    spec = importlib.util.spec_from_file_location("verify_rlc_figures", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # ``@dataclass`` resolves annotations through ``sys.modules``; a module
    # executed without being registered there raises inside dataclasses.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _gate()


def test_the_tree_is_clean(gate):
    assert gate.check() == []


def test_a_fabricated_latency_is_a_finding(gate):
    findings = gate._unsourced_latencies(
        {"docs/INTRINSIC_RECURRENCE.md": "at a median 47 ms and a maximum 318 ms"}
    )
    assert [f["figure"] for f in findings] == ["318 ms"]
    assert findings[0]["problem"] == "latency_unsourced"


def test_a_measured_latency_is_not_a_finding(gate):
    active = json.loads(
        (
            ROOT
            / "artifacts/closeout/latent_cortex/cp568_semantic_neural_active_r1"
            / "runtime_verification.json"
        ).read_text()
    )
    written = f"a median {active['p50_latency_ms']:.3f} ms"
    assert gate._unsourced_latencies({"docs/INTRINSIC_RECURRENCE.md": written}) == []


def test_latency_prose_outside_the_rlc_section_is_left_alone(gate):
    readme = "## Something Else\n\nBudget 9999 ms for the first token.\n"
    assert gate._unsourced_latencies({"README.md": readme}) == []


def test_every_quoted_figure_recomputes_from_its_artifact(gate):
    for figure in gate.FIGURES:
        assert figure.recompute(gate._load(figure.source)) == figure.quoted, (
            f"{figure.quoted} no longer matches {figure.source}"
        )


def test_the_ceiling_claim_is_derived_not_asserted(gate):
    adjudication = gate._load(gate.CP566)
    ordinary = gate._ordinary_by_family(adjudication)
    per_family = int(adjudication["tasks_per_family"])
    # The family that "gained nothing" is the one ordinary decode had already
    # answered completely. Stating the gain without this reads as a failure.
    assert ordinary["frontier_misleading_premise"] == per_family
    assert sum(ordinary.values()) == (
        adjudication["independent_exact_by_arm"]["ordinary_base"]
    )


def test_a_drifted_arm_count_is_a_finding(gate, tmp_path, monkeypatch):
    tampered = gate._load(gate.CP566)
    tampered["independent_exact_by_arm"]["treatment"] = 59
    target = tmp_path / "adjudication.json"
    target.write_text(json.dumps(tampered))

    real_load = gate._load

    def fake_load(relative):
        if relative == gate.CP566:
            return json.loads(target.read_text())
        return real_load(relative)

    monkeypatch.setattr(gate, "_load", fake_load)
    problems = {f["problem"] for f in gate.check()}
    assert "figure_drifted" in problems


def test_missing_evidence_is_reported_rather_than_skipped(gate, monkeypatch):
    real_load = gate._load

    def fake_load(relative):
        if relative == gate.CP567_RUNTIME:
            raise gate.MissingEvidenceError(relative)
        return real_load(relative)

    monkeypatch.setattr(gate, "_load", fake_load)
    problems = {f["problem"] for f in gate.check()}
    assert "evidence_missing" in problems


def test_the_p_value_renders_the_way_the_documents_write_it(gate):
    assert gate._sci(5.684341886080802e-14) == "5.7 × 10⁻¹⁴"
