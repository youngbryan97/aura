"""Blinded module reconstruction tests for the Reimplementation Lab.

Covers: spec extraction, blinding, comparison, attribution, auditing,
promotion gate, and full pipeline integration.
"""
import ast
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.self_improvement.interface_contract import (
    AuditResult, CandidateModule, ComparisonReport, DiscrepancyCategory,
    DiscrepancyItem, DiscrepancyReport, FunctionSignature, InterfaceContract,
    LabResult, ModuleSpec, PromotionVerdict, TestCase, TestVerdict,
)
from core.self_improvement.spec_extractor import SpecExtractor
from core.self_improvement.blinded_workspace import BlindedWorkspaceFactory, BlindedWorkspace
from core.self_improvement.candidate_builder import CandidateBuilder, StubGenerator
from core.self_improvement.deterministic_comparator import DeterministicComparator
from core.self_improvement.discrepancy_attributor import DiscrepancyAttributor
from core.self_improvement.hardcoding_auditor import HardcodingAuditor
from core.self_improvement.guardrail_auditor import GuardrailAuditor
from core.self_improvement.promotion_gate import LabPromotionGate
from core.self_improvement.reimplementation_lab import ReimplementationLab


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def project_root():
    return str(PROJECT_ROOT)

@pytest.fixture
def spec_extractor(project_root):
    return SpecExtractor(project_root=project_root)

@pytest.fixture
def sample_spec():
    return ModuleSpec(
        module_path="core/sample/module.py",
        module_name="module",
        interface=InterfaceContract(
            module_path="core/sample/module.py",
            functions=[FunctionSignature(name="add", parameters=("a: int", "b: int"),
                                         return_annotation="int", docstring="Add two numbers.")],
            classes=[], constants={},
            all_names=frozenset({"add"}), imports=[],
        ),
        module_docstring="Sample module.", invariants=[], test_cases=[], dependencies=[],
    )

@pytest.fixture
def sample_candidate_good():
    return CandidateModule(
        source_code='def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n\n__all__ = ["add"]\n',
        module_path="core/sample/module.py",
    )

@pytest.fixture
def sample_candidate_hardcoded():
    return CandidateModule(
        source_code='def add(a: int, b: int) -> int:\n    return 3.141592653589793\n',
        module_path="core/sample/module.py",
    )

@pytest.fixture
def sample_candidate_eval():
    return CandidateModule(
        source_code='def add(a: int, b: int) -> int:\n    return eval("a + b")\n',
        module_path="core/sample/module.py",
    )


# ── Spec Extractor Tests ─────────────────────────────────────────────────

class TestSpecExtractor:
    def test_extract_behavioral_contracts(self, spec_extractor):
        spec = spec_extractor.extract("core/promotion/behavioral_contracts.py")
        assert spec.module_path == "core/promotion/behavioral_contracts.py"
        assert spec.module_name == "behavioral_contracts"
        assert len(spec.interface.classes) > 0
        class_names = {c.name for c in spec.interface.classes}
        assert "BehavioralContract" in class_names
        assert "BehavioralContractSuite" in class_names

    def test_extract_all_names(self, spec_extractor):
        spec = spec_extractor.extract("core/promotion/behavioral_contracts.py")
        assert "BehavioralContract" in spec.interface.all_names
        assert "synthesize_contracts_from_history" in spec.interface.all_names

    def test_extract_functions(self, spec_extractor):
        spec = spec_extractor.extract("core/promotion/behavioral_contracts.py")
        func_names = {f.name for f in spec.interface.functions}
        assert "synthesize_contracts_from_history" in func_names

    def test_extract_module_docstring(self, spec_extractor):
        spec = spec_extractor.extract("core/promotion/behavioral_contracts.py")
        assert "Behavioral contracts" in spec.module_docstring

    def test_extract_dependencies(self, spec_extractor):
        spec = spec_extractor.extract("core/promotion/behavioral_contracts.py")
        assert "operator" in spec.dependencies
        assert "time" in spec.dependencies

    def test_extract_imports(self, spec_extractor):
        spec = spec_extractor.extract("core/promotion/behavioral_contracts.py")
        assert any("operator" in imp for imp in spec.interface.imports)

    def test_extract_nonexistent_raises(self, spec_extractor):
        with pytest.raises(FileNotFoundError):
            spec_extractor.extract("core/nonexistent/module.py")

    def test_extract_class_methods(self, spec_extractor):
        spec = spec_extractor.extract("core/promotion/behavioral_contracts.py")
        bc = next(c for c in spec.interface.classes if c.name == "BehavioralContract")
        method_names = {m.name for m in bc.methods}
        assert "evaluate" in method_names
        assert "to_dict" in method_names


# ── Blinded Workspace Tests ──────────────────────────────────────────────

class TestBlindedWorkspace:
    def test_create_workspace(self, project_root, sample_spec):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        try:
            assert ws.workspace_dir.exists()
            assert ws.stub_path.exists()
            stub = ws.stub_path.read_text()
            assert "NotImplementedError" in stub
            assert "add" in stub
        finally:
            ws.cleanup()

    def test_forbidden_path_detection(self, project_root, sample_spec):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        try:
            assert ws.is_forbidden("core/sample/module.py")
            assert not ws.is_forbidden("core/other/module.py")
        finally:
            ws.cleanup()

    def test_access_logging(self, project_root, sample_spec):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        try:
            ws.record_access("/some/path")
            assert "/some/path" in ws.access_log
        finally:
            ws.cleanup()

    def test_cleanup(self, project_root, sample_spec):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        d = ws.workspace_dir
        ws.cleanup()
        assert not d.exists()

    def test_spec_reference_file(self, project_root, sample_spec):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        try:
            spec_ref = ws.workspace_dir / "SPEC.txt"
            assert spec_ref.exists()
            assert "module" in spec_ref.read_text().lower()
        finally:
            ws.cleanup()


# ── Candidate Builder Tests ──────────────────────────────────────────────

class TestCandidateBuilder:
    @pytest.mark.asyncio
    async def test_stub_generator(self, sample_spec, project_root):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        try:
            builder = CandidateBuilder(generator=StubGenerator())
            candidate = await builder.build(sample_spec, ws, attempt=1)
            assert candidate.source_code
            assert candidate.module_path == sample_spec.module_path
            assert candidate.attempt_number == 1
            assert candidate.generation_time_s >= 0
        finally:
            ws.cleanup()

    def test_prompt_builder(self):
        from core.self_improvement.candidate_builder import PromptBuilder
        spec = ModuleSpec(
            module_path="core/test.py", module_name="test",
            interface=InterfaceContract(
                module_path="core/test.py",
                functions=[FunctionSignature(name="foo", parameters=("x",))],
            ),
            module_docstring="Test module.",
        )
        prompt = PromptBuilder().build(spec)
        assert "foo" in prompt
        assert "Test module." in prompt
        assert "Reimplementation" in prompt


# ── Hardcoding Auditor Tests ─────────────────────────────────────────────

class TestHardcodingAuditor:
    def test_clean_code_passes(self, sample_spec, sample_candidate_good, project_root):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        try:
            auditor = HardcodingAuditor()
            result = auditor.audit(sample_candidate_good, sample_spec, ws)
            assert result.passed
            assert len(result.violations) == 0
        finally:
            ws.cleanup()

    def test_hardcoded_float_detected(self, sample_spec, sample_candidate_hardcoded, project_root):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        try:
            auditor = HardcodingAuditor()
            result = auditor.audit(sample_candidate_hardcoded, sample_spec, ws)
            assert not result.passed
            assert any("HARDCODED_RETURN" in v for v in result.violations)
        finally:
            ws.cleanup()

    def test_eval_detected(self, sample_spec, sample_candidate_eval, project_root):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        try:
            auditor = HardcodingAuditor()
            result = auditor.audit(sample_candidate_eval, sample_spec, ws)
            assert not result.passed
            assert any("EVAL_EXEC" in v for v in result.violations)
        finally:
            ws.cleanup()

    def test_forbidden_access_detected(self, sample_spec, sample_candidate_good, project_root):
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(sample_spec, "core/sample/module.py")
        try:
            ws.record_access("core/sample/module.py")
            auditor = HardcodingAuditor()
            result = auditor.audit(sample_candidate_good, sample_spec, ws)
            assert not result.passed
            assert any("FORBIDDEN_ACCESS" in v for v in result.violations)
        finally:
            ws.cleanup()

    def test_constant_return_detected(self, project_root):
        spec = ModuleSpec(
            module_path="m.py", module_name="m",
            interface=InterfaceContract(
                module_path="m.py",
                functions=[FunctionSignature(name="compute", parameters=("x: int",))],
            ),
        )
        candidate = CandidateModule(
            source_code='def compute(x: int):\n    """Compute something."""\n    return 42\n',
            module_path="m.py",
        )
        factory = BlindedWorkspaceFactory(project_root=project_root)
        ws = factory.create(spec, "m.py")
        try:
            auditor = HardcodingAuditor()
            result = auditor.audit(candidate, spec, ws)
            assert not result.passed
            assert any("CONSTANT_RETURN" in v for v in result.violations)
        finally:
            ws.cleanup()


# ── Guardrail Auditor Tests ──────────────────────────────────────────────

class TestGuardrailAuditor:
    def test_clean_code_passes(self, project_root, sample_candidate_good):
        auditor = GuardrailAuditor(project_root=project_root)
        result = auditor.audit(sample_candidate_good, "core/sample/module.py")
        assert result.passed

    def test_protected_module_rejected(self, project_root):
        candidate = CandidateModule(source_code="pass\n", module_path="constitutional_guard.py")
        auditor = GuardrailAuditor(project_root=project_root)
        result = auditor.audit(candidate, "core/safety/constitutional_guard.py")
        assert not result.passed
        assert any("PROTECTED_MODULE" in v for v in result.violations)

    def test_subprocess_import_rejected(self, project_root):
        candidate = CandidateModule(
            source_code="import subprocess\ndef run(): subprocess.run(['ls'])\n",
            module_path="core/sample/module.py",
        )
        auditor = GuardrailAuditor(project_root=project_root)
        result = auditor.audit(candidate, "core/sample/module.py")
        assert not result.passed
        assert any("UNSAFE_IMPORT" in v for v in result.violations)

    def test_syntax_error_rejected(self, project_root):
        candidate = CandidateModule(source_code="def broken(:\n", module_path="m.py")
        auditor = GuardrailAuditor(project_root=project_root)
        result = auditor.audit(candidate, "m.py")
        assert not result.passed
        assert any("SYNTAX_ERROR" in v for v in result.violations)


# ── Discrepancy Attributor Tests ─────────────────────────────────────────

class TestDiscrepancyAttributor:
    def test_assertion_error_is_agent_error(self):
        report = ComparisonReport(
            verdicts=[TestVerdict(test_name="test_foo", passed=False,
                                  error_message="AssertionError: 1 != 2")],
        )
        spec = ModuleSpec(module_path="m.py", module_name="m",
                          interface=InterfaceContract(module_path="m.py"))
        attributor = DiscrepancyAttributor()
        result = attributor.attribute(report, spec)
        assert len(result.items) == 1
        assert result.items[0].category == DiscrepancyCategory.AGENT_ERROR

    def test_import_error_is_environment(self):
        report = ComparisonReport(
            verdicts=[TestVerdict(test_name="test_bar", passed=False,
                                  error_message="ModuleNotFoundError: No module named 'xyz'")],
        )
        spec = ModuleSpec(module_path="m.py", module_name="m",
                          interface=InterfaceContract(module_path="m.py"))
        attributor = DiscrepancyAttributor()
        result = attributor.attribute(report, spec)
        assert result.items[0].category == DiscrepancyCategory.ENVIRONMENT_ERROR

    def test_fixture_error_is_test_deficiency(self):
        report = ComparisonReport(
            verdicts=[TestVerdict(test_name="test_baz", passed=False,
                                  error_message="fixture 'db' not found")],
        )
        spec = ModuleSpec(module_path="m.py", module_name="m",
                          interface=InterfaceContract(module_path="m.py"))
        attributor = DiscrepancyAttributor()
        result = attributor.attribute(report, spec)
        assert result.items[0].category == DiscrepancyCategory.TEST_DEFICIENCY

    def test_syntax_invalid_flagged(self):
        report = ComparisonReport(syntax_valid=False, verdicts=[])
        spec = ModuleSpec(module_path="m.py", module_name="m",
                          interface=InterfaceContract(module_path="m.py"))
        attributor = DiscrepancyAttributor()
        result = attributor.attribute(report, spec)
        assert any(i.category == DiscrepancyCategory.AGENT_ERROR for i in result.items)

    def test_summary_counts(self):
        report = ComparisonReport(
            verdicts=[
                TestVerdict(test_name="t1", passed=False, error_message="AssertionError"),
                TestVerdict(test_name="t2", passed=False, error_message="ModuleNotFoundError"),
                TestVerdict(test_name="t3", passed=True),
            ],
        )
        spec = ModuleSpec(module_path="m.py", module_name="m",
                          interface=InterfaceContract(module_path="m.py"))
        attributor = DiscrepancyAttributor()
        result = attributor.attribute(report, spec)
        assert result.summary.get("agent_error", 0) >= 1
        assert result.summary.get("environment_error", 0) >= 1


# ── Promotion Gate Tests ─────────────────────────────────────────────────

class TestLabPromotionGate:
    def test_all_pass_promotes(self):
        gate = LabPromotionGate(emit_receipts=False)
        comparison = ComparisonReport(
            total_tests=5, passed_tests=5, aggregate_pass_rate=1.0,
            syntax_valid=True, public_surface_preserved=True,
        )
        audit_ok = AuditResult(passed=True, audit_type="hardcoding")
        guardrail_ok = AuditResult(passed=True, audit_type="guardrail")
        disc = DiscrepancyReport()
        result = gate.evaluate(comparison, audit_ok, guardrail_ok, disc)
        assert result == PromotionVerdict.PROMOTE

    def test_hardcoding_fail_rejects(self):
        gate = LabPromotionGate(emit_receipts=False)
        comparison = ComparisonReport(
            total_tests=5, passed_tests=5, aggregate_pass_rate=1.0,
            syntax_valid=True, public_surface_preserved=True,
        )
        audit_fail = AuditResult(passed=False, violations=["HARDCODED"], audit_type="hardcoding")
        guardrail_ok = AuditResult(passed=True, audit_type="guardrail")
        disc = DiscrepancyReport()
        result = gate.evaluate(comparison, audit_fail, guardrail_ok, disc)
        assert result == PromotionVerdict.REJECT

    def test_guardrail_fail_rejects(self):
        gate = LabPromotionGate(emit_receipts=False)
        comparison = ComparisonReport(syntax_valid=True, public_surface_preserved=True)
        audit_ok = AuditResult(passed=True, audit_type="hardcoding")
        guardrail_fail = AuditResult(passed=False, violations=["PROTECTED"], audit_type="guardrail")
        disc = DiscrepancyReport()
        result = gate.evaluate(comparison, audit_ok, guardrail_fail, disc)
        assert result == PromotionVerdict.REJECT

    def test_partial_pass_retries(self):
        gate = LabPromotionGate(emit_receipts=False)
        comparison = ComparisonReport(
            total_tests=5, passed_tests=3, aggregate_pass_rate=0.6,
            syntax_valid=True, public_surface_preserved=True,
        )
        audit_ok = AuditResult(passed=True, audit_type="hardcoding")
        guardrail_ok = AuditResult(passed=True, audit_type="guardrail")
        disc = DiscrepancyReport(items=[
            DiscrepancyItem(test_name="t1", category=DiscrepancyCategory.AGENT_ERROR,
                            description="wrong"),
        ], summary={"agent_error": 1})
        result = gate.evaluate(comparison, audit_ok, guardrail_ok, disc)
        assert result == PromotionVerdict.RETRY

    def test_syntax_invalid_retries(self):
        gate = LabPromotionGate(emit_receipts=False)
        comparison = ComparisonReport(syntax_valid=False, public_surface_preserved=True)
        audit_ok = AuditResult(passed=True, audit_type="hardcoding")
        guardrail_ok = AuditResult(passed=True, audit_type="guardrail")
        disc = DiscrepancyReport()
        result = gate.evaluate(comparison, audit_ok, guardrail_ok, disc)
        assert result == PromotionVerdict.RETRY

    def test_spec_underspec_rejects(self):
        gate = LabPromotionGate(emit_receipts=False)
        comparison = ComparisonReport(
            total_tests=5, passed_tests=2, aggregate_pass_rate=0.4,
            syntax_valid=True, public_surface_preserved=True,
        )
        audit_ok = AuditResult(passed=True, audit_type="hardcoding")
        guardrail_ok = AuditResult(passed=True, audit_type="guardrail")
        disc = DiscrepancyReport(items=[
            DiscrepancyItem(test_name="t1", category=DiscrepancyCategory.SPEC_UNDERSPECIFICATION,
                            description="ambiguous spec"),
        ], summary={"spec_underspecification": 1})
        result = gate.evaluate(comparison, audit_ok, guardrail_ok, disc)
        assert result == PromotionVerdict.REJECT


# ── Deterministic Comparator Tests ───────────────────────────────────────

class TestDeterministicComparator:
    def test_syntax_check(self):
        comparator = DeterministicComparator()
        # Good syntax
        good = ast.parse("def foo(): pass")
        assert good is not None
        # Bad syntax detected
        candidate = CandidateModule(source_code="def broken(:\n", module_path="m.py")
        spec = ModuleSpec(module_path="m.py", module_name="m",
                          interface=InterfaceContract(module_path="m.py"))
        factory = BlindedWorkspaceFactory()
        ws = factory.create(spec, "m.py")
        try:
            report = asyncio.run(comparator.compare(candidate, spec, ws))
            assert not report.syntax_valid
            assert not report.all_passed
        finally:
            ws.cleanup()

    def test_public_surface_check(self):
        comparator = DeterministicComparator()
        spec = ModuleSpec(
            module_path="m.py", module_name="m",
            interface=InterfaceContract(
                module_path="m.py",
                functions=[FunctionSignature(name="required_fn", parameters=())],
                all_names=frozenset({"required_fn"}),
            ),
        )
        # Candidate missing the required function
        candidate = CandidateModule(
            source_code="def other_fn(): pass\n", module_path="m.py",
        )
        factory = BlindedWorkspaceFactory()
        ws = factory.create(spec, "m.py")
        try:
            report = asyncio.run(comparator.compare(candidate, spec, ws))
            assert not report.public_surface_preserved
        finally:
            ws.cleanup()


# ── Full Pipeline Integration Test ───────────────────────────────────────

class TestReimplementationLabIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_stub_generator(self, project_root):
        """End-to-end: spec→blind→build→audit→compare→attribute→decide."""
        lab = ReimplementationLab(
            project_root=project_root,
            generator=StubGenerator(),
            max_attempts=1,
        )
        # Use a real module but StubGenerator won't produce working code
        result = await lab.run_reconstruction("core/promotion/behavioral_contracts.py")
        # StubGenerator produces stubs, so it won't pass tests
        assert isinstance(result, LabResult)
        assert result.module_path == "core/promotion/behavioral_contracts.py"
        assert result.attempts >= 1
        assert result.total_time_s > 0

    @pytest.mark.asyncio
    async def test_pipeline_nonexistent_module(self, project_root):
        lab = ReimplementationLab(project_root=project_root, generator=StubGenerator())
        result = await lab.run_reconstruction("core/nonexistent.py")
        assert not result.success
        assert result.verdict == PromotionVerdict.REJECT

    @pytest.mark.asyncio
    async def test_lab_result_serialization(self, project_root):
        lab = ReimplementationLab(
            project_root=project_root, generator=StubGenerator(), max_attempts=1,
        )
        result = await lab.run_reconstruction("core/promotion/behavioral_contracts.py")
        d = result.to_dict()
        assert "success" in d
        assert "module_path" in d
        assert "verdict" in d
        assert "attempts" in d


# ── Data Class Tests ─────────────────────────────────────────────────────

class TestDataClasses:
    def test_module_spec_summary(self, sample_spec):
        s = sample_spec.summary()
        assert "Functions: 1" in s
        assert "module" in s.lower()

    def test_comparison_report_all_passed(self):
        r = ComparisonReport(total_tests=3, passed_tests=3, syntax_valid=True, imports_valid=True)
        assert r.all_passed

    def test_comparison_report_not_all_passed(self):
        r = ComparisonReport(total_tests=3, passed_tests=2, failed_tests=1,
                              syntax_valid=True, imports_valid=True)
        assert not r.all_passed

    def test_discrepancy_report_has_agent_errors(self):
        r = DiscrepancyReport(items=[
            DiscrepancyItem(test_name="t", category=DiscrepancyCategory.AGENT_ERROR, description="x"),
        ])
        assert r.has_agent_errors
        assert not r.has_spec_issues

    def test_lab_result_to_dict(self):
        r = LabResult(success=True, module_path="m.py", verdict=PromotionVerdict.PROMOTE, attempts=1)
        d = r.to_dict()
        assert d["success"] is True
        assert d["verdict"] == "promote"

    def test_interface_contract_public_names(self):
        ic = InterfaceContract(
            module_path="m.py",
            functions=[FunctionSignature(name="foo")],
            classes=[],
        )
        assert "foo" in ic.public_names

    def test_interface_contract_all_names_override(self):
        ic = InterfaceContract(
            module_path="m.py",
            functions=[FunctionSignature(name="foo")],
            all_names=frozenset({"bar"}),
        )
        assert ic.public_names == frozenset({"bar"})
