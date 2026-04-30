"""core/self_improvement/interface_contract.py — Behavioral specification data classes.

These structures carry the full behavioral contract of an Aura module:
its public interface, docstrings, invariants, test cases, and trace examples.
They are the "blinded spec" that the Reimplementation Lab uses to drive
clean-room reconstruction — the candidate builder sees *only* these,
never the original source code.

Adapted from the "Read the Paper, Write the Code" pattern where the
specification replaces the paper's methods section and the interface
contract replaces the table template.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple


class DiscrepancyCategory(Enum):
    """Root-cause categories for reimplementation failures.

    Directly adapted from the paper's error taxonomy:
    agent error, extraction error, original-paper underspecification,
    missing data, environment error, unknown.
    """

    AGENT_ERROR = "agent_error"
    SPEC_UNDERSPECIFICATION = "spec_underspecification"
    TEST_DEFICIENCY = "test_deficiency"
    DATA_MISMATCH = "data_mismatch"
    ENVIRONMENT_ERROR = "environment_error"
    UNKNOWN = "unknown"


class PromotionVerdict(Enum):
    """Outcome of the promotion gate evaluation."""

    PROMOTE = "promote"
    RETRY = "retry"
    REJECT = "reject"


@dataclass(frozen=True)
class FunctionSignature:
    """Signature of a single public function or method."""

    name: str
    parameters: Tuple[str, ...] = ()
    is_async: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    return_annotation: Optional[str] = None
    docstring: Optional[str] = None
    decorators: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassSignature:
    """Signature of a public class."""

    name: str
    bases: Tuple[str, ...] = ()
    methods: Tuple[FunctionSignature, ...] = ()
    docstring: Optional[str] = None
    decorators: Tuple[str, ...] = ()


@dataclass
class InterfaceContract:
    """The public surface of a module — everything visible to importers.

    This is the "table template" in the paper analogy: the shape of the
    expected output without the actual values.
    """

    module_path: str
    functions: List[FunctionSignature] = field(default_factory=list)
    classes: List[ClassSignature] = field(default_factory=list)
    constants: Dict[str, str] = field(default_factory=dict)  # name → type string
    all_names: FrozenSet[str] = field(default_factory=frozenset)
    imports: List[str] = field(default_factory=list)

    @property
    def public_names(self) -> FrozenSet[str]:
        """All public names this module exports."""
        if self.all_names:
            return self.all_names
        names = set()
        for f in self.functions:
            names.add(f.name)
        for c in self.classes:
            names.add(c.name)
        names.update(self.constants.keys())
        return frozenset(names)


@dataclass
class BehavioralInvariant:
    """A named invariant with a callable check function.

    The check function receives a mapping of metrics/outputs and returns
    True if the invariant holds.
    """

    name: str
    description: str
    check_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    check_source: str = ""  # Human-readable source for the check
    critical: bool = True


@dataclass
class TestCase:
    """A single test case extracted from the test suite."""

    __test__ = False  # Prevent pytest from collecting this dataclass

    name: str
    source: str  # The test function source code
    file_path: str = ""
    expected_outcome: str = "pass"  # "pass", "fail", "skip"


@dataclass
class TraceExample:
    """Historical runtime trace for a module function."""

    function_name: str
    input_repr: str
    output_repr: str
    latency_ms: float = 0.0
    timestamp: float = 0.0


@dataclass
class ModuleSpec:
    """Full behavioral specification of a module.

    This is the complete "methods section" that the candidate builder
    uses to generate a replacement implementation. It contains everything
    needed to reconstruct the module's behavior *except* the original code.
    """

    module_path: str
    module_name: str
    interface: InterfaceContract
    module_docstring: str = ""
    invariants: List[BehavioralInvariant] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)
    trace_examples: List[TraceExample] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    extracted_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        """Human-readable summary of the spec."""
        parts = [
            f"Module: {self.module_path}",
            f"  Functions: {len(self.interface.functions)}",
            f"  Classes: {len(self.interface.classes)}",
            f"  Constants: {len(self.interface.constants)}",
            f"  Invariants: {len(self.invariants)}",
            f"  Test Cases: {len(self.test_cases)}",
            f"  Trace Examples: {len(self.trace_examples)}",
            f"  Dependencies: {len(self.dependencies)}",
        ]
        return "\n".join(parts)


@dataclass
class CandidateModule:
    """A generated replacement implementation."""

    source_code: str
    module_path: str
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time_s: float = 0.0
    attempt_number: int = 1


@dataclass
class TestVerdict:
    """Result of running a single test against the candidate."""

    __test__ = False  # Prevent pytest from collecting this dataclass

    test_name: str
    passed: bool
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    latency_ms: float = 0.0


@dataclass
class ComparisonReport:
    """Cell-by-cell behavioral comparison between candidate and spec."""

    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    verdicts: List[TestVerdict] = field(default_factory=list)
    aggregate_pass_rate: float = 0.0
    latency_budget_ok: bool = True
    syntax_valid: bool = True
    imports_valid: bool = True
    public_surface_preserved: bool = True
    metrics: Dict[str, float] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    @property
    def all_passed(self) -> bool:
        return self.failed_tests == 0 and self.syntax_valid and self.imports_valid


@dataclass
class DiscrepancyItem:
    """A single attributed discrepancy."""

    test_name: str
    category: DiscrepancyCategory
    description: str
    severity: str = "medium"  # "low", "medium", "high", "critical"


@dataclass
class DiscrepancyReport:
    """Attributed root-cause report for all failures."""

    items: List[DiscrepancyItem] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    @property
    def has_agent_errors(self) -> bool:
        return any(i.category == DiscrepancyCategory.AGENT_ERROR for i in self.items)

    @property
    def has_spec_issues(self) -> bool:
        return any(i.category == DiscrepancyCategory.SPEC_UNDERSPECIFICATION for i in self.items)


@dataclass
class AuditResult:
    """Result of hardcoding or guardrail audit."""

    passed: bool
    violations: List[str] = field(default_factory=list)
    audit_type: str = ""
    generated_at: float = field(default_factory=time.time)


@dataclass
class LabResult:
    """Complete result of a reimplementation lab run."""

    success: bool
    module_path: str
    candidate: Optional[CandidateModule] = None
    comparison: Optional[ComparisonReport] = None
    discrepancy: Optional[DiscrepancyReport] = None
    hardcoding_audit: Optional[AuditResult] = None
    guardrail_audit: Optional[AuditResult] = None
    verdict: PromotionVerdict = PromotionVerdict.REJECT
    attempts: int = 0
    total_time_s: float = 0.0
    receipt_id: Optional[str] = None
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "module_path": self.module_path,
            "verdict": self.verdict.value,
            "attempts": self.attempts,
            "total_time_s": self.total_time_s,
            "receipt_id": self.receipt_id,
            "comparison": {
                "total_tests": self.comparison.total_tests,
                "passed": self.comparison.passed_tests,
                "failed": self.comparison.failed_tests,
                "pass_rate": self.comparison.aggregate_pass_rate,
            }
            if self.comparison
            else None,
            "discrepancy_summary": self.discrepancy.summary
            if self.discrepancy
            else None,
            "hardcoding_passed": self.hardcoding_audit.passed
            if self.hardcoding_audit
            else None,
            "guardrail_passed": self.guardrail_audit.passed
            if self.guardrail_audit
            else None,
        }


__all__ = [
    "AuditResult",
    "BehavioralInvariant",
    "CandidateModule",
    "ClassSignature",
    "ComparisonReport",
    "DiscrepancyCategory",
    "DiscrepancyItem",
    "DiscrepancyReport",
    "FunctionSignature",
    "InterfaceContract",
    "LabResult",
    "ModuleSpec",
    "PromotionVerdict",
    "TestCase",
    "TestVerdict",
    "TraceExample",
]
