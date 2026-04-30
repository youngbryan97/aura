"""Aura Reimplementation Lab — Spec-to-Code Reproduction Engine.

Clean-room module reconstruction subsystem that enables Aura to regenerate,
repair, and improve parts of herself from behavioral specifications.

Adapted from "Read the Paper, Write the Code" (Kohler et al. 2026):
    specification → blinded isolation → independent implementation
    → sandbox execution → deterministic comparison
    → discrepancy attribution → promotion / rejection

The authority is deterministic: tests + trace comparison + sandbox execution
+ safety invariants + promotion gate. Not an LLM judge.

Usage:
    from core.self_improvement import ReimplementationLab
    lab = ReimplementationLab(project_root=".")
    result = await lab.run_reconstruction("core/promotion/behavioral_contracts.py")
"""
from core.self_improvement.interface_contract import (
    AuditResult,
    BehavioralInvariant,
    CandidateModule,
    ClassSignature,
    ComparisonReport,
    DiscrepancyCategory,
    DiscrepancyItem,
    DiscrepancyReport,
    FunctionSignature,
    InterfaceContract,
    LabResult,
    ModuleSpec,
    PromotionVerdict,
    TestCase,
    TestVerdict,
    TraceExample,
)
from core.self_improvement.spec_extractor import SpecExtractor
from core.self_improvement.blinded_workspace import BlindedWorkspace, BlindedWorkspaceFactory
from core.self_improvement.candidate_builder import (
    CandidateBuilder,
    CodeGenerator,
    PromptBuilder,
    StubGenerator,
)
from core.self_improvement.deterministic_comparator import DeterministicComparator
from core.self_improvement.discrepancy_attributor import DiscrepancyAttributor
from core.self_improvement.hardcoding_auditor import HardcodingAuditor
from core.self_improvement.guardrail_auditor import GuardrailAuditor
from core.self_improvement.promotion_gate import LabPromotionGate
from core.self_improvement.reimplementation_lab import ReimplementationLab

__all__ = [
    # Pipeline
    "ReimplementationLab",
    # Components
    "SpecExtractor",
    "BlindedWorkspace",
    "BlindedWorkspaceFactory",
    "CandidateBuilder",
    "CodeGenerator",
    "DeterministicComparator",
    "DiscrepancyAttributor",
    "HardcodingAuditor",
    "GuardrailAuditor",
    "LabPromotionGate",
    "PromptBuilder",
    "StubGenerator",
    # Data types
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
