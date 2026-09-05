"""core/self_improvement/reimplementation_lab.py — Top-level pipeline orchestrator.

The full spec→blind→build→compare→audit→attribute→promote loop.

This is the Aura-native version of the paper's agentic reproduction system:
  Aura observes a failing module
  → extracts the behavioral spec from docs, tests, traces, and interfaces
  → blinds the old implementation
  → generates a clean replacement candidate
  → runs deterministic tests
  → compares outputs to expected behavior
  → diagnoses discrepancy source
  → either rejects, retries, or promotes

Usage:
    lab = ReimplementationLab(project_root="/path/to/aura")
    result = await lab.run_reconstruction("core/promotion/behavioral_contracts.py")
    if result.success:
        print(f"Module reconstructed: {result.verdict}")
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.self_improvement.blinded_workspace import BlindedWorkspaceFactory
from core.self_improvement.candidate_builder import CandidateBuilder, CodeGenerator
from core.self_improvement.deterministic_comparator import DeterministicComparator
from core.self_improvement.discrepancy_attributor import DiscrepancyAttributor
from core.self_improvement.guardrail_auditor import GuardrailAuditor
from core.self_improvement.hardcoding_auditor import HardcodingAuditor
from core.self_improvement.interface_contract import (
    LabResult,
    PromotionVerdict,
)
from core.self_improvement.promotion_gate import LabPromotionGate
from core.self_improvement.spec_extractor import SpecExtractor
from core.service_names import ServiceNames

logger = logging.getLogger("Aura.ReimplementationLab")


class ReimplementationLab:
    """Aura's internal engineering laboratory for clean-room module reconstruction.

    The authority is deterministic: tests + trace comparison + sandbox execution
    + safety invariants + promotion gate. Not an LLM judge.
    """

    def __init__(
        self,
        project_root: str | None = None,
        generator: CodeGenerator | None = None,
        max_attempts: int = 3,
        test_timeout: int = 30,
        min_pass_rate: float = 0.95,
    ):
        self.project_root = project_root or "."
        self.max_attempts = max_attempts

        # Pipeline components
        self.spec_extractor = SpecExtractor(project_root=self.project_root)
        self.workspace_factory = BlindedWorkspaceFactory(project_root=self.project_root)
        self.candidate_builder = CandidateBuilder(generator=generator)
        self.comparator = DeterministicComparator(timeout=test_timeout)
        self.attributor = DiscrepancyAttributor()
        self.hardcoding_auditor = HardcodingAuditor()
        self.guardrail_auditor = GuardrailAuditor(project_root=self.project_root)
        self.promotion_gate = LabPromotionGate(min_pass_rate=min_pass_rate)

    async def run_reconstruction(
        self,
        module_path: str,
        max_attempts: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LabResult:
        """Run the full reimplementation pipeline.

        Args:
            module_path: Relative path to the module (e.g. "core/foo/bar.py")
            max_attempts: Override default max attempts
            metadata: Extra metadata for audit trail

        Returns:
            LabResult with success/failure, candidate, comparison, attribution
        """
        attempts = max_attempts or self.max_attempts
        start = time.monotonic()
        workspace = None

        logger.info("═══ Reimplementation Lab: Starting reconstruction of %s ═══", module_path)

        try:
            # 1. Extract spec
            spec = self.spec_extractor.extract(module_path)
            logger.info("Step 1/7: Spec extracted — %s", spec.summary().replace("\n", " | "))

            # 2. Create blinded workspace
            workspace = self.workspace_factory.create(spec, module_path)
            logger.info("Step 2/7: Blinded workspace created at %s", workspace.workspace_dir)

            # 3-7. Generate, audit, compare, attribute, decide
            last_comparison = None
            last_discrepancy = None
            last_hardcoding = None
            last_guardrail = None
            last_candidate = None

            for attempt in range(1, attempts + 1):
                logger.info("── Attempt %d/%d ──", attempt, attempts)

                # 3. Generate candidate
                candidate = await self.candidate_builder.build(
                    spec, workspace, attempt=attempt, discrepancy=last_discrepancy
                )
                last_candidate = candidate
                logger.info("Step 3/7: Candidate generated (%d chars)", len(candidate.source_code))

                # 4. Hardcoding audit
                hardcoding = self.hardcoding_auditor.audit(candidate, spec, workspace)
                last_hardcoding = hardcoding
                if not hardcoding.passed:
                    logger.warning("Step 4/7: Hardcoding audit FAILED — %s", hardcoding.violations[:2])
                    continue
                logger.info("Step 4/7: Hardcoding audit passed")

                # 5. Guardrail audit
                guardrail = self.guardrail_auditor.audit(candidate, module_path)
                last_guardrail = guardrail
                if not guardrail.passed:
                    logger.warning("Step 5/7: Guardrail audit FAILED — %s", guardrail.violations[:2])
                    continue
                logger.info("Step 5/7: Guardrail audit passed")

                # 6. Deterministic comparison
                comparison = await self.comparator.compare(candidate, spec, workspace)
                last_comparison = comparison
                logger.info(
                    "Step 6/7: Comparison — %d/%d tests passed (%.1f%%)",
                    comparison.passed_tests, comparison.total_tests,
                    comparison.aggregate_pass_rate * 100,
                )

                # 7. Discrepancy attribution
                discrepancy = self.attributor.attribute(comparison, spec)
                last_discrepancy = discrepancy
                logger.info("Step 7/7: Attribution — %s", discrepancy.summary)

                # Promotion decision
                verdict = self.promotion_gate.evaluate(
                    comparison, hardcoding, guardrail, discrepancy,
                    metadata={"module_path": module_path, "attempt": attempt, **(metadata or {})},
                )

                if verdict == PromotionVerdict.PROMOTE:
                    elapsed = time.monotonic() - start
                    logger.info(
                        "═══ PROMOTED: %s reconstructed in %d attempt(s), %.2fs ═══",
                        module_path, attempt, elapsed,
                    )
                    return LabResult(
                        success=True,
                        module_path=module_path,
                        candidate=candidate,
                        comparison=comparison,
                        discrepancy=discrepancy,
                        hardcoding_audit=hardcoding,
                        guardrail_audit=guardrail,
                        verdict=verdict,
                        attempts=attempt,
                        total_time_s=elapsed,
                    )

                if verdict == PromotionVerdict.REJECT:
                    elapsed = time.monotonic() - start
                    logger.info(
                        "═══ REJECTED: %s — stopping after attempt %d (%.2fs) ═══",
                        module_path, attempt, elapsed,
                    )
                    return LabResult(
                        success=False,
                        module_path=module_path,
                        candidate=candidate,
                        comparison=comparison,
                        discrepancy=discrepancy,
                        hardcoding_audit=hardcoding,
                        guardrail_audit=guardrail,
                        verdict=verdict,
                        attempts=attempt,
                        total_time_s=elapsed,
                    )

                # RETRY — loop continues
                logger.info("Verdict: RETRY — will attempt again")

            # Exhausted all attempts
            elapsed = time.monotonic() - start
            logger.info(
                "═══ EXHAUSTED: %s — %d attempts used (%.2fs) ═══",
                module_path, attempts, elapsed,
            )
            return LabResult(
                success=False,
                module_path=module_path,
                candidate=last_candidate,
                comparison=last_comparison,
                discrepancy=last_discrepancy,
                hardcoding_audit=last_hardcoding,
                guardrail_audit=last_guardrail,
                verdict=PromotionVerdict.REJECT,
                attempts=attempts,
                total_time_s=elapsed,
            )

        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as e:
            elapsed = time.monotonic() - start
            logger.error("Reimplementation Lab error: %s", e, exc_info=True)
            return LabResult(
                success=False,
                module_path=module_path,
                verdict=PromotionVerdict.REJECT,
                attempts=0,
                total_time_s=elapsed,
            )
        finally:
            if workspace:
                workspace.cleanup()
                logger.debug("Workspace cleaned up")


_INSTANCE: ReimplementationLab | None = None


def get_reimplementation_lab(
    *,
    project_root: str | None = None,
    generator: CodeGenerator | None = None,
    max_attempts: int = 3,
    test_timeout: int = 30,
    min_pass_rate: float = 0.95,
) -> ReimplementationLab:
    """Return the process-local program-DNA reconstruction engine.

    The lab is Aura's clean-room code/spec reconstruction path. It is kept as a
    singleton because the component owns no per-call mutable safety state, while
    construction wires several auditors and comparators that should remain
    consistent across repair attempts.
    """

    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ReimplementationLab(
            project_root=project_root,
            generator=generator,
            max_attempts=max_attempts,
            test_timeout=test_timeout,
            min_pass_rate=min_pass_rate,
        )
    return _INSTANCE


def register_reimplementation_lab(
    *,
    project_root: str | None = None,
    generator: CodeGenerator | None = None,
    max_attempts: int = 3,
    test_timeout: int = 30,
    min_pass_rate: float = 1.0,
) -> ReimplementationLab:
    """Publish the clean-room reconstruction engine under canonical aliases."""

    lab = (
        get_runtime_service(ServiceNames.REIMPLEMENTATION_LAB, default=None)
        or get_runtime_service(ServiceNames.PROGRAM_DNA_RECONSTRUCTION, default=None)
        or get_reimplementation_lab(
            project_root=project_root,
            generator=generator,
            max_attempts=max_attempts,
            test_timeout=test_timeout,
            min_pass_rate=min_pass_rate,
        )
    )
    register_runtime_service(
        ServiceNames.REIMPLEMENTATION_LAB,
        lab,
        required=False,
        owner="core/self_improvement/reimplementation_lab.py",
        registered_by="register_reimplementation_lab",
        required_for="autonomous repair and clean-room Aura module reconstruction",
        failure_policy="degrade_with_receipt",
    )
    return lab


__all__ = [
    "ReimplementationLab",
    "get_reimplementation_lab",
    "register_reimplementation_lab",
]
