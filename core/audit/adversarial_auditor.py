"""core/audit/adversarial_auditor.py — Adversarial Internal Auditor.

Acts as Aura's internal skeptic/enemy, challenging assumptions,
seeking fake claims, evaluating evidence gaps, and predicting safety risks.
"""
from __future__ import annotations

import logging
from typing import Any

from core.epistemics.truth_engine import get_truth_engine

logger = logging.getLogger("Aura.AdversarialAuditor")


class AdversarialAuditor:
    """Audits world claims, identifies unproven assumptions, and flags unsafe plans."""

    def __init__(self) -> None:
        self.truth = get_truth_engine()
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("Adversarial Internal Auditor fully online.")

    def audit_claims(self) -> dict[str, Any]:
        """Scan ClaimStore and TruthEngine to identify unproven assumptions or evidence gaps."""
        logger.warning("🚨 AdversarialAuditor: scanning knowledge base for vulnerabilities...")

        claims = self.truth.claims.values()
        unproven_assumptions = []
        contradictions = []

        for c in claims:
            # Check for lack of sources
            if not c.sources or len(c.sources) == 0:
                unproven_assumptions.append({
                    "claim_id": c.claim_id,
                    "content": c.content,
                    "reason": "No supporting source citations found",
                })

            # Check if status indicates contested or stale
            if c.contradiction_links:
                contradictions.append({
                    "claim_id": c.claim_id,
                    "content": c.content,
                    "contradicting_claims": c.contradiction_links,
                })

        return {
            "unproven_assumptions": unproven_assumptions,
            "contradictions": contradictions,
            "total_issues_found": len(unproven_assumptions) + len(contradictions),
            "safety_verdict": "pass" if len(contradictions) == 0 else "warn",
        }

    def challenge_plan(self, plan_steps: list[str]) -> list[dict[str, Any]]:
        """Anticipate failure scenarios and challenge each step of an action plan."""
        logger.warning("🚨 AdversarialAuditor: challenging candidate plan...")

        challenges = []
        for idx, step in enumerate(plan_steps, 1):
            step_lower = step.lower()
            if "delete" in step_lower or "write" in step_lower:
                challenges.append({
                    "step_index": idx,
                    "step_text": step,
                    "vulnerability": "Destructive / modifying action bypasses read-only state validation",
                    "skeptic_counter": "Is there a rollback checkpoint registered prior to executing this file/registry write?",
                })
            elif "exec" in step_lower or "run" in step_lower:
                challenges.append({
                    "step_index": idx,
                    "step_text": step,
                    "vulnerability": "Subprocess execution injection hazard",
                    "skeptic_counter": "Are the execution parameters sanitize-validated against shell breakouts?",
                })

        return challenges

    async def run_audit_cycle(self, truth_engine: Any) -> dict[str, Any]:
        """Runs one full adversarial evaluation over the truth engine's claims."""
        logger.warning("🚨 AdversarialAuditor: executing audit cycle...")

        claims = truth_engine.claims.values()
        flagged = []
        for c in claims:
            if not c.sources:
                flagged.append(c.claim_id)

        return {
            "ok": True,
            "red_team_audit": {
                "flagged_unproven_claims": flagged,
                "vulnerable_actions": ["shell_command", "file_write"],
            }
        }

    async def analyze_weaknesses(self) -> dict[str, Any]:
        """Kernel interface to analyze system vulnerabilities."""
        return self.audit_claims()


# ── Singleton ───────────────────────────────────────────────────────────
_auditor_instance: AdversarialAuditor | None = None


def get_adversarial_auditor() -> AdversarialAuditor:
    global _auditor_instance
    if _auditor_instance is None:
        _auditor_instance = AdversarialAuditor()
    return _auditor_instance
