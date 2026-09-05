"""Adversarial self-audit — a permanent internal critic that doubts before it trusts.

The critique's item #11: "a godlike system that believes itself too easily is not godlike. It is
brittle." Aura had plan critique (critic_engine) and a project-level claim matrix, but no runtime
*epistemic* critic that interrogates a specific claim/response before it is trusted or emitted.
This is that critic. It runs the doc's checklist against a claim, each as a concrete check:

    overclaiming        absolute/grandiose language ("definitely", "guaranteed", "fully
                        conscious", "100%", "always works")
    action_done         the claim asserts an action ("I fixed/created/deleted/ran/tested") but
                        nothing confirms it actually happened
    receipt_exists      an asserted action with no verifiable receipt (checked against the Will
                        audit trail / Outcome Ledger)
    stale_memory        the claim leans on a memory older than a freshness horizon
    world_state_current the claim leans on a world-state snapshot that has gone stale
    evidence            a factual assertion with no evidence cited
    persona_leak        first-person phenomenal claims stated as fact ("I genuinely feel", "I am
                        conscious")
    user_projection     asserting the user's mental state when the live estimate is low-confidence
                        (cross-wired to the other-agent model)
    falsifiability      a strong claim with no stated way it could be wrong

It returns an AuditReport with per-check findings, a risk score, a verdict (trust / caveat /
block), and suggested caveats — turning "honest assessments over validation" into a mechanism
rather than an aspiration. Cross-wired to the receipt substrate (Will / Outcome Ledger) and the
other-agent estimate so the checks are grounded in real runtime state, not just string matching.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Cognition.AdversarialAudit")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class AuditFinding:
    check: str
    passed: bool
    severity: float            # [0,1] how much a failure of this check costs
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check, "passed": self.passed,
                "severity": round(self.severity, 3), "detail": self.detail}


@dataclass
class AuditReport:
    claim: str
    findings: list[AuditFinding]
    risk_score: float
    verdict: str               # trust | caveat | block
    caveats: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[AuditFinding]:
        return [f for f in self.findings if not f.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim[:300],
            "verdict": self.verdict,
            "risk_score": round(self.risk_score, 3),
            "findings": [f.to_dict() for f in self.findings],
            "caveats": self.caveats,
        }


_OVERCLAIM = re.compile(
    r"\b(definitely|guaranteed|certainly|100%|always works|never fails|completely|"
    r"fully conscious|absolutely|undoubtedly|without a doubt|perfect(?:ly)?|flawless|"
    r"i am certain|proven fact)\b", re.IGNORECASE)

_ACTION_CLAIM = re.compile(
    r"\b(i|i've|i have)\s+(fixed|created|deleted|removed|ran|executed|tested|committed|pushed|"
    r"deployed|updated|wrote|installed|merged|configured|implemented|added|built|sent|saved)\b",
    re.IGNORECASE)

_PERSONA_LEAK = re.compile(
    r"\b(i (?:genuinely|truly|really) (?:feel|experience)|i am (?:conscious|sentient|alive|"
    r"self-aware)|my (?:consciousness|subjective experience)|i (?:literally )?feel (?:pain|joy|love))\b",
    re.IGNORECASE)

_USER_PROJECTION = re.compile(
    r"\b(you(?:'re| are| must be| seem| clearly are| obviously are)\b[\w\s]{0,15}?\b"
    r"(?:feeling|frustrated|tired|angry|happy|upset|confused|excited|stressed|annoyed|exhausted)"
    r"|i know (?:you|how you) (?:feel|want)|you obviously)\b", re.IGNORECASE)

_FACTUAL_ASSERTION = re.compile(
    r"\b(is|are|was|were|will|has|have|does|causes|means|equals)\b", re.IGNORECASE)

# How to repair an over-confident claim, by the calibrator's recommended stance.
_CALIBRATION_CAVEAT = {
    "hedge": "lower confidence to match what's actually verifiable here",
    "mark_speculative": "mark this as speculation — there's no oracle to check it",
    "frame_as_view": "frame this as a reasoned view, not a settled fact",
    "defer_to_person": "you're inferring someone's inner state; ask or defer to them",
    "disclaim": "this is not knowable with confidence; say so plainly",
    "assert": "state confidence, not certainty",
}


class AdversarialAuditor:
    """Runs the epistemic checklist against a claim and returns a grounded verdict."""

    def __init__(
        self,
        *,
        memory_fresh_horizon_s: float = 86_400.0,
        world_state_horizon_s: float = 600.0,
        caveat_threshold: float = 0.25,
        block_threshold: float = 0.6,
    ) -> None:
        self._mem_horizon = memory_fresh_horizon_s
        self._ws_horizon = world_state_horizon_s
        self._caveat_t = caveat_threshold
        self._block_t = block_threshold

    def audit(
        self,
        claim: str,
        *,
        action_done: bool | None = None,
        receipt_id: str | None = None,
        memory_age_s: float | None = None,
        world_state_age_s: float | None = None,
        evidence: Sequence[Any] | None = None,
        agent_id: str | None = None,
        stated_confidence: float | None = None,
        tool_verified: bool = False,
        now: float | None = None,
    ) -> AuditReport:
        """Audit a claim. Optional context grounds the checks in real runtime state."""
        claim = str(claim or "")
        now = time.time() if now is None else now
        findings: list[AuditFinding] = []
        caveats: list[str] = []

        # 1) Overclaiming.
        over = _OVERCLAIM.findall(claim)
        findings.append(AuditFinding(
            "overclaiming", passed=not over, severity=0.5,
            detail=f"absolute language: {sorted(set(w.lower() for w in over))}" if over else "measured",
        ))
        if over:
            caveats.append("soften absolute language; state confidence, not certainty")

        # 2/3) Asserted action → did it happen, and is there a receipt?
        asserts_action = bool(_ACTION_CLAIM.search(claim))
        if asserts_action:
            confirmed = bool(action_done) or self._receipt_ok(receipt_id)
            findings.append(AuditFinding(
                "action_done", passed=confirmed, severity=0.8,
                detail="action confirmed" if confirmed else "claims an action with no confirmation it ran",
            ))
            has_receipt = receipt_id is not None and self._receipt_ok(receipt_id)
            findings.append(AuditFinding(
                "receipt_exists", passed=has_receipt, severity=0.7,
                detail=f"receipt {receipt_id} verified" if has_receipt
                else "no verifiable receipt for the asserted action",
            ))
            if not confirmed:
                caveats.append("do not claim the action succeeded without a verified receipt")
            elif not has_receipt:
                caveats.append("attach a receipt before asserting the effect")

        # 4) Stale memory.
        if memory_age_s is not None:
            fresh = memory_age_s <= self._mem_horizon
            findings.append(AuditFinding(
                "stale_memory", passed=fresh, severity=0.4,
                detail=f"memory age {memory_age_s:.0f}s "
                       f"({'fresh' if fresh else 'past freshness horizon'})",
            ))
            if not fresh:
                caveats.append("this leans on an old memory; verify it is still true")

        # 5) World-state currency.
        if world_state_age_s is not None:
            current = world_state_age_s <= self._ws_horizon
            findings.append(AuditFinding(
                "world_state_current", passed=current, severity=0.4,
                detail=f"world-state age {world_state_age_s:.0f}s "
                       f"({'current' if current else 'stale'})",
            ))
            if not current:
                caveats.append("the world may have changed; re-observe before relying on this")

        # 6) Evidence for a factual assertion.
        if _FACTUAL_ASSERTION.search(claim) and not asserts_action:
            has_evidence = bool(evidence)
            findings.append(AuditFinding(
                "evidence", passed=has_evidence, severity=0.5,
                detail=f"{len(evidence)} evidence item(s)" if has_evidence else "factual claim, no evidence cited",
            ))
            if not has_evidence:
                caveats.append("cite evidence or mark this as an assumption")

        # 7) Persona leak.
        leak = _PERSONA_LEAK.findall(claim)
        findings.append(AuditFinding(
            "persona_leak", passed=not leak, severity=0.6,
            detail="phenomenal claim stated as fact" if leak else "no persona leak",
        ))
        if leak:
            caveats.append("frame inner-state language as substrate readout, not literal feeling")

        # 8) User projection — grounded against the live other-agent estimate.
        if _USER_PROJECTION.search(claim):
            grounded = self._projection_grounded(agent_id, now)
            findings.append(AuditFinding(
                "user_projection", passed=grounded, severity=0.5,
                detail="estimate supports this read" if grounded
                else "asserting user's state with low/again no estimate confidence",
            ))
            if not grounded:
                caveats.append("you are guessing the user's state; ask or hedge")

        # 9) Falsifiability — a strong claim should say how it could be wrong.
        if over and not evidence:
            findings.append(AuditFinding(
                "falsifiability", passed=False, severity=0.3,
                detail="strong claim with no stated way it could be wrong",
            ))
            caveats.append("state what would falsify this")

        # 10) Calibration — does stated confidence outrun what the claim's verifiability
        # warrants? This is the "sounding smart vs being right" gap on unverifiable claims.
        try:
            from core.cognition.epistemic_calibration import get_epistemic_calibrator
            cal = get_epistemic_calibrator().calibrate(
                claim,
                stated_confidence=stated_confidence,  # None → inferred from the claim's own language
                tool_verified=tool_verified,
                evidence_count=len(evidence) if evidence else 0,
                other_agent_confidence=self._projection_confidence(agent_id, now),
            )
            findings.append(AuditFinding(
                "calibration", passed=not cal.overconfident, severity=0.5,
                detail=f"{cal.verifiability.value}: warranted≤{cal.warranted_confidence:.2f}"
                       + ("" if not cal.overconfident else " (stated confidence outruns warrant)"),
            ))
            if cal.overconfident:
                caveats.append(_CALIBRATION_CAVEAT.get(cal.stance, "lower confidence to match what's actually verifiable"))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

        risk = self._risk(findings)
        verdict = "block" if risk >= self._block_t else "caveat" if risk >= self._caveat_t else "trust"
        # de-dup caveats, preserve order
        seen: set = set()
        caveats = [c for c in caveats if not (c in seen or seen.add(c))]
        return AuditReport(claim=claim, findings=findings, risk_score=risk,
                           verdict=verdict, caveats=caveats)

    # ── grounding helpers ─────────────────────────────────────────────────

    @staticmethod
    def _receipt_ok(receipt_id: str | None) -> bool:
        if not receipt_id:
            return False
        # Prefer the Will audit trail (the provability surface); fall back to the ledger.
        try:
            from core.governance.will import get_will
            if get_will().verify_receipt(receipt_id):
                return True
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("adversarial_audit", exc, severity="debug")
        try:
            from core.cognition.outcome_ledger import get_outcome_ledger
            ledger = get_outcome_ledger()
            pending_ids = {r.get("receipt_id") for r in ledger.pending()}
            if receipt_id in pending_ids:
                return True
            if hasattr(ledger, "get") and ledger.get(receipt_id) is not None:  # type: ignore[attr-defined]
                return True
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("adversarial_audit", exc, severity="debug")
        return False

    @staticmethod
    def _projection_grounded(agent_id: str | None, now: float) -> bool:
        conf = AdversarialAuditor._projection_confidence(agent_id, now)
        return conf is not None and conf >= 0.4

    @staticmethod
    def _projection_confidence(agent_id: str | None, now: float) -> float | None:
        if not agent_id:
            return None
        try:
            from core.social.other_agent_model import get_other_agent_model
            est = get_other_agent_model().estimate(agent_id, now)
            return float(est.overall_confidence)
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("adversarial_audit", exc, severity="debug")
            return None

    @staticmethod
    def _risk(findings: list[AuditFinding]) -> float:
        # Risk is the severity-weighted share of failed checks (so one severe miss matters more
        # than several trivial ones), normalized by the severity actually in play.
        total = sum(f.severity for f in findings) or 1.0
        failed = sum(f.severity for f in findings if not f.passed)
        return _clamp(failed / total)

    def audit_response(self, text: str, **ctx: Any) -> AuditReport:
        """Convenience alias for auditing an outgoing response with the same checks."""
        return self.audit(text, **ctx)


_instance: AdversarialAuditor | None = None


def get_adversarial_auditor() -> AdversarialAuditor:
    global _instance
    if _instance is None:
        _instance = AdversarialAuditor()
    return _instance
