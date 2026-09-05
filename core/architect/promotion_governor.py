"""Promotion gate for ASA shadow candidates.

This module writes candidate bytes into LIVE SOURCE, so every check here is
load-bearing: what it accepts, the running system becomes.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from core.architect.config import ASAConfig
from core.architect.errors import PromotionError
from core.architect.models import MutationTier, PromotionDecision, PromotionStatus, ProofReceipt, RefactorPlan, RollbackPacket
from core.architect.mutation_classifier import MutationClassifier
from core.architect.shadow_workspace import ShadowRun
from core.runtime.atomic_writer import atomic_write_bytes, atomic_write_text


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contained_target(repo_root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` inside ``repo_root``, or None if it escapes.

    CP126 89cfe03a: a plan target was joined straight onto repo_root and its
    parents were created before any live write, with nothing rejecting an
    absolute path or ``..`` traversal — a plan could therefore promote bytes
    ANYWHERE the process can write.
    """
    raw = str(rel or "").strip()
    if not raw:
        return None
    candidate = Path(raw.replace("\\", "/"))
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        return None
    root = repo_root.resolve(strict=False)
    target = (root / candidate).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


class PromotionGovernor:
    """Autonomously promote candidates only when proof obligations pass."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.classifier = MutationClassifier(self.config)

    def decide(self, plan: RefactorPlan, proof: ProofReceipt, rollback_packet: RollbackPacket) -> PromotionDecision:
        if plan.risk_tier >= MutationTier.T4_GOVERNANCE_SENSITIVE:
            return self._decision(plan, proof, PromotionStatus.PROPOSAL_ONLY, "T4/T5 surfaces are proposal-only")
        if plan.risk_tier > self.config.max_tier:
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"plan tier {plan.risk_tier.name} exceeds configured max {self.config.max_tier.name}")
        if not proof.passed:
            failed = [
                result.obligation_id
                for result in proof.results
                if not result.passed
                and not (plan.risk_tier <= MutationTier.T1_CLEANUP and result.status == "BOOT_HARNESS_UNAVAILABLE")
            ]
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"proof failed: {failed[:8]}")
        if not rollback_packet.dry_run_passed:
            return self._decision(plan, proof, PromotionStatus.REJECTED, "rollback dry-run did not pass")
        # CP126 6f938e23: the gate trusted the caller-supplied plan tier and
        # only looked for SEALED paths, so a PROTECTED (T4) file hidden inside
        # an under-tiered plan promoted at the lower tier. Reclassify every
        # target and enforce the declared eligibility flag.
        if not plan.promotion_eligible:
            return self._decision(plan, proof, PromotionStatus.PROPOSAL_ONLY, "plan is not promotion-eligible")
        reclassified = {path: self.classifier.classify_path(path) for path in plan.changed_files}
        sealed = [path for path, tier in reclassified.items() if tier is MutationTier.T5_SEALED]
        if sealed:
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"sealed files touched: {sealed}")
        protected = [
            path for path, tier in reclassified.items()
            if tier >= MutationTier.T4_GOVERNANCE_SENSITIVE
        ]
        if protected:
            return self._decision(plan, proof, PromotionStatus.PROPOSAL_ONLY, f"protected paths are proposal-only: {protected[:8]}")
        understated = [
            path for path, tier in reclassified.items() if tier > plan.risk_tier
        ]
        if understated:
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"plan tier understates its targets: {understated[:8]}")
        escaping = [path for path in plan.changed_files if _contained_target(self.config.repo_root, path) is None]
        if escaping:
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"targets escape the repository: {escaping[:8]}")

        # CP126 f69faa30: the receipt's own bindings were never checked — a
        # proof and a rollback packet from DIFFERENT runs (or for a different
        # plan or file set) satisfied the gate as long as their booleans said
        # pass.
        binding = self._binding_mismatch(plan, proof, rollback_packet)
        if binding:
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"receipt bindings do not match: {binding}")

        # CP126 3d88c6f7: a behavioral regression blocked only T2-and-below, so
        # a T3 plan could carry explicit regressions to SHADOW_PASSED. A
        # regression is a regression at every promotable tier.
        if not proof.behavior_delta.equivalent:
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"behavior regression: {proof.behavior_delta.regressions}")
        return self._decision(plan, proof, PromotionStatus.SHADOW_PASSED, "eligible for atomic promotion")

    @staticmethod
    def _binding_mismatch(
        plan: RefactorPlan, proof: ProofReceipt, rollback_packet: RollbackPacket
    ) -> str:
        """Why this proof/rollback pair does not describe THIS plan, or ''."""
        if str(proof.plan_id) != str(plan.id):
            return f"proof.plan_id {proof.plan_id!r} != plan.id {plan.id!r}"
        if str(rollback_packet.run_id) != str(proof.run_id):
            return f"rollback.run_id {rollback_packet.run_id!r} != proof.run_id {proof.run_id!r}"
        if proof.tier is not plan.risk_tier:
            return f"proof.tier {proof.tier.name} != plan tier {plan.risk_tier.name}"
        planned = set(plan.changed_files)
        covered = set(rollback_packet.changed_files)
        if planned != covered:
            missing = sorted(planned - covered)
            extra = sorted(covered - planned)
            return f"rollback file set differs (missing={missing[:4]}, extra={extra[:4]})"
        uncovered = sorted(planned - set(rollback_packet.candidate_hashes))
        if uncovered:
            return f"no proved candidate hash for {uncovered[:4]}"
        return ""

    def promote(self, plan: RefactorPlan, shadow: ShadowRun, proof: ProofReceipt, rollback_packet: RollbackPacket) -> PromotionDecision:
        decision = self.decide(plan, proof, rollback_packet)
        if decision.status is not PromotionStatus.SHADOW_PASSED:
            self._persist_decision(decision)
            return decision
        # CP126 cde68e74 + c54936d6. Promotion used to read each mutable
        # candidate snapshot and write its CURRENT bytes one file at a time,
        # with no check against the proved candidate hash and no rollback: a
        # snapshot edited after proof was promoted unverified, and a failure
        # part-way through left an already-promoted PREFIX in live source
        # while claiming "atomic completion".
        #
        # Stage everything first (read + verify every candidate against the
        # hash the proof was computed over), then write; if any write fails,
        # restore the originals we already replaced.
        staged: list[tuple[str, Path, bytes]] = []
        for rel in plan.changed_files:
            # An absent mapping yields "", and Path("") is "." — a DIRECTORY
            # that exists. Check the raw entry and require a regular file.
            candidate_raw = str(shadow.candidate_files.get(rel, "") or "").strip()
            candidate = Path(candidate_raw) if candidate_raw else None
            if candidate is None or not candidate.is_file():
                raise PromotionError(f"candidate snapshot missing for {rel}")
            target = _contained_target(self.config.repo_root, rel)
            if target is None:
                raise PromotionError(f"promotion target escapes the repository: {rel}")
            payload = candidate.read_bytes()
            expected = str(rollback_packet.candidate_hashes.get(rel) or "")
            actual = _sha256_bytes(payload)
            if not expected:
                raise PromotionError(f"no proved candidate hash for {rel}")
            if actual != expected:
                raise PromotionError(
                    f"candidate bytes for {rel} do not match the proved hash "
                    f"(expected {expected[:12]}, got {actual[:12]})"
                )
            staged.append((rel, target, payload))

        promoted: list[str] = []
        restore: list[tuple[Path, bytes | None]] = []
        try:
            for rel, target, payload in staged:
                previous = target.read_bytes() if target.exists() else None
                restore.append((target, previous))
                atomic_write_bytes(target, payload)
                promoted.append(rel)
        except (OSError, PromotionError) as exc:
            for target, previous in reversed(restore):
                try:
                    if previous is None:
                        if target.exists():
                            os.unlink(target)
                    else:
                        atomic_write_bytes(target, previous)
                except OSError:
                    # Report the ORIGINAL failure with the un-restored file
                    # named: a partially promoted tree must never be silent.
                    raise PromotionError(
                        f"promotion failed ({exc}) AND rollback of {target} failed — "
                        "live source may be partially promoted"
                    ) from exc
            raise PromotionError(f"promotion rolled back after failure: {exc}") from exc
        promoted_decision = PromotionDecision(
            run_id=proof.run_id,
            plan_id=plan.id,
            status=PromotionStatus.PROMOTED,
            reason="atomic file promotion completed after proof pass",
            receipt_hash=proof.decision_hash,
            promoted_files=tuple(promoted),
        )
        self._persist_decision(promoted_decision)
        return promoted_decision

    def _decision(self, plan: RefactorPlan, proof: ProofReceipt, status: PromotionStatus, reason: str) -> PromotionDecision:
        return PromotionDecision(
            run_id=proof.run_id,
            plan_id=plan.id,
            status=status,
            reason=reason,
            receipt_hash=proof.decision_hash,
        )

    def _persist_decision(self, decision: PromotionDecision) -> None:
        decisions = self.config.artifacts / "decisions"
        decisions.mkdir(parents=True, exist_ok=True)
        atomic_write_text(decisions / f"{decision.run_id}.json", json.dumps(asdict(decision), indent=2, sort_keys=True, default=str))
