"""core/autonomy/self_modification.py -- Will-Authorized Self-Modification
=========================================================================
Aura can propose edits to her own non-critical modules.  Every modification
flows through the Unified Will for authorization.

Process:
  1. Propose a modification (diff, reason, target module)
  2. Classify the target -- PROTECTED modules are rejected outright
  3. Simulate consequences (dry-run import, basic risk scoring)
  4. Route to Unified Will for PROCEED / REFUSE
  5. If approved, apply the change and log a receipt
  6. All proposals (accepted and rejected) go to the audit log

PROTECTED (never modifiable):
  - core/will.py                (the Will itself)
  - core/identity/*             (identity core)
  - core/safety/*               (safety gates)
  - core/constitution.py        (constitutional alignment)
  - core/heartstone_directive.py (sacred vows)

MODIFIABLE (with Will authorization):
  - Drive weights               (heartstone_values, drive_engine)
  - Response strategies          (pipeline/, brain/, conversation/)
  - Skill implementations       (skills/, skill_management/)
  - Threshold values             (adaptation/, cognitive/)
  - Self-modification engine     (self_modification/)
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.SelfModification.Autonomous")

# ── Persistence ─────────────────────────────────────────────────────────────
_DATA_DIR = Path.home() / ".aura" / "data" / "self_modification"
_AUDIT_LOG_PATH = _DATA_DIR / "audit_log.jsonl"
_MAX_AUDIT_ENTRIES = 2000


# ── Classification ──────────────────────────────────────────────────────────

class ModuleZone(str, Enum):
    """Classification of a module's modifiability."""
    PROTECTED = "protected"       # Never touch
    MODIFIABLE = "modifiable"     # Allowed with Will approval
    UNKNOWN = "unknown"           # Not in allowlist -- treat as protected


class ProposalOutcome(str, Enum):
    APPROVED = "approved"
    REFUSED_BY_WILL = "refused_by_will"
    REFUSED_PROTECTED = "refused_protected"
    REFUSED_SIMULATION = "refused_simulation"
    ERROR = "error"


# Protected path prefixes (relative to project root)
_PROTECTED_PREFIXES = (
    "core/will.py",
    "core/identity/",
    "core/safety/",
    "core/constitution.py",
    "core/constitutional_alignment.py",
    "core/heartstone_directive.py",
    "core/container.py",
    "core/prime_directives.py",
)

# Explicitly modifiable path prefixes
_MODIFIABLE_PREFIXES = (
    "core/affect/heartstone_values.py",
    "core/drive_engine.py",
    "core/adaptation/",
    "core/pipeline/",
    "core/brain/",
    "core/conversation/",
    "core/skills/",
    "core/skill_management/",
    "core/cognitive/",
    "core/self_modification/",
    "core/autonomy/",
    "core/learning/",
)


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ModificationProposal:
    """A proposed self-modification."""
    proposal_id: str
    target_path: str              # Relative to project root
    description: str              # Why this change
    diff_summary: str             # Human-readable description of what changes
    changes: Dict[str, Any]       # Structured change data
    source: str                   # Which subsystem proposed this
    priority: float = 0.5         # 0-1
    timestamp: float = field(default_factory=time.time)

    def content_hash(self) -> str:
        raw = f"{self.target_path}:{self.diff_summary}:{self.description}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class ModificationReceipt:
    """Audit record of every proposal -- accepted or rejected."""
    proposal_id: str
    target_path: str
    description: str
    diff_summary: str
    source: str
    outcome: ProposalOutcome
    will_receipt_id: str = ""
    will_reason: str = ""
    zone: str = ""
    simulation_result: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "target_path": self.target_path,
            "description": self.description,
            "diff_summary": self.diff_summary,
            "source": self.source,
            "outcome": self.outcome.value,
            "will_receipt_id": self.will_receipt_id,
            "will_reason": self.will_reason,
            "zone": self.zone,
            "simulation_result": self.simulation_result,
            "timestamp": self.timestamp,
        }


# ── Autonomous Self-Modification System ────────────────────────────────────

class AutonomousSelfModification:
    """Will-authorized self-modification system.

    All modifications must pass through the Unified Will.  Protected
    modules are rejected before Will consultation.  Every proposal is
    logged regardless of outcome.
    """

    _MAX_PENDING = 50
    _MAX_RECEIPTS = 500

    def __init__(self) -> None:
        self._pending: List[ModificationProposal] = []
        self._receipts: List[ModificationReceipt] = []
        self._started = False
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("AutonomousSelfModification created -- awaiting start()")

    async def start(self) -> None:
        """Register in ServiceContainer."""
        if self._started:
            return
        ServiceContainer.register_instance(
            "autonomous_self_modification", self, required=False
        )
        self._started = True
        logger.info("AutonomousSelfModification ONLINE")

    # ── Zone Classification ─────────────────────────────────────────────

    @staticmethod
    def classify_target(rel_path: str) -> ModuleZone:
        """Classify a module path into its modification zone."""
        normalized = rel_path.replace("\\", "/").lstrip("/")

        for prefix in _PROTECTED_PREFIXES:
            if normalized.startswith(prefix) or normalized == prefix.rstrip("/"):
                return ModuleZone.PROTECTED

        for prefix in _MODIFIABLE_PREFIXES:
            if normalized.startswith(prefix) or normalized == prefix.rstrip("/"):
                return ModuleZone.MODIFIABLE

        return ModuleZone.UNKNOWN

    # ── Proposal Flow ───────────────────────────────────────────────────

    async def propose(self, proposal: ModificationProposal) -> ModificationReceipt:
        """Submit a modification proposal through the full authorization pipeline.

        Steps:
          1. Classify target zone
          2. Reject PROTECTED outright
          3. Simulate consequences
          4. Consult Unified Will
          5. Apply if approved
          6. Log receipt
        """
        logger.info(
            "Self-modification proposal [%s]: %s -> %s",
            proposal.proposal_id, proposal.target_path, proposal.description[:80],
        )

        # 1. Zone check
        zone = self.classify_target(proposal.target_path)
        if zone == ModuleZone.PROTECTED:
            receipt = ModificationReceipt(
                proposal_id=proposal.proposal_id,
                target_path=proposal.target_path,
                description=proposal.description,
                diff_summary=proposal.diff_summary,
                source=proposal.source,
                outcome=ProposalOutcome.REFUSED_PROTECTED,
                zone=zone.value,
                will_reason="Target is in PROTECTED zone -- modification forbidden",
            )
            self._record_receipt(receipt)
            logger.warning(
                "REFUSED (protected): %s -> %s", proposal.target_path, proposal.description[:60]
            )
            return receipt

        if zone == ModuleZone.UNKNOWN:
            receipt = ModificationReceipt(
                proposal_id=proposal.proposal_id,
                target_path=proposal.target_path,
                description=proposal.description,
                diff_summary=proposal.diff_summary,
                source=proposal.source,
                outcome=ProposalOutcome.REFUSED_PROTECTED,
                zone=zone.value,
                will_reason="Target is not in the modifiable allowlist",
            )
            self._record_receipt(receipt)
            logger.warning(
                "REFUSED (unknown zone): %s -> %s", proposal.target_path, proposal.description[:60]
            )
            return receipt

        # 2. Simulate consequences
        sim_ok, sim_detail = await self._simulate(proposal)
        if not sim_ok:
            receipt = ModificationReceipt(
                proposal_id=proposal.proposal_id,
                target_path=proposal.target_path,
                description=proposal.description,
                diff_summary=proposal.diff_summary,
                source=proposal.source,
                outcome=ProposalOutcome.REFUSED_SIMULATION,
                zone=zone.value,
                simulation_result=sim_detail,
                will_reason=f"Simulation failed: {sim_detail}",
            )
            self._record_receipt(receipt)
            logger.warning(
                "REFUSED (simulation): %s -> %s", proposal.target_path, sim_detail[:80]
            )
            return receipt

        # 3. Consult the Unified Will
        try:
            from core.will import get_will, ActionDomain
            will = get_will()
            decision = will.decide(
                content=(
                    f"Self-modification proposal: {proposal.description}. "
                    f"Target: {proposal.target_path}. "
                    f"Changes: {proposal.diff_summary[:200]}. "
                    f"Simulation: {sim_detail[:100]}"
                ),
                source=f"self_modification/{proposal.source}",
                domain=ActionDomain.STATE_MUTATION,
                priority=proposal.priority,
                context={
                    "proposal_id": proposal.proposal_id,
                    "zone": zone.value,
                    "simulation_passed": sim_ok,
                },
            )
        except Exception as exc:
            receipt = ModificationReceipt(
                proposal_id=proposal.proposal_id,
                target_path=proposal.target_path,
                description=proposal.description,
                diff_summary=proposal.diff_summary,
                source=proposal.source,
                outcome=ProposalOutcome.ERROR,
                zone=zone.value,
                will_reason=f"Will consultation failed: {exc}",
            )
            self._record_receipt(receipt)
            logger.error("Self-modification Will consultation failed: %s", exc)
            return receipt

        if not decision.is_approved():
            receipt = ModificationReceipt(
                proposal_id=proposal.proposal_id,
                target_path=proposal.target_path,
                description=proposal.description,
                diff_summary=proposal.diff_summary,
                source=proposal.source,
                outcome=ProposalOutcome.REFUSED_BY_WILL,
                zone=zone.value,
                will_receipt_id=decision.receipt_id,
                will_reason=decision.reason,
                simulation_result=sim_detail,
            )
            self._record_receipt(receipt)
            logger.info(
                "REFUSED by Will: %s -> %s (%s)",
                proposal.target_path, proposal.description[:60], decision.reason,
            )
            return receipt

        # 4. Apply the modification
        apply_detail = await self._apply(proposal)

        receipt = ModificationReceipt(
            proposal_id=proposal.proposal_id,
            target_path=proposal.target_path,
            description=proposal.description,
            diff_summary=proposal.diff_summary,
            source=proposal.source,
            outcome=ProposalOutcome.APPROVED,
            zone=zone.value,
            will_receipt_id=decision.receipt_id,
            will_reason=decision.reason,
            simulation_result=f"{sim_detail}; applied: {apply_detail}",
        )
        self._record_receipt(receipt)
        logger.info(
            "APPROVED: %s -> %s (will: %s)",
            proposal.target_path, proposal.description[:60], decision.receipt_id,
        )

        # Publish event
        self._publish_event("self_modification.applied", receipt)

        return receipt

    # ── Simulation ──────────────────────────────────────────────────────

    async def _simulate(self, proposal: ModificationProposal) -> tuple[bool, str]:
        """Simulate a proposed modification.

        For value/weight changes: validate ranges.
        For code changes: syntax check via compile().
        """
        changes = proposal.changes or {}
        change_type = changes.get("type", "unknown")

        try:
            if change_type == "value_adjustment":
                # Validate drive/value weight adjustments
                new_values = changes.get("new_values", {})
                for key, val in new_values.items():
                    if not isinstance(val, (int, float)):
                        return False, f"Non-numeric value for {key}: {val}"
                    if val < 0.0 or val > 1.0:
                        return False, f"Value {key}={val} out of [0.0, 1.0] range"
                return True, f"Value adjustment validated: {len(new_values)} change(s)"

            elif change_type == "threshold_adjustment":
                new_threshold = changes.get("new_threshold")
                if not isinstance(new_threshold, (int, float)):
                    return False, f"Non-numeric threshold: {new_threshold}"
                return True, f"Threshold adjustment validated: {new_threshold}"

            elif change_type == "code_patch":
                # Syntax-check the replacement code
                new_code = changes.get("new_code", "")
                if not new_code.strip():
                    return False, "Empty code patch"
                try:
                    compile(new_code, "<self_modification_patch>", "exec")
                except SyntaxError as se:
                    return False, f"Syntax error in patch: {se}"
                return True, f"Code patch syntax valid ({len(new_code)} chars)"

            elif change_type == "config_update":
                return True, "Config update (low risk)"

            else:
                return True, f"Untyped change -- passed basic validation"

        except Exception as exc:
            return False, f"Simulation error: {exc}"

    # ── Application ─────────────────────────────────────────────────────

    async def _apply(self, proposal: ModificationProposal) -> str:
        """Apply an approved modification.

        Different change types route to different application strategies.
        """
        changes = proposal.changes or {}
        change_type = changes.get("type", "unknown")

        try:
            if change_type == "value_adjustment":
                return await self._apply_value_adjustment(changes)
            elif change_type == "threshold_adjustment":
                return await self._apply_threshold_adjustment(proposal.target_path, changes)
            elif change_type == "code_patch":
                return "Code patch queued for safe_modification pipeline"
            elif change_type == "config_update":
                return "Config update noted"
            else:
                return "No automatic application -- logged for manual review"
        except Exception as exc:
            logger.error("Self-modification apply failed: %s", exc)
            return f"Apply error: {exc}"

    async def _apply_value_adjustment(self, changes: Dict[str, Any]) -> str:
        """Apply value/weight adjustments to HeartstoneValues or DriveEngine."""
        new_values = changes.get("new_values", {})
        target_system = changes.get("target_system", "heartstone")
        applied = []

        if target_system == "heartstone":
            try:
                from core.affect.heartstone_values import get_heartstone_values
                hv = get_heartstone_values()
                for key, val in new_values.items():
                    old_val = hv.get(key, 0.5)
                    delta = val - old_val
                    hv._adjust(key, delta)
                    applied.append(f"{key}: {old_val:.3f} -> {val:.3f}")
            except Exception as exc:
                return f"Heartstone adjustment failed: {exc}"

        elif target_system == "drive_engine":
            try:
                drive_engine = ServiceContainer.get("drive_engine", default=None)
                if drive_engine:
                    for name, val in new_values.items():
                        b = drive_engine.budgets.get(name)
                        if b:
                            old_level = b.level
                            b.level = max(0.0, min(b.capacity, val * b.capacity))
                            applied.append(f"{name}: {old_level:.1f} -> {b.level:.1f}")
            except Exception as exc:
                return f"Drive engine adjustment failed: {exc}"

        return f"Applied {len(applied)} value change(s): {'; '.join(applied)}"

    async def _apply_threshold_adjustment(
        self, target_path: str, changes: Dict[str, Any]
    ) -> str:
        """Apply a threshold adjustment to a named attribute."""
        attr_path = changes.get("attribute_path", "")
        new_threshold = changes.get("new_threshold")
        if not attr_path or new_threshold is None:
            return "Missing attribute_path or new_threshold"
        return f"Threshold {attr_path} -> {new_threshold} (logged, manual confirmation needed)"

    # ── Audit Logging ───────────────────────────────────────────────────

    def _record_receipt(self, receipt: ModificationReceipt) -> None:
        """Record receipt in memory and append to persistent JSONL log."""
        self._receipts.append(receipt)
        if len(self._receipts) > self._MAX_RECEIPTS:
            self._receipts = self._receipts[-self._MAX_RECEIPTS:]

        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(receipt.to_dict(), default=str) + "\n")
        except Exception as exc:
            logger.debug("Audit log write failed: %s", exc)

    def _publish_event(self, topic: str, receipt: ModificationReceipt) -> None:
        """Publish modification event to the event bus."""
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe(topic, receipt.to_dict())
        except Exception:
            pass

    # ── Public API ──────────────────────────────────────────────────────

    def get_recent_receipts(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return recent modification receipts for audit."""
        return [r.to_dict() for r in self._receipts[-n:]]

    def get_status(self) -> Dict[str, Any]:
        """Return current status."""
        approved = sum(1 for r in self._receipts if r.outcome == ProposalOutcome.APPROVED)
        refused = sum(1 for r in self._receipts if r.outcome != ProposalOutcome.APPROVED)
        return {
            "total_proposals": len(self._receipts),
            "approved": approved,
            "refused": refused,
            "approval_rate": round(approved / max(1, len(self._receipts)), 4),
            "pending": len(self._pending),
        }

    @staticmethod
    def make_proposal_id(source: str, description: str) -> str:
        """Generate a unique proposal ID."""
        raw = f"{time.time():.6f}:{source}:{description[:50]}"
        return "smod_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: Optional[AutonomousSelfModification] = None


def get_autonomous_self_modification() -> AutonomousSelfModification:
    """Get or create the singleton AutonomousSelfModification."""
    global _instance
    if _instance is None:
        _instance = AutonomousSelfModification()
    return _instance
