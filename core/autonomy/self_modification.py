"""core/autonomy/self_modification.py -- Will-Authorized Self-Modification
=========================================================================
Aura can propose edits to her own non-critical modules.  Every modification
flows through the Unified Will for authorization.

Process:
  1. Propose a modification (diff, reason, target module)
  2. Classify the target -- PROTECTED modules are rejected outright
  3. Simulate consequences (hermetic AST inspection, range/schema validation)
  4. Route to Unified Will for PROCEED / REFUSE
  5. If approved, freeze the proposal and durably queue it for the
     SafeSelfModification quarantine/promotion pipeline
  6. All proposals (accepted and rejected) go to a hash-chained audit log

PROTECTED (never modifiable):
  - core/will.py                (the Will itself)
  - core/identity/*             (identity core)
  - core/safety/*               (safety gates)
  - core/security/*             (trust/security)
  - core/constitution.py        (constitutional alignment)
  - core/identity/heartstone.py (sacred vows)

MODIFIABLE (with Will authorization and queued promotion):
  - Drive weights               (heartstone_values, drive_engine)
  - Response strategies          (pipeline/, brain/, conversation/)
  - Skill implementations       (skills/, skill_management/)
  - Threshold values             (adaptation/, cognitive/)
  - Self-modification engine     (self_modification/)
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import posixpath
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.governance_context import local_internal_governed_scope
from core.memory.retention_policy import working_history_retention_policy
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.SelfModification.Autonomous")

# ── Persistence ─────────────────────────────────────────────────────────────
_DATA_DIR = state_root() / "data" / "self_modification"
_AUDIT_LOG_PATH = _DATA_DIR / "audit_log.jsonl"
_OUTBOX_PATH = _DATA_DIR / "pending_outbox.jsonl"
_MAX_AUDIT_ENTRIES = working_history_retention_policy("AURA_SELF_MODIFICATION_AUDIT_MAX").max_items
_RUNTIME_SELF_MODIFICATION_ENV = "AURA_ALLOW_RUNTIME_SELF_MODIFICATION"

# Genesis anchor for the tamper-evident audit chain.
_AUDIT_CHAIN_GENESIS = "0" * 64

# Errors that best-effort durable writes are allowed to swallow (after recording
# a degradation). A governance refusal surfaces as RuntimeError, so it MUST be
# in this set — the finding that motivated it was a governed-context write that
# escaped the OSError-only handler after Will approval.
_DURABLE_WRITE_ERRORS = (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError)


# ── Classification ──────────────────────────────────────────────────────────

class ModuleZone(StrEnum):
    """Classification of a module's modifiability."""
    PROTECTED = "protected"       # Never touch
    MODIFIABLE = "modifiable"     # Allowed with Will approval
    UNKNOWN = "unknown"           # Not in allowlist -- treat as protected


class ProposalOutcome(StrEnum):
    APPROVED = "approved"
    QUEUED_FOR_PIPELINE = "queued_for_pipeline"
    REFUSED_BY_WILL = "refused_by_will"
    REFUSED_PROTECTED = "refused_protected"
    REFUSED_SIMULATION = "refused_simulation"
    DROPPED_OVERFLOW = "dropped_overflow"
    ERROR = "error"


# Terminal states a downstream promotion consumer may report back.
_TERMINAL_PROMOTION_STATES = frozenset(
    {"promoted", "quarantined", "rolled_back", "failed"}
)


# Protected path prefixes (relative to project root)
_PROTECTED_PREFIXES = (
    "core/will.py",
    "core/identity/",
    "core/safety/",
    "core/security/",
    "core/constitution.py",
    "core/values/constitutional_alignment.py",
    "core/identity/heartstone.py",
    "core/container.py",
    "core/values/prime_directives.py",
)

# Direct function/attribute-call names that never belong in a self-authored
# patch. Attribute chains are matched by their dotted suffix as well, so
# `sys.modules`-style escapes are caught even when the base is aliased.
_BANNED_CODE_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "open",
    "input",
    "breakpoint",
    "os.system",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
}

# Import of any of these modules from inside a patch is a capability escalation
# — file, process, network, native memory, or interpreter-internals reach.
_BANNED_IMPORT_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "importlib",
    "pickle",
    "marshal",
    "builtins",
    "gc",
    "resource",
    "mmap",
    "signal",
    "pty",
    "fcntl",
    "multiprocessing",
    "threading",
}

# Dunder attributes that let code walk out of its lexical scope to globals,
# builtins, or the class hierarchy.
_BANNED_DUNDER_ATTRS = {
    "__globals__",
    "__builtins__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__code__",
    "__dict__",
    "__class__",
    "__import__",
    "__loader__",
    "__closure__",
    "__getattribute__",
}


def _runtime_self_modification_enabled() -> bool:
    raw = os.getenv(_RUNTIME_SELF_MODIFICATION_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _live_runtime_application_enabled() -> bool:
    """Live in-process mutation is intentionally disabled.

    The legacy runtime flag is still read by older callers, but it no longer
    authorizes direct value/config mutation in the foreground runtime. Approved
    proposals are queued for the SafeSelfModification quarantine/promotion path
    so the running process is never patched under itself.
    """
    return False


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _canonical_changes_json(changes: Any) -> str:
    """Deterministic, mutation-proof serialization of a change mapping."""
    try:
        return json.dumps(changes or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # A non-serializable change payload cannot be authorized safely.
        return json.dumps({"__unserializable__": repr(changes)[:512]}, sort_keys=True)


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
    changes: dict[str, Any]       # Structured change data
    source: str                   # Which subsystem proposed this
    priority: float = 0.5         # 0-1
    timestamp: float = field(default_factory=lambda: time.time())

    def content_hash(self) -> str:
        """Bind authorization to the WHOLE artifact, not just its prose.

        Covering the structured ``changes`` (the executable patch or value
        payload), source, and priority means a caller cannot alter what will be
        promoted after the Will has authorized a different content hash.
        """
        raw = "\x1f".join(
            (
                self.target_path,
                self.diff_summary,
                self.description,
                self.source,
                f"{float(self.priority):.6f}",
                _canonical_changes_json(self.changes),
            )
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def frozen(self, *, content_hash: str, will_receipt_id: str) -> QueuedProposal:
        """Snapshot into an immutable, durably-storable queued proposal."""
        return QueuedProposal(
            proposal_id=self.proposal_id,
            target_path=self.target_path,
            description=self.description,
            diff_summary=self.diff_summary,
            changes_json=_canonical_changes_json(self.changes),
            source=self.source,
            priority=float(self.priority),
            timestamp=self.timestamp,
            content_hash=content_hash,
            will_receipt_id=will_receipt_id,
        )


@dataclass(frozen=True)
class QueuedProposal:
    """An approved proposal, frozen at authorization time.

    ``changes_json`` is a canonical string, so the executable payload the Will
    authorized cannot be mutated in place while it waits in the outbox.
    """
    proposal_id: str
    target_path: str
    description: str
    diff_summary: str
    changes_json: str
    source: str
    priority: float
    timestamp: float
    content_hash: str
    will_receipt_id: str

    def to_record(self, status: str = "queued") -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "target_path": self.target_path,
            "description": self.description,
            "diff_summary": self.diff_summary,
            "changes_json": self.changes_json,
            "source": self.source,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "content_hash": self.content_hash,
            "will_receipt_id": self.will_receipt_id,
            "status": status,
        }

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> QueuedProposal:
        return cls(
            proposal_id=str(d.get("proposal_id", "")),
            target_path=str(d.get("target_path", "")),
            description=str(d.get("description", "")),
            diff_summary=str(d.get("diff_summary", "")),
            changes_json=str(d.get("changes_json", "{}")),
            source=str(d.get("source", "")),
            priority=float(d.get("priority", 0.5) or 0.0),
            timestamp=float(d.get("timestamp", 0.0) or 0.0),
            content_hash=str(d.get("content_hash", "")),
            will_receipt_id=str(d.get("will_receipt_id", "")),
        )


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
    content_hash: str = ""
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
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
            "content_hash": self.content_hash,
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
    _MAX_RECEIPTS = working_history_retention_policy("AURA_SELF_MODIFICATION_RECEIPT_MAX").max_items

    def __init__(self) -> None:
        self._pending: list[QueuedProposal] = []
        self._receipts: list[ModificationReceipt] = []
        self._started = False
        self._lock = threading.RLock()
        self._audit_seq = 0
        self._last_audit_hash = _AUDIT_CHAIN_GENESIS
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("AutonomousSelfModification created -- awaiting start()")

    # ── Runtime-state guard ─────────────────────────────────────────────

    def _ensure_runtime_state(self) -> None:
        """Make the instance safe to use even if ``__init__`` was bypassed.

        Recovery paths and tests build the service with ``__new__``; every
        method that touches mutable state calls this first so a missing
        attribute never turns into an AttributeError mid-authorization.
        """
        if not hasattr(self, "_lock"):
            self._lock = threading.RLock()
        if not hasattr(self, "_pending"):
            self._pending = []
        if not hasattr(self, "_receipts"):
            self._receipts = []
        if not hasattr(self, "_audit_seq"):
            self._audit_seq = 0
        if not hasattr(self, "_last_audit_hash"):
            self._last_audit_hash = _AUDIT_CHAIN_GENESIS

    async def start(self) -> None:
        """Register in ServiceContainer and recover durable state."""
        if getattr(self, "_started", False):
            return
        self._ensure_runtime_state()
        self._recover_audit_chain_tail()
        self._recover_pending_from_outbox()
        ServiceContainer.register_instance(
            "autonomous_self_modification", self, required=False
        )
        self._started = True
        logger.info(
            "AutonomousSelfModification ONLINE -- %d pending recovered, audit seq %d",
            len(self._pending),
            self._audit_seq,
        )

    # ── Zone Classification ─────────────────────────────────────────────

    @staticmethod
    def classify_target(rel_path: str) -> ModuleZone:
        """Classify a module path into its modification zone.

        The path is canonicalized before matching: back-slashes are folded,
        leading slashes stripped, and dot segments resolved. Any path that is
        absolute, escapes the project root, or still contains a ``..`` segment
        after normalization fails CLOSED to PROTECTED — a prefix such as
        ``core/autonomy/../security/x`` must never inherit a modifiable prefix.
        """
        if not isinstance(rel_path, str) or not rel_path.strip():
            return ModuleZone.PROTECTED

        candidate = rel_path.replace("\\", "/").strip()
        if candidate.startswith("/") or (len(candidate) > 1 and candidate[1] == ":"):
            # Absolute POSIX path or Windows drive path — not a project-relative
            # module reference.
            return ModuleZone.PROTECTED

        normalized = posixpath.normpath(candidate).lstrip("/")
        # normpath collapses ``a/../b`` to ``b`` and ``../x`` to ``../x``; any
        # residual parent reference means the path tried to escape the root.
        if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
            return ModuleZone.PROTECTED
        if "\x00" in normalized:
            return ModuleZone.PROTECTED

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
          2. Reject PROTECTED / UNKNOWN outright
          3. Simulate consequences (hermetic inspection)
          4. Consult Unified Will (authorization bound to the content hash)
          5. Verify audit storage, then durably queue the frozen proposal
          6. Log a hash-chained receipt
        """
        self._ensure_runtime_state()
        content_hash = proposal.content_hash()
        logger.info(
            "Self-modification proposal [%s] hash=%s: %s -> %s",
            proposal.proposal_id, content_hash, proposal.target_path,
            proposal.description[:80],
        )

        # 1. Zone check
        zone = self.classify_target(proposal.target_path)
        if zone in (ModuleZone.PROTECTED, ModuleZone.UNKNOWN):
            reason = (
                "Target is in PROTECTED zone -- modification forbidden"
                if zone == ModuleZone.PROTECTED
                else "Target is not in the modifiable allowlist"
            )
            receipt = self._new_receipt(
                proposal, ProposalOutcome.REFUSED_PROTECTED, zone, content_hash,
                will_reason=reason,
            )
            self._record_receipt(receipt)
            logger.warning(
                "REFUSED (%s): %s -> %s",
                zone.value, proposal.target_path, proposal.description[:60],
            )
            return receipt

        # 2. Simulate consequences
        sim_ok, sim_detail = await self._simulate(proposal)
        if not sim_ok:
            receipt = self._new_receipt(
                proposal, ProposalOutcome.REFUSED_SIMULATION, zone, content_hash,
                simulation_result=sim_detail,
                will_reason=f"Simulation failed: {sim_detail}",
            )
            self._record_receipt(receipt)
            logger.warning(
                "REFUSED (simulation): %s -> %s", proposal.target_path, sim_detail[:80]
            )
            return receipt

        # 3. Consult the Unified Will — authorization is bound to content_hash.
        try:
            from core.will import ActionDomain, get_will
            will = get_will()
            decision = will.decide(
                content=(
                    f"Self-modification proposal: {proposal.description}. "
                    f"Target: {proposal.target_path}. "
                    f"Content-hash: {content_hash}. "
                    f"Changes: {proposal.diff_summary[:200]}. "
                    f"Simulation: {sim_detail[:100]}"
                ),
                source=f"self_modification/{proposal.source}",
                domain=getattr(ActionDomain, "SELF_MODIFICATION", ActionDomain.STATE_MUTATION),
                priority=proposal.priority,
                context={
                    "proposal_id": proposal.proposal_id,
                    "zone": zone.value,
                    "content_hash": content_hash,
                    "simulation_passed": sim_ok,
                },
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('self_modification', exc)
            receipt = self._new_receipt(
                proposal, ProposalOutcome.ERROR, zone, content_hash,
                will_reason=f"Will consultation failed: {exc}",
            )
            self._record_receipt(receipt)
            logger.error("Self-modification Will consultation failed: %s", exc)
            return receipt

        if not decision.is_approved():
            receipt = self._new_receipt(
                proposal, ProposalOutcome.REFUSED_BY_WILL, zone, content_hash,
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

        # 3b. The artifact must not have changed between hashing and approval.
        post_hash = proposal.content_hash()
        if post_hash != content_hash:
            receipt = self._new_receipt(
                proposal, ProposalOutcome.ERROR, zone, content_hash,
                will_receipt_id=decision.receipt_id,
                will_reason=(
                    "Proposal artifact mutated after authorization "
                    f"(authorized {content_hash}, now {post_hash})"
                ),
                simulation_result=sim_detail,
            )
            self._record_receipt(receipt)
            logger.error(
                "REFUSED: self-modification artifact changed after Will approval "
                "(%s -> %s)", content_hash, post_hash,
            )
            self._publish_event("self_modification.refused", receipt)
            return receipt

        audit_ok, audit_detail = self._audit_log_ready()
        if not audit_ok:
            receipt = self._new_receipt(
                proposal, ProposalOutcome.ERROR, zone, content_hash,
                will_receipt_id=decision.receipt_id,
                will_reason=f"Audit log unavailable; refused runtime application: {audit_detail}",
                simulation_result=sim_detail,
            )
            self._record_receipt(receipt)
            logger.error(
                "REFUSED runtime self-modification because audit log is unavailable: %s",
                audit_detail,
            )
            self._publish_event("self_modification.refused", receipt)
            return receipt

        # 4. Freeze and durably queue the approved change for the audited
        # promotion pipeline. Runtime value/config mutation is never applied in
        # this process; the SafeSelfModification consumer quarantines, tests,
        # and promotes it out of band.
        change_type = str((proposal.changes or {}).get("type", "unknown"))
        frozen = proposal.frozen(
            content_hash=content_hash, will_receipt_id=decision.receipt_id
        )
        receipt = self._new_receipt(
            proposal, ProposalOutcome.QUEUED_FOR_PIPELINE, zone, content_hash,
            will_receipt_id=decision.receipt_id,
            will_reason=decision.reason,
            simulation_result=(
                f"{sim_detail}; queued: {change_type} requires SafeSelfModification "
                "quarantine, tests, receipts, and promotion"
            ),
        )
        self._record_receipt(receipt)
        self._queue_proposal(frozen)
        logger.warning(
            "QUEUED self-modification instead of live-applying: %s (%s)",
            proposal.target_path,
            change_type,
        )
        self._publish_event("self_modification.queued", receipt)
        return receipt

    def _new_receipt(
        self,
        proposal: ModificationProposal,
        outcome: ProposalOutcome,
        zone: ModuleZone,
        content_hash: str,
        *,
        will_receipt_id: str = "",
        will_reason: str = "",
        simulation_result: str = "",
    ) -> ModificationReceipt:
        return ModificationReceipt(
            proposal_id=proposal.proposal_id,
            target_path=proposal.target_path,
            description=proposal.description,
            diff_summary=proposal.diff_summary,
            source=proposal.source,
            outcome=outcome,
            zone=zone.value,
            content_hash=content_hash,
            will_receipt_id=will_receipt_id,
            will_reason=will_reason,
            simulation_result=simulation_result,
        )

    # ── Simulation ──────────────────────────────────────────────────────

    async def _simulate(self, proposal: ModificationProposal) -> tuple[bool, str]:
        """Simulate a proposed modification.

        For value/weight changes: finite-range validation.
        For code changes: hermetic AST inspection (syntax + capability denylist).
        For config changes: a typed schema.
        Unknown change types FAIL CLOSED — the least-understood forms must not
        be the easiest to slip past the Will.
        """
        changes = proposal.changes or {}
        change_type = changes.get("type", "unknown")

        try:
            if change_type == "value_adjustment":
                new_values = changes.get("new_values", {})
                if not isinstance(new_values, dict) or not new_values:
                    return False, "value_adjustment missing a new_values mapping"
                for key, val in new_values.items():
                    if isinstance(val, bool) or not isinstance(val, (int, float)):
                        return False, f"Non-numeric value for {key}: {val!r}"
                    if not math.isfinite(val):
                        return False, f"Non-finite value for {key}: {val!r}"
                    if val < 0.0 or val > 1.0:
                        return False, f"Value {key}={val} out of [0.0, 1.0] range"
                return True, f"Value adjustment validated: {len(new_values)} change(s)"

            elif change_type == "threshold_adjustment":
                new_threshold = changes.get("new_threshold")
                if isinstance(new_threshold, bool) or not isinstance(new_threshold, (int, float)):
                    return False, f"Non-numeric threshold: {new_threshold!r}"
                if not math.isfinite(new_threshold):
                    return False, f"Non-finite threshold: {new_threshold!r}"
                return True, f"Threshold adjustment validated: {new_threshold}"

            elif change_type == "code_patch":
                return self._inspect_code_patch(changes.get("new_code", ""))

            elif change_type == "config_update":
                return self._validate_config_update(changes)

            else:
                return False, f"Unknown change type '{change_type}' -- refused (fail-closed)"

        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation('self_modification', exc)
            return False, f"Simulation error: {exc}"

    @staticmethod
    def _inspect_code_patch(new_code: Any) -> tuple[bool, str]:
        """Hermetic static inspection of a candidate code patch.

        This is not a sandbox — it cannot prove semantic safety — but it closes
        the obvious capability escalations the old syntax-only check let through:
        dangerous imports, dynamic attribute access, dunder escapes to globals /
        builtins / the class hierarchy, file/process/network calls, and the most
        common unbounded-loop shape.
        """
        if not isinstance(new_code, str) or not new_code.strip():
            return False, "Empty code patch"
        try:
            tree = ast.parse(new_code, filename="<self_modification_patch>")
        except SyntaxError as se:
            return False, f"Syntax error in patch: {se}"

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _BANNED_IMPORT_MODULES:
                        violations.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in _BANNED_IMPORT_MODULES:
                    violations.append(f"from {node.module} import ...")
            elif isinstance(node, ast.Attribute):
                if node.attr in _BANNED_DUNDER_ATTRS:
                    violations.append(f"attribute {node.attr}")
                else:
                    dotted = _call_name(node)
                    if dotted in _BANNED_CODE_CALLS:
                        violations.append(dotted)
            elif isinstance(node, ast.Name) and node.id in _BANNED_DUNDER_ATTRS:
                violations.append(node.id)
            elif isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in _BANNED_CODE_CALLS:
                    violations.append(name)
            elif isinstance(node, ast.While):
                if _is_truthy_constant(node.test) and not _loop_has_exit(node):
                    violations.append("while True without break/return/raise")

        if violations:
            unique = sorted(set(violations))
            return False, f"Unsafe code patch call(s): {', '.join(unique)}"
        return True, f"Code patch syntax valid ({len(new_code)} chars)"

    @staticmethod
    def _validate_config_update(changes: dict[str, Any]) -> tuple[bool, str]:
        """Typed schema for config updates — the least-understood proposal form.

        A config update must name exactly one key and one JSON-scalar value.
        Numeric values must be finite; keys must be plain dotted identifiers
        (no dunder, no path traversal), so a 'config' change can never smuggle
        an executable payload or an unbounded object through the Will.
        """
        key = changes.get("config_key")
        if not isinstance(key, str) or not key.strip():
            return False, "config_update missing a string config_key"
        if ".." in key or "/" in key or key.startswith("__") or len(key) > 128:
            return False, f"config_update key not permitted: {key!r}"
        if not all(part.isidentifier() for part in key.split(".") if part):
            return False, f"config_update key is not a dotted identifier: {key!r}"
        if "config_value" not in changes:
            return False, "config_update missing config_value"
        value = changes.get("config_value")
        if isinstance(value, bool):
            pass  # bool is an acceptable scalar
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                return False, f"config_update value is non-finite: {value!r}"
        elif isinstance(value, str):
            if len(value) > 512:
                return False, "config_update string value exceeds 512 chars"
        elif value is None:
            pass
        else:
            return False, f"config_update value must be a JSON scalar, got {type(value).__name__}"
        return True, f"Config update validated: {key}"

    # ── Application (permanently gated off in this process) ──────────────

    async def _apply(self, proposal: ModificationProposal) -> str:
        """Apply an approved modification — disabled in the foreground runtime.

        Live in-process mutation is never performed here: approved changes are
        queued for the SafeSelfModification pipeline. These helpers remain only
        so a downstream consumer can call them under its own governed scope;
        called in-process they refuse rather than mutate protected live values.
        """
        if not _live_runtime_application_enabled():
            return "Live runtime application disabled -- proposal must go through the promotion pipeline"
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('self_modification', exc)
            logger.error("Self-modification apply failed: %s", exc)
            return f"Apply error: {exc}"

    async def _apply_value_adjustment(self, changes: dict[str, Any]) -> str:
        """Apply value/weight adjustments — refused unless live application is on."""
        if not _live_runtime_application_enabled():
            return "Live value mutation disabled -- refused"
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
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation('self_modification', exc)
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
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation('self_modification', exc)
                return f"Drive engine adjustment failed: {exc}"

        return f"Applied {len(applied)} value change(s): {'; '.join(applied)}"

    async def _apply_threshold_adjustment(
        self, target_path: str, changes: dict[str, Any]
    ) -> str:
        """Apply a threshold adjustment — refused unless live application is on."""
        if not _live_runtime_application_enabled():
            return "Live threshold mutation disabled -- refused"
        attr_path = changes.get("attribute_path", "")
        new_threshold = changes.get("new_threshold")
        if not attr_path or new_threshold is None:
            return "Missing attribute_path or new_threshold"
        return f"Threshold {attr_path} -> {new_threshold} (logged, manual confirmation needed)"

    # ── Durable promotion pipeline (consumer API) ───────────────────────

    def claim_next_pending(self) -> dict[str, Any] | None:
        """Claim the oldest queued proposal for the promotion pipeline.

        Returns the frozen proposal record (including its authorized
        ``content_hash``) and marks it claimed in the durable outbox so a
        restart does not re-hand it out. Returns None when nothing is queued.
        """
        self._ensure_runtime_state()
        with self._lock:
            if not self._pending:
                return None
            claimed = self._pending.pop(0)
        self._append_outbox_record(claimed, status="claimed")
        return claimed.to_record(status="claimed")

    def record_promotion_outcome(
        self, proposal_id: str, status: str, detail: str = ""
    ) -> bool:
        """Record a terminal promotion result reported by the consumer.

        ``status`` must be one of promoted / quarantined / rolled_back / failed.
        Writes a durable acknowledgement line and publishes an event so the
        producer learns the fate of each queued change.
        """
        self._ensure_runtime_state()
        status = str(status).strip().lower()
        if status not in _TERMINAL_PROMOTION_STATES:
            logger.warning("Rejected unknown promotion status %r for %s", status, proposal_id)
            return False
        record = {
            "proposal_id": str(proposal_id),
            "status": status,
            "detail": str(detail)[:512],
            "timestamp": time.time(),
        }
        self._durable_append(
            _OUTBOX_PATH, record, "autonomy.self_modification.promotion_ack"
        )
        self._publish_dict(f"self_modification.{status}", record)
        return True

    def _queue_proposal(self, frozen: QueuedProposal) -> None:
        """Store an approved, frozen proposal for external audited promotion.

        Durable first (survives restart), then in memory. When the in-memory
        backlog would exceed the cap, the OLDEST queued proposal is dead-lettered
        with a receipt and event rather than silently sliced away.
        """
        self._ensure_runtime_state()
        self._append_outbox_record(frozen, status="queued")
        with self._lock:
            self._pending.append(frozen)
            overflow: list[QueuedProposal] = []
            while len(self._pending) > self._MAX_PENDING:
                overflow.append(self._pending.pop(0))
        for dropped in overflow:
            self._dead_letter(dropped)

    def _dead_letter(self, dropped: QueuedProposal) -> None:
        """Surface an overflow drop as a receipt + event, never a silent loss."""
        self._append_outbox_record(dropped, status="dropped_overflow")
        receipt = ModificationReceipt(
            proposal_id=dropped.proposal_id,
            target_path=dropped.target_path,
            description=dropped.description,
            diff_summary=dropped.diff_summary,
            source=dropped.source,
            outcome=ProposalOutcome.DROPPED_OVERFLOW,
            will_receipt_id=dropped.will_receipt_id,
            zone=ModuleZone.MODIFIABLE.value,
            content_hash=dropped.content_hash,
            will_reason=(
                f"Pending backlog exceeded {self._MAX_PENDING}; oldest approved "
                "proposal dead-lettered before promotion"
            ),
        )
        self._record_receipt(receipt)
        self._publish_event("self_modification.dropped", receipt)
        record_degradation(
            "self_modification",
            RuntimeError(f"self-modification backlog overflow dropped {dropped.proposal_id}"),
            severity="warning",
            action="dead-lettered an approved self-modification proposal on backlog overflow",
        )
        logger.error(
            "DEAD-LETTER self-modification proposal %s (backlog > %d)",
            dropped.proposal_id, self._MAX_PENDING,
        )

    def _append_outbox_record(self, frozen: QueuedProposal, *, status: str) -> None:
        self._durable_append(
            _OUTBOX_PATH,
            frozen.to_record(status=status),
            "autonomy.self_modification.outbox",
        )

    def _recover_pending_from_outbox(self) -> None:
        """Rebuild the pending list from the durable outbox on start.

        The last status wins per proposal_id; only ``queued`` proposals that
        were never claimed or completed are re-admitted.
        """
        latest: dict[str, dict[str, Any]] = {}
        for record in self._read_jsonl(_OUTBOX_PATH):
            pid = record.get("proposal_id")
            if isinstance(pid, str) and pid:
                latest[pid] = record
        recovered = [
            QueuedProposal.from_record(rec)
            for rec in latest.values()
            if rec.get("status") == "queued" and rec.get("content_hash")
        ]
        recovered.sort(key=lambda p: p.timestamp)
        with self._lock:
            self._pending = recovered[-self._MAX_PENDING:]

    # ── Audit Logging (hash-chained, tamper-evident) ────────────────────

    @staticmethod
    def _audit_log_ready() -> tuple[bool, str]:
        """Verify durable audit storage before any live runtime mutation.

        Fails closed on ANY write barrier, not only OSError: the file-write
        gateway can raise RuntimeError or a governance refusal, and those must
        produce the intended fail-closed receipt rather than escaping after
        Will approval.
        """
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope(
                "autonomy.self_modification.audit_log_probe",
                domain="file_write",
                receipt_prefix="self-modification-audit-probe",
            ):
                get_file_write_gateway().append_text(
                    _AUDIT_LOG_PATH,
                    "",
                    encoding="utf-8",
                    source="autonomy.self_modification.audit_log_probe",
                )
            return True, "audit log writable"
        except _DURABLE_WRITE_ERRORS as exc:
            record_degradation("self_modification", exc)
            return False, str(exc)

    def _record_receipt(self, receipt: ModificationReceipt) -> None:
        """Record receipt in memory and append a hash-chained JSONL entry."""
        self._ensure_runtime_state()
        with self._lock:
            self._receipts.append(receipt)
            if len(self._receipts) > self._MAX_RECEIPTS:
                self._receipts = self._receipts[-self._MAX_RECEIPTS:]
            seq = self._audit_seq + 1
            prev_hash = self._last_audit_hash
            entry = receipt.to_dict()
            entry_hash = self._chain_hash(seq, prev_hash, entry)
            chained = {
                "seq": seq,
                "prev_hash": prev_hash,
                "entry": entry,
                "entry_hash": entry_hash,
            }

        wrote = self._durable_append(
            _AUDIT_LOG_PATH, chained, "autonomy.self_modification.record_receipt"
        )
        if wrote:
            with self._lock:
                # Advance the chain head only after a durable write, so a failed
                # append cannot leave an unrecoverable gap in the sequence.
                self._audit_seq = seq
                self._last_audit_hash = entry_hash

    @staticmethod
    def _chain_hash(seq: int, prev_hash: str, entry: dict[str, Any]) -> str:
        payload = json.dumps(
            {"seq": seq, "prev_hash": prev_hash, "entry": entry},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _recover_audit_chain_tail(self) -> None:
        """Restore chain head (seq + last hash) from the audit log tail."""
        last = None
        for record in self._read_jsonl(_AUDIT_LOG_PATH):
            if isinstance(record, dict) and "entry_hash" in record:
                last = record
        if last is not None:
            try:
                self._audit_seq = int(last.get("seq", 0))
                self._last_audit_hash = str(last.get("entry_hash") or _AUDIT_CHAIN_GENESIS)
            except (TypeError, ValueError):
                self._audit_seq = 0
                self._last_audit_hash = _AUDIT_CHAIN_GENESIS

    def verify_audit_chain(self) -> tuple[bool, str]:
        """Recompute the chain from disk; detect any break or tamper."""
        prev = _AUDIT_CHAIN_GENESIS
        seq_expected = 0
        count = 0
        for record in self._read_jsonl(_AUDIT_LOG_PATH):
            if not isinstance(record, dict) or "entry_hash" not in record:
                continue
            seq_expected += 1
            count += 1
            if int(record.get("seq", -1)) != seq_expected:
                return False, f"sequence break at entry {count} (seq {record.get('seq')})"
            if str(record.get("prev_hash")) != prev:
                return False, f"prev_hash mismatch at seq {seq_expected}"
            recomputed = self._chain_hash(seq_expected, prev, record.get("entry", {}))
            if recomputed != record.get("entry_hash"):
                return False, f"entry_hash mismatch at seq {seq_expected}"
            prev = str(record.get("entry_hash"))
        return True, f"audit chain intact ({count} entries)"

    # ── Shared durable I/O helpers ──────────────────────────────────────

    @staticmethod
    def _durable_append(path: Path, record: dict[str, Any], source: str) -> bool:
        """Governed, best-effort append of one JSONL record. Returns success."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope(
                source, domain="file_write", receipt_prefix="self-modification-log"
            ):
                get_file_write_gateway().append_text(
                    path,
                    json.dumps(record, default=str) + "\n",
                    encoding="utf-8",
                    source=source,
                )
            return True
        except _DURABLE_WRITE_ERRORS as exc:
            record_degradation('self_modification', exc)
            logger.debug("Durable append failed for %s: %s", source, exc)
            return False

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file, skipping unparsable lines. Empty on any error."""
        out: list[dict[str, Any]] = []
        try:
            if not path.exists():
                return out
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(obj, dict):
                        out.append(obj)
        except OSError as exc:
            record_degradation('self_modification', exc)
        return out

    def _publish_event(self, topic: str, receipt: ModificationReceipt) -> None:
        """Publish a receipt-carrying modification event to the event bus."""
        self._publish_dict(topic, receipt.to_dict())

    @staticmethod
    def _publish_dict(topic: str, payload: dict[str, Any]) -> None:
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe(topic, payload)
        except (ImportError, AttributeError, RuntimeError):
            pass  # no-op: intentional

    # ── Public API ──────────────────────────────────────────────────────

    def get_recent_receipts(self, n: int = 20) -> list[dict[str, Any]]:
        """Return recent modification receipts for audit."""
        self._ensure_runtime_state()
        return [r.to_dict() for r in self._receipts[-n:]]

    def get_status(self) -> dict[str, Any]:
        """Return current status.

        ``authorized`` counts proposals the Will approved (queued + any legacy
        APPROVED); ``authorization_rate`` uses that numerator so a Will-approved
        queueing is no longer reported as a zero approval rate.
        """
        self._ensure_runtime_state()
        approved = sum(1 for r in self._receipts if r.outcome == ProposalOutcome.APPROVED)
        queued = sum(1 for r in self._receipts if r.outcome == ProposalOutcome.QUEUED_FOR_PIPELINE)
        dropped = sum(1 for r in self._receipts if r.outcome == ProposalOutcome.DROPPED_OVERFLOW)
        authorized = approved + queued
        refused = sum(
            1
            for r in self._receipts
            if r.outcome
            not in {
                ProposalOutcome.APPROVED,
                ProposalOutcome.QUEUED_FOR_PIPELINE,
                ProposalOutcome.DROPPED_OVERFLOW,
            }
        )
        total = len(self._receipts)
        return {
            "total_proposals": total,
            "approved": approved,
            "queued": queued,
            "authorized": authorized,
            "dropped_overflow": dropped,
            "refused": refused,
            "approval_rate": round(authorized / max(1, total), 4),
            "pending": len(self._pending),
            "audit_seq": self._audit_seq,
            "live_runtime_application_enabled": _live_runtime_application_enabled(),
            "legacy_runtime_flag_set": _runtime_self_modification_enabled(),
        }

    @staticmethod
    def make_proposal_id(source: str, description: str) -> str:
        """Generate a unique proposal ID."""
        raw = f"{time.time():.6f}:{source}:{description[:50]}"
        return "smod_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── AST helpers ─────────────────────────────────────────────────────────────

def _is_truthy_constant(node: ast.AST) -> bool:
    """True for a literal that a ``while`` test treats as always-true."""
    if isinstance(node, ast.Constant):
        try:
            return bool(node.value)
        except (TypeError, ValueError):
            return False
    return False


def _loop_has_exit(loop: ast.While) -> bool:
    """Whether a while-loop body can terminate (break / return / raise).

    Nested loops own their own ``break``, so a break inside an inner loop does
    not count as an exit for the outer one.
    """
    for node in ast.walk(loop):
        if node is loop:
            continue
        if isinstance(node, (ast.Return, ast.Raise)):
            return True
        if isinstance(node, ast.Break) and _nearest_loop_is(loop, node):
            return True
    return False


def _nearest_loop_is(loop: ast.While, brk: ast.Break) -> bool:
    """True if ``brk``'s nearest enclosing loop is ``loop`` (not a nested one)."""
    for child in ast.walk(loop):
        if isinstance(child, (ast.For, ast.While, ast.AsyncFor)) and child is not loop:
            for inner in ast.walk(child):
                if inner is brk:
                    return False
    return True


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: AutonomousSelfModification | None = None


def get_autonomous_self_modification() -> AutonomousSelfModification:
    """Get or create the singleton AutonomousSelfModification."""
    global _instance
    if _instance is None:
        _instance = AutonomousSelfModification()
    return _instance
