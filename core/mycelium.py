"""
Mycelial Network v3.0 — Direct Root System
==========================================

Inspired by Physarum polycephalum (slime mold), this module provides:

1. **HardwiredPathways**: Regex-based intent→skill mappings with parameter extraction.
   These are the "direct roots": they skip the LLM reasoning loop, not the
   constitutional effect boundary. A matched pathway is a ROUTING proposal.
   Every consumer that turns one into an effect passes it through the same
   authority gate as any other action — ``allow_direct_user_shortcut`` for the
   incoming lane, ``approve_response`` for a direct reflex — and a pathway that
   the gate declines yields to the governed path.

   This file used to call them "unblockable" and say they "bypass the LLM
   reasoning loop entirely". The runtime never granted the first and the second
   was read as licence for the first (CP126 39f4805f). The description is now
   the contract, and ``tests/test_mycelium_routing_authority.py`` holds every
   effect-producing consumer to it.

2. **Physarum Reinforcement**: Pathways strengthen on success, weaken on failure.
   Conductivity naturally converges to the most reliable routes.

3. **Hyphae Network**: General-purpose connections between subsystems with
   rooted_flow context managers for stall detection and emergency override.

4. **Autonomous Discovery**: After non-hardwired skill executions succeed,
   the network proposes new pathways (slime mold exploration).

5. **Introspection API**: Full topology reporting for UI visualization and
   health monitoring.

Architecture:
   User Input
       ↓
   MycelialNetwork.match_hardwired()  ← FIRST (Hardwired Shortcuts, zero latency)
       ↓ (if no match)
   IntentRouter.classify()            ← SECOND (LLM-based reasoning, slower)
"""

import ast
import asyncio
import contextlib
import hashlib
import hmac
import inspect
import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from itertools import islice
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, ClassVar, Optional, TypeVar

from pydantic import BaseModel, Field

from core.governance_context import local_internal_governed_scope
from core.runtime.background_policy import foreground_only_runtime
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.flags import FlagKind, declare
from core.runtime.sqlite_support import connecting
from core.utils.concurrency import run_io_bound
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.Mycelium")

T = TypeVar("T")

# Regex safety bounds for routing patterns (registered or restored from the
# vault). A catastrophic-backtracking pattern searched against user text on the
# hot routing path is a ReDoS lever; these caps reject the worst offenders and
# the match wrapper bounds the input length actually scanned.
_MAX_PATTERN_LEN = 512
_MAX_ROUTE_INPUT_LEN = 4096
# Heuristic markers of catastrophic backtracking: nested/adjacent unbounded
# quantifiers. Not exhaustive, but blocks the classic (a+)+ / (.*)* shapes.
_REDOS_MARKERS = (
    re.compile(r"\([^)]*[+*][^)]*\)[+*]"),   # (…+…)+  /  (…*…)*
    re.compile(r"[+*]\s*[+*]"),              # ** / *+ / +* / ++
    re.compile(r"\{\d{4,}"),                 # {1000,} huge repetition
)


def _validate_route_pattern(pattern: str) -> str:
    """Reject oversized or catastrophic-backtracking routing patterns."""
    text = str(pattern or "")
    if len(text) > _MAX_PATTERN_LEN:
        raise ValueError(f"route pattern exceeds {_MAX_PATTERN_LEN} chars")
    for marker in _REDOS_MARKERS:
        if marker.search(text):
            raise ValueError("route pattern rejected: potential catastrophic backtracking")
    return text


def _compile_route_pattern(pattern: str) -> "re.Pattern[str]":
    """Compile a routing regex after ReDoS validation."""
    return re.compile(_validate_route_pattern(pattern), re.IGNORECASE)


def _safe_pattern_search(compiled: "re.Pattern[str]", text: str):
    """Search a routing regex against length-bounded input."""
    return compiled.search(str(text or "")[:_MAX_ROUTE_INPUT_LEN])


#: Where the vault's tamper-evidence key lives. Local and same-uid: this makes
#: an edited vault file detectable, it does NOT authenticate against an
#: adversary who can already read the user's home directory. The distinction is
#: carried through to the restore path, which reports what it verified rather
#: than asserting the topology is trusted (CP126 3901c6f3).
_VAULT_MAC_KEY_FILENAME = "mycelium_vault.key"
#: Key length, named once so the mint and the length check cannot disagree.
_VAULT_MAC_KEY_BYTES = 32
_VAULT_MAC_ALGORITHM = "hmac-sha256"


def _vault_mac_key(base_dir: Path) -> bytes | None:
    """Read or mint the vault tamper-evidence key, or None if it can't exist."""
    key_path = base_dir / "data" / _VAULT_MAC_KEY_FILENAME
    try:
        if key_path.exists():
            # NO .strip(). The key is 32 raw random bytes, and stripping treats
            # whitespace-valued bytes as padding: a key that happens to begin or
            # end with 0x20/0x09/0x0a/0x0b/0x0c/0x0d comes back SHORTER than it
            # was written. Measured at ~4.6% of generated keys, each of which
            # silently breaks vault tamper-evidence forever after — the MAC is
            # then computed with a key that never existed.
            existing = key_path.read_bytes()
            # Length-checked, not merely non-empty. The exclusive create below
            # cannot truncate an existing key, but it can still be interrupted
            # partway through its single write, leaving a short file. Accepting
            # that would pin the vault to a truncated key forever; treating it
            # as absent lets the miss below mint a whole one.
            if len(existing) == _VAULT_MAC_KEY_BYTES:
                return existing
        import secrets

        raw = secrets.token_bytes(_VAULT_MAC_KEY_BYTES)
        with local_internal_governed_scope(
            "mycelium.vault_mac_key",
            domain="file_write",
        ):
            authoritative = get_file_write_gateway().provision_private_bytes(
                key_path,
                raw,
                expected_size=_VAULT_MAC_KEY_BYTES,
                mode=0o600,
                source="mycelium.vault_mac_key",
            )
        return authoritative
    except (OSError, RuntimeError, ValueError) as exc:
        record_degradation(
            "mycelium",
            exc,
            severity="warning",
            action=(
                "wrote or read the root vault without tamper evidence; a "
                "restored topology will be reported as unattested"
            ),
            enforce_failure_policy=False,
        )
        return None


def _vault_mac(key: bytes, encoded: str) -> str:
    return hmac.new(key, encoded.encode("utf-8"), hashlib.sha256).hexdigest()


#: What the topology surfaces are and are not evidence of. Carried in the public
#: read models because the numbers travel further than the code that produced
#: them: a reverse-import count rendered as a large red node reads as "this is
#: load-bearing at runtime", and it is not that measurement (CP126 0eae8e2d).
_TOPOLOGY_EVIDENCE_DISCLOSURE = {
    "centrality": {
        "measures": "count of modules that statically import this one",
        "basis": "static_import_graph",
        "does_not_establish": [
            "runtime information flow",
            "intervention dependence",
            "functional integration",
        ],
    },
    "critical_modules": {
        "measures": "the highest static reverse-dependency counts",
        "basis": "static_import_graph",
        "does_not_establish": ["runtime criticality", "failure blast radius"],
    },
    "edges": {
        "observed": "has carried at least one pulse",
        "static_import": "derived from an import statement, never exercised",
        "declared": "named in configuration, never exercised",
    },
}

#: What may precede an instruction and still leave it an instruction: an address
#: to Aura, an interjection, a courtesy. Anything else before the verb means the
#: verb is being talked about rather than issued (CP126 961f7fae).
_ROUTE_UTTERANCE_PREFIX = (
    r"^\s*(?:(?:hey|hi|hello|ok|okay|yo)\b[\s,]*)?"
    r"(?:aura\b[\s,:]*)?"
    r"(?:(?:can|could|would|will)\s+you\s+)?"
    r"(?:please\s+)?"
    r"(?:go\s+(?:ahead\s+)?and\s+)?"
    r"(?:just\s+)?"
)


def _vault_allowed_fields(
    model: type[BaseModel], *, dropped: set[str], added: set[str]
) -> set[str]:
    """The field names a vault row for ``model`` may carry.

    Derived from the model rather than hand-listed. The hand-written copies
    drifted the moment anyone added a field: three counters added to
    HardwiredPathway serialized fine and were then rejected at restore, so the
    vault round-trip broke and only an integration test caught it. ``dropped``
    and ``added`` name the deliberate differences — monotonic timestamps are
    persisted as ages, because a monotonic clock does not survive a restart.
    """
    fields = set(model.model_fields) - dropped
    missing = dropped - set(model.model_fields)
    if missing:  # pragma: no cover - guards against a silent rename
        raise ValueError(f"vault schema drops fields {model.__name__} lacks: {missing}")
    return fields | added


def _live_skill_names() -> set[str] | None:
    """Skill names this build provides, or None when the registry can't be asked.

    None and an empty set are different answers: no registry means the question
    was not answered, and a restore must not delete routes on the strength of a
    question it could not ask (CP126 2f6e7791).
    """
    try:
        from core.runtime.service_registry import get_runtime_service

        registry = get_runtime_service("skill_registry", default=None)
        skills = getattr(registry, "skills", None)
        if not isinstance(skills, dict):
            return None
        return {str(name) for name in skills}
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return None


def _unit_reading(value: Any) -> float | None:
    """A telemetry reading in [0, 1], or None when there isn't one.

    None covers absent, non-numeric, non-finite and out-of-range alike, because
    the caller's only correct response to any of them is the same: do not act on
    a number you did not get (CP126 d926886e).
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    reading = float(value)
    if not math.isfinite(reading) or not 0.0 <= reading <= 1.0:
        return None
    return reading


_MODULE_NAME_SEPARATORS = re.compile(r"[./\\_]+")


def _module_name_parts(module_path: str) -> tuple[str, ...]:
    """Split a module path into lowercase name components."""
    return tuple(
        part for part in _MODULE_NAME_SEPARATORS.split(str(module_path).lower()) if part
    )


def _alias_matches_parts(alias: str, parts: tuple[str, ...]) -> bool:
    """Whether an alias equals a component or a contiguous run of components."""
    alias_parts = _module_name_parts(alias)
    if not alias_parts or len(alias_parts) > len(parts):
        return False
    span = len(alias_parts)
    return any(
        parts[start : start + span] == alias_parts
        for start in range(len(parts) - span + 1)
    )


def _cohesion_from_topology(
    strengths: list[float], confidences: list[float]
) -> float | None:
    """Fraction of the topology that is not weak, or None when there is none.

    CP126 40325f75. The old number was ``mean(edge strengths + route
    confidences)`` — an average across two different scales (strength runs to
    10.0, confidence to 1.0) reported as "system cohesion". Its one behavioural
    consumer compared it against 0.7 as though it were a fraction, so a healthy
    network sat near 1.5 and the "I feel a sense of fragmentation in my roots"
    line was effectively unreachable. Padding empty lists with ``[0.0]`` and
    ``[1.0]`` also manufactured a cohesion of 0.5 for a network with no
    topology at all.

    This is a fraction with a statable meaning: of the edges and routes that
    exist, how many are at or above the health they were established with. No
    topology means nothing to measure, which is None, not zero.
    """
    total = len(strengths) + len(confidences)
    if total == 0:
        return None
    healthy = sum(1 for value in strengths if value >= Hypha.DEFAULT_STRENGTH)
    healthy += sum(
        1 for value in confidences if value >= HardwiredPathway.PRUNE_THRESHOLD
    )
    return round(healthy / total, 3)


def _calling_site() -> str:
    """The first frame outside this module — who opened the flow.

    An absorbed failure is only actionable if the report names the code that
    absorbed it. ``mycelium.py`` frames are skipped so the site is the caller's,
    not the context manager's — and so is ``contextlib``, which sits between
    them because ``rooted_flow`` is an ``@asynccontextmanager``. Without that
    second skip every flow in the runtime reports the same line of the standard
    library, which names nothing.
    """
    try:
        frame = inspect.currentframe()
        skip = {__file__, contextlib.__file__}
        while frame is not None:
            filename = frame.f_code.co_filename
            if filename not in skip:
                return f"{filename}:{frame.f_lineno}"
            frame = frame.f_back
    except (AttributeError, ValueError):  # pragma: no cover - introspection guard
        pass
    return "<unknown>"


def _evidence_verifies_outcome(evidence: Any, success: bool) -> bool:
    """Whether ``evidence`` actually states the outcome it is offered for.

    CP126 2462d3c5. A caller boolean is a claim; this asks for the execution
    result behind it. An evidence object only verifies when it carries an
    explicit outcome field AGREEING with the claim — evidence that contradicts
    the caller is not evidence for the caller, and an object with no outcome
    field at all (a bare result, a string, None) verifies nothing.
    """
    if evidence is None:
        return False
    if isinstance(evidence, bool):
        # A second boolean is not corroboration.
        return False
    outcome: Any = None
    if isinstance(evidence, dict):
        for key in ("verified_success", "ok", "success"):
            if key in evidence:
                outcome = evidence[key]
                break
    else:
        for key in ("verified_success", "ok", "success"):
            if hasattr(evidence, key):
                outcome = getattr(evidence, key)
                break
    if not isinstance(outcome, bool):
        return False
    return outcome is bool(success)


from core.governance.durable_learning import (
    admit_learning_update,
    grade_from_evidence,
)
from core.runtime.lockdep import checked_lock
from core.runtime.turn_outcome import VerificationGrade


def _evidence_identity(evidence: Any) -> str | None:
    """A stable id for the evidence behind an outcome, when it carries one.

    Without an id, a durable update cannot be found again when its evidence
    is later withdrawn, so it cannot be rolled back — and the gate refuses
    to make it durable. Returning None here is honest, not a failure.
    """
    if evidence is None or isinstance(evidence, bool):
        return None
    for key in ("evidence_id", "receipt_id", "execution_id", "id"):
        value = (
            evidence.get(key)
            if isinstance(evidence, dict)
            else getattr(evidence, key, None)
        )
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return None


def _redact_source_path(path: Any) -> str | None:
    """Return a project-relative module path, never an absolute filesystem path.

    Public topology/report read models are API- and UI-facing; leaking
    absolute paths discloses the host's local filesystem layout (usernames,
    home dirs, deploy roots). Paths are relativized to the project base and
    anything outside it collapses to its basename.
    """
    if not path:
        return None
    try:
        from core.config import config

        base = config.paths.base_dir
        candidate = Path(str(path))
        try:
            return candidate.resolve().relative_to(Path(base).resolve()).as_posix()
        except (ValueError, OSError):
            return candidate.name
    except (ImportError, RuntimeError, TypeError, ValueError, OSError):
        return Path(str(path)).name

_DEFAULT_INFRASTRUCTURE_SCAN_DIRS = (
    "core",
    "interface",
    "skills",
    "aura",
    "llm",
    "senses",
    "autonomy_engine",
    "cloud",
    "infrastructure",
    "integration",
    "memory",
    "orchestrator",
    "proof_kernel",
    "research",
    "security",
    "storage",
    "training",
    "utils",
)
_VAULT_CLOCK_SKEW_TOLERANCE_S = 1.0

# CP126 944a6043: restore limits. Each is an order of magnitude above what a
# live generation of this codebase produces (~1.5k modules, a few dozen routes),
# so they bound a hostile or corrupt vault without constraining a real one. A
# restore that trips one is refused; the previous generation stands.
_VAULT_MAX_PATHWAYS = 5_000
_VAULT_MAX_HYPHAE = 50_000
_VAULT_MAX_MAPPED_FILES = 50_000
_VAULT_MAX_MODULE_IMPORTS = 2_000
#: Hypha.log() already caps the live trace at 100 entries.
_VAULT_MAX_HYPHA_TRACE = 100
#: Serialized bytes accepted from the vault row before any decoding.
_VAULT_MAX_ENCODED_BYTES = 64 * 1024 * 1024

# CP126 a56e4c5d: the mapping admission contract. The module ceiling matches the
# vault's, since a generation that cannot be persisted is not worth building.
# The budget is wall-clock across discovery and parsing: this is maintenance
# work, and maintenance that runs unbounded is not maintenance.
_MAPPING_MAX_MODULES = _VAULT_MAX_MAPPED_FILES
_MAPPING_BUDGET_S = 120.0
#: How long shutdown waits for a running mapper before clearing state anyway.
_MAPPER_DRAIN_BUDGET_S = 3.0
_ALLOW_FOREGROUND_MAPPING_FLAG = declare(
    "AURA_ALLOW_FOREGROUND_INFRASTRUCTURE_MAPPING",
    kind=FlagKind.BOOL,
    default=False,
    description="Allow mycelial infrastructure mapping in foreground-only mode",
    owner="core.mycelium",
)
_FOREGROUND_MAPPING_QUIET_FLAG = declare(
    "AURA_FOREGROUND_INFRASTRUCTURE_MAPPING_QUIET_S",
    kind=FlagKind.FLOAT,
    default=180.0,
    description="Foreground-only startup quiet window before mycelial mapping",
    owner="core.mycelium",
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class HardwiredPathway(BaseModel):
    """A direct connection from an intent pattern to a skill.

    Direct means it skips intent classification, not that it skips governance:
    a match is a routing proposal that the caller's authority gate still has to
    admit (CP126 39f4805f).
    """
    pathway_id: str
    pattern: Any  # Union[str, re.Pattern]
    skill_name: str
    param_map: dict[str, int | str] = Field(default_factory=dict)
    priority: float = 1.0
    source_file: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    # An untested pathway is not a proven-reliable one. Start below the
    # reinforce ceiling so confidence reflects earned, not assumed, reliability.
    confidence: float = 0.5
    activity_label: str = ""
    hit_count: int = 0
    miss_count: int = 0
    # CP126 2462d3c5: hit/miss count every reinforcement, including ones whose
    # only evidence is "the call did not raise". These count ONLY outcomes
    # backed by an execution result, so a pathway's earned reliability can be
    # told apart from its asserted one.
    verified_hits: int = 0
    verified_misses: int = 0
    unverified_reinforcements: int = 0
    created_at: float = Field(default_factory=time.time)
    last_matched: float = Field(default_factory=time.monotonic)
    direct_response: str | None = None  # Legacy non-user emergency response only
    color: str = "#4A90E2"                 # Default Aura Blue
    description: str = ""
    size: float = 1.0

    # Physarum thresholds
    REINFORCE_DELTA: ClassVar[float] = 0.05
    #: A success nobody verified is weaker evidence than one that was.
    UNVERIFIED_REINFORCE_DELTA: ClassVar[float] = 0.02
    WEAKEN_DELTA: ClassVar[float] = 0.15
    PRUNE_THRESHOLD: ClassVar[float] = 0.2
    MAX_CONFIDENCE: ClassVar[float] = 1.0
    MIN_CONFIDENCE: ClassVar[float] = 0.05

    model_config = {"arbitrary_types_allowed": True}

    def reinforce(self, success: bool, *, verified: bool = False):
        """Physarum-inspired conductivity update.

        CP126 2462d3c5: this trained routing confidence from a caller boolean
        with no execution receipt and no independent outcome check. The live
        callers made that concrete — the response lane passed success=True
        immediately after a tool call that had merely not raised, so a tool
        returning a failure RESULT still strengthened the pathway that chose
        it. Confidence then rose on evidence of "nothing threw".

        Unverified reinforcement still moves confidence, because refusing it
        would freeze routing everywhere the caller has no receipt. What it no
        longer does is look the same as a verified outcome: an unverified
        success earns a fraction of the delta, and the verified counters are
        kept separately so reliability can be read at its true strength.
        """
        if success:
            delta = self.REINFORCE_DELTA if verified else self.UNVERIFIED_REINFORCE_DELTA
            self.confidence = min(self.MAX_CONFIDENCE, self.confidence + delta)
            self.hit_count += 1
            if verified:
                self.verified_hits += 1
        else:
            # A failure is weakened at full strength either way: acting on a
            # pathway that did not work is the risk, and discounting the
            # penalty for missing evidence would keep a broken route alive.
            self.confidence = max(self.MIN_CONFIDENCE, self.confidence - self.WEAKEN_DELTA)
            self.miss_count += 1
            if verified:
                self.verified_misses += 1
        if not verified:
            self.unverified_reinforcements += 1

    def record_unverified_reinforcement(self) -> None:
        """Tally an outcome whose evidence was too weak to persist.

        The durable confidence is deliberately untouched. Only the counter
        moves, so ``evidence_grade`` can still say how much of this
        pathway's record rests on assertions — which is the number an
        operator needs when a route looks reliable and is not.
        """
        self.unverified_reinforcements += 1

    @property
    def is_weak(self) -> bool:
        return self.confidence < self.PRUNE_THRESHOLD

    @property
    def verified_success_rate(self) -> float | None:
        """Success rate over VERIFIED outcomes, or None when there are none.

        None rather than 0.0: no verified evidence is not a bad record
        (CP126 2462d3c5).
        """
        total = self.verified_hits + self.verified_misses
        return (self.verified_hits / total) if total > 0 else None

    @property
    def evidence_grade(self) -> str:
        """How much of this pathway's record rests on checked outcomes."""
        total = self.hit_count + self.miss_count
        if total == 0:
            return "untested"
        verified = self.verified_hits + self.verified_misses
        if verified == 0:
            return "asserted_only"
        return "verified" if verified >= total else "mixed"

    @property
    def success_rate(self) -> float:
        # No evidence yet ⇒ unknown, reported as 0.0 rather than a perfect
        # 1.0. An untested pathway must not appear maximally reliable.
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Legacy helper. Use .model_dump() instead."""
        data = self.model_dump()
        # For UI compatibility: frontend expects 'id'
        data["id"] = self.pathway_id
        # Ensure regex pattern is stringified for JSON compatibility
        if "pattern" in data and not isinstance(data["pattern"], str):
            data["pattern"] = getattr(self.pattern, 'pattern', str(self.pattern))
        return data


# ---------------------------------------------------------------------------
# Hypha (General-Purpose Connection)
# ---------------------------------------------------------------------------

class Hypha(BaseModel):
    """A connection within the mycelial network with dynamic strength."""
    name: str
    source: str
    target: str
    priority: float = 1.0
    strength: float = 1.0
    created_at: float = Field(default_factory=time.monotonic)
    last_pulse: float = Field(default_factory=time.monotonic)
    pulse_count: int = 0
    active: bool = True
    is_physical: bool = False
    source_file: str | None = None
    target_file: str | None = None
    color: str = "#4A90E2"
    description: str = ""
    size: float = 1.0
    trace: list[str] = Field(default_factory=list)

    #: The scale an edge's strength lives on. Named because ``get_system_cohesion``
    #: used to average this quantity with route confidence — which tops out at
    #: 1.0 — and report the result as a fraction (CP126 40325f75).
    MIN_STRENGTH: ClassVar[float] = 0.1
    MAX_STRENGTH: ClassVar[float] = 10.0
    DEFAULT_STRENGTH: ClassVar[float] = 1.0

    def pulse(self, success: bool = True):
        """Reinforce or prune the hypha based on successful transmission."""
        self.last_pulse = time.monotonic()
        self.pulse_count += 1
        if success:
            self.strength = min(self.MAX_STRENGTH, self.strength + 0.5)
        else:
            self.strength = max(self.MIN_STRENGTH, self.strength - 1.0)

    @property
    def is_weak(self) -> bool:
        """Weaker than it was when established — it has lost more than it gained."""
        return self.strength < self.DEFAULT_STRENGTH

    @property
    def evidence_basis(self) -> str:
        """What this edge is evidence OF.

        CP126 106a29f4, 0eae8e2d. Three different things were being presented
        as one: an edge derived from a static import, an edge somebody declared
        in a hardcoded list, and an edge that has actually carried traffic. Only
        the third is evidence of runtime information flow, and the topology
        surfaced no way to tell them apart — so a hardcoded
        ``("qualia", "phenomenology")`` link read exactly like a measured one.

        ``observed`` is derived from the edge's own pulse history rather than
        stored, so it cannot drift away from what actually happened.
        """
        if self.pulse_count > 0:
            return "observed"
        if self.is_physical:
            return "static_import"
        return "declared"

    def refresh_heartbeat(self):
        """Refresh liveness without mutating the learned strength of the edge."""
        self.last_pulse = time.monotonic()

    @property
    def thickness(self) -> float:
        """Dynamic representation of hypha health/strength (BUG-037)."""
        return 0.5 + (self.strength * 0.1)

    def log(self, msg: str):
        self.trace.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(self.trace) > 100:
            self.trace.pop(0)


class NeuralRoot(Hypha):
    """Owner-attested binding to a worker, service, or hardware endpoint."""
    hardware_id: str = "metal_default"
    pinned: bool = True
    root_kind: str = "hardware"
    liveness_contract: str = "on_demand"
    state: str = "unbound"
    owner_generation: str = ""
    attested_identity: dict[str, Any] = Field(default_factory=dict)
    last_activity_at: float = 0.0
    last_probe_at: float = 0.0
    last_probe_success_at: float = 0.0
    stale_after_s: float = 30.0
    last_error: str = ""


class RootedFlowHandle:
    """Owner-backed flow view that cannot mutate a detached hypha snapshot."""

    def __init__(
        self,
        network: "MycelialNetwork",
        source: str,
        target: str,
        priority: float,
    ):
        self._network = network
        self._source = source
        self._target = target
        self._priority = priority
        self._hypha_id = f"{source}->{target}"
        self._error: BaseException | None = None
        # CP126 34f01634: an absorbed failure leaves the caller's ``async with``
        # completing normally, so a failed action reads as a finished one unless
        # the caller remembers to ask. Remembering is not a safeguard. The
        # handle records where it was opened and whether anyone ever collected
        # the failure; the network sweeps the ones nobody did.
        self._activity: str = ""
        self._absorbed = False
        self._acknowledged = False
        self._swept = False
        self._caller_site: str = _calling_site()

    def _acknowledge(self) -> None:
        self._acknowledged = True

    @property
    def failed(self) -> bool:
        self._acknowledge()
        return self._error is not None

    @property
    def error(self) -> BaseException | None:
        self._acknowledge()
        return self._error

    @property
    def absorbed(self) -> bool:
        """Whether this flow's failure was swallowed instead of propagated."""
        return self._absorbed

    def _mark_failed(self, error: BaseException) -> None:
        self._error = error

    def _mark_absorbed(self, activity: str) -> None:
        self._absorbed = True
        self._activity = activity

    def raise_for_status(self) -> None:
        """Re-raise an absorbed flow failure at the caller's owned boundary."""
        self._acknowledge()
        if self._error is not None:
            raise self._error

    def _snapshot(self) -> Hypha:
        return self._network._record_rooted_flow_event(
            self._source,
            self._target,
            priority=self._priority,
        )

    def log(self, message: str) -> None:
        self._network._record_rooted_flow_event(
            self._source,
            self._target,
            priority=self._priority,
            message=message,
        )

    def pulse(self, success: bool = True) -> None:
        self._network._record_rooted_flow_event(
            self._source,
            self._target,
            priority=self._priority,
            success=success,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._snapshot(), name)


# ---------------------------------------------------------------------------
# Mycelial Network (Singleton)
# ---------------------------------------------------------------------------

#: How long a telemetry pulse may wait for the topology lock. Short on purpose:
#: the pulse is worth far less than the latency of waiting for it.
_PULSE_LOCK_TIMEOUT_S = 0.05
_DEFERRED_PULSE_LOCK = checked_lock("mycelium")


def _drain_deferred_pulse_handoff_locked(network: "MycelialNetwork") -> None:
    """Move wait-free pulse handoffs into the owned aggregate."""
    pending = network._deferred_pulses
    while True:
        try:
            key, successes, failures = network._deferred_pulse_handoff.get_nowait()
        except Empty:
            return
        prior_successes, prior_failures = pending.get(key, (0, 0))
        pending[key] = (
            prior_successes + successes,
            prior_failures + failures,
        )


def _defer_pulse(
    network: "MycelialNetwork",
    source: str,
    target: str | None,
    *,
    success: bool,
) -> None:
    """Retain a contended pulse without making its caller wait on topology."""
    key = f"{source}->{target or '*'}"
    if not _DEFERRED_PULSE_LOCK.acquire(blocking=False):
        network._deferred_pulse_handoff.put(
            (key, 1 if success else 0, 0 if success else 1)
        )
        return
    try:
        _drain_deferred_pulse_handoff_locked(network)
        pending = network._deferred_pulses
        successes, failures = pending.get(key, (0, 0))
        if success:
            successes += 1
        else:
            failures += 1
        pending[key] = (successes, failures)
        count = successes + failures
    finally:
        _DEFERRED_PULSE_LOCK.release()
    if count == 1:
        logger.info(
            "Mycelial pulse deferred under topology contention (%s); it will "
            "be merged into the next owned edge update without blocking the caller.",
            key,
        )


def _take_deferred_pulses(
    network: "MycelialNetwork",
    source: str,
    target: str | None,
) -> tuple[int, int]:
    key = f"{source}->{target or '*'}"
    if not _DEFERRED_PULSE_LOCK.acquire(blocking=False):
        return (0, 0)
    try:
        _drain_deferred_pulse_handoff_locked(network)
        return network._deferred_pulses.pop(key, (0, 0))
    finally:
        _DEFERRED_PULSE_LOCK.release()


#: Absorbed-failure handles awaiting acknowledgement, and the ones that aged out
#: of a sweep without ever being collected. Capped because an unacknowledged
#: absorption is a defect report, not a queue: past the cap the count still
#: rises but no further handles are retained.
_ABSORBED_FLOW_LOCK = checked_lock("mycelium")
_MAX_TRACKED_ABSORPTIONS = 256


def _track_absorbed_flow(
    network: "MycelialNetwork", handle: "RootedFlowHandle"
) -> None:
    with _ABSORBED_FLOW_LOCK:
        pending = network._absorbed_flows
        if len(pending) >= _MAX_TRACKED_ABSORPTIONS:
            network._absorbed_flow_overflow += 1
            return
        pending.append(handle)


def _sweep_absorbed_flows(network: "MycelialNetwork") -> list["RootedFlowHandle"]:
    """Return absorbed failures that survived a full sweep unacknowledged.

    Acknowledgement can only happen after the ``async with`` block returns, so a
    handle is never judged on the sweep that first sees it. Surviving two sweeps
    means no caller ever asked whether the flow failed.
    """
    with _ABSORBED_FLOW_LOCK:
        pending = network._absorbed_flows
        survivors: list[RootedFlowHandle] = []
        unclaimed: list[RootedFlowHandle] = []
        for handle in pending:
            if handle._acknowledged:
                continue
            if handle._swept:
                unclaimed.append(handle)
                continue
            handle._swept = True
            survivors.append(handle)
        network._absorbed_flows = survivors
    return unclaimed


def _merge_deferred_pulses(
    network: "MycelialNetwork",
    source: str,
    target: str | None,
    counts: tuple[int, int],
) -> None:
    if counts == (0, 0):
        return
    key = f"{source}->{target or '*'}"
    if not _DEFERRED_PULSE_LOCK.acquire(blocking=False):
        network._deferred_pulse_handoff.put((key, counts[0], counts[1]))
        return
    try:
        _drain_deferred_pulse_handoff_locked(network)
        successes, failures = network._deferred_pulses.get(key, (0, 0))
        network._deferred_pulses[key] = (
            successes + counts[0],
            failures + counts[1],
        )
    finally:
        _DEFERRED_PULSE_LOCK.release()


class MycelialNetwork:
    """The Unoverridable Root System."""

    _instance: ClassVar[Optional["MycelialNetwork"]] = None
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _vault_io_lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: ClassVar[bool] = False

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if MycelialNetwork._initialized:
            return

        with MycelialNetwork._lock:
            if MycelialNetwork._initialized:
                return

            self._async_lock: asyncio.Lock | None = None
            
            # Phase XXIII: Aegis Protection Flag
            object.__setattr__(self, "_aegis_locked", False)

            # --- Hardwired Pathways ---
            self.pathways: dict[str, HardwiredPathway] = {}
            self._pathway_order: list[str] = []

            # --- General Hyphae ---
            self.hyphae: dict[str, Hypha] = {}

            # --- Discovery Engine ---
            self._execution_log: list[dict[str, Any]] = []
            self._discovery_candidates: dict[str, int] = defaultdict(int)
            self._route_signal_log_state: dict[str, tuple[str, float, int]] = {}
            self._hypha_alert_times: dict[str, float] = {}
            self._deferred_pulses: dict[str, tuple[int, int]] = {}
            self._deferred_pulse_handoff: SimpleQueue[
                tuple[str, int, int]
            ] = SimpleQueue()
            self._absorbed_flows: list[RootedFlowHandle] = []
            self._absorbed_flow_overflow: int = 0
            self._unclaimed_absorptions: int = 0
            #: None until a vault generation has been published into this
            #: instance — "never restored" is not the same claim as "restored
            #: without tamper evidence".
            self._restored_from_vault_at: float | None = None
            self._restored_generation_attested: bool | None = None
            #: True when shutdown cleared topology state while the mapper was
            #: still running (CP126 53f15b88).
            self._shutdown_left_mapper_running: bool = False

            # --- Props ---
            self.ui_callback: Callable[[str], Coroutine] | None = None
            self.mapped_files: dict[str, dict[str, Any]] = {}
            self.infrastructure_mapped: bool = False
            self._centrality: dict[str, int] = {}
            self._critical_modules: list[str] = []
            self._cross_links: dict[str, list[str]] = {}
            self._is_mapping: bool = False
            # Mapping lifecycle and topology data share one lock. Separate locks
            # previously allowed publication and shutdown to acquire them in
            # opposite orders and made a coherent graph generation impossible.
            self._mapping_lock = MycelialNetwork._lock
            self._mapping_thread: threading.Thread | None = None
            self._mapping_admission_token: object | None = None
            self._mapping_generation: int = 0
            self._topology_revision: int = 0
            self._topology_structure_revision: int = 0
            self._last_vault_sync_revision: int | None = None
            self._last_vault_sync_at: float | None = None
            self._last_vault_sync_lag_revisions: int = 0
            self._mapping_started_at: float | None = None
            self._mapping_completed_at: float | None = None
            self._mapping_last_error: str | None = None
            self._created_at_monotonic = time.monotonic()
            self._deferred_mapping_reason: str | None = None
            self._stop_event = threading.Event()
            self._topology_counts_cache: dict[str, int] = {}
            self._topology_summary_cache: dict[str, int] = {}
            
            # Legacy compat
            self.direct_roots: dict[str, str] = {}
            
            # Reflex Core (SOMA)
            try:
                from core.soma.reflex_core import HardenedReflexCore
                self.reflex = HardenedReflexCore()
            except ImportError:
                self.reflex = None

            # --- Platform Binding ---
            self._neural_roots: list[NeuralRoot] = []

            self._publish_topology_read_models_locked()
            
            MycelialNetwork._initialized = True
            object.__setattr__(self, "_aegis_locked", True)
            self._setup_default_pathways()
            
            logger.info("🍄 [MYCELIUM] Network Online v4.0 (Hardened) — Enterprise Grade.")

    def _publish_topology_read_models_locked(self) -> None:
        endpoints = {
            endpoint
            for hypha in self.hyphae.values()
            for endpoint in (hypha.source, hypha.target)
            if endpoint
        }
        endpoints.update(self.mapped_files)
        annotated_pathways = sum(
            1 for pathway in self.pathways.values() if pathway.source_file
        )
        self._topology_counts_cache = {
            "pathways": len(self.pathways),
            "hyphae": len(self.hyphae),
            "mapped_files": len(self.mapped_files),
            "mapping_generation": self._mapping_generation,
        }
        self._topology_summary_cache = {
            "nodes": len(endpoints) + len(self.pathways),
            "links": len(self.hyphae) + annotated_pathways,
            "pathways": len(self.pathways),
            "mapping_generation": self._mapping_generation,
        }

    # ------------------------------------------------------------------
    # Session-local routing confidence.
    #
    # Deliberately NOT a Pathway field. Session state that lives inside the
    # durable model gets serialized into the vault by anything that walks
    # model_dump(), and then "this seemed to work once" is indistinguishable
    # from "this was checked" after a restart. Keeping it on the network, in
    # a plain dict, makes the boundary structural rather than a convention
    # somebody has to remember.
    # ------------------------------------------------------------------

    #: Bounded so a long-running process with churning pathway ids cannot
    #: grow this without limit.
    _MAX_SESSION_CONFIDENCE_ENTRIES: ClassVar[int] = 4096

    def _apply_session_reinforcement_locked(self, pathway_id: str, success: bool) -> None:
        """Move this session's view of a pathway without touching the vault."""
        self._apply_session_confidence_delta_locked(
            pathway_id,
            HardwiredPathway.UNVERIFIED_REINFORCE_DELTA
            if success
            else -HardwiredPathway.WEAKEN_DELTA,
        )

    def _apply_session_confidence_delta_locked(
        self, pathway_id: str, delta: float
    ) -> None:
        """Accumulate a session-only confidence adjustment, bounded."""
        if not math.isfinite(delta):
            return
        deltas = getattr(self, "_session_confidence", None)
        if deltas is None:
            deltas = {}
            object.__setattr__(self, "_session_confidence", deltas)
        current = float(deltas.get(pathway_id, 0.0))
        deltas[pathway_id] = max(-1.0, min(1.0, current + float(delta)))
        if len(deltas) > self._MAX_SESSION_CONFIDENCE_ENTRIES:
            for stale in list(deltas)[: len(deltas) - self._MAX_SESSION_CONFIDENCE_ENTRIES]:
                deltas.pop(stale, None)

    def session_confidence_delta(self, pathway_id: str) -> float:
        """This session's unverified adjustment for a pathway. Never persisted."""
        deltas = getattr(self, "_session_confidence", None) or {}
        return float(deltas.get(str(pathway_id), 0.0))

    def effective_confidence(self, pathway_id: str) -> float:
        """Durable confidence plus this session's unverified evidence.

        What routing should ask. ``Pathway.confidence`` alone answers "what
        has Aura earned the right to believe", which is the right question
        for persistence and the wrong one for choosing a route right now.
        """
        pathway = self.pathways.get(str(pathway_id))
        if pathway is None:
            return 0.0
        combined = float(pathway.confidence) + self.session_confidence_delta(pathway_id)
        return max(HardwiredPathway.MIN_CONFIDENCE * 0.0, min(HardwiredPathway.MAX_CONFIDENCE, combined))

    def _mark_topology_mutated_locked(self, *, structure_changed: bool = False) -> None:
        self._topology_revision += 1
        if structure_changed:
            self._topology_structure_revision += 1
            self._publish_topology_read_models_locked()

    def _setup_default_pathways(self):
        """Register action routes; conversation remains owned by CognitiveEngine.

        CP126 961f7fae. These patterns were unanchored, and routing searches
        rather than matches — so ``system check`` fired the self-repair action
        from "what does a system check actually do?", "don't run a system check
        yet" and "IT already did a system check", and ``google`` fired a web
        search from "my google account password". An action dispatched from a
        substring of a sentence that was not a request for it is the whole
        defect.

        Each pattern is now anchored to the start of the utterance, past an
        optional address or courtesy lead-in, so it fires on an instruction and
        not on a mention. ``tests/test_mycelium_routing_authority.py``
        carries the negative controls.
        """
        self.register_pathway(
            "direct_web_search",
            _ROUTE_UTTERANCE_PREFIX
            + r"(?:search (?:the web )?for|look up|google|find info on)\s+(.+)",
            "search_web",
            priority=1.5,
            activity_label="🔍 Searching the Intelligence Web"
        )
        self.register_pathway(
            "direct_self_repair",
            _ROUTE_UTTERANCE_PREFIX
            + r"(?:run a self-diag|diagnose yourself|repair yourself"
            r"|(?:run|do) a system check|system check\b|fix system\b)",
            "self_repair",
            priority=1.5,
            activity_label="🧬 Running Self-Diagnostics"
        )
    #: Container attributes that may not be REBOUND once the network is live.
    _AEGIS_PROTECTED_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {"pathways", "hyphae", "_pathway_order"}
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse to rebind the live topology containers.

        CP126 1947a19c. This was documented as "Singleton True-Lock (Memory
        Protection)", which reads as a guarantee that the topology cannot be
        altered. It is not that and cannot be: routing LEARNS, so the contents
        of ``pathways`` and ``hyphae`` are mutated constantly and by design.
        What the guard actually does is narrower and still worth having — it
        stops the container itself being swapped out from under live readers
        holding a reference to it.

        Sanctioned replacement (vault restore, shutdown, generation publish)
        goes through :meth:`_aegis_replace`, so the guard has one named door
        rather than being ambiently side-stepped with ``object.__setattr__``.
        ``tests/test_mycelium_aegis_scope.py`` holds the door count down.
        """
        # Allow initialization to proceed naturally
        if not getattr(self, "_aegis_locked", False):
            super().__setattr__(name, value)
            return

        if name in self._AEGIS_PROTECTED_ATTRS:
            logger.critical("🛡️ AEGIS: Unauthorized attempt to overwrite %s!", name)
            raise PermissionError(f"Aegis True-Lock: Cannot overwrite core Mycelial attribute '{name}'")

        super().__setattr__(name, value)

    def _aegis_replace(self, name: str, value: Any) -> None:
        """Replace a protected container from a sanctioned publication path.

        The one door through the Aegis rebinding guard. Callers must already
        hold ``MycelialNetwork._lock``: the point of the guard is that no reader
        observes a half-swapped topology, and that only holds if the swap
        happens under the same lock readers take.
        """
        object.__setattr__(self, name, value)

    def _active_owner_locked(self) -> Optional["MycelialNetwork"]:
        """Resolve stale references to the one currently published singleton."""
        current = MycelialNetwork._instance
        stop_event = getattr(current, "_stop_event", None)
        if current is None or stop_event is None or stop_event.is_set():
            return None
        return current

    def _active_owner(self) -> Optional["MycelialNetwork"]:
        with MycelialNetwork._lock:
            return self._active_owner_locked()


    def setup(self, *, force: bool = False) -> bool:
        """Schedule the single owned infrastructure map when policy permits."""
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            return owner.setup(force=force)
        with self._mapping_lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                return owner.setup(force=force)
            if not force and self._foreground_mapping_deferred():
                return False
            thread = self._mapping_thread
            if self._is_mapping or (
                thread is not None and thread.is_alive()
            ) or (self.infrastructure_mapped and not force):
                return False

            from core.config import config

            mapping_base = config.paths.base_dir
            logger.info(
                "🍄 [MYCELIUM] Scheduling infrastructure mapping at: %s",
                mapping_base,
            )
            admission_token = object()
            self._mapping_admission_token = admission_token
            self._is_mapping = True
            self._mapping_started_at = time.time()
            self._mapping_last_error = None
            self._deferred_mapping_reason = None
            try:
                thread = threading.Thread(
                    target=self._mapping_worker,
                    args=(str(mapping_base),),
                    kwargs={"force": force, "_admission_token": admission_token},
                    daemon=True,
                    name="MyceliumInfrastructureMap",
                )
                self._mapping_thread = thread
                thread.start()
            except Exception:  # noqa: BLE001 - restore admission state for any start failure
                if self._mapping_admission_token is admission_token:
                    self._mapping_admission_token = None
                    self._is_mapping = False
                self._mapping_thread = None
                raise
            return True

    def _mapping_worker(
        self,
        base_dir: str,
        *,
        force: bool = False,
        _admission_token: object | None = None,
    ) -> None:
        """Run the optional mapper without leaving a false running state."""
        try:
            self.map_infrastructure(
                base_dir,
                force=force,
                _admission_token=_admission_token,
            )
        except Exception as exc:  # noqa: BLE001 - owner-thread liveness boundary
            message = f"{type(exc).__name__}: {exc}"
            with self._mapping_lock:
                boundary_recorded = self._mapping_last_error == message
                self._mapping_last_error = message
                retained_generation = self.infrastructure_mapped
            if not boundary_recorded:
                record_degradation(
                    "mycelium",
                    exc,
                    severity="warning",
                    action=(
                        "retained the prior complete infrastructure generation after "
                        "owned mapper refresh failure"
                        if retained_generation
                        else "left infrastructure graph unmapped after owned mapper failure"
                    ),
                )
            logger.error("🍄 [MYCELIUM] Infrastructure mapping failed: %s", exc, exc_info=True)
        finally:
            with self._mapping_lock:
                # A worker may lose admission to a direct caller. It must never
                # clear that caller's latch. It may only release the reservation
                # that setup assigned specifically to this worker.
                if (
                    _admission_token is not None
                    and self._mapping_admission_token is _admission_token
                ):
                    self._mapping_admission_token = None
                    self._is_mapping = False
                if self._mapping_thread is threading.current_thread():
                    self._mapping_thread = None

    # ======================================================================
    # HARDWIRED PATHWAYS — The Core Intent Router
    # ======================================================================

    def register_pathway(
        self,
        pathway_id: str,
        pattern: str,
        skill_name: str,
        param_map: dict[str, Any] | None = None,
        priority: float = 1.0,
        activity_label: str = "",
        direct_response: str | None = None,
    ) -> None:
        """Register a hardwired intent→skill pathway with regex param extraction.

        Args:
            pathway_id: Unique identifier (e.g., "image_gen_primary")
            pattern: Regex pattern string with capture groups for params
            skill_name: Target skill name (e.g., "generate_image")
            param_map: Maps skill param names to regex group indices or
                literal values for always-on params.
            priority: Higher priority pathways are checked first.
            activity_label: UI message shown when this pathway fires.
            direct_response: Legacy emergency response for non-user origins.
                Production user conversation must remain on CognitiveEngine.
        """
        compiled = _compile_route_pattern(pattern)
        pw = HardwiredPathway(
            pathway_id=pathway_id,
            pattern=compiled,
            skill_name=skill_name,
            param_map=param_map or {},
            priority=priority,
            activity_label=activity_label or f"Aura is executing {skill_name}...",
            direct_response=direct_response,
        )
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.register_pathway(
                    pathway_id,
                    pattern,
                    skill_name,
                    param_map=param_map,
                    priority=priority,
                    activity_label=activity_label,
                    direct_response=direct_response,
                )
            self.pathways[pathway_id] = pw

            # Sanctioned rebinding: under MycelialNetwork._lock, via the one door.
            self._aegis_replace(
                "_pathway_order",
                sorted(
                    self.pathways.keys(),
                    key=lambda k: self.pathways[k].priority,
                    reverse=True,
                ),
            )
            self.direct_roots[pathway_id] = skill_name
            self._mark_topology_mutated_locked(structure_changed=True)

        logger.info(
            "🍄 [MYCELIUM] Pathway Hardwired: '%s' → %s (priority=%.1f, groups=%s)",
            pathway_id, skill_name, priority, list((param_map or {}).keys()),
        )


    def match_hardwired(self, text: str) -> tuple[HardwiredPathway, dict[str, Any]] | None:
        """Match user text against all hardwired pathways with parameter extraction (Issue 77)."""
        if not isinstance(text, str) or not text.strip():
            return None

        # ISSUE-77: Strict Message Validation
        if len(text) > 4096:
            logger.warning("🍄 [MYCELIUM] Message too long for hardwired matching (%d chars)", len(text))
            return None
            
        text_clean = text.strip()

        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return None
            if owner is not self:
                return owner.match_hardwired(text)
            candidates = tuple(
                (
                    pathway_id,
                    pathway,
                    pathway.model_copy(deep=True) if pathway is not None else None,
                )
                for pathway_id in self._pathway_order
                for pathway in (self.pathways.get(pathway_id),)
            )

        for pw_id, original, pw in candidates:
            if original is None or pw is None:
                continue

            # Skip pathways that have decayed below minimum confidence.
            # EFFECTIVE, not durable: a route that failed this session on
            # unverified evidence must be avoided now even though that
            # evidence was too weak to change what is persisted.
            if self.effective_confidence(pw_id) < pw.MIN_CONFIDENCE:
                continue

            match = _safe_pattern_search(pw.pattern, text_clean)
            if match:
                # Extract params from capture groups
                params: dict[str, Any] = {}
                for param_name, mapping in pw.param_map.items():
                    if isinstance(mapping, int):
                        try:
                            value = match.group(mapping)
                            if value:
                                params[param_name] = value.strip()
                        except (IndexError, AttributeError):
                            logger.warning(
                                "🍄 [MYCELIUM] Param extraction failed for '%s' group %s in pathway '%s'",
                                param_name, mapping, pw_id,
                            )
                    else:
                        params[param_name] = mapping

                with MycelialNetwork._lock:
                    owner = self._active_owner_locked()
                    if owner is None:
                        return None
                    if owner is not self:
                        return owner.match_hardwired(text)
                    current = self.pathways.get(pw_id)
                    if current is not original:
                        continue
                    current.last_matched = time.monotonic()
                    self._mark_topology_mutated_locked()
                    result = current.model_copy(deep=True)

                logger.info(
                    "🍄 [MYCELIUM] ⚡ HardwiredPathway MATCHED: '%s' → skill=%s, params=%s, confidence=%.2f",
                    pw_id, pw.skill_name, params, pw.confidence,
                )

                return (result, params)

        return None

    # ======================================================================
    # HYPHAE NETWORK — Subsystem Connectivity
    # ======================================================================

    def establish_connection(self, source: str, target: str, priority: float = 1.0) -> Hypha:
        """Establish a subsystem hypha and return a detached read model."""
        hypha_id = f"{source}->{target}"
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.establish_connection(source, target, priority=priority)
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                hypha = Hypha(
                    name=hypha_id,
                    source=source,
                    target=target,
                    priority=priority,
                )
                self.hyphae[hypha_id] = hypha
                self._mark_topology_mutated_locked(structure_changed=True)
                logger.info("🍄 [MYCELIUM] Hypha established: %s", hypha_id)
            return hypha.model_copy(deep=True)

    def add_hypha(self, source: str, target: str, link_type: str = "general", metadata: dict | None = None):
        """Enterprise method for adding a hypha with rich metadata."""
        hypha_id = f"{source}->{target}"
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.add_hypha(source, target, link_type=link_type, metadata=metadata)
            if hypha_id not in self.hyphae:
                self.hyphae[hypha_id] = Hypha(
                    name=hypha_id,
                    source=source,
                    target=target,
                    trace=[f"Link Type: {link_type}"]
                )
                self._mark_topology_mutated_locked(structure_changed=True)
                logger.info("🍄 [MYCELIUM] Hypha added: %s (%s)", hypha_id, link_type)

    def get_hypha(self, source: str, target: str = None) -> Hypha | None:
        """Return a detached hypha read model."""
        if target is None and "->" in source:
            hypha_id = source
        else:
            hypha_id = f"{source}->{target}"
            
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return None
            if owner is not self:
                return owner.get_hypha(source, target)
            hypha = self.hyphae.get(hypha_id)
            return hypha.model_copy(deep=True) if hypha is not None else None

    @staticmethod
    def _hypha_id(source: str, target: str | None = None) -> str:
        return source if target is None and "->" in source else f"{source}->{target}"

    def pulse_hypha(
        self,
        source: str,
        target: str | None = None,
        *,
        success: bool = True,
    ) -> bool:
        """Pulse the current owned edge without leaking a mutable object.

        NEVER blocks the caller waiting for the topology lock. A contended
        pulse is retained in a side buffer and merged into the next owned edge
        update, while waiting for the topology lock on the event loop would
        cost the whole mind.

        Measured live: the loop sat in ``pulse_hypha`` -> ``MycelialNetwork._lock``
        during a desktop task and the hypervisor reported "severe event-loop lag
        97.192s". Ninety-seven seconds of a frozen runtime for a counter.
        """
        hypha_id = self._hypha_id(source, target)
        if not MycelialNetwork._lock.acquire(timeout=_PULSE_LOCK_TIMEOUT_S):
            _defer_pulse(self, source, target, success=success)
            return False
        try:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                _merge_deferred_pulses(
                    owner,
                    source,
                    target,
                    _take_deferred_pulses(self, source, target),
                )
                return owner.pulse_hypha(source, target, success=success)
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                return False
            deferred_successes, deferred_failures = _take_deferred_pulses(
                self, source, target
            )
            if deferred_successes or deferred_failures:
                hypha.pulse_count += deferred_successes + deferred_failures
                hypha.last_pulse = time.monotonic()
                hypha.strength = min(
                    10.0,
                    max(
                        0.1,
                        hypha.strength
                        + (0.5 * deferred_successes)
                        - float(deferred_failures),
                    ),
                )
            hypha.pulse(success=success)
            self._mark_topology_mutated_locked()
            return True
        finally:
            MycelialNetwork._lock.release()

    def log_hypha(self, source: str, target: str | None, message: str) -> bool:
        """Append an owned trace entry to the current edge."""
        hypha_id = self._hypha_id(source, target)
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                return owner.log_hypha(source, target, message)
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                return False
            hypha.log(message)
            self._mark_topology_mutated_locked()
            return True

    def _record_rooted_flow_event(
        self,
        source: str,
        target: str,
        *,
        priority: float,
        message: str | None = None,
        success: bool | None = None,
    ) -> Hypha:
        """Atomically bind one flow event to the currently published owner."""
        hypha_id = self._hypha_id(source, target)
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner._record_rooted_flow_event(
                    source,
                    target,
                    priority=priority,
                    message=message,
                    success=success,
                )
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                self.establish_connection(source, target, priority=priority)
                hypha = self.hyphae[hypha_id]
            if success is not None:
                hypha.pulse(success=success)
                self._mark_topology_mutated_locked()
            if message is not None:
                hypha.log(message)
                self._mark_topology_mutated_locked()
            return hypha.model_copy(deep=True)

    def set_hypha_strength(
        self,
        source: str,
        target: str | None,
        strength: float,
    ) -> bool:
        """Set current edge strength through the topology owner."""
        hypha_id = self._hypha_id(source, target)
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                return owner.set_hypha_strength(source, target, strength)
            hypha = self.hyphae.get(hypha_id)
            if hypha is None:
                return False
            hypha.strength = max(0.1, min(10.0, float(strength)))
            self._mark_topology_mutated_locked()
            return True

    def link_layer(self, layer_name: str, module_class: Any):
        """High-level linking for transcendence modules."""
        logger.info("🍄 [MYCELIUM] Linking Transcendence Layer: '%s' -> %s", layer_name, module_class.__name__)
        # This typically involves registering the module's presence for the discovery engine
        # and creating primary hyphae to the core cognition engine.
        self.establish_connection(layer_name, "cognition", priority=0.9)
        self.establish_connection("cognition", layer_name, priority=0.8)

    def route_signal(self, source: str, target: str, payload: dict[str, Any]):
        """Directly route a cognitive signal between subsystems."""
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            return owner.route_signal(source, target, payload)
        hypha_id = f"{source}->{target}"
        try:
            if self.get_hypha(hypha_id) is None:
                self.establish_connection(source, target)
            self._log_route_signal(source, target, payload)
            if self.pulse_hypha(hypha_id, success=True):
                return True
            # The singleton may have been replaced between lookup and pulse.
            self.establish_connection(source, target)
            return self.pulse_hypha(hypha_id, success=True)
        except RuntimeError:
            return False

    def _log_route_signal(self, source: str, target: str, payload: dict[str, Any]) -> None:
        """Emit route-signal telemetry on state change instead of every pulse."""
        key = f"{source}->{target}"
        payload_text = str(payload)[:160]
        now = time.monotonic()
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return
            if owner is not self:
                return owner._log_route_signal(source, target, payload)
            previous_payload, previous_at, suppressed = (
                self._route_signal_log_state.get(key, ("", 0.0, 0))
            )
            repeated = payload_text == previous_payload and (now - previous_at) < 30.0
            if repeated:
                self._route_signal_log_state[key] = (
                    previous_payload,
                    previous_at,
                    suppressed + 1,
                )
            else:
                self._route_signal_log_state[key] = (payload_text, now, 0)
        if repeated:
            logger.debug(
                "🍄 [MYCELIUM] Repeated signal pulse suppressed: %s | Payload: %s",
                key,
                payload_text,
            )
            return

        if suppressed:
            logger.info(
                "🍄 [MYCELIUM] 📡 Signal Routed: %s -> %s | Payload: %s | repeated=%d",
                source,
                target,
                payload_text,
                suppressed,
            )
            return
        logger.info(
            "🍄 [MYCELIUM] 📡 Signal Routed: %s -> %s | Payload: %s",
            source,
            target,
            payload_text,
        )

    async def emit_reflex(self, signal_type: str, metadata: dict = None):
        """Broadcast a critical reflex signal across the mycelial network."""
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            return await owner.emit_reflex(signal_type, metadata)
        if self.reflex:
            await self.reflex.trigger_reflex(signal_type, metadata)
            return True
        else:
            logger.warning("No Reflex Core online to handle signal: %s", signal_type)
            return False

    async def emit(self, signal_type: str, metadata: dict = None):
        """Compatibility event-bus bridge for callers that treat mycelium like a bus."""
        owner = self._active_owner()
        if owner is None:
            return None
        if owner is not self:
            return await owner.emit(signal_type, metadata)
        payload = dict(metadata or {})
        payload.setdefault("signal_type", signal_type)
        try:
            from core.event_bus import EventPriority, get_event_bus

            await get_event_bus().publish(signal_type, payload, priority=EventPriority.COGNITIVE)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('mycelium', exc)
            logger.debug("🍄 [MYCELIUM] emit bridge publish failed: %s", exc)
        return payload

    def _should_monitor_hypha(self, hypha: Hypha) -> bool:
        """Alarm only on edges with a real continuous-liveness contract.

        Logical/event edges are expected to go quiet when their event does not
        occur. Prior traffic is evidence of use, not evidence that an edge
        must carry traffic every five minutes. Physical roots are the only
        current hyphae with an independently maintained heartbeat.
        """
        return bool(
            isinstance(hypha, NeuralRoot)
            and hypha.liveness_contract == "heartbeat"
            and hypha.state in {"ready_idle", "stale", "error"}
        )

    def establish_neural_root(self, source: str, hardware_id: str = "gpu_metal") -> NeuralRoot:
        """Record historical topology without claiming current liveness.

        Live owners must call :meth:`attest_neural_root` after validating the
        endpoint they actually own. This compatibility method deliberately
        creates an unbound root.
        """
        hypha_id = f"{source}->hardware:{hardware_id}"
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.establish_neural_root(source, hardware_id=hardware_id)
            existing = self.hyphae.get(hypha_id)
            if isinstance(existing, NeuralRoot):
                return existing.model_copy(deep=True)
            nr = NeuralRoot(
                name=hypha_id,
                source=source,
                target=f"hardware:{hardware_id}",
                hardware_id=hardware_id,
                root_kind="hardware",
                liveness_contract="on_demand",
                state="unbound",
                pinned=True,
                priority=5.0,
            )
            self.hyphae[hypha_id] = nr
            self._neural_roots = [
                hypha
                for hypha in self.hyphae.values()
                if isinstance(hypha, NeuralRoot)
            ]
            self._mark_topology_mutated_locked(structure_changed=True)
        logger.info("🍄 [MYCELIUM] 🌿 Neural Root topology recorded: %s (unbound)", hypha_id)
        return nr.model_copy(deep=True)

    @staticmethod
    def _neural_root_id(source: str, root_kind: str, target_id: str) -> str:
        source = str(source or "").strip()
        root_kind = str(root_kind or "").strip().lower()
        target_id = str(target_id or "").strip()
        if not source or root_kind not in {"worker", "service", "hardware"} or not target_id:
            raise ValueError("neural root requires source, supported kind, and target identity")
        return f"{source}->{root_kind}:{target_id}"

    @staticmethod
    def _canonical_root_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(evidence, dict):
            raise TypeError("neural root evidence must be a dictionary")
        try:
            encoded = json.dumps(
                evidence,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("neural root evidence must be finite JSON data") from exc
        if len(encoded.encode("utf-8")) > 16_384:
            raise ValueError("neural root evidence exceeds 16 KiB")
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise ValueError("neural root evidence must decode to an object")
        return decoded

    def attest_neural_root(
        self,
        source: str,
        *,
        root_kind: str,
        target_id: str,
        owner_generation: str,
        evidence: dict[str, Any],
        liveness_contract: str = "heartbeat",
        stale_after_s: float = 30.0,
    ) -> NeuralRoot:
        """Bind one endpoint to current, owner-supplied runtime evidence."""
        hypha_id = self._neural_root_id(source, root_kind, target_id)
        generation = str(owner_generation or "").strip()
        if not generation:
            raise ValueError("neural root attestation requires owner generation")
        contract = str(liveness_contract or "").strip().lower()
        if contract not in {"heartbeat", "on_demand"}:
            raise ValueError("unsupported neural root liveness contract")
        stale_after = float(stale_after_s)
        if not math.isfinite(stale_after) or stale_after <= 0.0:
            raise ValueError("neural root stale_after_s must be positive and finite")
        canonical_evidence = self._canonical_root_evidence(evidence)
        now = time.monotonic()
        recovered = False
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.attest_neural_root(
                    source,
                    root_kind=root_kind,
                    target_id=target_id,
                    owner_generation=generation,
                    evidence=canonical_evidence,
                    liveness_contract=contract,
                    stale_after_s=stale_after,
                )
            existing = self.hyphae.get(hypha_id)
            structure_changed = not isinstance(existing, NeuralRoot)
            if isinstance(existing, NeuralRoot):
                root = existing
                recovered = root.state in {"stale", "error"}
            else:
                root = NeuralRoot(
                    name=hypha_id,
                    source=str(source).strip(),
                    target=f"{str(root_kind).strip().lower()}:{str(target_id).strip()}",
                    hardware_id=str(target_id).strip(),
                    root_kind=str(root_kind).strip().lower(),
                    priority=5.0,
                    pinned=True,
                )
                self.hyphae[hypha_id] = root
            root.liveness_contract = contract
            root.state = "ready_idle"
            root.owner_generation = generation
            root.attested_identity = canonical_evidence
            root.last_activity_at = now
            root.last_probe_at = now
            root.last_probe_success_at = now
            root.last_pulse = now
            root.stale_after_s = stale_after
            root.last_error = ""
            root.active = True
            root.pulse_count += 1
            self._neural_roots = [
                hypha
                for hypha in self.hyphae.values()
                if isinstance(hypha, NeuralRoot)
            ]
            self._mark_topology_mutated_locked(structure_changed=structure_changed)
        if recovered:
            logger.info(
                "🍄 [MYCELIUM] Neural Root recovered from owner attestation: %s",
                hypha_id,
            )
        elif structure_changed:
            logger.info(
                "🍄 [MYCELIUM] 🌿 Neural Root owner-attested: %s",
                hypha_id,
            )
        return root.model_copy(deep=True)

    def pulse_neural_root(
        self,
        source: str,
        *,
        root_kind: str,
        target_id: str,
        owner_generation: str,
        success: bool = True,
        evidence: dict[str, Any] | None = None,
    ) -> bool:
        """Refresh only the root owned by the matching live generation."""
        hypha_id = self._neural_root_id(source, root_kind, target_id)
        now = time.monotonic()
        recovered = False
        failed = False
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                return owner.pulse_neural_root(
                    source,
                    root_kind=root_kind,
                    target_id=target_id,
                    owner_generation=owner_generation,
                    success=success,
                    evidence=evidence,
                )
            root = self.hyphae.get(hypha_id)
            if (
                not isinstance(root, NeuralRoot)
                or root.owner_generation != str(owner_generation or "").strip()
            ):
                return False
            previous_state = root.state
            root.last_activity_at = now
            root.last_probe_at = now
            root.last_pulse = now
            root.pulse_count += 1
            if evidence:
                root.attested_identity.update(
                    self._canonical_root_evidence(evidence)
                )
            if success:
                root.last_probe_success_at = now
                root.last_error = ""
                root.state = "ready_idle"
                root.active = True
                recovered = previous_state in {"stale", "error"}
            else:
                root.state = "error"
                root.active = False
                root.last_error = str((evidence or {}).get("error") or "owner probe failed")
                failed = previous_state != "error"
            self._mark_topology_mutated_locked()
        if recovered:
            logger.info("🍄 [MYCELIUM] Neural Root recovered: %s", hypha_id)
        elif failed:
            logger.warning(
                "🍄 [MYCELIUM] Neural Root owner probe failed: %s (%s)",
                hypha_id,
                root.last_error,
            )
        return True

    def unbind_neural_roots(self, source: str, *, owner_generation: str = "") -> int:
        """Invalidate current evidence without deleting historical topology."""
        changed = 0
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return 0
            if owner is not self:
                return owner.unbind_neural_roots(
                    source,
                    owner_generation=owner_generation,
                )
            generation = str(owner_generation or "").strip()
            for root in self._neural_roots:
                if root.source != str(source).strip():
                    continue
                if generation and root.owner_generation != generation:
                    continue
                if root.state == "unbound":
                    continue
                root.state = "unbound"
                root.active = False
                root.owner_generation = ""
                root.last_error = "owner_generation_retired"
                changed += 1
            if changed:
                self._mark_topology_mutated_locked()
        return changed

    async def hardware_pulse(self):
        """Compatibility hook: evaluate owner heartbeats; never invent one."""
        await self._pulse_once()

    def reinforce(
        self, pathway_id: str, success: bool, *, evidence: Any = None
    ):
        """Physarum-inspired conductivity update after skill execution.

        ``evidence`` is the execution result that justifies ``success``. When
        it actually carries an outcome the reinforcement counts as VERIFIED;
        without it the update still applies but at reduced weight and is
        tallied separately (CP126 2462d3c5).

        Enterprise Enhancement: Also pulses all physical hyphae connected to
        the pathway's source module, so the import graph strengthens where
        it matters at runtime.
        
        Transcendental Enhancement: Reinforcement is weighted by qualia norm.
        Pathways fired during high phenomenal intensity learn faster.
        """
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return
            if owner is not self:
                return owner.reinforce(pathway_id, success, evidence=evidence)
            pw = self.pathways.get(pathway_id)
            if not pw:
                return
            # CP126 / durable-learning governance: the strength of the
            # evidence decides the SCOPE of the change, not just its size.
            #
            # Before this, "nothing threw" moved the same persisted number
            # that a checked outcome moved — only by less. A route could
            # therefore become durably more trusted for producing failures,
            # one unverified success at a time, and nothing in the topology
            # recorded that its confidence rested on assertions.
            #
            # Now an unverified outcome steers THIS SESSION and dies with the
            # process. The broken-route safety property is kept: a session
            # weakening still lowers effective confidence immediately, so a
            # pathway that just failed is avoided now — it simply does not
            # rewrite what Aura believes tomorrow on the strength of a
            # caller's say-so.
            grade = grade_from_evidence(evidence, success=success)
            admission = admit_learning_update(
                "mycelium",
                str(pathway_id),
                operation="reinforce" if success else "weaken",
                success=bool(success),
                grade=grade,
                verifier="mycelium.execution_evidence" if grade > VerificationGrade.ASSERTED else None,
                evidence_id=_evidence_identity(evidence),
                inverse=(
                    {"operation": "set_confidence", "confidence": float(pw.confidence)}
                    if grade > VerificationGrade.ASSERTED
                    else None
                ),
            )
            if admission.is_durable:
                pw.reinforce(success, verified=True)
                self._mark_topology_mutated_locked()
            elif admission.applies_now:
                self._apply_session_reinforcement_locked(str(pathway_id), success)
                pw.record_unverified_reinforcement()

        # --- QUALIA-WEIGHTED REINFORCEMENT (session lane only) ---
        # CP126 d926886e. Phenomenal intensity says how vividly Aura was
        # experiencing when a route fired. It says nothing about whether the
        # route worked, so it is not evidence about the route — and this block
        # was writing pw.confidence directly, straight past the evidence gate
        # the code above had just applied. An ASSERTED-only outcome that the
        # gate had confined to this session became durable anyway, as long as
        # q_norm happened to be high.
        #
        # Two further defects made it worse. Neither q_norm nor arousal was
        # validated — an out-of-range or non-finite reading scaled the bonus
        # without limit, and a missing experiencer substituted 0.5 as though an
        # unmeasured arousal were a measured middling one. And the clamps were
        # min(10.0)/max(0.1) while a pathway's declared range is
        # [MIN_CONFIDENCE, MAX_CONFIDENCE] = [0.05, 1.0], so qualia could push
        # a route to ten times its own ceiling and permanently distort is_weak,
        # cohesion and routing rank.
        #
        # It now modulates only the session view, within the declared range,
        # and only from readings that are actually present and in range.
        try:
            from core.container import ServiceContainer
            qualia = ServiceContainer.get("qualia_synthesizer", default=None)
            q_norm = _unit_reading(getattr(qualia, "q_norm", None))
            if q_norm is not None and q_norm > 0.5:
                experiencer = ServiceContainer.get(
                    "phenomenological_experiencer", default=None
                )
                arousal = _unit_reading(getattr(experiencer, "current_arousal", None))
                if arousal is None:
                    # Unmeasured arousal is not middling arousal. Skip the
                    # weighting rather than inventing the missing half of it.
                    logger.debug(
                        "🍄 [MYCELIUM] Qualia weighting skipped for '%s': arousal "
                        "is unmeasured.",
                        pathway_id,
                    )
                else:
                    qualia_bonus = (q_norm - 0.5) * 0.1 * (arousal * 2.0)
                    with MycelialNetwork._lock:
                        if self._active_owner_locked() is not self:
                            return
                        if self.pathways.get(pathway_id) is None:
                            return
                        self._apply_session_confidence_delta_locked(
                            str(pathway_id),
                            qualia_bonus if success else -qualia_bonus * 0.5,
                        )
                    logger.debug(
                        "🍄 [MYCELIUM] 🧠 Qualia-weighted session adjustment: "
                        "'%s' ±%.3f (q=%.2f, a=%.2f)",
                        pathway_id, qualia_bonus, q_norm, arousal
                    )
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('mycelium', e)
            capture_and_log(e, {'module': __name__})

        # --- RUNTIME PHYSICAL HYPHAE REINFORCEMENT ---
        with MycelialNetwork._lock:
            if self._active_owner_locked() is not self:
                return
            current = self.pathways.get(pathway_id)
            if current is None:
                return
            source_file = current.source_file
            infrastructure_mapped = self.infrastructure_mapped
        if source_file and infrastructure_mapped:
            source_module = None
            for mk, info in self.get_mapped_files_snapshot().items():
                if info.get("path") == source_file:
                    source_module = mk
                    break

            if source_module:
                pulsed = 0
                with MycelialNetwork._lock:
                    if self._active_owner_locked() is not self:
                        return
                    for h in self.hyphae.values():
                        if h.is_physical and (
                            h.source == source_module or h.target == source_module
                        ):
                            h.pulse(success)
                            pulsed += 1
                    if pulsed:
                        self._mark_topology_mutated_locked()
                if pulsed > 0:
                    logger.debug(
                        "🍄 [MYCELIUM] ⚡ Runtime pulse: %d physical hyphae for '%s' (%s)",
                        pulsed, source_module, "✓" if success else "✗",
                    )

        with MycelialNetwork._lock:
            if self._active_owner_locked() is not self:
                return
            current = self.pathways.get(pathway_id)
            confidence = current.confidence if current is not None else pw.confidence
            success_rate = current.success_rate if current is not None else pw.success_rate
            is_weak = current.is_weak if current is not None else pw.is_weak
        if is_weak:
            logger.warning(
                "🍄 [MYCELIUM] ⚠️ Pathway '%s' is WEAK (confidence=%.2f, rate=%.0f%%). "
                "Consider reviewing or removing.",
                pathway_id, confidence, success_rate * 100,
            )
        else:
            logger.debug(
                "🍄 [MYCELIUM] Pathway '%s' reinforced: confidence=%.2f (%s)",
                pathway_id, confidence, "✓" if success else "✗",
            )


    # --- Legacy Compatibility Shims ---

    def register_direct_root(self, pattern: str, skill_name: str):
        """Legacy shim: converts old substring patterns to basic regex pathways."""
        safe_pattern = re.escape(pattern)
        self.register_pathway(
            pathway_id=f"legacy_{pattern.replace(' ', '_')}",
            pattern=safe_pattern,
            skill_name=skill_name,
            param_map={},
            priority=0.5,  # Lower priority than proper regexes
            activity_label=f"Aura is executing {skill_name}...",
        )

    def match_direct_root(self, text: str) -> str | None:
        """Legacy shim: returns just the skill name for old orchestrator code."""
        result = self.match_hardwired(text)
        if result:
            return result[0].skill_name
        return None

    # ======================================================================
    # DISCOVERY ENGINE — Slime Mold Exploration
    # ======================================================================

    def record_execution(self, message: str, skill_name: str, params: dict[str, Any], success: bool):
        """Record a non-hardwired skill execution for pathway discovery.

        Called by the orchestrator after the state machine successfully routes
        a message to a skill via LLM classification (i.e., the slow path).
        If the same skill is used repeatedly with similar messages, the network
        proposes a new hardwired pathway.
        """
        if not success:
            return

        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return
            if owner is not self:
                return owner.record_execution(message, skill_name, params, success)
            self._execution_log.append({
                "message": message,
                "skill": skill_name,
                "params": dict(params),
                "timestamp": time.monotonic(),
            })
            if len(self._execution_log) > 500:
                self._execution_log = self._execution_log[-250:]
            self._discovery_candidates[skill_name] += 1
            should_propose = self._discovery_candidates[skill_name] >= 5

        if should_propose:
            self._propose_pathway(skill_name)

    def _propose_pathway(self, skill_name: str):
        """Analyze recent executions to propose a new hardwired pathway."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                return
            if owner is not self:
                return owner._propose_pathway(skill_name)
            relevant_count = sum(
                1 for event in self._execution_log if event["skill"] == skill_name
            )
            if relevant_count < 3:
                return
            existing_count = sum(
                1 for pathway in self.pathways.values()
                if pathway.skill_name == skill_name
            )
            if existing_count >= 5:
                return
            self._discovery_candidates[skill_name] = 0

        logger.info(
            "🍄 [MYCELIUM] 🌱 Discovery: skill '%s' used %d times via slow path. "
            "Consider adding a hardwired pathway for common patterns.",
            skill_name, relevant_count,
        )

    # ======================================================================
    # GENERAL HYPHAE — Subsystem Connections
    # ======================================================================

    def set_ui_callback(self, callback: Callable[[str], Coroutine]):
        """Connect the Mycelium directly to the UI for failsafe message delivery."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not self:
                return owner.set_ui_callback(callback)
            self.ui_callback = callback
        logger.info("🍄 [MYCELIUM] Direct UI Hypha Connected.")


    def establish_unification_hyphae(self):
        """Phase 25: Sovereign Unification Hyphae.
        
        Links canonical subsystems into the root network to ensure they are 
        visible and tracked even before dynamic mapping completes.
        Names here match SubsystemAudit.SUBSYSTEMS for identity synchronization.
        """
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            return owner.establish_unification_hyphae()
        unification_links = [
            ("orchestrator", "personality_engine", 3.0, "#FF69B4", "Core identity and persona control"),
            ("orchestrator", "memory_facade", 3.0, "#F5A623", "Long-term knowledge and episodic recall"),
            ("orchestrator", "affect_engine", 2.5, "#D0021B", "Emotional state and motivation substrate"),
            ("orchestrator", "drive_controller", 2.0, "#BD10E0", "Biological-inspired drives and urgency"),
            ("orchestrator", "liquid_substrate", 2.0, "#7ED321", "Dynamic arousal and focus management"),
            ("orchestrator", "sovereign_scanner", 2.0, "#50E3C2", "Reactive intent detection and safety"),
            ("personality_engine", "cognition", 2.5, "#4A90E2", "Identity guiding thought generation"),
            ("cognition", "autonomy", 3.0, "#9013FE", "Decision making and goal selection"),
            ("autonomy", "cognition", 3.0, "#9013FE", "Feedback loop for autonomous action"),
            ("mind_tick", "mycelium", 2.5, "#F8E71C", "Universal heartbeat and connectivity"),
            ("orchestrator", "critic_engine", 3.0, "#50E3C2", "Recursive self-correction and plan verification"),
            # --- Personhood & Resilience Roots ---
            ("orchestrator", "personhood", 3.0, "#FF007F", "Spontaneous thought and subjective agency"),
            ("orchestrator", "voice_presence", 3.0, "#00FFFF", "Vocal embodiment and immediate expression"),
            ("orchestrator", "stability_guardian", 3.0, "#39FF14", "Real-time health monitoring and stall prevention"),
            ("orchestrator", "research_cycle", 2.5, "#FFFF00", "Autonomous knowledge pursuit during idle"),
        ]
        for src, tgt, prio, color, desc in unification_links:
            self.establish_connection(src, tgt, priority=prio)
            with MycelialNetwork._lock:
                owner = self._active_owner_locked()
                if owner is None:
                    return False
                if owner is not self:
                    return owner.establish_unification_hyphae()
                hypha = self.hyphae.get(f"{src}->{tgt}")
                if hypha is not None:
                    hypha.color = color
                    hypha.description = desc
                    hypha.strength = 5.0
                    self._mark_topology_mutated_locked()
        logger.info("🍄 [MYCELIUM] ✅ Core Unification Hyphae established (%d links)", len(unification_links))
        return True

    def shutdown(self):
        """Phase 28: Total Neural Root Cleanup (Issue 76).
        Ensures all hardware pins and active hyphae are safely disconnected.
        """
        logger.info("🍄 [MYCELIUM] Neutralizing all neural roots and hyphae...")
        with MycelialNetwork._lock:
            self._stop_event.set()
            if MycelialNetwork._instance is self:
                MycelialNetwork._instance = None
                MycelialNetwork._initialized = False
            mapping_thread = self._mapping_thread
        if (
            mapping_thread is not None
            and mapping_thread.is_alive()
            and mapping_thread is not threading.current_thread()
        ):
            mapping_thread.join(timeout=_MAPPER_DRAIN_BUDGET_S)
            if mapping_thread.is_alive():
                # CP126 53f15b88: the state below is cleared whether or not the
                # mapper drained, and a thread that outlives shutdown still
                # holds references to the dicts being emptied. The clear is safe
                # — the mapper re-checks ownership and the stop event under the
                # topology lock before it publishes, so it cannot write into a
                # torn-down network — but "shutdown finished with a worker still
                # running" is a fact the runtime has to be able to see, not a
                # warning in a log nobody is reading.
                self._shutdown_left_mapper_running = True
                record_degradation(
                    "mycelium",
                    RuntimeError(
                        f"infrastructure mapper did not drain within "
                        f"{_MAPPER_DRAIN_BUDGET_S:.0f}s of shutdown"
                    ),
                    severity="error",
                    action=(
                        "cleared topology state with the mapper thread still "
                        "alive; its publication is latched closed and will be "
                        "discarded"
                    ),
                    enforce_failure_policy=False,
                )
                logger.warning(
                    "🍄 [MYCELIUM] Mapper did not drain within shutdown budget; "
                    "the publication latch remains closed."
                )
        with MycelialNetwork._lock:
            self.infrastructure_mapped = False
            self._is_mapping = False
            if mapping_thread is None or not mapping_thread.is_alive():
                self._mapping_thread = None
            self._execution_log.clear()
            self._discovery_candidates.clear()
            self._route_signal_log_state.clear()
            self._hypha_alert_times.clear()
            with _DEFERRED_PULSE_LOCK:
                self._deferred_pulses.clear()
            with _ABSORBED_FLOW_LOCK:
                self._absorbed_flows = []
            self.pathways.clear()
            self._aegis_replace("_pathway_order", [])
            self.direct_roots.clear()
            self.hyphae.clear()
            self.mapped_files.clear()
            self._centrality.clear()
            self._critical_modules.clear()
            self._cross_links.clear()
            self._neural_roots.clear()
            self.ui_callback = None
            self._mark_topology_mutated_locked(structure_changed=True)
        logger.info("🍄 [MYCELIUM] Network Offline.")

    def on_stop(self) -> None:
        """ServiceContainer lifecycle hook."""
        self.shutdown()

    def establish_consciousness_hyphae(self):
        """Phase 5: Transcendental Consciousness Hyphae.
        Specifically links modules related to qualia and phenomenology.
        """
        links = [
            ("qualia", "phenomenology", 3.0),
            ("consciousness", "global_workspace", 2.5),
            ("sentience", "autonomy", 2.0),
        ]
        for src, tgt, prio in links:
            self.establish_connection(src, tgt, priority=prio)
        # CP126 106a29f4: these are declared edges, not measured integration.
        # They start with evidence_basis "declared" and only become "observed"
        # once something actually pulses them; the log says which it is.
        logger.info(
            "🍄 [MYCELIUM] 👁️ Declared %d consciousness hyphae (no traffic "
            "observed on them yet).",
            len(links),
        )

    @asynccontextmanager
    async def rooted_flow(self, source: str, target: str, activity: str = None,
                          timeout: float = 60.0, priority: float = 1.0,
                          absorb_failures: bool | None = None):
        """Wraps a process in a mycelial root. If it stalls, the root overrides.

        ``absorb_failures`` decides whether a failure inside the block reaches
        the caller. Left unset it is derived from ``priority``, which is the
        historical rule and the one the live lanes are tuned for — but the
        derivation is now named rather than implied by a magic threshold, and
        every absorption is tracked until somebody collects it. Absorbing a
        failure without anyone asking is the shape of a failed action that
        looks finished (CP126 34f01634).
        """
        try:
            timeout_s = float(timeout)
        except (TypeError, ValueError):
            timeout_s = 60.0
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            timeout_s = 60.0
        activity_label = str(activity or f"{source}->{target}")
        owner = self._active_owner()
        if owner is None:
            raise RuntimeError("retired mycelium instance has no active owner")
        if owner is not self:
            async with owner.rooted_flow(
                source,
                target,
                activity=activity_label,
                timeout=timeout_s,
                priority=priority,
                absorb_failures=absorb_failures,
            ) as handle:
                yield handle
            return
        hypha_id = f"{source}->{target}"
        handle = RootedFlowHandle(self, source, target, priority)
        handle.log(f"INITIATING: {activity_label}")

        try:
            async with asyncio.timeout(timeout_s):
                yield handle
            handle.pulse(success=True)
            handle.log(f"SUCCESS: {activity_label}")
        except asyncio.CancelledError:
            try:
                handle.log(f"CANCELLED: {activity_label}")
            except Exception as topology_error:  # noqa: BLE001 - preserve cancellation
                record_degradation("mycelium.rooted_flow_telemetry", topology_error)
            raise
        except Exception as e:  # noqa: BLE001 - rooted-flow failure boundary
            handle._mark_failed(e)
            record_degradation('mycelium', e)
            hypha = None
            try:
                handle.pulse(success=False)
                handle.log(f"STALL/FAILURE: {activity_label} - {e}")
                hypha = handle._snapshot()
            except Exception as topology_error:  # noqa: BLE001 - preserve original error
                record_degradation("mycelium.rooted_flow_telemetry", topology_error)
                logger.error(
                    "🍄 [MYCELIUM] Could not persist rooted-flow failure for %s: %s",
                    hypha_id,
                    topology_error,
                    exc_info=True,
                )
            logger.error("🍄 [MYCELIUM] Critical Stall in %s (%s). Triggering Override.", hypha_id, e)
            recovery_owner = self._active_owner()
            if hypha is not None and recovery_owner is not None:
                recovery_timeout_s = min(5.0, max(0.1, timeout_s))
                try:
                    async with asyncio.timeout(recovery_timeout_s):
                        await recovery_owner._emergency_override(
                            hypha, activity_label, str(e)
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as recovery_error:  # noqa: BLE001 - recovery boundary
                    record_degradation(
                        "mycelium.emergency_override",
                        recovery_error,
                        severity="error",
                        action=(
                            "bounded emergency override failure without masking the "
                            "original rooted-flow error"
                        ),
                    )
                    logger.error(
                        "🍄 [MYCELIUM] Emergency override failed for %s: %s",
                        hypha_id,
                        recovery_error,
                        exc_info=True,
                    )
            if absorb_failures is None:
                # Historical rule, now stated: a priority root is a failsafe
                # bypass, so its failure is absorbed rather than propagated.
                absorbing = hypha is not None and hypha.priority >= 1.0
            else:
                absorbing = bool(absorb_failures)
            if absorbing:
                handle._mark_absorbed(activity_label)
                _track_absorbed_flow(self, handle)
                return  # Absorbed error — the handle carries it
            raise

    async def _emergency_override(self, hypha: Hypha, activity: str, error_msg: str):
        """Force a result through the Mycelium when the standard path stalls."""
        logger.warning("⚡ [ROOT OVERRIDE] Forcing path completion: %s → %s", hypha.name, activity)
        
        # Bridge to Hardened Reflex Core
        if self.reflex:
            await self.reflex.trigger_reflex("STALL_DETECTED", {
                "hypha": hypha.name,
                "activity": activity,
                "error": error_msg
            })
            
        if "response" in activity.lower() and self.ui_callback:
            # Honest failure: the operation did NOT complete. Do not claim the
            # block was bypassed when the original action remains failed.
            msg = (
                "⚠️ I hit a stall while processing your request and couldn't "
                f"complete it ({error_msg}). I've logged the failure and will "
                "recover the affected path — please try again."
            )
            await self.ui_callback(msg)
        # A stalled path must NOT be reinforced. Rerouting away from a failed
        # route is the correct adaptation; strengthening it (previously
        # +2.0, net-positive even after the -1.0 failure pulse) entrenched a
        # path that did not complete.

    # ======================================================================
    # INFRASTRUCTURE MAPPING — Codebase Unification
    # ======================================================================

    def map_infrastructure(
        self,
        base_dir: str,
        scan_dirs: list[str] | None = None,
        *,
        force: bool = False,
        _admission_token: object | None = None,
    ) -> bool:
        """Publish one complete code-map generation or retain the previous one.

        This public boundary owns admission and cleanup for both direct callers
        and the background worker. No exception can leave the mapping latch set.

        Args:
            base_dir: Absolute path to the project root (e.g., autonomy_engine/).
            scan_dirs: Optional subdirectories under ``base_dir`` to scan.
                The default covers every production source root plus root modules.
        """
        owner = self._active_owner()
        if owner is None:
            return False
        if owner is not self:
            if _admission_token is not None:
                return False
            return owner.map_infrastructure(base_dir, scan_dirs, force=force)
        with self._mapping_lock:
            owner = self._active_owner_locked()
            if owner is None:
                return False
            if owner is not self:
                if _admission_token is not None:
                    return False
                return owner.map_infrastructure(base_dir, scan_dirs, force=force)
            if not force and self._foreground_mapping_deferred():
                return False
            if _admission_token is None:
                if self._is_mapping or (self.infrastructure_mapped and not force):
                    return False
                admission_token = object()
                self._mapping_admission_token = admission_token
                self._is_mapping = True
            else:
                admission_token = _admission_token
                if (
                    not self._is_mapping
                    or self._mapping_admission_token is not admission_token
                ):
                    return False
            previously_mapped = self.infrastructure_mapped
            self._mapping_started_at = time.time()
            self._mapping_last_error = None
            self._deferred_mapping_reason = None

        try:
            return self._map_infrastructure_generation(
                base_dir,
                scan_dirs,
                previously_mapped=previously_mapped,
            )
        except Exception as exc:  # noqa: BLE001 - public lifecycle boundary records then re-raises
            with self._mapping_lock:
                self.infrastructure_mapped = previously_mapped
                self._mapping_last_error = f"{type(exc).__name__}: {exc}"
            record_degradation(
                "mycelium",
                exc,
                severity="warning",
                action=(
                    "retained the prior complete infrastructure generation after "
                    "mapping failure"
                    if previously_mapped
                    else "left infrastructure graph unmapped after mapping failure"
                ),
            )
            raise
        finally:
            with self._mapping_lock:
                if self._mapping_admission_token is admission_token:
                    self._mapping_admission_token = None
                    self._is_mapping = False

    def _map_infrastructure_generation(
        self,
        base_dir: str,
        scan_dirs: list[str] | None,
        *,
        previously_mapped: bool,
    ) -> bool:
        """Build a private infrastructure generation and publish it atomically."""
        # Optimization: Use a local cache for AST results to avoid re-parsing if called multiple times
        # though singleton pattern usually prevents this.
        
        base = Path(base_dir).resolve()
        if scan_dirs is None:
            scan_dirs = list(_DEFAULT_INFRASTRUCTURE_SCAN_DIRS)

        start_time_map = time.monotonic()
        logger.info("🍄 [MYCELIUM] 🗺️ Infrastructure Mapping starting from: %s", base)

        # 1. Discover all .py files.
        all_files: dict[str, Path] = {}  # module_key → file_path
        if scan_dirs == list(_DEFAULT_INFRASTRUCTURE_SCAN_DIRS):
            for py_file in base.glob("*.py"):
                if not py_file.name.startswith("__"):
                    all_files[py_file.stem] = py_file
        # CP126 a56e4c5d: discovery and AST parsing had cancellation but no
        # budget and no ceiling, so a base_dir with a large vendored or
        # generated tree under it turned a maintenance sweep into an unbounded
        # read-and-parse of everything reachable. The generation is refused
        # rather than truncated — half a dependency graph published as a whole
        # one is worse than the previous generation standing.
        for subdir in scan_dirs:
            scan_root = base / subdir
            if not scan_root.exists():
                logger.debug("🍄 [MYCELIUM] Scan directory not found: %s", scan_root)
                continue
            for py_file in scan_root.rglob("*.py"):
                if self._stop_event.is_set():
                    logger.info("🍄 [MYCELIUM] Infrastructure mapping cancelled during discovery.")
                    return False
                if len(all_files) >= _MAPPING_MAX_MODULES:
                    raise ValueError(
                        f"infrastructure scan exceeds {_MAPPING_MAX_MODULES} modules; "
                        "refusing to publish a partial generation"
                    )
                if time.monotonic() - start_time_map > _MAPPING_BUDGET_S:
                    raise TimeoutError(
                        f"infrastructure discovery exceeded its {_MAPPING_BUDGET_S:.0f}s "
                        "budget"
                    )
                if py_file.name.startswith("__"):
                    continue
                try:
                    rel = py_file.relative_to(base)
                    module_key = str(rel.with_suffix("")).replace(os.sep, ".")
                    all_files[module_key] = py_file
                except ValueError:
                    continue

        logger.info("🍄 [MYCELIUM] Discovered %d Python modules.", len(all_files))

        # 2. Parse imports and build dependency edges
        dependency_graph: dict[str, list[str]] = {}
        mapped_files: dict[str, dict[str, Any]] = {}
        for module_key, file_path in all_files.items():
            if self._stop_event.is_set():
                logger.info("🍄 [MYCELIUM] Infrastructure mapping cancelled during parsing.")
                return False
            if time.monotonic() - start_time_map > _MAPPING_BUDGET_S:
                raise TimeoutError(
                    f"infrastructure parsing exceeded its {_MAPPING_BUDGET_S:.0f}s budget "
                    f"after {len(dependency_graph)} of {len(all_files)} modules"
                )
            deps, source_sha256 = self._extract_imports(file_path, base)
            dependency_graph[module_key] = deps

            # Build privately. Readers must never observe a half-published map.
            mapped_files[module_key] = {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
                "imports": deps,
                # None when the module could not be read or parsed: an empty
                # import list from an unreadable file is not evidence that the
                # module imports nothing.
                "source_sha256": source_sha256,
            }

        # 3. Create physical Hypha connections for import relationships
        physical_hyphae: dict[str, Hypha] = {}
        for module_key, deps in dependency_graph.items():
            for dep in deps:
                if dep in all_files:
                    hypha_name = f"import:{module_key}->{dep}"
                    h = Hypha(
                        name=hypha_name,
                        source=module_key,
                        target=dep,
                        priority=0.5,
                        is_physical=True,
                    )
                    h.source_file = str(all_files[module_key])
                    h.target_file = str(all_files[dep])
                    physical_hyphae[hypha_name] = h
        physical_connections = len(physical_hyphae)

        # 4. Compute Module Centrality (reverse dependency index)
        #    Centrality = how many other modules depend on this one.
        #    High centrality = load-bearing pillar; failure has wide blast radius.
        reverse_deps: dict[str, int] = {}
        for _module_key, deps in dependency_graph.items():
            for dep in deps:
                if dep in all_files:
                    reverse_deps[dep] = reverse_deps.get(dep, 0) + 1

        centrality = {k: int(v) for k, v in reverse_deps.items()}

        # Tag the top-20 most critical modules
        critical_modules = [
            module
            for module, _count in sorted(
                reverse_deps.items(), key=lambda item: item[1], reverse=True
            )[:20]
        ]

        # Store centrality in mapped_files for API exposure
        for module_key, module_data in mapped_files.items():
            module_data["centrality"] = reverse_deps.get(module_key, 0)
            module_data["is_critical"] = module_key in critical_modules

        # 5. Cross-Layer Linking: connect logical subsystem hyphae to physical backing
        #    Maps abstract subsystem names (e.g., "cognition") to the directory/module
        #    patterns they correspond to in the codebase.
        SUBSYSTEM_ALIASES: dict[str, list[str]] = {
            "cognition": ["cognitive", "brain", "cognitive_engine", "cognitive_integration"],
            "personality": ["personality", "persona", "identity"],
            "memory": ["memory", "dual_memory", "episodic"],
            "affect": ["affect", "emotion", "mood"],
            "autonomy": ["autonomy", "autonomic", "volition", "agency"],
            "perception": ["perception", "senses", "sensory", "screen_observer"],
            "consciousness": ["consciousness", "awareness", "existential", "qualia", "subjectivity", "sentience"],
            "self_modification": ["self_modification", "self_mod", "evolution", "mutate"],
            "skills": ["skills", "capability", "skill_management"],
            "scanner": ["scanner", "cognitive.scanner"],
            "mycelium": ["mycelium"],
            "guardian": ["guardian", "autonomy_guardian"],
            "state_machine": ["state_machine", "orchestrator.state"],
            "drive_engine": ["drive", "motivation", "drives"],
            "telemetry": ["telemetry", "thought_stream", "neural_feed"],
            "system": ["orchestrator", "main", "container"],
            "core_logic": ["orchestrator", "pipeline", "cognitive"],
            "skill_execution": ["capability_engine", "skill_execution"],
            # Phase XXII: Transcendence subsystems
            "meta_evolution": ["meta_cognition", "meta_evolution"],
            "hephaestus": ["hephaestus", "skill_management"],
            "networking": ["networking"],
            "model_selector": ["model_selector", "llm", "brain"],
            "curiosity": ["curiosity", "curiosity_engine", "exploration"],
            # Phase II: Deep consciousness sub-modules
            "cel": ["constitutive_expression", "cel"],
            "iit_phi": ["iit_surrogate", "riiu", "phi"],
            "workspace": ["global_workspace", "gwt"],
            "ganglion": ["ganglion_node", "ganglion"],
            "executive": ["executive_inhibitor", "executive"],
            "qualia_engine": ["qualia_engine"],
            "quantum_entropy": ["quantum_entropy"],
            "opacity": ["structural_opacity", "opacity"],
        }

        def _matches_subsystem(subsystem_name: str, module_path: str) -> bool:
            """Whether a module belongs to a named subsystem.

            CP126 9d0e7313. This was raw substring containment, so ``cel``
            matched ``cancel``, ``parcel`` and ``excel``; ``phi`` matched
            ``philosophy``; ``identity`` matched
            ``resident_recurrent_sft_adapter_identity``. Every false hit became
            a cross-layer link, and cross-layer links are presented as
            integration.

            Matching is on name components — a module is split on ``.``, ``/``
            and ``_``, and an alias matches when it equals a component or a
            contiguous run of them. ``llm`` still matches ``core.brain.llm_router``
            and ``cognitive_engine`` still matches ``core.cognitive_engine``;
            ``cel`` no longer matches ``cancel``.
            """
            aliases = SUBSYSTEM_ALIASES.get(subsystem_name, [subsystem_name])
            parts = _module_name_parts(module_path)
            return any(_alias_matches_parts(alias, parts) for alias in aliases)

        def _build_cross_links(
            logical_hyphae: dict[str, Hypha],
        ) -> dict[str, list[str]]:
            links: dict[str, list[str]] = {}
            for logical_name, logical_hypha in logical_hyphae.items():
                backing_physical: list[str] = []
                for physical_name, physical_hypha in physical_hyphae.items():
                    source_matches = _matches_subsystem(
                        logical_hypha.source, physical_hypha.source
                    )
                    target_matches = _matches_subsystem(
                        logical_hypha.target, physical_hypha.target
                    )
                    if source_matches and target_matches:
                        backing_physical.append(physical_name)
                if backing_physical:
                    links[logical_name] = backing_physical
            return links

        # M-15 FIX: Prevent false-positive mapping if zero modules found
        if not all_files:
            logger.warning("🍄 [MYCELIUM] ❌ Infrastructure mapping found 0 modules! Retrying in next cycle.")
            with self._mapping_lock:
                self.infrastructure_mapped = previously_mapped
                self._mapping_last_error = "no_modules_discovered"
            return False

        # Cross-linking is O(logical × physical), so compute it outside the
        # topology lock. A compact endpoint signature detects structural races;
        # dynamic pulse updates do not force needless retries.
        annotated = 0
        for _attempt in range(5):
            with MycelialNetwork._lock:
                logical_hyphae = {
                    name: hypha.model_copy(deep=True)
                    for name, hypha in self.hyphae.items()
                    if not hypha.is_physical
                }
                logical_signature = tuple(
                    sorted(
                        (name, hypha.source, hypha.target)
                        for name, hypha in logical_hyphae.items()
                    )
                )
                pathway_skills = {
                    pathway_id: pathway.skill_name
                    for pathway_id, pathway in self.pathways.items()
                }
                pathway_signature = tuple(
                    sorted(
                        (
                            pathway_id,
                            id(pathway),
                            pathway.skill_name,
                        )
                        for pathway_id, pathway in self.pathways.items()
                    )
                )
            cross_links = _build_cross_links(logical_hyphae)
            pathway_annotations = self._build_pathway_annotations(
                pathway_skills,
                all_files,
                dependency_graph,
            )

            # Publish one coherent generation. UI, health, reinforcement, and
            # vault readers see either the previous graph or this complete graph.
            with MycelialNetwork._lock:
                if self._active_owner_locked() is not self:
                    logger.info(
                        "🍄 [MYCELIUM] Infrastructure publication cancelled after owner replacement."
                    )
                    return False
                if self._stop_event.is_set():
                    logger.info(
                        "🍄 [MYCELIUM] Infrastructure mapping cancelled before publication."
                    )
                    return False
                current_logical = {
                    name: hypha
                    for name, hypha in self.hyphae.items()
                    if not hypha.is_physical
                }
                current_signature = tuple(
                    sorted(
                        (name, hypha.source, hypha.target)
                        for name, hypha in current_logical.items()
                    )
                )
                if current_signature != logical_signature:
                    continue
                current_pathway_signature = tuple(
                    sorted(
                        (
                            pathway_id,
                            id(pathway),
                            pathway.skill_name,
                        )
                        for pathway_id, pathway in self.pathways.items()
                    )
                )
                if current_pathway_signature != pathway_signature:
                    continue

                # Preserve learned dynamic state for unchanged import edges.
                for name, replacement in physical_hyphae.items():
                    existing = self.hyphae.get(name)
                    if (
                        existing is None
                        or not existing.is_physical
                        or existing.source != replacement.source
                        or existing.target != replacement.target
                    ):
                        continue
                    replacement.strength = existing.strength
                    replacement.created_at = existing.created_at
                    replacement.last_pulse = existing.last_pulse
                    replacement.pulse_count = existing.pulse_count
                    replacement.active = existing.active
                    replacement.color = existing.color
                    replacement.description = existing.description
                    replacement.size = existing.size
                    replacement.trace = list(existing.trace)

                previous_mapped_paths = {
                    str(module.get("path"))
                    for module in self.mapped_files.values()
                    if module.get("path")
                }
                for pathway_id, pathway in self.pathways.items():
                    annotation = pathway_annotations.get(pathway_id)
                    if annotation is not None:
                        source_file, dependencies = annotation
                        pathway.source_file = source_file
                        pathway.dependencies = list(dependencies)
                    elif pathway.source_file in previous_mapped_paths:
                        pathway.source_file = None
                        pathway.dependencies = []

                next_hyphae = dict(current_logical)
                next_hyphae.update(physical_hyphae)
                self._aegis_replace("hyphae", next_hyphae)
                self.mapped_files = mapped_files
                self._centrality = centrality
                self._critical_modules = critical_modules
                self._cross_links = cross_links
                self.infrastructure_mapped = True
                self._mapping_completed_at = time.time()
                self._mapping_last_error = None
                self._deferred_mapping_reason = None
                self._mapping_generation += 1
                self._mark_topology_mutated_locked(structure_changed=True)
                annotated = len(pathway_annotations)
                break
        else:
            raise RuntimeError(
                "logical topology changed repeatedly while publishing infrastructure map"
            )
        elapsed = time.monotonic() - start_time_map
        logger.info(
            "🍄 [MYCELIUM] 🗺️ Infrastructure Mapping COMPLETE (%.2fs): "
            "%d modules, %d physical connections, %d pathways annotated, "
            "%d critical indicators tagged.",
            elapsed, len(all_files), physical_connections, annotated, len(critical_modules)
        )
        return True

    @staticmethod
    def _build_pathway_annotations(
        pathway_skills: dict[str, str],
        all_files: dict[str, Path],
        dependency_graph: dict[str, list[str]],
    ) -> dict[str, tuple[str, list[str]]]:
        """Attribute a pathway to the module that implements its skill.

        CP126 88be9de3. The rule was containment in either direction with a
        ``break`` on the first hit, so ``web.py`` and ``search.py`` both claimed
        ``web_search``, and which one won depended on directory walk order —
        the same codebase could annotate differently between two runs. A source
        attribution that changes without the source changing is not evidence of
        anything.

        Now: an exact match on the module's own name wins, a full-module-key
        match is the fallback, and an ambiguous result annotates nothing. No
        owner is a truthful answer; an arbitrary one is not.
        """
        annotations: dict[str, tuple[str, list[str]]] = {}
        for pathway_id, skill_name in pathway_skills.items():
            skill_parts = _module_name_parts(skill_name)
            if not skill_parts:
                continue
            exact: list[str] = []
            tail: list[str] = []
            for module_key, file_path in all_files.items():
                if _module_name_parts(file_path.stem) == skill_parts:
                    exact.append(module_key)
                elif _module_name_parts(module_key)[-len(skill_parts) :] == skill_parts:
                    tail.append(module_key)
            candidates = exact or tail
            if len(candidates) != 1:
                if len(candidates) > 1:
                    logger.debug(
                        "🍄 [MYCELIUM] Pathway '%s' has %d candidate source "
                        "modules; leaving it unattributed.",
                        pathway_id,
                        len(candidates),
                    )
                continue
            module_key = candidates[0]
            annotations[pathway_id] = (
                str(all_files[module_key]),
                dependency_graph.get(module_key, []),
            )
        return annotations

    def _foreground_mapping_deferred(self) -> bool:
        if not foreground_only_runtime():
            return False
        if bool(_ALLOW_FOREGROUND_MAPPING_FLAG.value()):
            return False
        quiet_s = float(_FOREGROUND_MAPPING_QUIET_FLAG.value())
        age_s = max(
            0.0,
            time.monotonic()
            - float(getattr(self, "_created_at_monotonic", time.monotonic())),
        )
        if age_s < max(0.0, quiet_s):
            self._deferred_mapping_reason = (
                f"foreground_quiet_window:{age_s:.1f}s/{max(0.0, quiet_s):.1f}s"
            )
            logger.info(
                "🍄 [MYCELIUM] Infrastructure mapping deferred (%s).",
                self._deferred_mapping_reason,
            )
            return True
        return False

    def _extract_imports(
        self, file_path: Path, base_dir: Path
    ) -> tuple[list[str], str | None]:
        """Extract import targets, and the digest of the bytes they came from.

        CP126 fe556a31. ``errors="ignore"`` silently deleted every byte that
        would not decode, so the AST was built from a program that differs from
        the one on disk — and nothing downstream could tell, because the result
        looked like a clean parse of a real module. Undecodable source is now a
        parse failure, which is what it is.

        The returned digest is over the exact bytes parsed, so the topology
        derived from a module is tied to the source generation it was derived
        from (CP126 5894e929). ``None`` means the module was not successfully
        read — an unmapped module, not a module that maps to nothing.
        """
        imports: list[str] = []
        try:
            raw = file_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            source = raw.decode("utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, ValueError, OSError) as e:
            logger.debug("🍄 [MYCELIUM] AST parse failed for %s: %s", file_path.name, e)
            return imports, None

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Resolve relative imports
                    if node.level > 0:
                        try:
                            rel = file_path.parent.relative_to(base_dir)
                            parts = list(rel.parts)
                            # Go up 'level - 1' parents
                            if node.level > 1:
                                parts = parts[:-(node.level - 1)] if len(parts) >= node.level - 1 else parts
                            base_module = ".".join(parts)
                            full_module = f"{base_module}.{node.module}" if base_module else node.module
                            imports.append(full_module)
                        except (ValueError, IndexError):
                            imports.append(node.module)
                    else:
                        imports.append(node.module)

        return imports, digest

    def get_mapped_files_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return one detached infrastructure-map generation for concurrent readers."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_mapped_files_snapshot()
            return self._mapped_files_snapshot_locked()

    def _mapped_files_snapshot_locked(self) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for module_key, module_data in self.mapped_files.items():
            detached = dict(module_data)
            imports = detached.get("imports")
            if isinstance(imports, list):
                detached["imports"] = list(imports)
            snapshot[module_key] = detached
        return snapshot

    def get_route_cache_token(self) -> tuple[int, int]:
        """Return the active topology owner's identity and structure revision."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_route_cache_token()
            return id(self), self._topology_structure_revision

    def get_graph_snapshot(self) -> dict[str, Any]:
        """Return topology and code map from one published generation."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_graph_snapshot()
            return {
                "topology": self._network_topology_snapshot_locked(),
                "mapped_files": self._mapped_files_snapshot_locked(),
                "centrality": dict(self._centrality),
                "critical_modules": list(self._critical_modules),
                "mapping_generation": self._mapping_generation,
                "mapping_state": self._mapping_state_locked(),
                "mapping_last_error": self._mapping_last_error,
                "topology_revision": self._topology_revision,
                "topology_structure_revision": self._topology_structure_revision,
            }

    def get_runtime_snapshot(self) -> dict[str, Any]:
        """Return the complete API read model under one topology lock."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_runtime_snapshot()
            return {
                "topology": self._network_topology_snapshot_locked(),
                "infrastructure": self._infrastructure_report_snapshot_locked(),
            }

    def get_topology_counts(self) -> dict[str, int]:
        """Return the atomically replaced count read model without graph copying."""
        owner = MycelialNetwork._instance
        owner_stop = getattr(owner, "_stop_event", None)
        if (
            owner is not None
            and owner is not self
            and owner_stop is not None
            and not owner_stop.is_set()
        ):
            return owner.get_topology_counts()
        return dict(self._topology_counts_cache)

    def get_topology_summary(self) -> dict[str, int]:
        """Return the precomputed user-facing topology summary lock-free."""
        owner = MycelialNetwork._instance
        owner_stop = getattr(owner, "_stop_event", None)
        if (
            owner is not None
            and owner is not self
            and owner_stop is not None
            and not owner_stop.is_set()
        ):
            return owner.get_topology_summary()
        return dict(self._topology_summary_cache)

    def get_rooted_flow_integrity(self) -> dict[str, int]:
        """How many absorbed failures are outstanding, and how many went unclaimed.

        ``unclaimed`` is the number a caller never collected — each one is a
        failed action that its caller treated as complete.
        """
        owner = self._active_owner()
        if owner is not None and owner is not self:
            return owner.get_rooted_flow_integrity()
        with _ABSORBED_FLOW_LOCK:
            awaiting = len(self._absorbed_flows)
            overflow = self._absorbed_flow_overflow
        return {
            "absorptions_awaiting_acknowledgement": awaiting,
            "absorptions_unclaimed": self._unclaimed_absorptions,
            "absorptions_untracked_overflow": overflow,
        }

    def get_hypha_signal_snapshot(self, *, limit: int) -> list[tuple[float, float]]:
        """Return detached strength/recency inputs for bounded numeric consumers."""
        bounded_limit = max(0, int(limit))
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_hypha_signal_snapshot(limit=bounded_limit)
            return [
                (float(hypha.strength), float(hypha.last_pulse))
                for hypha in islice(self.hyphae.values(), bounded_limit)
            ]

    def _mapping_state_locked(self) -> str:
        if self._is_mapping:
            return "refreshing" if self.infrastructure_mapped else "running"
        if self.infrastructure_mapped:
            return "ready_with_refresh_error" if self._mapping_last_error else "ready"
        if self._mapping_last_error:
            return "failed"
        if self._deferred_mapping_reason:
            return "deferred"
        return "idle"

    def get_infrastructure_report(self) -> dict[str, Any]:
        """Return a summary of the infrastructure mapping for API/UI consumption."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_infrastructure_report()
            return self._infrastructure_report_snapshot_locked()

    def _infrastructure_report_snapshot_locked(self) -> dict[str, Any]:
        mapped_files = self._mapped_files_snapshot_locked()
        physical_hyphae = {
            name: {
                "source": hypha.source,
                "target": hypha.target,
                # Redacted to project-relative paths — never absolute.
                "source_file": _redact_source_path(hypha.source_file),
                "target_file": _redact_source_path(hypha.target_file),
                "strength": float(round(hypha.strength, 2)),
            }
            for name, hypha in self.hyphae.items()
            if hypha.is_physical
        }
        annotated_pathways = [
            pathway.pathway_id
            for pathway in self.pathways.values()
            if pathway.source_file
        ]
        return {
            "mapped": self.infrastructure_mapped,
            "mapping_state": self._mapping_state_locked(),
            "mapping_generation": self._mapping_generation,
            "topology_revision": self._topology_revision,
            "topology_structure_revision": self._topology_structure_revision,
            "deferred_reason": self._deferred_mapping_reason,
            "mapping_started_at": self._mapping_started_at,
            "mapping_completed_at": self._mapping_completed_at,
            "mapping_last_error": self._mapping_last_error,
            "total_modules": len(mapped_files),
            "physical_connections": len(physical_hyphae),
            "annotated_pathways": annotated_pathways,
            "critical_modules": [
                {"module": module, "centrality": self._centrality.get(module, 0)}
                for module in self._critical_modules
            ],
            "cross_layer_links": {
                logical: len(physical_list)
                for logical, physical_list in self._cross_links.items()
            },
            "modules": {k: _redact_source_path(v.get("path")) for k, v in mapped_files.items()},
            "physical_hyphae_sample": dict(list(physical_hyphae.items())[:20]),
            "vault_sync": {
                "revision": self._last_vault_sync_revision,
                "committed_at": self._last_vault_sync_at,
                "lag_revisions_at_commit": self._last_vault_sync_lag_revisions,
            },
            "shutdown_left_mapper_running": self._shutdown_left_mapper_running,
            "vault_restore": {
                "restored_at": self._restored_from_vault_at,
                # None = never restored. False = restored from a generation
                # carrying no tamper evidence, which is not the same as verified.
                "attested": self._restored_generation_attested,
            },
        }

    # ======================================================================
    # MAINTENANCE — Background Health
    # ======================================================================

    @staticmethod
    def _foreground_defers_pulse() -> bool:
        """Hypha maintenance can always wait 30s; conversation cannot."""
        try:
            from core.runtime.foreground_guard import foreground_activity_reason

            return bool(foreground_activity_reason())
        except (ImportError, RuntimeError, AttributeError):
            return False

    async def _pulse_once(self):
        """One pulse pass: refresh critical hyphae, report weak pathways."""
        owner = self._active_owner()
        if owner is None:
            return
        if owner is not self:
            await owner._pulse_once()
            return
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            now = time.monotonic()
            weak_pathways: list[tuple[str, float]] = []
            with MycelialNetwork._lock:
                owner = self._active_owner_locked()
                if owner is None:
                    return
                if owner is not self:
                    reroute = owner
                else:
                    reroute = None
                root_state_changed = False
                if reroute is not None:
                    weak_pathways = []
                else:
                    for name, hypha in self.hyphae.items():
                        if (
                            self._should_monitor_hypha(hypha)
                            and isinstance(hypha, NeuralRoot)
                            and hypha.state == "ready_idle"
                            and (
                                hypha.last_probe_success_at <= 0.0
                                or now - hypha.last_probe_success_at > hypha.stale_after_s
                            )
                        ):
                            hypha.state = "stale"
                            hypha.active = False
                            hypha.last_error = "owner_heartbeat_stale"
                            logger.warning(
                                "🍄 [MYCELIUM] Neural Root stale: %s "
                                "(last verified %.1fs ago; contract %.1fs).",
                                name,
                                (
                                    now - hypha.last_probe_success_at
                                    if hypha.last_probe_success_at > 0.0
                                    else float("inf")
                                ),
                                hypha.stale_after_s,
                            )
                            root_state_changed = True

                    weak_pathways = [
                        (pathway_id, pathway.confidence)
                        for pathway_id, pathway in self.pathways.items()
                        if pathway.is_weak
                        and pathway.hit_count + pathway.miss_count > 5
                    ]
                    if root_state_changed:
                        self._mark_topology_mutated_locked()
            if reroute is not None:
                await reroute._pulse_once()
                return
            for pathway_id, confidence in weak_pathways:
                logger.warning(
                    "🍄 [MYCELIUM] Weak pathway detected: '%s' (confidence=%.2f)",
                    pathway_id,
                    confidence,
                )
            self._report_unclaimed_absorptions()

    def _report_unclaimed_absorptions(self) -> None:
        """Name the failures that were swallowed and never collected.

        CP126 34f01634. The absorption itself is a deliberate failsafe; what was
        not deliberate is that nothing distinguished "the caller checked and
        handled it" from "the caller never asked and carried on as though the
        action had completed". Each unclaimed handle names the call site so the
        second case is fixable rather than invisible.
        """
        for handle in _sweep_absorbed_flows(self):
            self._unclaimed_absorptions += 1
            error = handle._error
            record_degradation(
                "mycelium.rooted_flow",
                error if isinstance(error, Exception) else RuntimeError(str(error)),
                severity="error",
                action=(
                    f"absorbed a failed rooted flow ({handle._hypha_id}: "
                    f"{handle._activity or 'unnamed activity'}) that "
                    f"{handle._caller_site} never collected — the caller "
                    "continued as though the action had completed"
                ),
                enforce_failure_policy=False,
            )
            logger.error(
                "🍄 [MYCELIUM] Unclaimed absorbed failure on %s (%s), opened at "
                "%s: %s",
                handle._hypha_id,
                handle._activity or "unnamed activity",
                handle._caller_site,
                error,
            )

    async def pulse_check(self):
        """Periodic background check to keep critical hyphae alive and prune weak pathways."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(30)
                if self._foreground_defers_pulse():
                    # Auto-pulse log bursts were firing mid-conversation in
                    # the 110GB-incident transcript; maintenance waits.
                    continue
                await self._pulse_once()
            except asyncio.CancelledError:
                # Cleanup for MemoryGovernor if it's running. Never cancel-and-
                # await THIS task from inside its own cancellation handler —
                # self-await raises/deadlocks the shutdown path.
                task = getattr(self, "_task", None)
                current = asyncio.current_task()
                if task is not None and task is not current:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError as _e:
                        logger.debug('Ignored asyncio.CancelledError in mycelium.py: %s', _e)
                    finally:
                        self._task = None
                elif task is current:
                    self._task = None

                # v8.1.0: Ensure total cleanup of any leaked worker handles
                try:
                    if hasattr(self, '_critical_cleanup') and callable(self._critical_cleanup):
                        await self._critical_cleanup()
                        logger.info("🛡️ Memory Governor shutdown complete. All worker handles leaked/active were purged.")
                except (RuntimeError, AttributeError, TypeError) as e:
                    record_degradation('mycelium', e)
                    logger.error("Error during Memory Governor shutdown: %s", e)
                logger.info("🍄 [MYCELIUM] Pulse check loop shutting down.")
                break
            except (OSError, ConnectionError, TimeoutError, RuntimeError, AttributeError, TypeError, ValueError, KeyError, LookupError) as e:
                # Broadened: a narrow I/O-only family let ordinary RuntimeError/
                # TypeError from a probe permanently kill the maintenance loop.
                record_degradation('mycelium', e)
                logger.error("🍄 [MYCELIUM] Pulse check error: %s", e, exc_info=True)
                await asyncio.sleep(10)  # Backoff on error

    # ======================================================================
    # INTROSPECTION — Topology & Health Reporting
    # ======================================================================

    def get_network_topology(self) -> dict[str, Any]:
        """Full network state for UI visualization and health monitoring."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_network_topology()
            return self._network_topology_snapshot_locked()

    @staticmethod
    def _redact_read_model_paths(data: dict[str, Any]) -> dict[str, Any]:
        """Redact absolute path fields from a public read-model dict."""
        for field_name in ("source_file", "target_file", "file", "path"):
            if field_name in data and data[field_name]:
                data[field_name] = _redact_source_path(data[field_name])
        return data

    @staticmethod
    def _redact_read_model_payloads(data: dict[str, Any]) -> dict[str, Any]:
        """Replace payload surfaces with their shape.

        CP126 479423bb. ``to_dict()`` and ``model_dump()`` were returned
        wholesale to API consumers. The edge trace is the concrete leak: the
        response lane opens its rooted flow with
        ``activity=f"Process: {message[:20]}"``, so every hypha trace carried
        the opening characters of the user's message into a public topology
        endpoint. ``direct_response`` is a canned reply body and ``param_map``
        is route-extraction detail; neither is read by any consumer of this
        model. The counts stay so the graph keeps the signal without the
        content.
        """
        trace = data.pop("trace", None)
        if trace is not None:
            data["trace_entries"] = len(trace) if isinstance(trace, list) else 0
        if "direct_response" in data:
            data["has_direct_response"] = bool(data.pop("direct_response"))
        param_map = data.pop("param_map", None)
        if param_map is not None:
            data["param_count"] = len(param_map) if isinstance(param_map, dict) else 0
        return data

    @classmethod
    def _public_read_model(cls, data: dict[str, Any]) -> dict[str, Any]:
        return cls._redact_read_model_payloads(cls._redact_read_model_paths(data))

    def _network_topology_snapshot_locked(self) -> dict[str, Any]:
        # Public read models must not leak absolute filesystem paths, and must
        # not carry route payloads or edge traces (CP126 363d2438, 479423bb).
        pathways = {
            pathway_id: self._public_read_model(pathway.to_dict())
            for pathway_id, pathway in self.pathways.items()
        }
        hyphae = {
            name: self._public_read_model(
                {**hypha.model_dump(), "evidence_basis": hypha.evidence_basis}
            )
            for name, hypha in self.hyphae.items()
        }
        cross_layer_linked = len(self._cross_links)
        infrastructure_mapped = self.infrastructure_mapped
        critical_modules = list(self._critical_modules[:10])
        discovery_candidates = dict(self._discovery_candidates)
        ui_connected = self.ui_callback is not None
        physical_count = sum(
            1 for hypha in hyphae.values() if hypha.get("is_physical")
        )
        logical_count = len(hyphae) - physical_count
        # No ``or [0.0]`` / ``or [1.0]`` padding: an absent measurement is not a
        # measurement of zero, and it is not a measurement of one either.
        strengths = [float(hypha.get("strength", 0.0)) for hypha in hyphae.values()]
        confidences = [
            float(pathway.get("confidence", 0.0))
            for pathway in pathways.values()
        ]

        return {
            "pathways": pathways,
            "pathway_count": len(pathways),
            "hyphae": hyphae,
            "topology_revision": self._topology_revision,
            "topology_structure_revision": self._topology_structure_revision,
            "hyphae_summary": {
                "total": len(hyphae),
                "logical": logical_count,
                "physical": physical_count,
                "cross_layer_linked": cross_layer_linked,
                "infrastructure_mapped": infrastructure_mapped,
                # CP126 106a29f4: a hardcoded ("qualia","phenomenology") edge
                # used to be indistinguishable from one that has carried traffic.
                "by_evidence_basis": {
                    basis: sum(
                        1
                        for edge in hyphae.values()
                        if edge.get("evidence_basis") == basis
                    )
                    for basis in ("observed", "static_import", "declared")
                },
            },
            "critical_modules": critical_modules,
            "topology_evidence": _TOPOLOGY_EVIDENCE_DISCLOSURE,
            "discovery_candidates": discovery_candidates,
            "ui_connected": ui_connected,
            "system_cohesion": _cohesion_from_topology(strengths, confidences),
            "cohesion_basis": {"edges": len(strengths), "routes": len(confidences)},
            "total_pathway_hits": sum(
                int(pathway.get("hit_count", 0)) for pathway in pathways.values()
            ),
            "total_pathway_misses": sum(
                int(pathway.get("miss_count", 0)) for pathway in pathways.values()
            ),
        }

    def get_unity_report(self) -> dict[str, Any]:
        """Backward-compatible unity report."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is not None and owner is not self:
                return owner.get_unity_report()
            hyphae = {
                name: {
                    "strength": hypha.strength,
                    "last_active": time.monotonic() - hypha.last_pulse,
                }
                for name, hypha in self.hyphae.items()
            }
            pathway_count = len(self.pathways)
            pathway_confidences = [
                pathway.confidence for pathway in self.pathways.values()
            ]
            ui_connected = self.ui_callback is not None
        strengths = [entry["strength"] for entry in hyphae.values()]
        return {
            "hyphae": hyphae,
            "pathways": pathway_count,
            "ui_connected": ui_connected,
            "system_cohesion": _cohesion_from_topology(strengths, pathway_confidences),
            "cohesion_basis": {
                "edges": len(strengths),
                "routes": len(pathway_confidences),
            },
        }

    def get_system_cohesion(self) -> float | None:
        """Fraction of the topology that is not weak, in [0, 1].

        ``None`` when there is no topology to measure. Callers that turn this
        into a statement about how Aura feels must check for that — a network
        with no edges and no routes has not been measured as fragmented, it has
        not been measured at all.
        """
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not None and owner is not self:
                return owner.get_system_cohesion()
            strengths = [h.strength for h in self.hyphae.values()]
            confidences = [pw.confidence for pw in self.pathways.values()]
        return _cohesion_from_topology(strengths, confidences)

    def get_cohesion_report(self) -> dict[str, Any]:
        """Cohesion with the sample it was computed from and its definition."""
        with MycelialNetwork._lock:
            owner = self._active_owner_locked()
            if owner is None:
                raise RuntimeError("retired mycelium instance has no active owner")
            if owner is not None and owner is not self:
                return owner.get_cohesion_report()
            strengths = [h.strength for h in self.hyphae.values()]
            confidences = [pw.confidence for pw in self.pathways.values()]
        value = _cohesion_from_topology(strengths, confidences)
        return {
            "value": value,
            "measured": value is not None,
            "edges": len(strengths),
            "weak_edges": sum(
                1 for s in strengths if s < Hypha.DEFAULT_STRENGTH
            ),
            "routes": len(confidences),
            "weak_routes": sum(
                1 for c in confidences if c < HardwiredPathway.PRUNE_THRESHOLD
            ),
            "definition": (
                "fraction of edges and routes at or above the health they were "
                "established with"
            ),
            "range": [0.0, 1.0],
        }

    def _calculate_cohesion(self) -> float | None:
        """Backward-compatible internal alias for the owner-backed read API."""
        return self.get_system_cohesion()

    # ======================================================================
    # PILLAR 3: THE ROOT VAULT (Aegis Persistence)
    # ======================================================================

    def _vault_snapshot_locked(self) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        captured_at_unix = time.time()
        pathways: dict[str, dict[str, Any]] = {}
        for key, pathway in self.pathways.items():
            data = pathway.to_dict()
            data.pop("id", None)
            created_at = self._vault_number(
                pathway.created_at,
                f"live pathway creation timestamp: {key}",
                minimum=0.0,
                maximum=captured_at_unix + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            last_matched = self._vault_number(
                pathway.last_matched,
                f"live pathway last-matched timestamp: {key}",
                minimum=0.0,
                maximum=now_monotonic + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            created_age = max(0.0, captured_at_unix - created_at)
            last_matched_age = max(0.0, now_monotonic - last_matched)
            if last_matched_age > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S:
                raise ValueError(
                    f"live pathway last match predates creation: {key}"
                )
            data["created_at"] = created_at
            data["last_matched_age_s"] = last_matched_age
            data.pop("last_matched", None)
            pathways[key] = data

        hyphae: dict[str, dict[str, Any]] = {}
        for key, hypha in self.hyphae.items():
            data = hypha.model_dump()
            if isinstance(hypha, NeuralRoot):
                data.update(
                    {
                        "active": False,
                        "state": "unbound",
                        "owner_generation": "",
                        "last_activity_at": 0.0,
                        "last_probe_at": 0.0,
                        "last_probe_success_at": 0.0,
                        "last_error": (
                            "persisted_historical_topology_requires_owner_attestation"
                        ),
                    }
                )
            created_at = self._vault_number(
                hypha.created_at,
                f"live hypha creation timestamp: {key}",
                minimum=0.0,
                maximum=now_monotonic + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            last_pulse = self._vault_number(
                hypha.last_pulse,
                f"live hypha pulse timestamp: {key}",
                minimum=0.0,
                maximum=now_monotonic + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            created_age = max(0.0, now_monotonic - created_at)
            last_pulse_age = max(0.0, now_monotonic - last_pulse)
            if last_pulse_age > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S:
                raise ValueError(f"live hypha pulse predates creation: {key}")
            data["created_age_s"] = created_age
            data["last_pulse_age_s"] = last_pulse_age
            data.pop("created_at", None)
            data.pop("last_pulse", None)
            hyphae[key] = data

        return {
            "schema_version": 3,
            "captured_at_unix": captured_at_unix,
            "pathways": pathways,
            "hyphae": hyphae,
            "mapped_files": self._mapped_files_snapshot_locked(),
            "centrality": dict(self._centrality),
            "critical_modules": list(self._critical_modules),
            "cross_links": {
                key: list(value) for key, value in self._cross_links.items()
            },
            "infrastructure_mapped": self.infrastructure_mapped,
            "mapping_generation": self._mapping_generation,
            "topology_revision": self._topology_revision,
        }

    @staticmethod
    def _vault_age(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite non-negative number")
        age = float(value)
        if not math.isfinite(age) or age < 0.0:
            raise ValueError(f"{label} must be a finite non-negative number")
        return age

    @staticmethod
    def _vault_number(
        value: Any,
        label: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be a finite number")
        if minimum is not None and number < minimum:
            raise ValueError(f"{label} is below its minimum")
        if maximum is not None and number > maximum:
            raise ValueError(f"{label} exceeds its maximum")
        return number

    @staticmethod
    def _vault_count(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    @staticmethod
    def _vault_optional_string(value: Any, label: str) -> str | None:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{label} must be a string or null")
        return value

    @classmethod
    def _restore_pathways(
        cls,
        raw: Any,
        *,
        now_monotonic: float,
        captured_at_unix: float,
        elapsed_since_capture_s: float,
    ) -> dict[str, HardwiredPathway]:
        if not isinstance(raw, dict):
            raise ValueError("vault pathways must be an object")
        allowed = _vault_allowed_fields(
            HardwiredPathway, dropped={"last_matched"}, added={"last_matched_age_s"}
        )
        restored: dict[str, HardwiredPathway] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise ValueError("vault pathway entries must be named objects")
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(
                    f"vault pathway {key} contains unknown fields: "
                    f"{', '.join(sorted(unknown))}"
                )
            fields = {name: item for name, item in value.items() if name in allowed}
            if str(fields.get("pathway_id") or "") != key:
                raise ValueError(f"vault pathway identity mismatch: {key}")
            pattern = fields.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"vault pathway pattern is missing: {key}")
            try:
                # Restored patterns pass the same ReDoS validation as freshly
                # registered ones — a persisted catastrophic regex must not
                # enter the live routing path at restore.
                fields["pattern"] = _compile_route_pattern(pattern)
            except (re.error, ValueError) as exc:
                raise ValueError(f"vault pathway regex is invalid: {key}") from exc
            skill_name = fields.get("skill_name")
            if not isinstance(skill_name, str) or not skill_name:
                raise ValueError(f"vault pathway skill is missing: {key}")
            param_map = fields.get("param_map", {})
            if not isinstance(param_map, dict) or any(
                not isinstance(name, str)
                or not name
                or isinstance(mapping, bool)
                or not isinstance(mapping, (int, str))
                or (isinstance(mapping, int) and mapping < 0)
                for name, mapping in param_map.items()
            ):
                raise ValueError(f"vault pathway parameter map is malformed: {key}")
            dependencies = fields.get("dependencies", [])
            if not isinstance(dependencies, list) or any(
                not isinstance(dependency, str) for dependency in dependencies
            ):
                raise ValueError(f"vault pathway dependencies are malformed: {key}")
            fields["priority"] = cls._vault_number(
                fields.get("priority", 1.0),
                f"vault pathway priority: {key}",
                minimum=0.0,
            )
            fields["confidence"] = cls._vault_number(
                fields.get("confidence", 1.0),
                f"vault pathway confidence: {key}",
                minimum=0.0,
                maximum=10.0,
            )
            fields["size"] = cls._vault_number(
                fields.get("size", 1.0),
                f"vault pathway size: {key}",
                minimum=0.0,
            )
            fields["created_at"] = cls._vault_number(
                fields.get("created_at"),
                f"vault pathway creation timestamp: {key}",
                minimum=0.0,
                maximum=captured_at_unix + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            fields["hit_count"] = cls._vault_count(
                fields.get("hit_count", 0), f"vault pathway hit count: {key}"
            )
            fields["miss_count"] = cls._vault_count(
                fields.get("miss_count", 0), f"vault pathway miss count: {key}"
            )
            # The evidence split is the part of the record that says how much
            # of the confidence was actually checked. A vault that restored
            # the totals but dropped the split would come back looking fully
            # corroborated, which is exactly the claim the counters exist to
            # deny.
            for counter, total in (
                ("verified_hits", "hit_count"),
                ("verified_misses", "miss_count"),
            ):
                fields[counter] = cls._vault_count(
                    fields.get(counter, 0), f"vault pathway {counter}: {key}"
                )
                if fields[counter] > fields[total]:
                    raise ValueError(
                        f"vault pathway {counter} exceeds {total}: {key}"
                    )
            fields["unverified_reinforcements"] = cls._vault_count(
                fields.get("unverified_reinforcements", 0),
                f"vault pathway unverified reinforcements: {key}",
            )
            for field_name in ("activity_label", "color", "description"):
                if not isinstance(fields.get(field_name, ""), str):
                    raise ValueError(
                        f"vault pathway {field_name} is malformed: {key}"
                    )
            fields["source_file"] = cls._vault_optional_string(
                fields.get("source_file"), f"vault pathway source file: {key}"
            )
            fields["direct_response"] = cls._vault_optional_string(
                fields.get("direct_response"),
                f"vault pathway direct response: {key}",
            )
            last_matched_age = cls._vault_age(
                fields.pop("last_matched_age_s", None),
                f"vault pathway last-matched age: {key}",
            )
            created_age = max(0.0, captured_at_unix - fields["created_at"])
            if (
                last_matched_age
                > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S
            ):
                raise ValueError(
                    f"vault pathway last match predates creation: {key}"
                )
            fields["last_matched"] = (
                now_monotonic - last_matched_age - elapsed_since_capture_s
            )
            restored[key] = HardwiredPathway(**fields)
        return restored

    @classmethod
    def _restore_hyphae(
        cls,
        raw: Any,
        *,
        now_monotonic: float,
        elapsed_since_capture_s: float,
    ) -> dict[str, Hypha]:
        if not isinstance(raw, dict):
            raise ValueError("vault hyphae must be an object")
        allowed = _vault_allowed_fields(
            NeuralRoot,  # the widest shape: a plain Hypha is a subset of it
            dropped={"created_at", "last_pulse"},
            added={"created_age_s", "last_pulse_age_s"},
        )
        restored: dict[str, Hypha] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise ValueError("vault hypha entries must be named objects")
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"vault hypha contains unknown fields: {key}")
            fields = {name: item for name, item in value.items() if name in allowed}
            if str(fields.get("name") or "") != key:
                raise ValueError(f"vault hypha identity mismatch: {key}")
            if not isinstance(fields.get("source"), str) or not fields["source"]:
                raise ValueError(f"vault hypha source is missing: {key}")
            if not isinstance(fields.get("target"), str) or not fields["target"]:
                raise ValueError(f"vault hypha target is missing: {key}")
            fields["priority"] = cls._vault_number(
                fields.get("priority", 1.0),
                f"vault hypha priority: {key}",
                minimum=0.0,
            )
            fields["strength"] = cls._vault_number(
                fields.get("strength", 1.0),
                f"vault hypha strength: {key}",
                minimum=0.1,
                maximum=10.0,
            )
            fields["size"] = cls._vault_number(
                fields.get("size", 1.0),
                f"vault hypha size: {key}",
                minimum=0.0,
            )
            fields["pulse_count"] = cls._vault_count(
                fields.get("pulse_count", 0), f"vault hypha pulse count: {key}"
            )
            for field_name in ("active", "is_physical"):
                if not isinstance(fields.get(field_name, False), bool):
                    raise ValueError(f"vault hypha {field_name} is malformed: {key}")
            for field_name in ("source_file", "target_file"):
                fields[field_name] = cls._vault_optional_string(
                    fields.get(field_name), f"vault hypha {field_name}: {key}"
                )
            for field_name in ("color", "description"):
                if not isinstance(fields.get(field_name, ""), str):
                    raise ValueError(f"vault hypha {field_name} is malformed: {key}")
            trace = fields.get("trace", [])
            if not isinstance(trace, list) or any(
                not isinstance(entry, str) for entry in trace
            ):
                raise ValueError(f"vault hypha trace is malformed: {key}")
            created_age = cls._vault_age(
                fields.pop("created_age_s", None),
                f"vault hypha creation age: {key}",
            )
            last_pulse_age = cls._vault_age(
                fields.pop("last_pulse_age_s", None),
                f"vault hypha pulse age: {key}",
            )
            if last_pulse_age > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S:
                raise ValueError(f"vault hypha pulse predates creation: {key}")
            fields["created_at"] = (
                now_monotonic - created_age - elapsed_since_capture_s
            )
            fields["last_pulse"] = (
                now_monotonic - last_pulse_age - elapsed_since_capture_s
            )
            is_neural_root = "hardware_id" in fields or "pinned" in fields
            if is_neural_root:
                hardware_id = fields.get("hardware_id")
                if not isinstance(hardware_id, str) or not hardware_id:
                    raise ValueError(f"vault neural-root hardware id is malformed: {key}")
                if not isinstance(fields.get("pinned"), bool):
                    raise ValueError(f"vault neural-root pinned flag is malformed: {key}")
                root_kind = fields.get("root_kind", "hardware")
                if root_kind not in {"worker", "service", "hardware"}:
                    raise ValueError(f"vault neural-root kind is malformed: {key}")
                if fields["target"] != f"{root_kind}:{hardware_id}":
                    raise ValueError(f"vault neural-root target is malformed: {key}")
                contract = fields.get("liveness_contract", "on_demand")
                if contract not in {"heartbeat", "on_demand"}:
                    raise ValueError(f"vault neural-root contract is malformed: {key}")
                identity = fields.get("attested_identity", {})
                if not isinstance(identity, dict):
                    raise ValueError(f"vault neural-root identity is malformed: {key}")
                identity = cls._canonical_root_evidence(identity)
                for string_field in ("state", "owner_generation", "last_error"):
                    if not isinstance(fields.get(string_field, ""), str):
                        raise ValueError(
                            f"vault neural-root {string_field} is malformed: {key}"
                        )
                stale_after = cls._vault_number(
                    fields.get("stale_after_s", 30.0),
                    f"vault neural-root stale interval: {key}",
                    minimum=0.001,
                )
                for timestamp_name in (
                    "last_activity_at",
                    "last_probe_at",
                    "last_probe_success_at",
                ):
                    cls._vault_number(
                        fields.get(timestamp_name, 0.0),
                        f"vault neural-root {timestamp_name}: {key}",
                        minimum=0.0,
                    )
                # Operational evidence belongs to the process generation that
                # produced it. Preserve identity as history, but require the
                # current owner to re-attest before this root can be live.
                fields["root_kind"] = root_kind
                fields["liveness_contract"] = contract
                fields["state"] = "unbound"
                fields["owner_generation"] = ""
                fields["attested_identity"] = dict(identity)
                fields["last_activity_at"] = 0.0
                fields["last_probe_at"] = 0.0
                fields["last_probe_success_at"] = 0.0
                fields["stale_after_s"] = stale_after
                fields["last_error"] = "restored_historical_topology_requires_owner_attestation"
                fields["active"] = False
            model = NeuralRoot if is_neural_root else Hypha
            restored[key] = model(**fields)
        return restored

    @staticmethod
    def _stage_restored_generation(
        topology: dict[str, Any],
        target: "MycelialNetwork",
        *,
        attested: bool,
    ) -> None:
        """Check a decoded generation against the running system before publishing.

        CP126 2f6e7791. One accepted generation replaced the live routing table
        wholesale — no deployment decision, no compatibility check. Attestation
        (3901c6f3) established that the bytes are the ones Aura wrote; it says
        nothing about whether they still fit the build that is about to run
        them.

        Two staged checks, both about the thing that actually goes wrong:

        1. An unattested generation may SEED a network that has not learned
           anything yet, but may not overwrite one that has. The test is
           earned state — routes with recorded outcomes, or a completed
           infrastructure map — not merely "populated": a freshly booted
           network already carries its default routes, and refusing on those
           would strand every install whose vault predates attestation. Once
           such an install syncs once, its vault is attested and the question
           stops arising.
        2. A route whose skill this build no longer has is a dead route that
           fails at dispatch. Those are dropped here with a named count, rather
           than published and discovered one failed user request at a time.
        """
        pathways = topology.get("pathways") or {}
        with MycelialNetwork._lock:
            earned = sum(
                pathway.hit_count + pathway.miss_count
                for pathway in target.pathways.values()
            )
            mapped = target.infrastructure_mapped
        if not attested and (earned or mapped):
            raise ValueError(
                f"refusing to overwrite learned routing state ({earned} recorded "
                f"outcomes, mapped={mapped}) with an unattested vault generation"
            )

        known_skills = _live_skill_names()
        if known_skills is None:
            # No registry to ask. Publishing unchecked is the historical
            # behaviour and the right one here — an unavailable registry is not
            # evidence that a skill is missing.
            logger.info(
                "🛡️ AEGIS: Skill registry unavailable; restoring %d routes "
                "without a compatibility check.",
                len(pathways),
            )
            return

        stale = [
            key
            for key, pathway in pathways.items()
            if pathway.skill_name not in known_skills
        ]
        for key in stale:
            pathways.pop(key, None)
        if stale:
            logger.warning(
                "🛡️ AEGIS: Dropped %d restored route(s) whose skill this build "
                "does not have: %s",
                len(stale),
                ", ".join(sorted(stale)[:10]),
            )
            record_degradation(
                "mycelium",
                ValueError(
                    f"{len(stale)} restored routes reference skills this build "
                    "does not provide"
                ),
                severity="warning",
                action="dropped the stale routes instead of publishing dead ones",
                enforce_failure_policy=False,
            )

    @staticmethod
    def _verify_vault_attestation(
        encoded: str, attestation_row: Any, *, base_dir: Path
    ) -> bool:
        """Whether this generation carries tamper evidence that checks out.

        Three outcomes, and they are deliberately not two. A MAC that verifies
        is attested. A MAC that does not verify is a refusal — the vault was
        edited after Aura wrote it, and installing routing rules and direct
        responses from it is exactly the attack. No MAC at all is neither: it
        is a vault written before attestation existed, which restores but is
        reported as unattested rather than quietly counted as verified
        (CP126 3901c6f3).
        """
        algorithm = str(attestation_row[0] or "") if attestation_row else ""
        stored_mac = str(attestation_row[1] or "") if attestation_row else ""
        if not stored_mac:
            logger.warning(
                "🛡️ AEGIS: Root vault carries no tamper evidence; restoring an "
                "UNATTESTED topology generation."
            )
            return False
        if algorithm != _VAULT_MAC_ALGORITHM:
            raise ValueError(
                f"vault attestation uses an unsupported algorithm: {algorithm!r}"
            )
        key = _vault_mac_key(base_dir)
        if key is None:
            raise ValueError(
                "vault carries tamper evidence but the key is unavailable; "
                "refusing to restore rather than skipping the check"
            )
        if not hmac.compare_digest(_vault_mac(key, encoded), stored_mac):
            raise ValueError(
                "vault tamper evidence does not match its topology generation"
            )
        return True

    @staticmethod
    def _enforce_vault_cardinality(payload: dict[str, Any]) -> None:
        """Bound the vault before decoding it.

        Serialized size first: the whole payload is already in memory as a
        string by the time we get here, but rejecting it before building tens of
        thousands of pydantic models is the difference between a refused restore
        and a boot that dies. Then per-collection counts, then the two
        unbounded per-element lists (imports and edge traces).
        """
        for field, limit in (
            ("pathways", _VAULT_MAX_PATHWAYS),
            ("hyphae", _VAULT_MAX_HYPHAE),
            ("mapped_files", _VAULT_MAX_MAPPED_FILES),
            ("centrality", _VAULT_MAX_MAPPED_FILES),
            ("cross_links", _VAULT_MAX_MAPPED_FILES),
            ("critical_modules", _VAULT_MAX_MAPPED_FILES),
        ):
            value = payload.get(field)
            if isinstance(value, (dict, list)) and len(value) > limit:
                raise ValueError(
                    f"vault {field} exceeds the restore limit "
                    f"({len(value)} > {limit})"
                )
        hyphae = payload.get("hyphae")
        if isinstance(hyphae, dict):
            for key, value in hyphae.items():
                trace = value.get("trace") if isinstance(value, dict) else None
                if isinstance(trace, list) and len(trace) > _VAULT_MAX_HYPHA_TRACE:
                    raise ValueError(f"vault hypha trace exceeds its limit: {key}")
        mapped_files = payload.get("mapped_files")
        if isinstance(mapped_files, dict):
            for key, value in mapped_files.items():
                imports = value.get("imports") if isinstance(value, dict) else None
                if isinstance(imports, list) and len(imports) > _VAULT_MAX_MODULE_IMPORTS:
                    raise ValueError(f"vault module imports exceed their limit: {key}")

    @classmethod
    def _decode_vault_topology(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("schema_version") != 3:
            raise ValueError("unsupported mycelium vault schema")
        # CP126 944a6043: every field was type-checked and none was counted, so
        # a vault with a million pathways type-validated perfectly and then
        # exhausted memory during boot — the one moment where there is nothing
        # to fall back to. Bounds are checked before any element is built.
        cls._enforce_vault_cardinality(payload)
        if set(payload) != {
            "schema_version",
            "captured_at_unix",
            "pathways",
            "hyphae",
            "mapped_files",
            "centrality",
            "critical_modules",
            "cross_links",
            "infrastructure_mapped",
            "mapping_generation",
            "topology_revision",
        }:
            raise ValueError("mycelium vault fields are invalid")
        captured_at = payload.get("captured_at_unix")
        if (
            isinstance(captured_at, bool)
            or not isinstance(captured_at, (int, float))
            or not math.isfinite(float(captured_at))
            or float(captured_at) <= 0.0
        ):
            raise ValueError("vault capture timestamp is malformed")
        now_monotonic = time.monotonic()
        now_unix = time.time()
        if float(captured_at) > now_unix + _VAULT_CLOCK_SKEW_TOLERANCE_S:
            raise ValueError("vault capture timestamp is implausibly in the future")
        elapsed_since_capture_s = max(0.0, now_unix - float(captured_at))
        pathways = cls._restore_pathways(
            payload.get("pathways"),
            now_monotonic=now_monotonic,
            captured_at_unix=float(captured_at),
            elapsed_since_capture_s=elapsed_since_capture_s,
        )
        hyphae = cls._restore_hyphae(
            payload.get("hyphae"),
            now_monotonic=now_monotonic,
            elapsed_since_capture_s=elapsed_since_capture_s,
        )
        raw_mapped_files = payload.get("mapped_files")
        centrality = payload.get("centrality")
        critical_modules = payload.get("critical_modules")
        cross_links = payload.get("cross_links")
        if not isinstance(raw_mapped_files, dict):
            raise ValueError("vault mapped-files surface is malformed")
        mapped_files: dict[str, dict[str, Any]] = {}
        mapped_paths: set[str] = set()
        for key, value in raw_mapped_files.items():
            if not isinstance(key, str) or not key or not isinstance(value, dict):
                raise ValueError("vault mapped-files surface is malformed")
            path = value.get("path")
            imports = value.get("imports")
            size_bytes = value.get("size_bytes")
            module_centrality = value.get("centrality")
            is_critical = value.get("is_critical")
            source_sha256 = value.get("source_sha256")
            if set(value) != {
                "path",
                "size_bytes",
                "imports",
                "centrality",
                "is_critical",
                "source_sha256",
            }:
                raise ValueError(f"vault mapped-file fields are invalid: {key}")
            # CP126 5894e929: the digest ties the restored topology to the
            # source generation it was derived from. None is the honest value
            # for a module that could not be read; a malformed one is not.
            if source_sha256 is not None and (
                not isinstance(source_sha256, str)
                or len(source_sha256) != 64
                or not all(c in "0123456789abcdef" for c in source_sha256)
            ):
                raise ValueError(f"vault mapped-file digest is malformed: {key}")
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise ValueError(f"vault mapped-file path is malformed: {key}")
            if path in mapped_paths:
                raise ValueError(f"vault mapped-file path is duplicated: {key}")
            if not isinstance(imports, list) or any(
                not isinstance(dependency, str) for dependency in imports
            ):
                raise ValueError(f"vault mapped-file imports are malformed: {key}")
            if (
                isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 0
            ):
                raise ValueError(f"vault mapped-file size is malformed: {key}")
            if (
                isinstance(module_centrality, bool)
                or not isinstance(module_centrality, int)
                or module_centrality < 0
            ):
                raise ValueError(f"vault mapped-file centrality is malformed: {key}")
            if not isinstance(is_critical, bool):
                raise ValueError(f"vault mapped-file critical flag is malformed: {key}")
            mapped_paths.add(path)
            mapped_files[key] = {
                "path": path,
                "size_bytes": size_bytes,
                "imports": list(imports),
                "centrality": module_centrality,
                "is_critical": is_critical,
                "source_sha256": source_sha256,
            }
        if not isinstance(centrality, dict) or any(
            key not in mapped_files
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in centrality.items()
        ):
            raise ValueError("vault centrality surface is malformed")
        if not isinstance(critical_modules, list) or any(
            not isinstance(module, str) or module not in mapped_files
            for module in critical_modules
        ):
            raise ValueError("vault critical-module surface is malformed")
        if len(set(critical_modules)) != len(critical_modules):
            raise ValueError("vault critical-module surface contains duplicates")
        computed_centrality: dict[str, int] = {}
        for module in mapped_files.values():
            for dependency in module["imports"]:
                if dependency in mapped_files:
                    computed_centrality[dependency] = (
                        computed_centrality.get(dependency, 0) + 1
                    )
        if any(
            module["centrality"] != computed_centrality.get(key, 0)
            for key, module in mapped_files.items()
        ):
            raise ValueError("vault module centrality disagrees with its import graph")
        expected_centrality = dict(computed_centrality)
        if centrality != expected_centrality:
            raise ValueError("vault centrality disagrees with the module map")
        expected_critical = {
            key for key, value in mapped_files.items() if value["is_critical"]
        }
        if set(critical_modules) != expected_critical:
            raise ValueError("vault critical modules disagree with the module map")
        ranked_centralities = sorted(computed_centrality.values(), reverse=True)
        expected_critical_count = min(20, len(ranked_centralities))
        if len(critical_modules) != expected_critical_count:
            raise ValueError("vault critical modules disagree with centrality ranking")
        if ranked_centralities:
            cutoff = ranked_centralities[expected_critical_count - 1]
            if any(
                computed_centrality.get(module, 0) < cutoff
                for module in critical_modules
            ) or any(
                value > cutoff and module not in expected_critical
                for module, value in computed_centrality.items()
            ):
                raise ValueError("vault critical modules disagree with centrality ranking")
        if not isinstance(cross_links, dict):
            raise ValueError("vault cross-link surface is malformed")
        for logical_name, physical_names in cross_links.items():
            if (
                not isinstance(logical_name, str)
                or logical_name not in hyphae
                or hyphae[logical_name].is_physical
                or not isinstance(physical_names, list)
                or any(not isinstance(name, str) for name in physical_names)
                or len(set(physical_names)) != len(physical_names)
            ):
                raise ValueError("vault cross-link owner is malformed")
            if any(
                name not in hyphae or not hyphae[name].is_physical
                for name in physical_names
            ):
                raise ValueError("vault cross-link target is malformed")
        infrastructure_mapped = payload.get("infrastructure_mapped")
        if not isinstance(infrastructure_mapped, bool):
            raise ValueError("vault infrastructure state is malformed")
        physical_hyphae = [hypha for hypha in hyphae.values() if hypha.is_physical]
        if infrastructure_mapped and not mapped_files:
            raise ValueError("mapped vault contains no modules")
        if infrastructure_mapped and any(
            hypha.source not in mapped_files or hypha.target not in mapped_files
            for hypha in physical_hyphae
        ):
            raise ValueError("vault physical topology is not backed by its module map")
        if not infrastructure_mapped and (mapped_files or physical_hyphae):
            raise ValueError("unmapped vault contains published physical topology")
        expected_physical_names = {
            f"import:{source}->{target}"
            for source, module in mapped_files.items()
            for target in module["imports"]
            if target in mapped_files
        }
        actual_physical_names = {
            name for name, hypha in hyphae.items() if hypha.is_physical
        }
        if actual_physical_names != expected_physical_names:
            raise ValueError("vault physical topology disagrees with the module map")
        for name, hypha in hyphae.items():
            expected_name = (
                f"import:{hypha.source}->{hypha.target}"
                if hypha.is_physical
                else f"{hypha.source}->{hypha.target}"
            )
            if name != expected_name:
                raise ValueError(f"vault hypha identity is inconsistent: {name}")
            if hypha.is_physical and (
                hypha.source_file != mapped_files[hypha.source]["path"]
                or hypha.target_file != mapped_files[hypha.target]["path"]
            ):
                raise ValueError(f"vault physical hypha files are inconsistent: {name}")
        for pathway in pathways.values():
            if pathway.source_file is not None and pathway.source_file not in mapped_paths:
                raise ValueError(
                    f"vault pathway source is outside the module map: {pathway.pathway_id}"
                )
            if pathway.source_file is not None:
                module = next(
                    value
                    for value in mapped_files.values()
                    if value["path"] == pathway.source_file
                )
                if pathway.dependencies != module["imports"]:
                    raise ValueError(
                        "vault pathway dependencies disagree with its source module: "
                        f"{pathway.pathway_id}"
                    )
        mapping_generation = payload.get("mapping_generation")
        if (
            isinstance(mapping_generation, bool)
            or not isinstance(mapping_generation, int)
            or mapping_generation < 0
        ):
            raise ValueError("vault mapping generation is negative")
        topology_revision = payload.get("topology_revision")
        if (
            isinstance(topology_revision, bool)
            or not isinstance(topology_revision, int)
            or topology_revision < 0
        ):
            raise ValueError("vault topology revision is malformed")
        return {
            "pathways": pathways,
            "hyphae": hyphae,
            "mapped_files": mapped_files,
            "centrality": dict(centrality),
            "critical_modules": list(critical_modules),
            "cross_links": {
                key: list(value) for key, value in cross_links.items()
            },
            "infrastructure_mapped": infrastructure_mapped,
            "mapping_generation": mapping_generation,
            "topology_revision": topology_revision,
        }

    async def vault_sync(self) -> bool:
        """Persist one complete, versioned topology generation."""
        import json
        import sqlite3

        from core.config import config

        aegis_cfg = getattr(config, "aegis", None)
        vault_path = (
            getattr(aegis_cfg, "vault_path", None)
            or "data/mycelium_vault.db"
        )
        db_path = config.paths.base_dir / vault_path

        def _sync_worker() -> tuple[int, int]:
            with MycelialNetwork._vault_io_lock:
                with MycelialNetwork._lock:
                    if (
                        MycelialNetwork._instance is not self
                        or self._stop_event.is_set()
                    ):
                        raise RuntimeError(
                            "retired mycelium instance cannot write the root vault"
                        )
                    topology = self._vault_snapshot_locked()
                    snapshot_revision = int(topology["topology_revision"])
                # Validate our own serialized contract before replacing the last
                # known-good generation.
                self._decode_vault_topology(topology)
                encoded = json.dumps(topology, allow_nan=False, sort_keys=True)
                key = _vault_mac_key(config.paths.base_dir)
                mac = _vault_mac(key, encoded) if key else ""
                db_path.parent.mkdir(parents=True, exist_ok=True)
                with connecting(sqlite3.connect(db_path)) as conn:
                    conn.execute("PRAGMA busy_timeout=5000;")
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=FULL;")
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS aegis_vault "
                        "(key TEXT PRIMARY KEY, data TEXT, timestamp REAL)"
                    )
                    # The MAC lives in its own row keyed to the generation it
                    # covers, so an older schema reads the topology unchanged and
                    # a restore can tell "no evidence was written" apart from
                    # "the evidence does not match".
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS aegis_vault_attestation "
                        "(key TEXT PRIMARY KEY, algorithm TEXT, mac TEXT, "
                        "sha256 TEXT, timestamp REAL)"
                    )
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "REPLACE INTO aegis_vault (key, data, timestamp) "
                        "VALUES (?, ?, ?)",
                        ("topology_v3", encoded, time.time()),
                    )
                    conn.execute(
                        "REPLACE INTO aegis_vault_attestation "
                        "(key, algorithm, mac, sha256, timestamp) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            "topology_v3",
                            _VAULT_MAC_ALGORITHM if mac else "",
                            mac,
                            hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                            time.time(),
                        ),
                    )
                    with MycelialNetwork._lock:
                        if (
                            MycelialNetwork._instance is not self
                            or self._stop_event.is_set()
                        ):
                            raise RuntimeError(
                                "mycelium retired before root-vault commit"
                            )
                        current_revision = self._topology_revision
                        conn.commit()
                        self._last_vault_sync_revision = snapshot_revision
                        self._last_vault_sync_at = time.time()
                        self._last_vault_sync_lag_revisions = max(
                            0,
                            current_revision - snapshot_revision,
                        )
                        return snapshot_revision, current_revision

        try:
            snapshot_revision, current_revision = await run_io_bound(_sync_worker)
        except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mycelium", exc)
            logger.error("🛡️ AEGIS: Vault Sync Failed! %s", exc)
            return False
        if current_revision != snapshot_revision:
            logger.debug(
                "🛡️ AEGIS: Vault committed coherent revision %d while live topology "
                "advanced to %d; the next interval will capture the newer state.",
                snapshot_revision,
                current_revision,
            )
        logger.debug("🛡️ AEGIS: Vault Sync Complete.")
        return True

    @classmethod
    async def restore_from_vault(cls) -> bool:
        """Validate and atomically publish one persisted topology generation."""
        import json
        import sqlite3

        from core.config import config

        aegis_cfg = getattr(config, "aegis", None)
        vault_path = (
            getattr(aegis_cfg, "vault_path", None)
            or "data/mycelium_vault.db"
        )
        db_path = config.paths.base_dir / vault_path
        if not db_path.exists():
            logger.critical("🛡️ AEGIS FATAL: Cannot restore; Root Vault missing!")
            return False

        with cls._lock:
            target_instance = cls._instance
            if target_instance is None:
                logger.critical(
                    "🛡️ AEGIS: Restoration aborted — Mycelium is not initialized."
                )
                return False
            mapping_thread = target_instance._mapping_thread
            if target_instance._is_mapping or (
                mapping_thread is not None and mapping_thread.is_alive()
            ):
                logger.critical(
                    "🛡️ AEGIS: Restoration deferred while a map generation is active."
                )
                return False
            if target_instance._stop_event.is_set():
                logger.critical("🛡️ AEGIS: Restoration refused during shutdown.")
                return False
            target_revision = target_instance._topology_revision

        def _restore_worker() -> None:
            # Vault and topology publication use the same lock order as sync:
            # vault first, then topology. The vault lock remains held until the
            # decoded generation is either rejected or fully published.
            with cls._vault_io_lock:
                with connecting(sqlite3.connect(db_path)) as conn:
                    row = conn.execute(
                        "SELECT data FROM aegis_vault WHERE key = ?",
                        ("topology_v3",),
                    ).fetchone()
                    try:
                        attestation_row = conn.execute(
                            "SELECT algorithm, mac FROM aegis_vault_attestation "
                            "WHERE key = ?",
                            ("topology_v3",),
                        ).fetchone()
                    except sqlite3.Error:
                        # Vault written before attestation existed.
                        attestation_row = None
                if not row:
                    raise ValueError("versioned topology generation is missing")
                encoded = row[0]
                if len(encoded) > _VAULT_MAX_ENCODED_BYTES:
                    raise ValueError(
                        "vault generation exceeds the restore size limit "
                        f"({len(encoded)} > {_VAULT_MAX_ENCODED_BYTES} bytes)"
                    )
                attested = cls._verify_vault_attestation(
                    encoded, attestation_row, base_dir=config.paths.base_dir
                )
                topology = cls._decode_vault_topology(json.loads(encoded))
                cls._stage_restored_generation(
                    topology, target_instance, attested=attested
                )

                with cls._lock:
                    instance = cls._instance
                    if instance is not target_instance:
                        raise RuntimeError("vault restoration target was replaced")
                    mapping_thread = instance._mapping_thread
                    if instance._is_mapping or (
                        mapping_thread is not None and mapping_thread.is_alive()
                    ):
                        raise RuntimeError(
                            "vault restoration raced an active map generation"
                        )
                    if instance._stop_event.is_set():
                        raise RuntimeError("vault restoration raced shutdown")
                    if instance._topology_revision != target_revision:
                        raise RuntimeError(
                            "vault restoration raced a newer in-memory topology revision"
                        )
                    instance._aegis_replace("pathways", topology["pathways"])
                    instance._aegis_replace("hyphae", topology["hyphae"])
                    instance._aegis_replace(
                        "_pathway_order",
                        sorted(
                            topology["pathways"],
                            key=lambda key: topology["pathways"][key].priority,
                            reverse=True,
                        ),
                    )
                    instance.direct_roots = {
                        key: pathway.skill_name
                        for key, pathway in topology["pathways"].items()
                    }
                    instance._neural_roots = [
                        hypha
                        for hypha in topology["hyphae"].values()
                        if isinstance(hypha, NeuralRoot)
                    ]
                    instance.mapped_files = topology["mapped_files"]
                    instance._centrality = topology["centrality"]
                    instance._critical_modules = topology["critical_modules"]
                    instance._cross_links = topology["cross_links"]
                    instance.infrastructure_mapped = topology["infrastructure_mapped"]
                    instance._mapping_generation = max(
                        instance._mapping_generation + 1,
                        topology["mapping_generation"],
                    )
                    instance._topology_revision = max(
                        instance._topology_revision + 1,
                        topology["topology_revision"] + 1,
                    )
                    instance._topology_structure_revision += 1
                    instance._mapping_completed_at = time.time()
                    instance._mapping_last_error = None
                    instance._deferred_mapping_reason = None
                    instance._restored_from_vault_at = time.time()
                    instance._restored_generation_attested = attested
                    instance._publish_topology_read_models_locked()
                    object.__setattr__(instance, "_aegis_locked", True)

        logger.critical("🛡️ AEGIS: Initiating Emergency Vault Restoration...")
        try:
            await run_io_bound(_restore_worker)
        except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("mycelium", exc)
            logger.critical("🛡️ AEGIS FATAL: Restoration Failed! %s", exc)
            return False
        logger.critical("🛡️ AEGIS: Restoration Successful. Mycelium Unity restored.")
        return True
