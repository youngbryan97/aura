"""What each cognitive phase is allowed to do, declared once and then checked.

Aura has never been short of explicit criteria. Initiative generation carries a
60-second impulse throttle, a 900-second idle requirement, a memory ceiling, a
failure-pressure ceiling, conversation-depth suppression, a consecutive-message
hard stop and a φ-scaled curiosity threshold. Executive closure carries
priority floors, need-pressure floors and commitment hysteresis. The arbiter
scores eight named dimensions with declared weights and emits the whole vector
with a rationale. None of that is an LLM deciding by feel.

What was missing is that those criteria are **executable but not declarable**.
``PhaseSpec`` carried a name, an attribute and a class. Nothing said which
fields a phase reads, which it may write, what its branches are, what
authority it needs, or what must still be true afterwards. So the criteria
existed in nine different shapes across the codebase and could be read only by
reading the code — which is what "cognition can't account for itself" actually
meant, and it is worth doing something about.

A :class:`CognitiveTransformContract` is that declaration:

    C = (reads, writes, preconditions, branches, thresholds,
         defaults, authority, side_effects, invariants)

versioned, hashed, and immutable. Every execution emits a
:class:`TransformationReceipt`:

    τ = (contract_hash, state_before_hash, inputs, branch, criteria,
         state_after_hash, observed_writes)

**The contract is checked, not merely published.** That distinction is the
whole design. A declaration nothing verifies is documentation with a type
annotation, and this codebase has been bitten repeatedly by exactly that shape
— a claim that outlived the code making it true. So:

* ``observed_writes`` is measured by digesting the declared fields before and
  after the phase runs, not reported by the phase.
* Any changed field the contract did not declare lands in
  ``undeclared_writes`` and is recorded as a degradation.
* ``thresholds`` holds the live constants by reference. A contract that copies
  a number is a second source of truth, and the copy is the one that goes
  stale, so the gate in tests/test_cognitive_contracts.py requires the phase
  module to export the constant and requires contracted phases marked
  ``thresholds_exhaustive`` to have no bare numeric comparison left in
  ``execute``.

**Scope, stated exactly.** Enforcement is bounded by the union of every
declared read and write across all contracts. That bound buys the case that
matters — cross-phase interference, where one phase writes a field another
phase declared it reads — without digesting all of AuraState on the
foreground path.

The bound had one unintended consequence, and it was the thing blocking the
other twenty-eight phases. ``watched_fields()`` is derived FROM the
contracts, so an uncontracted phase could only ever be seen touching fields
some already-written contract happened to name. With a single contract
registered that was eleven paths, and ``write_profile`` — documented as "the
productive end of the ratchet", the tool whose whole purpose is to ground the
NEXT contract in measurement — reported almost nothing for the phases it
exists to describe. The method was right and structurally could not run.

:func:`discovery_paths` closes that loop. A phase with no contract yet is
observed against the enumerated state surface rather than against other
phases' declarations, so its real write set can be measured and its contract
written from what it did. The wider set is used only for uncontracted
phases; once a contract exists, enforcement narrows back to the declared
fields where the cost is justified.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "BranchSpec",
    "CognitiveTransformContract",
    "TransformationReceipt",
    "all_contracts",
    "contract_for",
    "observed_field_digest",
    "register_contract",
    "watched_fields",
    "contract_coverage_report",
    "discovery_paths",
    "reset_discovery_paths_for_test",
    "UNCONTRACTED_PHASES",
]

logger = logging.getLogger(__name__)

#: Contract declarations are a schema. Bumping this is a deliberate act that
#: invalidates every stored contract hash, so historic receipts stay readable
#: as belonging to an older shape rather than silently comparing equal.
CONTRACT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class BranchSpec:
    """One path a phase can take, and what decides it.

    ``criterion`` is prose describing the test; ``outcome`` is what the branch
    does. Both are declarative — the receipt records which branch was actually
    taken, so the pair can be compared against behaviour rather than trusted.
    """

    name: str
    criterion: str
    outcome: str


@dataclass(frozen=True, slots=True)
class CognitiveTransformContract:
    """The declared semantics of one cognitive transformation."""

    #: Phase class name. The key everything else joins on.
    name: str
    #: Bumped by the author when the phase's semantics change.
    version: str
    #: One line: what this transformation is for.
    purpose: str
    #: Dotted AuraState paths this phase consults.
    reads: tuple[str, ...] = field(default_factory=tuple)
    #: Dotted AuraState paths this phase may modify. Anything else it changes
    #: is a contract violation, not a feature.
    writes: tuple[str, ...] = field(default_factory=tuple)
    #: What must hold before the phase does anything.
    preconditions: tuple[str, ...] = field(default_factory=tuple)
    #: Every path through the phase, with what selects it.
    branches: tuple[BranchSpec, ...] = field(default_factory=tuple)
    #: Named constants that decide the branches, BY REFERENCE to the live
    #: module constant. Never a copied literal.
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    #: What the phase falls back to when an input is missing.
    defaults: Mapping[str, Any] = field(default_factory=dict)
    #: Which authority admits this phase's output, or "" when none is needed.
    authority: str = ""
    #: Effects outside AuraState — a queued message, a file, a tool call.
    side_effects: tuple[str, ...] = field(default_factory=tuple)
    #: What must hold after the phase runs.
    invariants: tuple[str, ...] = field(default_factory=tuple)
    #: Where the thresholds came from: a measurement, a paper, or a judgement
    #: call. "judgement" is an honest answer; silence is not.
    calibration_source: str = "judgement"
    #: True when ``thresholds`` names EVERY constant the phase branches on. The
    #: gate then requires no bare numeric comparison in ``execute``. False is
    #: allowed and must say why in ``calibration_source``; the count of False
    #: contracts is ratcheted downward by the test suite.
    thresholds_exhaustive: bool = False
    #: The module the phase lives in, so the gate can find its constants.
    module: str = ""

    @property
    def content_hash(self) -> str:
        """A stable digest of the declaration itself.

        Receipts carry it so a stored receipt can be told apart from one
        emitted under a contract that has since been edited. Threshold VALUES
        are included: changing a number changes the semantics, and a receipt
        that claimed otherwise would be the false-provenance failure this
        module exists to prevent.
        """

        payload = "|".join(
            (
                CONTRACT_SCHEMA_VERSION,
                self.name,
                self.version,
                ",".join(sorted(self.reads)),
                ",".join(sorted(self.writes)),
                ",".join(sorted(self.preconditions)),
                ",".join(sorted(f"{b.name}:{b.criterion}:{b.outcome}" for b in self.branches)),
                ",".join(f"{key}={self.thresholds[key]!r}" for key in sorted(self.thresholds)),
                ",".join(f"{key}={self.defaults[key]!r}" for key in sorted(self.defaults)),
                self.authority,
                ",".join(sorted(self.side_effects)),
                ",".join(sorted(self.invariants)),
            )
        )
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "contract_hash": self.content_hash,
            "purpose": self.purpose,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "preconditions": list(self.preconditions),
            "branches": [
                {"name": b.name, "criterion": b.criterion, "outcome": b.outcome}
                for b in self.branches
            ],
            "thresholds": {key: self.thresholds[key] for key in sorted(self.thresholds)},
            "defaults": {key: self.defaults[key] for key in sorted(self.defaults)},
            "authority": self.authority,
            "side_effects": list(self.side_effects),
            "invariants": list(self.invariants),
            "calibration_source": self.calibration_source,
            "thresholds_exhaustive": self.thresholds_exhaustive,
            "module": self.module,
        }


@dataclass(frozen=True, slots=True)
class TransformationReceipt:
    """One execution of one contract, recorded from outside the phase.

    Nothing here is self-reported except ``branch`` and ``criteria``, which
    only the phase can know. The hashes and ``observed_writes`` are measured by
    the caller, so a phase cannot record having stayed inside its contract.
    """

    transform: str
    contract_hash: str
    contract_version: str
    state_before_hash: str
    state_after_hash: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    branch: str = ""
    criteria: Mapping[str, Any] = field(default_factory=dict)
    declared_writes: tuple[str, ...] = field(default_factory=tuple)
    observed_writes: tuple[str, ...] = field(default_factory=tuple)
    undeclared_writes: tuple[str, ...] = field(default_factory=tuple)
    duration_ms: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    at: float = field(default_factory=time.time)

    @property
    def honoured_contract(self) -> bool:
        """Whether this execution stayed inside what was declared."""
        return not self.undeclared_writes and not self.error

    @property
    def changed_state(self) -> bool:
        return self.state_before_hash != self.state_after_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform,
            "contract_hash": self.contract_hash,
            "contract_version": self.contract_version,
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after_hash,
            "inputs": dict(self.inputs),
            "branch": self.branch,
            "criteria": dict(self.criteria),
            "declared_writes": list(self.declared_writes),
            "observed_writes": list(self.observed_writes),
            "undeclared_writes": list(self.undeclared_writes),
            "duration_ms": round(float(self.duration_ms), 3),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "honoured_contract": self.honoured_contract,
            "at": self.at,
        }


#: Phases that do not yet declare a contract. **This set only shrinks.**
#:
#: Written as a baseline rather than as an empty target because the honest
#: alternative was worse. Declaring 28 contracts in one pass means reading 28
#: phases plus the engines several of them delegate into, and writing down what
#: the author believed each one touches — which is precisely the shape of
#: defect this repository keeps finding: a declaration that outlived the code
#: making it true. A contract nobody verified is not better than no contract;
#: it is a false claim with a dataclass around it.
#:
#: So the mechanism ships with one exhaustively verified exemplar, and
#: ``core.runtime.cognitive_provenance.write_profile()`` measures what every
#: other phase actually writes while the system runs. Contracts come off this
#: list backed by observation. ``tests/test_cognitive_contracts.py`` fails if
#: the list grows or if a name on it is not a real phase.
UNCONTRACTED_PHASES: frozenset[str] = frozenset(
    {
        "BondingPhase",
        "CognitiveRoutingPhase",
        "ConversationalDynamicsPhase",
        "EternalGrowthEngine",
        "EternalMemoryPhase",
        "ExecutiveClosurePhase",
        "GodModeToolPhase",
        "InferencePhase",
        "LearningPhase",
        "MemoryRetrievalPhase",
        "NativeMultimodalBridge",
        "PerfectEmotionPhase",
        "RepairPhase",
        "ResponseGenerationPhase",
        "SelfReviewPhase",
        "SensoryIngestionPhase",
        "ShadowExecutionPhase",
        "TrueEvolutionPhase",
    }
)


_CONTRACTS: dict[str, CognitiveTransformContract] = {}
_WATCHED: tuple[str, ...] | None = None


def register_contract(contract: CognitiveTransformContract) -> CognitiveTransformContract:
    """Register one phase's contract. Re-registration of a DIFFERENT contract
    under the same name is refused: two declarations for one phase is the
    ambiguity the registry exists to remove.
    """

    global _WATCHED
    existing = _CONTRACTS.get(contract.name)
    if existing is not None and existing.content_hash != contract.content_hash:
        raise ValueError(
            f"two different contracts registered for {contract.name!r}: "
            f"{existing.content_hash} and {contract.content_hash}"
        )
    _CONTRACTS[contract.name] = contract
    _WATCHED = None
    return contract


def contract_for(phase_name: Any) -> CognitiveTransformContract | None:
    return _CONTRACTS.get(str(phase_name or "").strip())


def all_contracts() -> dict[str, CognitiveTransformContract]:
    """Every registered contract, with the registry guaranteed complete.

    Contracts register at import time, so a caller that had not imported a
    phase module saw a partial registry and concluded the phase was
    uncontracted. That made the answer depend on import order — the same
    result reading differently depending on who asked, which is worse than
    no answer because it looks like a fact.
    """
    _ensure_registry_loaded()
    return dict(_CONTRACTS)


_LOADING = False


def _ensure_registry_loaded() -> None:
    """Import the contract-declaring modules once, re-entrantly safe.

    The imports trigger `register_contract`, which must not recurse back
    into this.
    """
    global _LOADING
    if _LOADING:
        return
    _LOADING = True
    try:
        from core.runtime.phase_contract_registry import ensure_contracts_loaded

        ensure_contracts_loaded()
    except ImportError:  # pragma: no cover - registry is part of the package
        pass


def watched_fields() -> tuple[str, ...]:
    """Every state path any contract declares an interest in.

    The observation set. Bounded on purpose — see the module docstring — and
    derived rather than listed, so a contract that declares a new field starts
    being watched without anything else changing.
    """

    global _WATCHED
    if _WATCHED is None:
        paths: set[str] = set()
        for contract in _CONTRACTS.values():
            paths.update(contract.reads)
            paths.update(contract.writes)
        _WATCHED = tuple(sorted(paths))
    return _WATCHED


def contract_coverage_report() -> dict[str, Any]:
    """How much of the pipeline declares itself, from the code that enforces it.

    Exists so a statement about how self-describing Aura's cognition is can be
    checked instead of believed — the same reason ``pipeline_rate_report``
    exists next door, and it was written after "29 phases per turn" turned out
    to be about eleven.
    """

    from core.runtime.phase_contract_registry import ensure_contracts_loaded
    from core.runtime.pipeline_blueprint import kernel_phase_attribute_order, phase_class_for_attribute

    # Contracts register at import time, so the count used to depend on who
    # had imported what. Two phases declared contracts that only appeared if
    # something happened to pull their module in, which made this report
    # understate coverage depending on the caller.
    unloadable = ensure_contracts_loaded()

    pipeline = tuple(
        phase_class_for_attribute(attribute) for attribute in kernel_phase_attribute_order()
    )
    contracted = sorted(name for name in pipeline if name in _CONTRACTS)
    uncontracted = sorted(name for name in pipeline if name not in _CONTRACTS)
    exhaustive = sorted(
        name for name, contract in _CONTRACTS.items() if contract.thresholds_exhaustive
    )
    return {
        "pipeline_phases": len(pipeline),
        "contracted": contracted,
        "uncontracted": uncontracted,
        "contracted_count": len(contracted),
        "coverage_fraction": round(len(contracted) / max(1, len(pipeline)), 4),
        "thresholds_exhaustive": exhaustive,
        "watched_fields": list(watched_fields()),
        "baseline_size": len(UNCONTRACTED_PHASES),
        "unloadable_contract_modules": list(unloadable),
        "note": (
            "UNCONTRACTED_PHASES only shrinks. Contracts are written from "
            "cognitive_provenance.write_profile() observations rather than "
            "from reading a phase and guessing what it touches."
        ),
    }


def _resolve(state: Any, path: str) -> Any:
    node: Any = state
    for part in path.split("."):
        if node is None:
            return None
        if isinstance(node, Mapping):
            node = node.get(part)
            continue
        node = getattr(node, part, None)
    return node


def _digest_value(value: Any) -> str:
    try:
        rendered = repr(value)
    except Exception:  # noqa: BLE001 — a __repr__ that raises must not kill a tick
        rendered = f"<unrepresentable {type(value).__name__}>"
    # Bounded: a working-memory list can be large, and this runs per phase.
    return hashlib.sha1(rendered[:4096].encode("utf-8", errors="ignore")).hexdigest()[:12]


#: Top-level state attributes worth descending into when discovering what an
#: UNCONTRACTED phase touches. Bounded deliberately: this is the bootstrap
#: surface, not a full object-graph walk on the foreground path.
_DISCOVERY_ROOTS: tuple[str, ...] = (
    "affect",
    "cognition",
    "identity",
    "motivation",
    "soma",
    "world",
    "health",
    "vitality",
    "response_modifiers",
)

#: Scalars that live directly on the state and are worth watching whole.
_DISCOVERY_SCALARS: tuple[str, ...] = (
    "phi",
    "phi_estimate",
    "free_energy",
    "loop_cycle",
    "transition_cause",
)

#: A root with more children than this is summarised by its own digest rather
#: than enumerated, so one dict-shaped field cannot dominate every receipt.
_DISCOVERY_FANOUT_LIMIT = 48

_DISCOVERY_PATHS: tuple[str, ...] | None = None


def discovery_paths(state: Any) -> tuple[str, ...]:
    """The state surface to watch for a phase that declares nothing yet.

    ``watched_fields()`` is derived FROM the contracts, which made the
    bootstrap circular: an uncontracted phase's writes were only visible if
    some already-written contract happened to name the same field. With one
    contract registered, that was eleven paths — so ``write_profile`` reported
    almost nothing for the twenty-eight phases it exists to describe, and the
    "productive end of the ratchet" could not produce.

    Enumerating the real state surface is what closes that loop: the phase is
    observed against the fields it could plausibly touch, and the contract is
    then written from what it DID rather than from reading the code and
    hoping. That was always the stated method; this is what makes it possible.
    """

    global _DISCOVERY_PATHS
    if _DISCOVERY_PATHS is not None:
        return _DISCOVERY_PATHS
    if state is None:
        return ()

    paths: list[str] = [*_DISCOVERY_SCALARS]
    for root in _DISCOVERY_ROOTS:
        node = getattr(state, root, None)
        if node is None:
            continue
        children = _child_names(node)
        if not children or len(children) > _DISCOVERY_FANOUT_LIMIT:
            # Watch the container itself. A change anywhere inside still
            # registers; it simply attributes to the root, which is honest
            # about the resolution actually being offered.
            paths.append(root)
            continue
        paths.extend(f"{root}.{child}" for child in children)
    _DISCOVERY_PATHS = tuple(sorted(set(paths)))
    return _DISCOVERY_PATHS


def _child_names(node: Any) -> tuple[str, ...]:
    if isinstance(node, Mapping):
        return tuple(sorted(str(key) for key in node))
    if dataclasses.is_dataclass(node):
        return tuple(sorted(f.name for f in dataclasses.fields(node)))
    slots = getattr(type(node), "__slots__", None)
    if slots:
        return tuple(sorted(str(name) for name in slots))
    return ()


def reset_discovery_paths_for_test() -> None:
    global _DISCOVERY_PATHS
    _DISCOVERY_PATHS = None


def observed_field_digest(
    state: Any,
    paths: tuple[str, ...] | None = None,
    *,
    discover: bool = False,
) -> dict[str, str]:
    """Digest the watched fields of a state, for before/after comparison.

    Values are hashed rather than kept. The point is to detect that a field
    changed, and retaining working memory or a full affect vector per phase per
    tick would cost more than the answer is worth.

    ``discover`` widens the set to the enumerated state surface, for phases
    that have no contract yet. Without it those phases report no writes at all
    and no contract can be grounded in measurement.
    """

    if state is None:
        return {}
    if paths is not None:
        targets: tuple[str, ...] = paths
    elif discover:
        targets = tuple(sorted(set(watched_fields()) | set(discovery_paths(state))))
    else:
        targets = watched_fields()
    digest: dict[str, str] = {}
    for path in targets:
        digest[path] = _digest_value(_resolve(state, path))
    return digest


def diff_digests(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[str, ...]:
    """Paths whose value changed between the two digests."""

    changed = [
        path
        for path, value in after.items()
        if before.get(path, "\x00missing") != value
    ]
    return tuple(sorted(changed))


def state_fingerprint(state: Any) -> str:
    """A cheap whole-state marker, for "did this phase change anything at all".

    Deliberately not a full serialization: ``version`` plus the identity of the
    state object plus its update stamp answers the question a receipt asks
    without walking the object graph on the foreground path.
    """

    if state is None:
        return ""
    parts = (
        str(getattr(state, "state_id", "")),
        str(getattr(state, "version", "")),
        str(getattr(state, "updated_at", "")),
    )
    return hashlib.sha1("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:16]
