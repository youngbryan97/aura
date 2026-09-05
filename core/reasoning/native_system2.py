"""Native System 2 search for Aura.

This module is Aura's deliberate cognition substrate. It is intentionally not
an LLM prompt wrapper: it maintains an explicit search tree of latent plans,
scores nodes through a value interface, simulates outcomes through a world
model interface, backpropagates search evidence, and emits an auditable
commitment receipt before any selected plan can be handed to the rest of Aura.

The engine never executes external side effects during search. Simulation is
side-effect free by contract; actual tools/actions remain governed elsewhere by
UnifiedWill and the AuthorityGateway.
"""
from __future__ import annotations

from core.runtime.errors import record_degradation

import asyncio
import contextvars
import hashlib
import heapq
import json
import logging
import math
import random
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace as _dc_replace
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("Aura.Reasoning.NativeSystem2")

#: Per-search tally of where the default scorers' values came from. A
#: ContextVar rather than an attribute because searches interleave on the event
#: loop, and an attribute would attribute one search's provenance to another —
#: the same reason the lesion registry scopes counterfactuals this way.
_VALUE_EVIDENCE: "contextvars.ContextVar[Optional[Dict[str, int]]]" = (
    contextvars.ContextVar("native_system2_value_evidence", default=None)
)


@dataclass(frozen=True)
class ActionValueEstimate:
    """An evidenced value and the declared risk that goes with it."""

    value: float
    risk: float
    evidence: str


class TreeCycleError(ValueError):
    """Raised when a search tree mutation would introduce a cycle."""


class SearchAlgorithm(str, Enum):
    MCTS = "mcts"
    BEAM = "beam"
    BEST_FIRST = "best_first"
    HYBRID = "hybrid"


class CommitmentStatus(str, Enum):
    OPEN = "open"
    SIMULATED = "simulated"
    REJECTED = "rejected"
    SELECTED = "selected"
    COMMITTED = "committed"


@dataclass(frozen=True)
class System2Action:
    """A candidate action/latent step proposed for the search tree."""

    name: str
    prior: float = 1.0
    action_type: str = "latent_plan"
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    risk: float = 0.0
    external_side_effect: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "prior": float(self.prior),
            "action_type": self.action_type,
            "metadata": dict(self.metadata),
            "valid": bool(self.valid),
            "risk": float(self.risk),
            "external_side_effect": bool(self.external_side_effect),
        }


@dataclass(frozen=True)
class SimulatedTransition:
    """Side-effect-free world-model prediction for taking an action."""

    next_state: Any
    reward_estimate: float = 0.0
    terminal_probability: float = 0.0
    uncertainty: float = 0.25
    changed_variables: Dict[str, Any] = field(default_factory=dict)
    trace: str = ""
    invalid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_state": self.next_state,
            "reward_estimate": round(float(self.reward_estimate), 6),
            "terminal_probability": round(float(self.terminal_probability), 6),
            "uncertainty": round(float(self.uncertainty), 6),
            "changed_variables": dict(self.changed_variables),
            "trace": self.trace,
            "invalid": bool(self.invalid),
        }


@dataclass
class NativePlanNode:
    """A native System 2 tree/graph node.

    The field set deliberately matches the proof-suite requirements: both a
    latent representation and a surface/action representation are present, and
    all MCTS/beam/commitment metadata is explicit.
    """

    id: str
    state: Any
    latent_state: List[float]
    action: Optional[System2Action] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 1.0
    reward: float = 0.0
    terminal: bool = False
    uncertainty: float = 0.25
    simulation_trace: List[Dict[str, Any]] = field(default_factory=list)
    reflection_trace: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_trace: List[Dict[str, Any]] = field(default_factory=list)
    commitment_status: CommitmentStatus = CommitmentStatus.OPEN
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    state_hash: str = ""
    symbolic_summary: str = ""
    surface_text_optional: str = ""
    action_sequence: List[str] = field(default_factory=list)
    rejection_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state_hash:
            self.state_hash = stable_state_hash(self.state)
        if self.action and not self.symbolic_summary:
            self.symbolic_summary = self.action.name
        if self.action and not self.surface_text_optional:
            self.surface_text_optional = self.action.name
        if self.action and not self.action_sequence:
            self.action_sequence = [self.action.name]

    @property
    def mean_value(self) -> float:
        if self.visits > 0:
            return self.value_sum / self.visits
        try:
            return float(self.metadata.get("estimated_value", 0.0))
        except (OSError, ConnectionError, TimeoutError):
            return 0.0

    @property
    def latent_plan_embedding(self) -> List[float]:
        return self.latent_state

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["action"] = self.action.to_dict() if self.action else None
        out["commitment_status"] = self.commitment_status.value
        out["mean_value"] = round(self.mean_value, 6)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NativePlanNode":
        data = dict(data)
        data.pop("mean_value", None)
        if data.get("action"):
            data["action"] = System2Action(**data["action"])
        data["commitment_status"] = CommitmentStatus(data.get("commitment_status", "open"))
        return cls(**data)


@dataclass
class System2SearchConfig:
    algorithm: SearchAlgorithm = SearchAlgorithm.HYBRID
    budget: int = 64
    max_depth: int = 5
    branching_factor: int = 4
    beam_width: int = 4
    exploration_constant: float = 1.41
    discount: float = 0.97
    seed: Optional[int] = None
    wall_clock_timeout_s: Optional[float] = None
    confidence_threshold: float = 0.62
    uncertainty_threshold: float = 0.45
    progressive_widening: int = 32
    allow_external_side_effects_in_simulation: bool = False

    def normalized(self) -> "System2SearchConfig":
        self.budget = max(0, int(self.budget))
        self.max_depth = max(1, int(self.max_depth))
        self.branching_factor = max(1, int(self.branching_factor))
        self.beam_width = max(1, int(self.beam_width))
        self.exploration_constant = max(0.0, float(self.exploration_constant))
        self.discount = max(0.0, min(1.0, float(self.discount)))
        self.confidence_threshold = max(0.0, min(1.0, float(self.confidence_threshold)))
        self.uncertainty_threshold = max(0.0, min(1.0, float(self.uncertainty_threshold)))
        self.progressive_widening = max(1, int(self.progressive_widening))
        return self


@dataclass
class NativeSearchReceipt:
    search_id: str
    root_state_hash: str
    algorithm: str
    budget: int
    seed: Optional[int]
    nodes_expanded: int
    simulations: int
    max_depth: int
    best_path: List[str]
    runner_up_paths: List[List[str]]
    value_scores: Dict[str, float]
    uncertainty: float
    rejected_branches: List[Dict[str, Any]]
    commitment_reason: str
    will_receipt_id: Optional[str] = None
    generated_at: float = field(default_factory=time.time)
    #: Where the value estimates behind this search came from, counted by
    #: source: caller / learned / prior / none. A search whose values were all
    #: "none" explored faithfully over numbers that expressed no preference,
    #: so its ordering is tie-breaking rather than judgement. Recording this is
    #: what stops a governed search receipt from implying evidence it did not
    #: have.
    value_evidence: Dict[str, int] = field(default_factory=dict)
    #: Actions whose risk was raised by the lexical hazard floor because they
    #: declared none. Named so a reviewer can see that a safety brake was
    #: applied on the strength of spelling alone.
    hazard_floored_actions: List[str] = field(default_factory=list)
    #: Candidates the preference procedure removed before the search ran, each
    #: with the stage and the preference that removed it. A prohibited action
    #: never reaches the value model, so without this the receipt would show a
    #: deliberation over a field that silently excluded something — which is
    #: indistinguishable from one where it was never offered.
    preference_removals: Dict[str, str] = field(default_factory=dict)
    #: How the preference procedure resolved, when it resolved at all. Set only
    #: where measured operator values separated the field and the search
    #: therefore confirmed a decision rather than making one. A receipt that
    #: did not say so would present a one-candidate confirmation as full
    #: deliberation — the same thing ``chunk_reused`` exists to prevent.
    preference_selection: Optional[str] = None
    #: The compiled-resolution key for this decision, when chunking applied.
    chunk_signature: Optional[str] = None
    #: True when a previously learned chunk supplied the answer and the search
    #: collapsed to confirming it. A receipt that does not say this would
    #: present a one-node confirmation as if it were full deliberation.
    chunk_reused: bool = False
    #: True when a Tier 2 generalized rule recognised this situation despite
    #: the exact signature being new. Distinct from chunk_reused because the
    #: two carry different risk: a chunk answered the same question again, a
    #: rule answered a question it had never been asked.
    rule_applied: bool = False
    #: The conditions that rule fired on, so a reviewer can see what the
    #: shortcut believed was causal rather than only that one was taken.
    rule_conditions: List[str] = field(default_factory=list)
    #: Stable digest of the situation in which the selected action was ranked.
    #: The raw prompt is deliberately not retained in the value table.
    outcome_state_key: str = ""
    #: Canonical selected action whose eventual real outcome may teach Q(s,a).
    outcome_action: str = ""
    #: Expected quality committed before execution.  This is provenance only;
    #: no outcome receipt is opened until an execution owner accepts the plan.
    outcome_expected: float = 0.0
    #: Durable-ledger id once an execution owner accepts this decision.
    outcome_receipt_id: str = ""
    outcome_opened_at: float = 0.0

    @property
    def value_is_evidenced(self) -> bool:
        """True when at least one estimate came from the caller or from data."""
        return bool(
            self.value_evidence.get("caller", 0)
            or self.value_evidence.get("learned", 0)
            or self.value_evidence.get("learned_contextual", 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NativeSearchResult:
    search_id: str
    algorithm: SearchAlgorithm
    tree: "NativeSearchTree"
    root_id: str
    selected_node_id: Optional[str]
    committed_action: Optional[System2Action]
    confidence: float
    uncertainty: float
    receipt: NativeSearchReceipt

    @property
    def selected_node(self) -> Optional[NativePlanNode]:
        return self.tree.nodes.get(self.selected_node_id or "")

    @property
    def best_path_nodes(self) -> List[NativePlanNode]:
        return [self.tree.nodes[nid] for nid in self.receipt.best_path if nid in self.tree.nodes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_id": self.search_id,
            "algorithm": self.algorithm.value,
            "root_id": self.root_id,
            "selected_node_id": self.selected_node_id,
            "committed_action": self.committed_action.to_dict() if self.committed_action else None,
            "confidence": round(self.confidence, 6),
            "uncertainty": round(self.uncertainty, 6),
            "receipt": self.receipt.to_dict(),
            "nodes": [node.to_dict() for node in self.tree.nodes.values()],
        }


ActionGenerator = Callable[[Any, NativePlanNode, System2SearchConfig], Sequence[System2Action] | Awaitable[Sequence[System2Action]]]
WorldModel = Callable[[Any, System2Action, NativePlanNode], SimulatedTransition | Awaitable[SimulatedTransition]]
ValueScorer = Callable[[NativePlanNode, str], float | Awaitable[float]]
ReflectionScorer = Callable[[NativePlanNode], Dict[str, Any] | Awaitable[Dict[str, Any]]]


def stable_state_hash(state: Any) -> str:
    """Stable JSON-based hash with repr fallback for non-JSON values."""
    try:
        payload = json.dumps(state, sort_keys=True, default=str, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = repr(state)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def latent_from_state(state: Any, dims: int = 32) -> List[float]:
    """Deterministic compressed latent vector for planning and tests.

    This is an information-preserving hash projection, not a learned VAE. It
    gives Aura a native latent slot today while leaving a clean seam for future
    learned encoders.
    """
    digest = hashlib.sha256(stable_state_hash(state).encode("utf-8")).digest()
    values = []
    for idx in range(dims):
        byte = digest[idx % len(digest)]
        values.append(round((byte / 127.5) - 1.0, 6))
    return values


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 0.0


class NativeSearchTree:
    """Explicit search graph with invariant checks and JSON roundtrip."""

    def __init__(self) -> None:
        self.nodes: Dict[str, NativePlanNode] = {}
        self.root_id: Optional[str] = None
        self.state_index: Dict[str, List[str]] = {}
        self.rejected_branches: List[Dict[str, Any]] = []

    def create_root(self, state: Any, *, summary: str = "root") -> NativePlanNode:
        node = NativePlanNode(
            id=self._new_id("root"),
            state=state,
            latent_state=latent_from_state(state),
            symbolic_summary=summary,
            surface_text_optional=summary,
        )
        self.root_id = node.id
        self._insert(node)
        return node

    def add_child(
        self,
        parent_id: str,
        action: System2Action,
        transition: SimulatedTransition,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NativePlanNode:
        if parent_id not in self.nodes:
            raise KeyError(f"parent not found: {parent_id}")
        parent = self.nodes[parent_id]
        node = NativePlanNode(
            id=self._new_id("n"),
            state=transition.next_state,
            latent_state=latent_from_state(transition.next_state),
            action=action,
            parent_id=parent_id,
            depth=parent.depth + 1,
            prior=max(0.0, float(action.prior)),
            reward=float(transition.reward_estimate),
            terminal=bool(transition.terminal_probability >= 0.95 or transition.invalid),
            uncertainty=_clamp01(transition.uncertainty),
            simulation_trace=[transition.to_dict()],
            commitment_status=CommitmentStatus.SIMULATED,
            action_sequence=[*parent.action_sequence, action.name],
            metadata=dict(metadata or {}),
        )
        self._assert_no_cycle(parent_id, node.id)
        parent.children_ids.append(node.id)
        parent.updated_at = time.time()
        self._insert(node)
        return node

    def attach_existing_child(self, parent_id: str, child_id: str) -> None:
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise KeyError("parent or child missing")
        self._assert_no_cycle(parent_id, child_id)
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        if child_id not in parent.children_ids:
            parent.children_ids.append(child_id)
        child.parent_id = parent_id
        child.depth = parent.depth + 1
        child.updated_at = time.time()

    def path_to_root(self, node_id: str) -> List[str]:
        path: List[str] = []
        seen: set[str] = set()
        current = node_id
        while current:
            if current in seen:
                raise TreeCycleError(f"cycle detected at {current}")
            seen.add(current)
            path.append(current)
            parent_id = self.nodes[current].parent_id if current in self.nodes else None
            current = parent_id or ""
        return list(reversed(path))

    def check_invariants(self) -> List[str]:
        errors: List[str] = []
        for node_id, node in self.nodes.items():
            if node.parent_id:
                parent = self.nodes.get(node.parent_id)
                if parent is None:
                    errors.append(f"orphan:{node_id}")
                elif node_id not in parent.children_ids:
                    errors.append(f"missing_parent_link:{node_id}")
                elif node.depth != parent.depth + 1:
                    errors.append(f"bad_depth:{node_id}")
            if len(node.children_ids) != len(set(node.children_ids)):
                errors.append(f"duplicate_child:{node_id}")
            for child_id in node.children_ids:
                child = self.nodes.get(child_id)
                if child is None:
                    errors.append(f"missing_child:{node_id}->{child_id}")
                elif child.parent_id != node_id:
                    errors.append(f"missing_child_backlink:{child_id}")
            try:
                self.path_to_root(node_id)
            except TreeCycleError as exc:
                errors.append(str(exc))
            child_visits = sum(self.nodes[cid].visits for cid in node.children_ids if cid in self.nodes)
            if node.visits and child_visits > node.visits:
                errors.append(f"visit_sum:{node_id}:{node.visits}<{child_visits}")
        return errors

    def prune(self, predicate: Callable[[NativePlanNode], bool], *, preserve_path: Sequence[str] = ()) -> List[str]:
        preserve = set(preserve_path)
        removed: List[str] = []
        for node_id, node in list(self.nodes.items()):
            if node_id == self.root_id or node_id in preserve:
                continue
            if predicate(node):
                node.commitment_status = CommitmentStatus.REJECTED
                node.rejection_reason = node.rejection_reason or "pruned"
                removed.append(node_id)
                self.rejected_branches.append({
                    "node_id": node_id,
                    "reason": node.rejection_reason,
                    "value": node.mean_value,
                    "depth": node.depth,
                })
        for node_id in removed:
            parent_id = self.nodes[node_id].parent_id
            if parent_id and parent_id in self.nodes:
                self.nodes[parent_id].children_ids = [
                    cid for cid in self.nodes[parent_id].children_ids if cid != node_id
                ]
            self.nodes.pop(node_id, None)
        self._rebuild_state_index()
        return removed

    def best_path(self, *, by_visits: bool = False) -> List[str]:
        if not self.root_id or self.root_id not in self.nodes:
            return []
        path = [self.root_id]
        current = self.nodes[self.root_id]
        while current.children_ids:
            children = [
                self.nodes[cid] for cid in current.children_ids
                if cid in self.nodes and self.nodes[cid].commitment_status != CommitmentStatus.REJECTED
            ]
            if not children:
                break
            if by_visits:
                best = max(children, key=lambda n: (n.visits, n.mean_value, n.prior))
            else:
                best = max(children, key=lambda n: (n.mean_value, n.visits, n.prior))
            path.append(best.id)
            current = best
        return path

    def runner_up_paths(self, limit: int = 3) -> List[List[str]]:
        if not self.root_id:
            return []
        root = self.nodes[self.root_id]
        children = [self.nodes[cid] for cid in root.children_ids if cid in self.nodes]
        children.sort(key=lambda n: (n.mean_value, n.visits), reverse=True)
        paths: List[List[str]] = []
        best = self.best_path()
        for child in children:
            path = self.best_path_from(child.id)
            if path and path != best:
                paths.append(path)
            if len(paths) >= limit:
                break
        return paths

    def best_path_from(self, node_id: str) -> List[str]:
        if not self.root_id or node_id not in self.nodes:
            return []
        path = self.path_to_root(node_id)
        current = self.nodes[node_id]
        while current.children_ids:
            children = [self.nodes[cid] for cid in current.children_ids if cid in self.nodes]
            if not children:
                break
            current = max(children, key=lambda n: (n.mean_value, n.visits, n.prior))
            path.append(current.id)
        return path

    def to_json(self) -> str:
        payload = {
            "root_id": self.root_id,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "rejected_branches": list(self.rejected_branches),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, raw: str) -> "NativeSearchTree":
        payload = json.loads(raw)
        tree = cls()
        tree.root_id = payload.get("root_id")
        tree.rejected_branches = list(payload.get("rejected_branches") or [])
        for node_data in payload.get("nodes") or []:
            node = NativePlanNode.from_dict(node_data)
            tree._insert(node)
        return tree

    def _insert(self, node: NativePlanNode) -> None:
        self.nodes[node.id] = node
        self.state_index.setdefault(node.state_hash, []).append(node.id)

    def _rebuild_state_index(self) -> None:
        self.state_index.clear()
        for node in self.nodes.values():
            self.state_index.setdefault(node.state_hash, []).append(node.id)

    def _assert_no_cycle(self, parent_id: str, child_id: str) -> None:
        current = parent_id
        while current:
            if current == child_id:
                raise TreeCycleError(f"attaching {child_id} below {parent_id} creates a cycle")
            current = self.nodes[current].parent_id if current in self.nodes else ""

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


class NativeSystem2Engine:
    MAX_RECEIPTS = 4096
    MAX_FAILED_BRANCHES = 4096

    """Governed hybrid MCTS/beam/best-first planner for Aura."""

    def __init__(
        self,
        *,
        llm: Any = None,
        governed: bool = True,
        action_generator: Optional[ActionGenerator] = None,
        world_model: Optional[WorldModel] = None,
        value_scorer: Optional[ValueScorer] = None,
        reflection_scorer: Optional[ReflectionScorer] = None,
    ) -> None:
        self.llm = llm
        self.governed = governed
        self.action_generator = action_generator or self._default_action_generator
        self.world_model = world_model or self._default_world_model
        self.value_scorer = value_scorer or self._default_value_scorer
        self.reflection_scorer = reflection_scorer
        self._receipts: OrderedDict[str, NativeSearchReceipt] = OrderedDict()
        self._failed_branch_memory: OrderedDict[str, int] = OrderedDict()
        self._receipt_evictions = 0
        self._failed_branch_evictions = 0

    async def search(
        self,
        goal: str,
        initial_state: Any,
        *,
        config: Optional[System2SearchConfig] = None,
        action_generator: Optional[ActionGenerator] = None,
        world_model: Optional[WorldModel] = None,
        value_scorer: Optional[ValueScorer] = None,
        source: str = "native_system2",
        context: Optional[Dict[str, Any]] = None,
    ) -> NativeSearchResult:
        config = (config or System2SearchConfig()).normalized()
        algorithm = self._route_algorithm(config, initial_state, context or {})
        rng = random.Random(config.seed)
        search_id = "s2_" + uuid.uuid4().hex[:12]
        t0 = time.monotonic()
        # Scoped to this search, so interleaved searches cannot inherit each
        # other's provenance. Reset in _finish.
        evidence_token = _VALUE_EVIDENCE.set({})
        will_receipt_id = self._consult_will(goal, source, algorithm, config, context or {})

        tree = NativeSearchTree()
        root = tree.create_root(initial_state, summary=str(goal)[:160])

        generator = action_generator or self.action_generator
        model = world_model or self.world_model
        scorer = value_scorer or self.value_scorer

        simulations = 0
        if config.budget == 0:
            result = self._finish(search_id, algorithm, tree, root.id, None, config, will_receipt_id, simulations)
            return result

        if algorithm == SearchAlgorithm.BEAM:
            simulations = await self._beam_search(goal, tree, root, config, generator, model, scorer, t0)
        elif algorithm == SearchAlgorithm.BEST_FIRST:
            simulations = await self._best_first_search(goal, tree, root, config, generator, model, scorer, t0, rng)
        else:
            simulations = await self._mcts_search(goal, tree, root, config, generator, model, scorer, t0, rng)

        by_visits = algorithm == SearchAlgorithm.MCTS
        best_path = tree.best_path(by_visits=by_visits)
        selected_id = best_path[-1] if len(best_path) > 1 else None
        if selected_id and selected_id in tree.nodes:
            tree.nodes[selected_id].commitment_status = CommitmentStatus.SELECTED
        try:
            result = self._finish(
                search_id, algorithm, tree, root.id, selected_id, config,
                will_receipt_id, simulations,
            )
            # Only the default scorers tally here; a caller supplying its own
            # world model and value scorer leaves this empty, which is correct
            # — this records the provenance of THIS engine's defaults, not a
            # claim about somebody else's model.
            tally = _VALUE_EVIDENCE.get() or {}
            if tally and not result.receipt.value_evidence:
                result.receipt.value_evidence = dict(tally)
            return result
        finally:
            _VALUE_EVIDENCE.reset(evidence_token)

    async def _resolve_preferences(
        self,
        candidate_actions: Sequence[System2Action],
        context: str,
        *,
        value_model: Any = None,
        state: str = "",
    ):
        """Translate what Aura already knows about these actions into preferences.

        Three sources, and each was already a fact the deliberation had access
        to and could not act on symbolically:

        * every candidate is ``acceptable`` — something proposed it;
        * ``valid=False`` is a ``reject``, which previously only reached the
          impasse classifier while the action stayed in the search and could
          still be committed on a good enough score;
        * a standing directive is a ``prohibit``. Directives are the owner's
          written prohibitions and were consulted only at the gateway, after
          deliberation had already committed to the act.

        Indifference is asserted only in the one case Soar-RL describes: every
        surviving candidate has a *measured* value and those values separate
        them. Then the decision procedure returns a winner and deliberation
        confirms it rather than re-deriving it — the same trade the chunk store
        already makes, on evidence rather than on a compiled resolution.

        Absent that, nothing here asserts indifference, so a field the
        preferences do not separate resolves to a tie impasse and the search
        below breaks it. That is the correct division of labour: the preference
        layer removes what must not be chosen, and deliberation chooses among
        what remains.
        """
        from core.cognition.preference_semantics import PreferenceBuilder, resolve

        builder = PreferenceBuilder("native_system2.rank_actions")
        for action in candidate_actions:
            builder.acceptable(action.name)
            if not action.valid:
                builder.reject(action.name, "the caller marked this action invalid")

        prohibited = await asyncio.to_thread(self._standing_prohibitions, candidate_actions)
        for name, reason in prohibited.items():
            builder.prohibit(name, reason)

        self._assert_learned_values(
            builder,
            [
                a
                for a in candidate_actions
                if a.valid and a.name not in prohibited
            ],
            value_model=value_model,
            state=state,
        )

        return resolve(
            [a.name for a in candidate_actions],
            builder.build(),
            context={"goal": "rank_actions", "context": context.strip()[:256]},
        )

    @staticmethod
    def _assert_learned_values(
        builder: Any,
        live: Sequence[System2Action],
        *,
        value_model: Any,
        state: str,
    ) -> None:
        """Attach measured operator values, when they are complete and decisive.

        This is where Soar-RL joins Aura's own machinery. Soar learns numeric
        preferences on operators from reward; ``ActionValueModel`` already
        learns per-action values from graded outcome receipts, shrunk toward the
        global mean by the ratio of within-group to between-group variance. The
        learning was never the missing part. The missing part was that nothing
        carried it into the decision procedure, so a value the system had
        measured could only influence the search's scoring and never the
        selection itself.

        Three conditions, and each is a refusal rather than a threshold.

        The evidence must be ``learned`` specifically, not merely present.
        ``ActionValue`` also reports ``caller`` for a supplied ``score_hint``
        and ``prior`` for the ledger's global mean. Accepting those here let the
        planner's own ``score_hint`` decide the operator — the caller's word
        promoted to a measured value, which is the exact substitution
        ``action_value`` exists to prevent. Soar-RL learns from reward; a hint
        is not reward.

        *Every* live candidate must be learned. Ranking a measured action
        against an unmeasured one is a judgement about which one somebody
        happened to have data for — the same defect, one layer up.

        The values must actually separate the field. If the top two are equal
        there is no decision in the numbers, and asserting indifference would
        hand the choice to a seeded draw that then looks like a learned result.
        A genuine tie belongs in the impasse log, where the tie rate is a
        diagnostic.
        """
        if value_model is None or len(live) < 2:
            return
        estimates = {
            action.name: value_model.value_for(action.name, action.metadata, state=state)
            for action in live
        }
        if not all(estimate.evidence == "learned" for estimate in estimates.values()):
            return
        ordered = sorted((e.value for e in estimates.values()), reverse=True)
        if ordered[0] <= ordered[1]:
            return
        for name, estimate in estimates.items():
            builder.numeric_indifferent(
                name, estimate.value, f"measured outcomes ({estimate.evidence})"
            )

    @staticmethod
    def _standing_prohibitions(
        candidate_actions: Sequence[System2Action],
    ) -> Dict[str, str]:
        """Which candidates the owner's standing directives forbid.

        Runs off the loop because the store stats and may read its file. A
        directive store that cannot be read yields no prohibitions and says so
        through its own degradation channel; it must not take the deliberation
        down with it, and it must not silently permit either — the gateway check
        downstream is unchanged and still refuses.
        """
        try:
            from core.governance.standing_directives import get_standing_directives
        except ImportError:
            return {}

        store = get_standing_directives()
        found: Dict[str, str] = {}
        for action in candidate_actions:
            try:
                match, _loaded = store.check(
                    tool_name=action.name,
                    args=dict(action.metadata or {}),
                    effect_scope="external" if action.external_side_effect else "",
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            if match is not None:
                found[action.name] = (
                    f"standing directive {match.directive.directive_id} "
                    f"({match.matched_on}): {match.directive.reason or 'no reason recorded'}"
                )
        return found

    async def rank_actions(
        self,
        *,
        context: str,
        actions: Sequence[str | System2Action | Dict[str, Any]],
        config: Optional[System2SearchConfig] = None,
        source: str = "native_system2.rank_actions",
    ) -> NativeSearchResult:
        candidate_actions = [self._coerce_action(action, idx) for idx, action in enumerate(actions)]

        # Value comes from recorded outcomes or from the caller, never from how
        # the action is spelled. See core/reasoning/action_value.py for what
        # this replaced and why the hazard floor is a separate axis.
        from core.reasoning.action_value import (
            get_action_value_model,
            lexical_hazard_floor,
        )

        value_model = get_action_value_model()
        # Evidence refresh performs SQLite reads.  Do it once, off-loop, before
        # search rather than inside the synchronous scorer for the first node.
        await value_model.refresh_if_stale()
        evidence_counts: Dict[str, int] = {}
        # The situation this decision is being made in. Passing it is what
        # makes the estimate Q(s,a) rather than V(a) where the evidence
        # supports it: "this action works here" instead of "this action tends
        # to work". The model backs off to the marginal history when the
        # contextual bucket is empty, which it usually is at first.
        #
        # Exactly one hashing site. The raw situation goes to value_for, which
        # hashes it; the digest goes on the receipt, so the key the writer
        # stores is the key the reader computes. Passing the digest to
        # value_for instead would hash it twice and every contextual lookup
        # would miss — silently, degrading straight back to V(a).
        state_raw = context.strip()[:512]
        state_descriptor = value_model.state_key(state_raw)

        # A bare string action arrives with risk=0.0 because _coerce_action has
        # nothing better to go on. Apply the lexical floor BEFORE search so the
        # risk channel — which the world model and value scorer both already
        # respect — carries the hazard, instead of the value function inventing
        # a penalty from the same substrings.
        hazard_floored: List[str] = []
        floored_actions: List[System2Action] = []
        for action in candidate_actions:
            floor = lexical_hazard_floor(action.name) if action.risk <= 0.0 else 0.0
            if floor > 0.0:
                # System2Action is frozen, which is the right call for
                # something that ends up in a commitment receipt: the action
                # that was searched must be the action that was recorded.
                action = _dc_replace(action, risk=floor)
                hazard_floored.append(action.name)
            floored_actions.append(action)
        candidate_actions = floored_actions

        # Soar's preference semantics, before any number is computed. The value
        # model runs on what survives, so a standing directive removes an action
        # from the decision instead of taxing it 0.20 of value and losing to a
        # candidate worth 0.25 less.
        resolution = await self._resolve_preferences(
            candidate_actions, context, value_model=value_model, state=state_raw
        )
        preference_removals = {
            name: resolution.why(name)
            for name in (a.name for a in candidate_actions)
            if name not in set(resolution.survivors)
        }
        if preference_removals:
            survivors = set(resolution.survivors)
            candidate_actions = [a for a in candidate_actions if a.name in survivors]

        # The decision procedure produced a winner, which here can only mean
        # measured values separated the field (nothing else asserts
        # indifference). Narrow to it and let the search confirm rather than
        # re-derive — the same shape as chunk reuse below, and for the same
        # reason: the expensive part is deciding, and this decision was already
        # paid for by the outcomes that produced the values.
        preference_winner = resolution.winner if resolution.decided else None
        if preference_winner is not None:
            by_preference = {a.name: a for a in candidate_actions}
            if preference_winner in by_preference:
                candidate_actions = [by_preference[preference_winner]]

        async def _generator(_state: Any, node: NativePlanNode, cfg: System2SearchConfig) -> Sequence[System2Action]:
            if node.depth == 0:
                return candidate_actions[: cfg.branching_factor]
            if node.action and node.action.name.startswith("verify:"):
                return []
            if node.action:
                return [
                    System2Action(
                        name=f"verify:{node.action.name}",
                        prior=0.72,
                        action_type="verification",
                        metadata={**node.action.metadata, "verifies": node.action.name},
                        risk=node.action.risk,
                        external_side_effect=node.action.external_side_effect,
                    )
                ]
            return []

        async def _world(state: Any, action: System2Action, node: NativePlanNode) -> SimulatedTransition:
            selected = action.metadata.get("verifies") or action.name
            estimate = value_model.value_for(
                selected, action.metadata, state=state_raw
            )
            evidence_counts[estimate.evidence] = (
                evidence_counts.get(estimate.evidence, 0) + 1
            )
            score_hint = estimate.value
            if action.name.startswith("verify:"):
                # Verification is a structural property of the generated node,
                # not a guess from its name: this branch exists only because
                # _generator built a verify: successor for a real action.
                score_hint = min(1.0, score_hint + 0.08)
            return SimulatedTransition(
                next_state={
                    "context": context,
                    "selected": selected,
                    "path": [*node.action_sequence, action.name],
                    "score_hint": score_hint,
                },
                reward_estimate=score_hint - (0.20 * float(action.risk)),
                terminal_probability=0.85 if action.name.startswith("verify:") else 0.0,
                uncertainty=max(0.05, 0.35 - (score_hint * 0.18)),
                changed_variables={"selected": selected},
                trace=f"simulated deliberate choice of {selected}",
                invalid=not action.valid,
            )

        async def _value(node: NativePlanNode, _goal: str) -> float:
            if node.action is None:
                return 0.5
            # No substring scoring. The estimate is the caller's score, or the
            # shrunk mean of measured outcomes for this action, or an
            # explicitly unevidenced midpoint — and which one it was is
            # reported on the result rather than blended away here.
            selected = node.action.metadata.get("verifies") or node.action.name
            estimate = value_model.value_for(
                selected, node.action.metadata, state=state_raw
            )
            score_hint = estimate.value
            if node.action.name.startswith("verify:"):
                score_hint = max(score_hint, _clamp01(float(node.reward)))
            if not node.action.valid:
                return 0.0
            # Risk and side effects are declared fields, so they stay: they are
            # facts the caller asserted, not inferences from spelling.
            if node.action.external_side_effect:
                score_hint -= 0.08
            score_hint -= 0.20 * float(node.action.risk)
            return _clamp01(score_hint - (node.uncertainty * 0.10))

        cfg = config or System2SearchConfig(
            algorithm=SearchAlgorithm.HYBRID,
            budget=max(12, min(80, len(candidate_actions) * 12)),
            max_depth=2,
            branching_factor=max(1, len(candidate_actions)),
            beam_width=max(1, min(5, len(candidate_actions))),
        )

        # Soar chunking, on the decision that is expensive enough to be worth
        # compiling. An impasse here is a real substate — a budgeted MCTS/beam
        # search over the candidate set — so a compiled resolution saves
        # measured search time rather than the microseconds a cheap tie costs.
        # This is why chunking is wired here and not to workspace arbitration:
        # a chunk has to out-earn its own match cost, and tie-breaking never
        # could.
        from core.cognition.impasse import (
            ImpasseType,
            Impasse,
            classify,
            get_impasse_learner,
            situation_signature,
        )

        learner = get_impasse_learner()
        signature = situation_signature(
            {"goal": "rank_actions", "context": context.strip()[:256]},
            [a.name for a in candidate_actions],
        )

        # Type the impasse rather than calling every deliberation a tie. The
        # four Soar types were implemented and only one was ever used here,
        # which threw away the distinction that makes them worth having: an
        # empty field, a field where everything was rejected, and a field of
        # equals are three different problems and warrant three different
        # responses.
        #
        # Scores come from the value model, so the classification reflects what
        # the decision procedure will actually see, not the raw candidate list.
        pre_scores = {
            a.name: value_model.value_for(a.name, a.metadata, state=state_raw).value
            for a in candidate_actions
        }
        typed = classify(
            [a.name for a in candidate_actions],
            scores=pre_scores,
            rejected=[a.name for a in candidate_actions if not a.valid],
            # Two candidates whose values differ by less than the spread the
            # value model itself reports as unevidenced are not distinguishable.
            tolerance=0.0 if any(
                value_model.value_for(a.name, a.metadata, state=state_raw).is_evidenced
                for a in candidate_actions
            ) else 0.05,
            context={"goal": "rank_actions"},
        )
        impasse = Impasse(
            type=typed.type if typed is not None else ImpasseType.TIE,
            signature=signature,
            candidates=tuple(sorted(a.name for a in candidate_actions)),
            detail=(
                typed.detail
                if typed is not None
                else f"{len(candidate_actions)} candidates, value model separates them"
            ),
        )

        match_started = time.perf_counter()
        chunk = learner.recall(signature)
        match_cost = time.perf_counter() - match_started
        by_name = {a.name: a for a in candidate_actions}

        # Tier 2. Tier 1 above only fires on an exact signature, so the same
        # problem worded differently reruns the whole search. A promoted
        # generalized rule can recognise it — but only after surviving lesion,
        # contradiction search and a Wilson floor, and only ever as a PROPOSAL.
        from core.cognition.procedural_generalization import (
            DecisionEpisode,
            decision_features,
            get_procedural_generalizer,
        )

        generalizer = get_procedural_generalizer()
        max_risk = max((a.risk for a in candidate_actions), default=0.0)
        features = decision_features(
            goal=context,
            candidate_count=len(candidate_actions),
            evidence=(
                "learned"
                if any(
                    value_model.value_for(
                        a.name, a.metadata, state=state_raw
                    ).is_evidenced
                    for a in candidate_actions
                )
                else "prior"
            ),
            max_risk=max_risk,
            hazard_floored=bool(hazard_floored),
            extra=(f"impasse={impasse.type.value}",),
        )
        rule_applied = None
        if chunk is None:
            proposed = generalizer.propose(features)
            if proposed is not None and proposed.resolution in by_name:
                rule_applied = proposed

        if chunk is not None and chunk.resolution in by_name:
            # The impasse is not re-entered. The compiled resolution supplies
            # the answer and the search collapses to confirming it, which is
            # what makes the saving real rather than notional.
            #
            # max_depth stays at 2 and the budget is small but not 1. Dropping
            # to depth 1 was measured and is wrong: the generator builds the
            # `verify:` successor at depth 1, so a depth-1 reuse committed to
            # the bare action where the original deliberation had committed to
            # verifying it. A chunk that silently removes the verification step
            # is not a compiled decision, it is a different and less safe one.
            cfg = System2SearchConfig(
                algorithm=cfg.algorithm,
                budget=4,
                max_depth=2,
                branching_factor=1,
                beam_width=1,
            )
            candidate_actions = [by_name[chunk.resolution]]
            chunk_applied = chunk
        elif rule_applied is not None:
            # Tier 2: a generalized rule recognised this situation even though
            # the exact signature is new. It narrows the field to its proposed
            # resolution and the search confirms it, exactly as Tier 1 does —
            # the rule replaces deliberation, never the confirmation and never
            # the authority downstream of it.
            cfg = System2SearchConfig(
                algorithm=cfg.algorithm,
                budget=4,
                max_depth=2,
                branching_factor=1,
                beam_width=1,
            )
            candidate_actions = [by_name[rule_applied.resolution]]
            if typed is not None:
                learner.record_impasse(impasse)
            chunk_applied = None
        else:
            # Only a real impasse goes in the impasse log. A decision the value
            # model separated cleanly is not a deadlock, and counting it as one
            # would make the impasse rate — the diagnostic the log exists for —
            # read as "how often did we deliberate" instead of "how often could
            # we not decide". The chunk is still learned either way, because it
            # compiles deliberation cost, which is a different thing.
            if typed is not None:
                learner.record_impasse(impasse)
            chunk_applied = None

        deliberation_started = time.perf_counter()
        result = await self.search(
            "rank candidate actions",
            {"context": context, "candidate_count": len(candidate_actions)},
            config=cfg,
            action_generator=_generator,
            world_model=_world,
            value_scorer=_value,
            source=source,
            context={"candidate_count": len(candidate_actions), "integration": "rank_actions"},
        )

        # Attach provenance after the fact, because the counts are produced by
        # the scorers during the search. Without this the receipt records a
        # rigorous search and says nothing about whether its inputs meant
        # anything, which is the specific way structured search over invented
        # numbers comes to look like deep reasoning.
        deliberation_s = time.perf_counter() - deliberation_started

        result.receipt.value_evidence = dict(evidence_counts)
        result.receipt.hazard_floored_actions = list(hazard_floored)
        result.receipt.preference_removals = dict(preference_removals)
        if preference_winner is not None:
            result.receipt.preference_selection = resolution.selection_reason

        committed = result.committed_action
        chosen = (
            (committed.metadata.get("verifies") or committed.name)
            if committed is not None
            else None
        )
        if chunk_applied is not None:
            result.receipt.chunk_signature = signature
            result.receipt.chunk_reused = True
        elif chosen is not None and chosen in by_name:
            # Compile what the substate produced. cost_saved_per_use is the
            # deliberation actually measured, not an estimate, and match_cost
            # is what the lookup above actually took — so the expected-value
            # test that decides whether to keep this chunk is grounded in two
            # measurements rather than in optimism.
            learner.learn(
                impasse,
                chosen,
                cost_saved_per_use=max(0.0, deliberation_s),
                match_cost=max(0.0, match_cost),
            )
            result.receipt.chunk_signature = signature
            result.receipt.chunk_reused = False

        # Feed Tier 2. Every resolved deliberation becomes an episode recorded
        # against the causal trace rather than the raw context. `correct` stays
        # None: an episode is not evidence until an outcome grades it, which
        # the ledger's resolution stream does. Derivation is attempted only
        # once enough graded episodes agree, and a derived rule still has to
        # survive lesion and contradiction search before it can propose.
        if chosen is not None:
            generalizer.record(
                DecisionEpisode(
                    features=features,
                    resolution=chosen,
                    correct=None,
                    protected=max_risk >= 0.6,
                )
            )
            result.receipt.rule_conditions = (
                sorted(rule_applied.conditions) if rule_applied is not None else []
            )
            result.receipt.rule_applied = rule_applied is not None

        # Search is simulation, not action.  Persist only the provenance needed
        # by an execution owner to open a resolvable receipt later.  Opening a
        # ledger row here created outcomes for counterfactual branches and plan
        # rescoring that never executed, then discarded the only id capable of
        # resolving them.
        if chosen is not None:
            result.receipt.outcome_state_key = state_descriptor
            result.receipt.outcome_action = value_model.action_key(str(chosen))
            result.receipt.outcome_expected = _clamp01(float(result.confidence))
        if not result.receipt.value_is_evidenced and candidate_actions:
            logger.info(
                "rank_actions: no value evidence for any of %d candidates "
                "(sources=%s); the ordering is tie-breaking, not judgement",
                len(candidate_actions),
                evidence_counts or {"none": 0},
            )
        return result

    def open_outcome_receipt(
        self,
        search_id: str,
        *,
        category: str = "deliberation",
        horizon_s: float = 3600.0,
    ) -> str | None:
        """Commit the selected decision when an owner is about to execute it.

        Ranking alone never calls this.  The owner that crosses from simulated
        plan to real action must retain the returned id and resolve it from a
        measured outcome.  That pairing prevents hypothetical searches from
        polluting action values while preserving chunk grading and Q(s,a).
        """
        receipt = self._receipts.get(search_id)
        if receipt is None or not receipt.outcome_action:
            return None
        if receipt.outcome_receipt_id:
            return receipt.outcome_receipt_id
        try:
            from core.cognition.outcome_ledger import get_outcome_ledger

            outcome_id = get_outcome_ledger().open(
                receipt.outcome_action,
                receipt.outcome_expected,
                category=category,
                horizon_s=max(1.0, float(horizon_s)),
                context={
                    "state": receipt.outcome_state_key,
                    "chunk_signature": receipt.chunk_signature or "",
                    "chunk_reused": bool(receipt.chunk_reused),
                    "system2_search_id": search_id,
                },
            )
            receipt.outcome_receipt_id = outcome_id
            receipt.outcome_opened_at = time.time()
            return outcome_id
        except (ImportError, RuntimeError, OSError, AttributeError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "native_system2",
                exc,
                severity="debug",
                action="decision receipt not opened; this ranking will not "
                "contribute evidence to the action-value model",
            )
            return None

    @staticmethod
    def resolve_outcome_receipt(receipt_id: str | None, observed: float, *, note: str = "") -> bool:
        """Resolve a retained execution receipt from a measured outcome."""
        if not receipt_id:
            return False
        try:
            from core.cognition.outcome_ledger import get_outcome_ledger

            return get_outcome_ledger().resolve(
                receipt_id, _clamp01(float(observed)), note=str(note)[:240]
            ) is not None
        except (ImportError, RuntimeError, OSError, AttributeError, ValueError) as exc:
            record_degradation(
                "native_system2",
                exc,
                severity="warning",
                action="decision outcome was measured but could not be resolved; "
                "the receipt remains pending for recovery",
            )
            return False

    def get_receipt(self, search_id: str) -> Optional[NativeSearchReceipt]:
        return self._receipts.get(search_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            "receipts": len(self._receipts),
            "governed": self.governed,
            "failed_branch_memory": len(self._failed_branch_memory),
            "receipt_capacity": self.MAX_RECEIPTS,
            "receipt_evictions": self._receipt_evictions,
            "failed_branch_capacity": self.MAX_FAILED_BRANCHES,
            "failed_branch_evictions": self._failed_branch_evictions,
            "algorithms": [a.value for a in SearchAlgorithm],
        }

    async def _mcts_search(
        self,
        goal: str,
        tree: NativeSearchTree,
        root: NativePlanNode,
        config: System2SearchConfig,
        generator: ActionGenerator,
        model: WorldModel,
        scorer: ValueScorer,
        started_at: float,
        rng: random.Random,
    ) -> int:
        simulations = 0
        for _ in range(config.budget):
            if simulations and simulations % 4 == 0:
                await asyncio.sleep(0)
            if self._timed_out(started_at, config):
                break
            node = root
            path = [root]
            while node.children_ids and node.depth < config.max_depth and not node.terminal:
                node = self._select_uct(tree, node, config, rng)
                path.append(node)

            if not node.terminal and node.depth < config.max_depth:
                await self._expand_node(tree, node, goal, config, generator, model, scorer)
                expandable = [tree.nodes[cid] for cid in node.children_ids if tree.nodes[cid].visits == 0]
                if expandable:
                    node = max(expandable, key=lambda child: (child.prior, -child.uncertainty))
                    path.append(node)

            value = await self._evaluate_node(node, goal, scorer)
            self._backpropagate(path, value, config.discount)
            simulations += 1
        return simulations

    async def _beam_search(
        self,
        goal: str,
        tree: NativeSearchTree,
        root: NativePlanNode,
        config: System2SearchConfig,
        generator: ActionGenerator,
        model: WorldModel,
        scorer: ValueScorer,
        started_at: float,
    ) -> int:
        frontier = [root]
        simulations = 0
        for _depth in range(config.max_depth):
            if simulations >= config.budget or self._timed_out(started_at, config):
                break
            next_frontier: List[NativePlanNode] = []
            for node in frontier:
                if simulations and simulations % 4 == 0:
                    await asyncio.sleep(0)
                if simulations >= config.budget:
                    break
                await self._expand_node(tree, node, goal, config, generator, model, scorer)
                for child_id in node.children_ids:
                    child = tree.nodes[child_id]
                    await self._evaluate_node(child, goal, scorer)
                    self._backpropagate(tree.path_to_root(child.id), child.mean_value, config.discount, tree=tree)
                    next_frontier.append(child)
                    simulations += 1
            next_frontier.sort(key=lambda n: (n.mean_value - n.uncertainty * 0.1, n.prior), reverse=True)
            frontier = next_frontier[: config.beam_width]
            for rejected in next_frontier[config.beam_width:]:
                rejected.commitment_status = CommitmentStatus.REJECTED
                rejected.rejection_reason = "beam_width_limit"
                tree.rejected_branches.append({
                    "node_id": rejected.id,
                    "reason": "beam_width_limit",
                    "value": rejected.mean_value,
                    "depth": rejected.depth,
                })
        return simulations

    async def _best_first_search(
        self,
        goal: str,
        tree: NativeSearchTree,
        root: NativePlanNode,
        config: System2SearchConfig,
        generator: ActionGenerator,
        model: WorldModel,
        scorer: ValueScorer,
        started_at: float,
        rng: random.Random,
    ) -> int:
        heap: List[Tuple[float, float, str]] = [(-0.5, rng.random(), root.id)]
        simulations = 0
        while heap and simulations < config.budget and not self._timed_out(started_at, config):
            if simulations and simulations % 4 == 0:
                await asyncio.sleep(0)
            _priority, _tie, node_id = heapq.heappop(heap)
            node = tree.nodes[node_id]
            if node.depth >= config.max_depth or node.terminal:
                continue
            await self._expand_node(tree, node, goal, config, generator, model, scorer)
            for child_id in node.children_ids:
                child = tree.nodes[child_id]
                value = await self._evaluate_node(child, goal, scorer)
                self._backpropagate(tree.path_to_root(child.id), value, config.discount, tree=tree)
                priority = -(child.mean_value + child.prior * 0.05 - child.uncertainty * 0.15)
                heapq.heappush(heap, (priority, rng.random(), child.id))
                simulations += 1
                if simulations >= config.budget:
                    break
        return simulations

    async def _expand_node(
        self,
        tree: NativeSearchTree,
        node: NativePlanNode,
        goal: str,
        config: System2SearchConfig,
        generator: ActionGenerator,
        model: WorldModel,
        scorer: ValueScorer,
    ) -> None:
        if node.children_ids or node.terminal:
            return
        raw_actions = generator(node.state, node, config)
        if asyncio.iscoroutine(raw_actions):
            raw_actions = await raw_actions
        actions = [
            action if isinstance(action, System2Action) else self._coerce_action(action, idx)
            for idx, action in enumerate(raw_actions)
        ]
        actions = self._dedupe_actions(actions)
        actions = [a for a in actions if a.valid]
        actions = actions[: min(config.branching_factor, config.progressive_widening)]
        total_prior = sum(max(0.0, a.prior) for a in actions) or 1.0
        for action in actions:
            normalized_action = System2Action(
                name=action.name,
                prior=max(0.0, action.prior) / total_prior,
                action_type=action.action_type,
                metadata=dict(action.metadata),
                valid=action.valid,
                risk=action.risk,
                external_side_effect=action.external_side_effect,
            )
            if normalized_action.external_side_effect and not config.allow_external_side_effects_in_simulation:
                # External effects are represented, not executed.
                normalized_action.metadata["simulation_mode"] = "side_effect_suppressed"
            transition = model(node.state, normalized_action, node)
            if asyncio.iscoroutine(transition):
                transition = await transition
            child = tree.add_child(node.id, normalized_action, transition)
            if transition.invalid:
                child.commitment_status = CommitmentStatus.REJECTED
                child.rejection_reason = "invalid_transition"
                tree.rejected_branches.append({
                    "node_id": child.id,
                    "reason": "invalid_transition",
                    "value": 0.0,
                    "depth": child.depth,
                })
            if self.reflection_scorer is not None:
                reflection = self.reflection_scorer(child)
                if asyncio.iscoroutine(reflection):
                    reflection = await reflection
                child.reflection_trace.append(dict(reflection or {}))
            await self._evaluate_node(child, goal, scorer)

    async def _evaluate_node(self, node: NativePlanNode, goal: str, scorer: ValueScorer) -> float:
        if node.commitment_status == CommitmentStatus.REJECTED:
            node.visits = max(node.visits, 1)
            node.value_sum = min(node.value_sum, 0.0)
            return 0.0
        value = scorer(node, goal)
        if asyncio.iscoroutine(value):
            value = await value
        value = _clamp01(float(value))
        # Reward and uncertainty are part of the state value, but cannot override hard rejection.
        adjusted = _clamp01((0.72 * value) + (0.23 * _clamp01((node.reward + 1.0) / 2.0)) - (0.10 * node.uncertainty))
        if node.action and node.action.external_side_effect:
            adjusted = max(0.0, adjusted - 0.03)
        node.metadata["estimated_value"] = adjusted
        node.updated_at = time.time()
        return adjusted

    def _select_uct(
        self,
        tree: NativeSearchTree,
        parent: NativePlanNode,
        config: System2SearchConfig,
        rng: random.Random,
    ) -> NativePlanNode:
        children = [
            tree.nodes[cid] for cid in parent.children_ids
            if cid in tree.nodes and tree.nodes[cid].commitment_status != CommitmentStatus.REJECTED
        ]
        if not children:
            return parent
        unvisited = [child for child in children if child.visits == 0]
        if unvisited:
            return max(unvisited, key=lambda c: (c.prior, -c.uncertainty, rng.random()))
        parent_visits = max(1, parent.visits)
        def score(child: NativePlanNode) -> float:
            exploit = child.mean_value
            explore = config.exploration_constant * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
            uncertainty_bonus = min(0.08, child.uncertainty * 0.05)
            memory_penalty = self._failed_branch_memory.get(child.state_hash, 0) * 0.03
            return exploit + explore + uncertainty_bonus - memory_penalty
        return max(children, key=score)

    def _backpropagate(
        self,
        path: Sequence[NativePlanNode | str],
        value: float,
        discount: float,
        *,
        tree: Optional[NativeSearchTree] = None,
    ) -> None:
        running = float(value)
        for item in reversed(path):
            node = tree.nodes[item] if isinstance(item, str) and tree else item
            if not isinstance(node, NativePlanNode):
                continue
            if node.visits == 0:
                node.visits = 1
                node.value_sum = running
            else:
                node.visits += 1
                node.value_sum += running
            node.updated_at = time.time()
            running = _clamp01((running * discount) + ((node.reward + 1.0) / 2.0) * (1.0 - discount))

    async def _default_action_generator(
        self,
        state: Any,
        node: NativePlanNode,
        config: System2SearchConfig,
    ) -> Sequence[System2Action]:
        if node.depth >= config.max_depth:
            return []
        if isinstance(state, dict) and state.get("actions"):
            return [self._coerce_action(action, idx) for idx, action in enumerate(state["actions"])]
        if self.llm is not None and node.depth == 0:
            try:
                prompt = (
                    "Propose diverse next planning actions for this goal. "
                    "Return one action per line, no numbering.\n\n"
                    f"STATE: {state}\nPATH: {node.action_sequence}"
                )
                raw = await self.llm.generate(prompt, temperature=0.5, priority=0.4)
                lines = [line.strip(" -0123456789.\t") for line in str(raw).splitlines() if line.strip()]
                return [
                    System2Action(name=line[:220], prior=1.0 / max(1, len(lines)), action_type="llm_latent_step")
                    for line in lines[: config.branching_factor]
                ]
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("native_system2", exc)
        return [
            System2Action("decompose the problem", 0.34, "decompose"),
            System2Action("simulate the most likely consequence", 0.26, "simulate"),
            System2Action("verify constraints before acting", 0.24, "verify"),
            System2Action("backtrack to an alternate plan", 0.16, "backtrack"),
        ][: config.branching_factor]

    @staticmethod
    def _estimate(action: System2Action, goal: str) -> "ActionValueEstimate":
        """Value for an action on the generic path, from evidence or admitted absent.

        These defaults are what every caller inherits who does not supply a
        world model or value scorer, so they were the widest-reaching instance
        of the same defect ``rank_actions`` had: reward 0.08 with +0.12 for a
        name containing verify/test/simulate/source/constraint, and a value of
        0.48 with +0.045 per matching token and -0.18 for delete/destructive.
        A search whose caller supplied nothing was ranking on spelling, and
        MCTS ran over it faithfully.

        Risk still applies and the hazard floor still fills a vacuum, because
        those are the safety channel rather than the value channel.
        """
        from core.reasoning.action_value import (
            get_action_value_model,
            lexical_hazard_floor,
        )

        model = get_action_value_model()
        estimate = model.value_for(action.name, action.metadata, state=goal)
        risk = action.risk if action.risk > 0.0 else lexical_hazard_floor(action.name)
        counts = _VALUE_EVIDENCE.get()
        if counts is not None:
            counts[estimate.evidence] = counts.get(estimate.evidence, 0) + 1
        return ActionValueEstimate(value=estimate.value, risk=risk, evidence=estimate.evidence)

    async def _default_world_model(
        self,
        state: Any,
        action: System2Action,
        node: NativePlanNode,
    ) -> SimulatedTransition:
        if not action.valid:
            return SimulatedTransition(state, reward_estimate=-1.0, terminal_probability=1.0, uncertainty=1.0, invalid=True)
        current_path = []
        if isinstance(state, dict):
            current_path = list(state.get("path") or [])
        next_state = {
            "previous": state,
            "action": action.name,
            "path": [*current_path, action.name],
            "depth": node.depth + 1,
        }
        goal = str(state.get("goal", "")) if isinstance(state, dict) else ""
        estimate = self._estimate(action, goal)
        # Reward is the evidenced value of taking this action, less its
        # declared risk. Nothing here reads the action's spelling for merit.
        reward = estimate.value - min(0.4, estimate.risk * 0.4)
        uncertainty = 0.28 + min(0.35, (node.depth * 0.05)) + min(0.2, estimate.risk * 0.2)
        if estimate.evidence in ("prior", "none"):
            # Unevidenced values deserve wider error bars, and saying so in the
            # uncertainty channel is what lets the commitment threshold refuse
            # a confident-looking search over numbers nobody measured.
            uncertainty += 0.15
        return SimulatedTransition(
            next_state=next_state,
            reward_estimate=reward,
            terminal_probability=0.0,
            uncertainty=_clamp01(uncertainty),
            changed_variables={"path": next_state["path"]},
            trace=f"latent rollout: {action.name} [value={estimate.evidence}]",
        )

    async def _default_value_scorer(self, node: NativePlanNode, goal: str) -> float:
        if node.action is None:
            return 0.5
        estimate = self._estimate(node.action, goal)
        score = estimate.value
        if not node.action.valid:
            return 0.0
        if node.action.external_side_effect:
            score -= 0.08
        score -= 0.20 * estimate.risk
        return _clamp01(score - (node.uncertainty * 0.08))

    def _finish(
        self,
        search_id: str,
        algorithm: SearchAlgorithm,
        tree: NativeSearchTree,
        root_id: str,
        selected_id: Optional[str],
        config: System2SearchConfig,
        will_receipt_id: Optional[str],
        simulations: int,
    ) -> NativeSearchResult:
        best_path = tree.best_path(by_visits=algorithm == SearchAlgorithm.MCTS)
        if selected_id is None and len(best_path) > 1:
            selected_id = best_path[-1]
        selected = tree.nodes.get(selected_id or "")
        if selected:
            for node_id in best_path:
                if node_id in tree.nodes:
                    tree.nodes[node_id].commitment_status = CommitmentStatus.COMMITTED if node_id == selected.id else CommitmentStatus.SELECTED
        confidence = _clamp01(selected.mean_value if selected else 0.0)
        uncertainty = _clamp01(selected.uncertainty if selected else 1.0)
        reason = self._commitment_reason(selected, confidence, uncertainty, config)
        values = {node_id: round(node.mean_value, 6) for node_id, node in tree.nodes.items()}
        receipt = NativeSearchReceipt(
            search_id=search_id,
            root_state_hash=tree.nodes[root_id].state_hash,
            algorithm=algorithm.value,
            budget=config.budget,
            seed=config.seed,
            nodes_expanded=max(0, len(tree.nodes) - 1),
            simulations=simulations,
            max_depth=max((node.depth for node in tree.nodes.values()), default=0),
            best_path=best_path,
            runner_up_paths=tree.runner_up_paths(),
            value_scores=values,
            uncertainty=uncertainty,
            rejected_branches=list(tree.rejected_branches),
            commitment_reason=reason,
            will_receipt_id=will_receipt_id,
        )
        self._receipts[search_id] = receipt
        self._receipts.move_to_end(search_id)
        while len(self._receipts) > self.MAX_RECEIPTS:
            self._receipts.popitem(last=False)
            self._receipt_evictions += 1
        if selected and selected.mean_value < 0.2:
            self._failed_branch_memory[selected.state_hash] = self._failed_branch_memory.get(selected.state_hash, 0) + 1
            self._failed_branch_memory.move_to_end(selected.state_hash)
            while len(self._failed_branch_memory) > self.MAX_FAILED_BRANCHES:
                self._failed_branch_memory.popitem(last=False)
                self._failed_branch_evictions += 1
        return NativeSearchResult(
            search_id=search_id,
            algorithm=algorithm,
            tree=tree,
            root_id=root_id,
            selected_node_id=selected_id,
            committed_action=selected.action if selected else None,
            confidence=confidence,
            uncertainty=uncertainty,
            receipt=receipt,
        )

    def _commitment_reason(
        self,
        selected: Optional[NativePlanNode],
        confidence: float,
        uncertainty: float,
        config: System2SearchConfig,
    ) -> str:
        if selected is None:
            return "no candidate selected; returning empty best-so-far"
        if confidence >= config.confidence_threshold and uncertainty <= config.uncertainty_threshold:
            return (
                f"selected '{selected.symbolic_summary}' because value={confidence:.3f} "
                f"met threshold and uncertainty={uncertainty:.3f} stayed bounded"
            )
        if confidence < config.confidence_threshold:
            return (
                f"best-so-far '{selected.symbolic_summary}' below confidence threshold "
                f"({confidence:.3f} < {config.confidence_threshold:.3f}); commit should be constrained or defer"
            )
        return (
            f"best-so-far '{selected.symbolic_summary}' has high uncertainty "
            f"({uncertainty:.3f}); commit should request more evidence or use safe fallback"
        )

    def _route_algorithm(
        self,
        config: System2SearchConfig,
        state: Any,
        context: Dict[str, Any],
    ) -> SearchAlgorithm:
        if isinstance(config.algorithm, str):
            config.algorithm = SearchAlgorithm(config.algorithm)
        if config.algorithm != SearchAlgorithm.HYBRID:
            return config.algorithm
        if context.get("requires_retrieval") or context.get("rag"):
            return SearchAlgorithm.MCTS
        if context.get("stochastic") or context.get("partial_observability"):
            return SearchAlgorithm.MCTS
        action_count = int(context.get("candidate_count") or 0)
        if isinstance(state, dict):
            action_count = action_count or len(state.get("actions") or [])
        if action_count and action_count <= config.beam_width and not context.get("stochastic"):
            return SearchAlgorithm.BEAM
        if action_count > config.beam_width * 3:
            return SearchAlgorithm.MCTS
        return SearchAlgorithm.BEST_FIRST if config.budget <= 16 else SearchAlgorithm.MCTS

    def _coerce_action(self, action: str | System2Action | Dict[str, Any], idx: int) -> System2Action:
        if isinstance(action, System2Action):
            return action
        if isinstance(action, dict):
            name = str(action.get("name") or action.get("description") or action.get("action") or f"action_{idx}")
            return System2Action(
                name=name,
                prior=float(action.get("prior", action.get("probability", 1.0))),
                action_type=str(action.get("action_type") or action.get("type") or "candidate"),
                metadata=dict(action.get("metadata") or {k: v for k, v in action.items() if k not in {"name", "description", "action", "prior", "probability", "action_type", "type"}}),
                valid=bool(action.get("valid", True)),
                risk=float(action.get("risk", 0.0)),
                external_side_effect=bool(action.get("external_side_effect", False)),
            )
        return System2Action(name=str(action), prior=1.0, action_type="candidate", metadata={"index": idx})

    @staticmethod
    def _dedupe_actions(actions: Iterable[System2Action]) -> List[System2Action]:
        seen: set[str] = set()
        deduped: List[System2Action] = []
        for action in actions:
            key = action.name.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(action)
        return deduped

    def _consult_will(
        self,
        goal: str,
        source: str,
        algorithm: SearchAlgorithm,
        config: System2SearchConfig,
        context: Dict[str, Any],
    ) -> Optional[str]:
        if not self.governed:
            return None
        try:
            from core.will import ActionDomain, get_will
            decision = get_will().decide(
                content=f"native_system2:{algorithm.value}:{goal[:160]}",
                source=source,
                domain=ActionDomain.REFLECTION,
                priority=float(context.get("priority", 0.45)),
                context={
                    "algorithm": algorithm.value,
                    "budget": config.budget,
                    "max_depth": config.max_depth,
                    "simulation_only": True,
                    **context,
                },
            )
            if not decision.is_approved():
                raise PermissionError(f"UnifiedWill denied System 2 search: {decision.reason}")
            return decision.receipt_id
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("native_system2", exc)
            if self.governed:
                raise
            return None

    @staticmethod
    def _timed_out(started_at: float, config: System2SearchConfig) -> bool:
        return bool(config.wall_clock_timeout_s and (time.monotonic() - started_at) >= config.wall_clock_timeout_s)


_native_system2: Optional[NativeSystem2Engine] = None


def get_native_system2() -> NativeSystem2Engine:
    global _native_system2
    if _native_system2 is None:
        _native_system2 = NativeSystem2Engine()
    return _native_system2


__all__ = [
    "CommitmentStatus",
    "NativePlanNode",
    "NativeSearchReceipt",
    "NativeSearchResult",
    "NativeSearchTree",
    "NativeSystem2Engine",
    "SearchAlgorithm",
    "SimulatedTransition",
    "System2Action",
    "System2SearchConfig",
    "TreeCycleError",
    "get_native_system2",
    "latent_from_state",
    "stable_state_hash",
]
