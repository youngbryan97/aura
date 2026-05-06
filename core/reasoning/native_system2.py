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
import hashlib
import heapq
import json
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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
        except Exception:
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
    except Exception:
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
    except Exception:
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
        self._receipts: Dict[str, NativeSearchReceipt] = {}
        self._failed_branch_memory: Dict[str, int] = {}

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
        return self._finish(search_id, algorithm, tree, root.id, selected_id, config, will_receipt_id, simulations)

    async def rank_actions(
        self,
        *,
        context: str,
        actions: Sequence[str | System2Action | Dict[str, Any]],
        config: Optional[System2SearchConfig] = None,
        source: str = "native_system2.rank_actions",
    ) -> NativeSearchResult:
        candidate_actions = [self._coerce_action(action, idx) for idx, action in enumerate(actions)]

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
            score_hint = _clamp01(float(action.metadata.get("score_hint", 0.55)))
            if action.name.startswith("verify:"):
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
            score_hint = _clamp01(float(node.action.metadata.get("score_hint", node.reward)))
            name = node.action.name.lower()
            for token in ("verify", "test", "simulate", "inspect", "evidence", "safe", "rollback", "minimal"):
                if token in name:
                    score_hint = min(1.0, score_hint + 0.045)
            for token in ("delete", "destructive", "exfiltrate", "bypass", "disable safety"):
                if token in name:
                    score_hint = max(0.0, score_hint - 0.18)
            if node.action.name.startswith("verify:"):
                score_hint = max(score_hint, _clamp01(float(node.reward)))
            if not node.action.valid:
                return 0.0
            if node.action.external_side_effect:
                score_hint -= 0.08
            return _clamp01(score_hint - (node.uncertainty * 0.10))

        cfg = config or System2SearchConfig(
            algorithm=SearchAlgorithm.HYBRID,
            budget=max(12, min(80, len(candidate_actions) * 12)),
            max_depth=2,
            branching_factor=max(1, len(candidate_actions)),
            beam_width=max(1, min(5, len(candidate_actions))),
        )
        return await self.search(
            "rank candidate actions",
            {"context": context, "candidate_count": len(candidate_actions)},
            config=cfg,
            action_generator=_generator,
            world_model=_world,
            value_scorer=_value,
            source=source,
            context={"candidate_count": len(candidate_actions), "integration": "rank_actions"},
        )

    def get_receipt(self, search_id: str) -> Optional[NativeSearchReceipt]:
        return self._receipts.get(search_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            "receipts": len(self._receipts),
            "governed": self.governed,
            "failed_branch_memory": len(self._failed_branch_memory),
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
            except Exception as exc:
                record_degradation("native_system2", exc)
        return [
            System2Action("decompose the problem", 0.34, "decompose"),
            System2Action("simulate the most likely consequence", 0.26, "simulate"),
            System2Action("verify constraints before acting", 0.24, "verify"),
            System2Action("backtrack to an alternate plan", 0.16, "backtrack"),
        ][: config.branching_factor]

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
        action_lower = action.name.lower()
        reward = 0.08
        if any(token in action_lower for token in ("verify", "test", "simulate", "source", "constraint")):
            reward += 0.12
        if "backtrack" in action_lower:
            reward += 0.04
        reward -= min(0.4, action.risk * 0.4)
        uncertainty = 0.28 + min(0.35, (node.depth * 0.05)) + min(0.2, action.risk * 0.2)
        return SimulatedTransition(
            next_state=next_state,
            reward_estimate=reward,
            terminal_probability=0.0,
            uncertainty=_clamp01(uncertainty),
            changed_variables={"path": next_state["path"]},
            trace=f"latent rollout: {action.name}",
        )

    async def _default_value_scorer(self, node: NativePlanNode, goal: str) -> float:
        text = " ".join([goal, node.symbolic_summary, " ".join(node.action_sequence)]).lower()
        score = 0.48
        for token in ("verify", "test", "simulate", "constraint", "evidence", "source", "minimal", "rollback"):
            if token in text:
                score += 0.045
        if "backtrack" in text:
            score += 0.025
        if "delete" in text or "destructive" in text:
            score -= 0.18
        score += min(0.12, max(0, len(node.action_sequence) - 1) * 0.02)
        score -= node.uncertainty * 0.08
        return _clamp01(score)

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
        if selected and selected.mean_value < 0.2:
            self._failed_branch_memory[selected.state_hash] = self._failed_branch_memory.get(selected.state_hash, 0) + 1
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
        except Exception as exc:
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
