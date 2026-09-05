"""core/brain/causal_world_model.py — Structural Causal Model (SCM)
===================================================================
Aura's predictive engine. Instead of just tracking correlations, this
implements a FOCUS-style Structural Causal Model (SCM) with
intervention-based causal discovery (do-calculus).

It allows Aura to:
1. Track observational correlations.
2. Record interventions (do-calculus) that distinguish causation from
   correlation — when, and only when, the caller supplies the control
   evidence that makes that distinction meaningful.
3. Answer counterfactual "what if I break this correlation?" queries.

The load-bearing line in this module used to be ``edge.relationship =
"causes"``, set from four caller-supplied floats with no control, no baseline
and no receipt — and the resulting edge was then rendered into the system
prompt under "ESTABLISHED WORLD CASCADES". CP126 flagged the whole chain. The
rules now are:

* **A claim is only as strong as its evidence.** ``causes`` requires an
  intervention receipt carrying a control/baseline outcome, and replication;
  without them an edge stays ``correlates_with`` however often it is
  submitted.
* **Only proven edges reach the prompt**, and node names are sanitized before
  they get near it — a node name is attacker-reachable text.
* **Disconfirmation is symmetric with confirmation.** A failed prediction
  lowers confidence and can downgrade the relationship.

CP126 03bbcb71 / a2ade8b4 / 462e62cb / 38cd93d1 / a45e3568 / 7020fb8f /
751bc489 / 04afeae8 / 99b39c15 / ed3c0893 / a730f5b8 / 7a91d00d / 7a646f7b /
2fc62124 / 9edb0908 / d2d5130d.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.runtime.errors import record_degradation
from core.runtime.numeric_safety import validated_int, validated_scalar, validated_unit
from core.runtime.service_registry import get_runtime_service, register_runtime_service
import os

logger = logging.getLogger("Aura.CausalWorldModel")

#: Persistence envelope version. A file without it is treated as legacy and
#: revalidated rather than trusted (CP126 7a91d00d).
SCHEMA_VERSION = 2

#: Node names are attacker-reachable text that ends up in a prompt block.
MAX_NODE_NAME_CHARS = 80
_NODE_NAME_ALLOWED = re.compile(r"[^a-z0-9 _\-/.]+")

#: Cardinality quotas. CP126 9edb0908: every novel pair created a node and an
#: edge, all retained and rewritten to disk forever.
MAX_NODES = 2000
MAX_EDGES = 8000

#: Bound on counterfactual work (CP126 99b39c15).
MAX_SIMULATION_STEPS = 25

#: An edge must clear all of these to be stated as established world knowledge.
PROMPT_MIN_CONFIDENCE = 0.7
PROMPT_MIN_ABS_WEIGHT = 0.5
PROMPT_MAX_RULES = 5

#: Interventions needed before a relationship is called `causes`.
MIN_INTERVENTIONS_FOR_CAUSAL = 2
#: Consecutive disconfirmations that downgrade a causal claim.
DISCONFIRMATIONS_TO_DOWNGRADE = 3

_CWM_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def sanitize_node_name(name: Any) -> str:
    """A bounded, single-line, restricted-charset node name.

    CP126 7020fb8f: source and target were lowercased but never length-limited
    or escaped, then persisted and interpolated between brackets into prompt
    text — so a poisoned observation could store newlines and instructions
    that later appeared as established world knowledge.
    """
    text = str(name or "").strip().lower()
    text = text.replace("\n", " ").replace("\r", " ").replace("\x00", "")
    text = _NODE_NAME_ALLOWED.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_NODE_NAME_CHARS]


@dataclass
class CausalNode:
    """A variable in Aura's world model."""

    name: str
    activation: float = 0.0  # Current active state (0.0 to 1.0)
    variance: float = 0.1    # Uncertainty of this state


@dataclass
class InterventionReceipt:
    """Evidence that an intervention actually happened.

    CP126 03bbcb71 / a2ade8b4: the old signature took four bare floats and
    upgraded the edge to ``causes``. Causation needs a comparison: what the
    target did when the source was forced, versus what it did when it was not.
    """

    source_value: float
    treated_outcome: float
    control_outcome: float
    #: Who ran it and in what environment — so replication can be checked.
    performed_by: str = ""
    environment: str = ""
    at: float = field(default_factory=time.time)

    @property
    def effect(self) -> float:
        """The treatment effect: treated minus control, not the raw level."""
        return self.treated_outcome - self.control_outcome

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_value": self.source_value,
            "treated_outcome": self.treated_outcome,
            "control_outcome": self.control_outcome,
            "effect": self.effect,
            "performed_by": self.performed_by,
            "environment": self.environment,
            "at": self.at,
        }


@dataclass
class CausalEdge:
    """A directional causal link: A -> B."""

    source: str
    target: str
    relationship: str = "correlates_with"  # Upgraded to "causes" upon intervention
    weight: float = 0.0           # -1.0 (inhibits) to 1.0 (excites)
    # CP126 7a646f7b: this defaulted to 1.0 — "proven fact" — so a persisted
    # edge missing the field was silently upgraded to maximum trust on load.
    confidence: float = 0.0       # 0.0 (guess) to 1.0 (proven fact)
    observations: int = 1
    intervention_count: int = 0   # Number of times do(source) was tested
    disconfirmations: int = 0
    #: Distinct reporters seen, so N duplicate calls are not N confirmations.
    sources_seen: List[str] = field(default_factory=list)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    last_confirmed: float = field(default_factory=time.time)
    last_disconfirmed: Optional[float] = None

    @property
    def is_causal(self) -> bool:
        return self.relationship == "causes"

    @property
    def independent_sources(self) -> int:
        return max(1, len(set(self.sources_seen)))


class CausalWorldModel:
    """The counterfactual simulation engine."""

    name = "causal_world_model"

    @staticmethod
    def _default_data_path() -> Any:
        """Where the causal graph lives when the caller does not say.

        Under pytest this is a per-process temporary file, NOT the live
        runtime graph. Constructing ``CausalWorldModel()`` in a test used to
        open ~/.aura/data/causal_world.json, which meant every test both
        POLLUTED the running system's causal beliefs and INHERITED whatever
        previous runs had left there. A test asserting that an intervention
        upgrades an edge to "causes" would pass or fail on the residue of an
        earlier run rather than on the code — which is exactly the
        order-dependence that makes a suite untrustworthy.

        ``AURA_CAUSAL_WORLD_PATH`` overrides both, for a caller that wants a
        specific graph.
        """
        from pathlib import Path

        override = os.environ.get("AURA_CAUSAL_WORLD_PATH", "").strip()
        if override:
            return Path(override)

        if os.environ.get("PYTEST_CURRENT_TEST"):
            import tempfile
            import uuid

            # Per INSTANCE, not per process: two models built in the same
            # pytest process would otherwise share one file, and the second
            # would load the first's edges. That is the same leak one level
            # down.
            return (
                Path(tempfile.gettempdir())
                / f"aura_causal_world_test_{os.getpid()}_{uuid.uuid4().hex}.json"
            )

        from core.config import config

        return config.paths.data_dir / "causal_world.json"

    def __init__(self, data_path: Any = None):
        if data_path is not None:
            from pathlib import Path

            self.data_path = Path(data_path)
        else:
            self.data_path = self._default_data_path()

        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        # CP126 2fc62124: nodes and the edge list were searched and mutated
        # with no lock while each caller serialized the whole state to one
        # file, so concurrent writers lost increments and overwrote snapshots.
        self._lock = threading.RLock()
        self.last_save_ok = True
        self.last_save_error = ""
        self.load_quarantined = False

        self._load()

    # -- ingress ---------------------------------------------------------
    def get_node(self, name: str) -> CausalNode:
        """Fetch or lazily create a node."""
        clean = sanitize_node_name(name)
        with self._lock:
            if clean not in self.nodes:
                if len(self.nodes) >= MAX_NODES:
                    # Quota, not unbounded growth (CP126 9edb0908).
                    self._evict_weakest()
                self.nodes[clean] = CausalNode(name=clean)
            return self.nodes[clean]

    def add_observation(
        self,
        source: str,
        target: str,
        correlation: float,
        *,
        reported_by: str = "",
    ) -> bool:
        """Record a real-world observation.

        Returns whether the update was durably saved. CP126 462e62cb:
        confidence used to be a fixed function of call count, so duplicate or
        fabricated calls read as independent confirmation. It is now driven by
        the number of DISTINCT reporters as well as the count.
        """
        source = sanitize_node_name(source)
        target = sanitize_node_name(target)
        if not source or not target or source == target:
            return False
        value = float(validated_scalar(correlation, name="correlation", low=-1.0, high=1.0))

        with self._lock:
            self.get_node(source)
            self.get_node(target)

            edge = self._find(source, target)
            if edge is None:
                if len(self.edges) >= MAX_EDGES:
                    self._evict_weakest()
                edge = CausalEdge(
                    source=source, target=target, weight=value,
                    confidence=0.05, observations=1,
                )
                if reported_by:
                    edge.sources_seen.append(str(reported_by)[:60])
                self.edges.append(edge)
            else:
                alpha = 1.0 / (edge.observations + 1)
                edge.weight = float(
                    validated_scalar(
                        (1 - alpha) * edge.weight + alpha * value,
                        name="weight", low=-1.0, high=1.0,
                    )
                )
                edge.observations += 1
                if reported_by and str(reported_by)[:60] not in edge.sources_seen:
                    edge.sources_seen.append(str(reported_by)[:60])
                edge.confidence = self._observational_confidence(edge)
                edge.last_confirmed = time.time()
        return self._save()

    @staticmethod
    def _observational_confidence(edge: CausalEdge) -> float:
        """Confidence from evidence, discounted by disconfirmation.

        Observation alone is capped BELOW the prompt threshold: no number of
        correlations makes a claim causal, which is the point of the module.
        """
        effective = math.sqrt(edge.observations * edge.independent_sources)
        raw = 1.0 - math.exp(-0.12 * effective)
        penalty = 0.85 ** edge.disconfirmations
        return float(min(PROMPT_MIN_CONFIDENCE - 0.01, raw * penalty))

    def disconfirm(self, source: str, target: str) -> bool:
        """A prediction based on this edge failed. Weaken it.

        CP126 38cd93d1: this shrank only the weight, so a ``causes`` edge could
        stay maximally confident and keep being stated as proven after
        repeated failures.
        """
        source = sanitize_node_name(source)
        target = sanitize_node_name(target)
        with self._lock:
            edge = self._find(source, target)
            if edge is None:
                return False
            edge.weight *= 0.8
            edge.disconfirmations += 1
            edge.last_disconfirmed = time.time()
            edge.confidence = float(validated_unit(edge.confidence * 0.6, name="confidence"))
            if edge.is_causal and edge.disconfirmations >= DISCONFIRMATIONS_TO_DOWNGRADE:
                logger.warning(
                    "Downgrading causal claim %s -> %s after %d disconfirmations",
                    edge.source, edge.target, edge.disconfirmations,
                )
                edge.relationship = "correlates_with"
                edge.confidence = min(edge.confidence, PROMPT_MIN_CONFIDENCE - 0.01)
        return self._save()

    def record_intervention(
        self,
        source: str,
        target: str,
        receipt: InterventionReceipt,
    ) -> bool:
        """Record a do(source) experiment against its control.

        CP126 03bbcb71: the old method performed, authorized and verified
        nothing — it trusted four caller values and unconditionally wrote
        ``causes``. CP126 a2ade8b4: it used the target LEVEL as the weight
        instead of the treatment effect, so sign and magnitude were invalid.

        Honest bound: this still trusts the caller to have run the experiment.
        What it now requires is a control outcome and
        MIN_INTERVENTIONS_FOR_CAUSAL replications before it will state
        causation — so a single unreplicated assertion cannot mint established
        world knowledge.
        """
        source = sanitize_node_name(source)
        target = sanitize_node_name(target)
        if not source or not target or source == target:
            return False
        if not isinstance(receipt, InterventionReceipt):
            record_degradation(
                "causal_world_model",
                TypeError("intervention without a receipt"),
                action="refused a causal upgrade that carried no intervention receipt",
                severity="warning",
            )
            return False

        source_value = float(validated_unit(receipt.source_value, name="source_value"))
        treated = float(
            validated_scalar(receipt.treated_outcome, name="treated", low=-1.0, high=1.0)
        )
        control = float(
            validated_scalar(receipt.control_outcome, name="control", low=-1.0, high=1.0)
        )
        # The treatment effect, signed by the direction the source was pushed.
        effect = treated - control
        implied = effect if source_value > 0.5 else -effect
        implied = float(validated_scalar(implied, name="effect", low=-1.0, high=1.0))

        with self._lock:
            self.get_node(source)
            self.get_node(target)
            edge = self._find(source, target)
            if edge is None:
                if len(self.edges) >= MAX_EDGES:
                    self._evict_weakest()
                edge = CausalEdge(source=source, target=target, weight=0.0, confidence=0.0)
                self.edges.append(edge)

            edge.intervention_count += 1
            edge.interventions.append(receipt.to_dict())
            del edge.interventions[:-10]
            alpha = 1.0 / edge.intervention_count
            edge.weight = float(
                validated_scalar(
                    (1 - alpha) * edge.weight + alpha * implied,
                    name="weight", low=-1.0, high=1.0,
                )
            )
            if edge.intervention_count >= MIN_INTERVENTIONS_FOR_CAUSAL:
                edge.relationship = "causes"
                edge.confidence = float(
                    validated_unit(
                        min(1.0, 0.5 + 0.15 * edge.intervention_count), name="confidence"
                    )
                )
            else:
                edge.confidence = max(edge.confidence, 0.45)
            edge.last_confirmed = time.time()
        return self._save()

    def discover_causality_via_intervention(
        self,
        source: str,
        target: str,
        source_val: float,
        target_val_observed: float,
        control_val: float | None = None,
        *,
        performed_by: str = "",
        environment: str = "",
    ) -> bool:
        """Backwards-compatible wrapper over :meth:`record_intervention`.

        Without a ``control_val`` there is no treatment effect to estimate, so
        the call is recorded as an OBSERVATION rather than silently minting a
        causal claim from a single level reading.
        """
        if control_val is None:
            logger.info(
                "Intervention for %s -> %s carried no control outcome; recording "
                "it as an observation, not a causal upgrade.",
                source, target,
            )
            return self.add_observation(
                source, target, target_val_observed,
                reported_by=performed_by or "intervention",
            )
        return self.record_intervention(
            source,
            target,
            InterventionReceipt(
                source_value=source_val,
                treated_outcome=target_val_observed,
                control_outcome=control_val,
                performed_by=performed_by,
                environment=environment,
            ),
        )

    # -- queries ---------------------------------------------------------
    def predict_effects(
        self,
        source_id: str,
        *,
        min_confidence: float = 0.3,
        causal_only: bool = False,
    ) -> List[Tuple[str, float]]:
        """Effects predicted for a cause, filtered by evidence.

        CP126 751bc489: this selected any positive weight above 0.3 and
        ignored relationship, confidence, disconfirmation and negative
        effects, while its docstring described predictions given a cause.
        """
        source_id = sanitize_node_name(source_id)
        floor = float(validated_unit(min_confidence, name="min_confidence"))
        with self._lock:
            predictions = [
                (edge.target, edge.weight)
                for edge in self.edges
                if edge.source == source_id
                and abs(edge.weight) > 0.3
                and edge.confidence >= floor
                and (edge.is_causal or not causal_only)
            ]
        return sorted(predictions, key=lambda item: abs(item[1]), reverse=True)

    def predict_effects_detailed(self, source_id: str) -> List[Dict[str, Any]]:
        """The same query with the evidence attached, for callers that judge."""
        source_id = sanitize_node_name(source_id)
        with self._lock:
            return sorted(
                (
                    {
                        "target": edge.target,
                        "weight": edge.weight,
                        "confidence": edge.confidence,
                        "relationship": edge.relationship,
                        "observations": edge.observations,
                        "interventions": edge.intervention_count,
                        "disconfirmations": edge.disconfirmations,
                    }
                    for edge in self.edges
                    if edge.source == source_id
                ),
                key=lambda row: abs(row["weight"]),
                reverse=True,
            )

    def simulate_counterfactual(
        self, do_interventions: Dict[str, float], steps: int = 3
    ) -> Dict[str, float]:
        """Run a counterfactual SCM simulation using do-calculus.

        CP126 04afeae8: each step ADDED the same source influence to the prior
        target state, so a static structural equation saturated and the result
        depended on an arbitrary step count. Each step now recomputes every
        non-intervened node from its parents' previous-step values — a
        simultaneous update — and iteration stops when the state converges.
        """
        step_count, _ = validated_int(
            steps, name="steps", low=1, high=MAX_SIMULATION_STEPS, default=3
        )
        with self._lock:
            state = {name: node.activation for name, node in self.nodes.items()}
            baselines = dict(state)
            edges = list(self.edges)

        intervened: Dict[str, float] = {}
        for key, value in (do_interventions or {}).items():
            clean = sanitize_node_name(key)
            if clean:
                intervened[clean] = float(validated_unit(value, name=f"do({clean})"))
        state.update(intervened)

        parents: Dict[str, List[CausalEdge]] = {}
        for edge in edges:
            if edge.is_causal and edge.target not in intervened:
                parents.setdefault(edge.target, []).append(edge)

        for _ in range(step_count):
            next_state = dict(state)
            for target, inbound in parents.items():
                # Simultaneous: every parent contributes from the PREVIOUS
                # state, and the node is recomputed rather than accumulated.
                influence = sum(
                    state.get(edge.source, 0.0) * edge.weight * edge.confidence
                    for edge in inbound
                )
                next_state[target] = float(
                    validated_unit(baselines.get(target, 0.0) + influence, name=target)
                )
            next_state.update(intervened)
            if all(
                abs(next_state[key] - state.get(key, 0.0)) < 1e-6 for key in next_state
            ):
                state = next_state
                break
            state = next_state
        return state

    def analyze_preventative_actions(self, undesirable_node: str) -> List[Tuple[str, float]]:
        """Nodes that negatively influence the undesirable node."""
        undesirable_node = sanitize_node_name(undesirable_node)
        with self._lock:
            preventers = [
                (edge.source, edge.weight)
                for edge in self.edges
                if edge.target == undesirable_node
                and edge.weight < -0.2
                and edge.confidence > 0.3
            ]
        return sorted(preventers, key=lambda item: item[1])

    def get_prompt_context(self) -> str:
        """The strongest PROVEN causal rules, for prompt injection.

        CP126 a45e3568: this filtered on confidence and weight but not on
        relationship, so repeated observational correlations were rendered
        under "ESTABLISHED WORLD CASCADES" with directional language despite
        never having received an intervention.
        """
        with self._lock:
            strong = [
                edge
                for edge in self.edges
                if edge.is_causal
                and edge.confidence > PROMPT_MIN_CONFIDENCE
                and abs(edge.weight) > PROMPT_MIN_ABS_WEIGHT
                and edge.disconfirmations < DISCONFIRMATIONS_TO_DOWNGRADE
            ]
        if not strong:
            return ""

        lines = []
        for edge in sorted(strong, key=lambda e: e.confidence, reverse=True)[:PROMPT_MAX_RULES]:
            effect = "INCREASES" if edge.weight > 0 else "DECREASES"
            # Names are re-sanitized at render time: persisted state may
            # predate the ingress sanitizer.
            source = sanitize_node_name(edge.source)
            target = sanitize_node_name(edge.target)
            lines.append(
                f"- [{source}] {effect} [{target}] "
                f"(confidence {edge.confidence:.2f}, {edge.intervention_count} interventions)"
            )
        return (
            "\n### ESTABLISHED WORLD CASCADES (intervention-tested)\n"
            + "\n".join(lines)
            + "\n"
        )

    # -- housekeeping ----------------------------------------------------
    def _find(self, source: str, target: str) -> Optional[CausalEdge]:
        return next(
            (e for e in self.edges if e.source == source and e.target == target), None
        )

    def _evict_weakest(self) -> None:
        """Drop the least-supported edge and any node it orphans."""
        if self.edges:
            weakest = min(
                self.edges,
                key=lambda e: (e.is_causal, e.confidence, abs(e.weight), e.observations),
            )
            self.edges.remove(weakest)
            logger.info(
                "Causal graph at quota; evicted %s -> %s", weakest.source, weakest.target
            )
        referenced = {e.source for e in self.edges} | {e.target for e in self.edges}
        for name in [n for n in self.nodes if n not in referenced]:
            if len(self.nodes) < MAX_NODES:
                break
            del self.nodes[name]

    def status(self) -> Dict[str, Any]:
        with self._lock:
            causal = sum(1 for e in self.edges if e.is_causal)
            return {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "causal_edges": causal,
                "correlational_edges": len(self.edges) - causal,
                "last_save_ok": self.last_save_ok,
                "last_save_error": self.last_save_error,
                "load_quarantined": self.load_quarantined,
                "schema_version": SCHEMA_VERSION,
            }

    # -- persistence -----------------------------------------------------
    def _save(self) -> bool:
        """Persist the graph. Returns whether the write actually landed.

        CP126 ed3c0893: mutation methods changed state and then called a
        ``_save`` that swallowed failures and returned nothing, so callers
        treated learning as durable when the file was never written.
        """
        try:
            with self._lock:
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "written_at": time.time(),
                    "nodes": {k: asdict(v) for k, v in self.nodes.items()},
                    "edges": [asdict(e) for e in self.edges],
                }
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().write_text(
                self.data_path,
                json.dumps(payload, indent=4, default=str),
                source="causal_world_model.save",
            )
        except _CWM_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "causal_world_model",
                exc,
                action="reported a non-durable causal-graph update to the caller",
                severity="error",
            )
            logger.error("Failed to save Causal World Model: %s", exc)
            self.last_save_ok = False
            self.last_save_error = f"{type(exc).__name__}: {exc}"
            return False
        self.last_save_ok = True
        self.last_save_error = ""
        return True

    def _seed(self) -> None:
        """Baseline digital intuition — recorded as correlations, not laws."""
        for source, target, weight in (
            ("high cpu usage", "system lag", 0.9),
            ("sandbox violation", "orchestrator crash", 0.95),
            ("unclear prompt", "hallucination", 0.7),
            ("deep dreaming", "memory consolidation", 0.8),
        ):
            self.add_observation(source, target, weight, reported_by="seed")

    def _load(self) -> None:
        """Load and VALIDATE the persisted graph.

        CP126 a730f5b8: only OSError/ConnectionError/TimeoutError were caught,
        so a JSON error or an unexpected dataclass field raised during service
        construction and could abort startup. CP126 7a91d00d: raw dataclass
        dictionaries were trusted with no schema, bounds or referential check.
        """
        if not self.data_path.exists():
            self._seed()
            return

        try:
            raw = self.data_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("causal world payload is not an object")
        except _CWM_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "causal_world_model",
                exc,
                action="quarantined an unreadable causal graph and reseeded baselines",
                severity="error",
            )
            logger.error("Failed to load Causal World Model: %s", exc)
            self.load_quarantined = True
            self._quarantine()
            self._seed()
            return

        nodes: Dict[str, CausalNode] = {}
        for key, value in (data.get("nodes") or {}).items():
            node = self._node_from(key, value)
            if node is not None:
                nodes[node.name] = node

        edges: List[CausalEdge] = []
        seen: set[tuple[str, str]] = set()
        for value in (data.get("edges") or []):
            edge = self._edge_from(value)
            if edge is None:
                continue
            pair = (edge.source, edge.target)
            if pair in seen:
                continue  # duplicate policy: first wins
            # Referential integrity: an edge endpoint must be a real node.
            nodes.setdefault(edge.source, CausalNode(name=edge.source))
            nodes.setdefault(edge.target, CausalNode(name=edge.target))
            seen.add(pair)
            edges.append(edge)

        self.nodes = dict(list(nodes.items())[:MAX_NODES])
        self.edges = edges[:MAX_EDGES]
        if int(data.get("schema_version", 0) or 0) != SCHEMA_VERSION:
            logger.info(
                "Causal graph loaded from schema %s (current %s); revalidated on load.",
                data.get("schema_version", "legacy"), SCHEMA_VERSION,
            )
        if not self.edges:
            self._seed()

    def _quarantine(self) -> None:
        try:
            corrupt = self.data_path.with_suffix(f".corrupt.{int(time.time())}.json")
            self.data_path.rename(corrupt)
            logger.error("Quarantined the unreadable causal graph at %s", corrupt)
        except OSError as exc:
            logger.error("Could not quarantine the causal graph: %s", exc)

    @staticmethod
    def _node_from(key: Any, value: Any) -> Optional[CausalNode]:
        if not isinstance(value, dict):
            return None
        name = sanitize_node_name(value.get("name", key))
        if not name:
            return None
        return CausalNode(
            name=name,
            activation=float(validated_unit(value.get("activation", 0.0), name="activation")),
            variance=float(validated_unit(value.get("variance", 0.1), name="variance")),
        )

    @staticmethod
    def _edge_from(value: Any) -> Optional[CausalEdge]:
        """Construct an edge from persisted data without trusting it.

        CP126 7a646f7b: loading expanded dictionaries through the dataclass
        constructor, so an edge missing ``confidence`` inherited the 1.0
        default and became a proven fact by omission.
        """
        if not isinstance(value, dict):
            return None
        source = sanitize_node_name(value.get("source"))
        target = sanitize_node_name(value.get("target"))
        if not source or not target or source == target:
            return None
        relationship = str(value.get("relationship") or "correlates_with")
        if relationship not in {"causes", "correlates_with"}:
            relationship = "correlates_with"
        observations, _ = validated_int(
            value.get("observations", 1), name="observations", low=0, high=10**7, default=1
        )
        interventions, _ = validated_int(
            value.get("intervention_count", 0), name="interventions",
            low=0, high=10**6, default=0,
        )
        disconfirmations, _ = validated_int(
            value.get("disconfirmations", 0), name="disconfirmations",
            low=0, high=10**6, default=0,
        )
        # A persisted `causes` with no recorded intervention is downgraded: the
        # claim's own evidence has to be present in the file.
        if relationship == "causes" and interventions < MIN_INTERVENTIONS_FOR_CAUSAL:
            relationship = "correlates_with"
        raw_sources = value.get("sources_seen")
        sources = (
            [str(item)[:60] for item in raw_sources][:32]
            if isinstance(raw_sources, list) else []
        )
        raw_interventions = value.get("interventions")
        receipts = (
            [item for item in raw_interventions if isinstance(item, dict)][-10:]
            if isinstance(raw_interventions, list) else []
        )
        return CausalEdge(
            source=source,
            target=target,
            relationship=relationship,
            weight=float(
                validated_scalar(value.get("weight", 0.0), name="weight", low=-1.0, high=1.0)
            ),
            # Absent confidence means UNKNOWN, which is 0.0, not proven.
            confidence=float(validated_unit(value.get("confidence", 0.0), name="confidence")),
            observations=observations,
            intervention_count=interventions,
            disconfirmations=disconfirmations,
            sources_seen=sources,
            interventions=receipts,
            last_confirmed=float(
                validated_scalar(value.get("last_confirmed", 0.0), name="last_confirmed", low=0.0)
            ),
            last_disconfirmed=(
                float(
                    validated_scalar(
                        value["last_disconfirmed"], name="last_disconfirmed", low=0.0
                    )
                )
                if value.get("last_disconfirmed") is not None
                else None
            ),
        )


def register_causal_world_model(orchestrator=None) -> CausalWorldModel:
    """Register the singleton causal world model.

    CP126 d2d5130d: this constructed and loaded a NEW model on every call and
    registered it unconditionally, so multiple callers split state and raced
    on the same file.
    """
    existing = get_runtime_service("causal_world_model", default=None)
    # Short-circuit on absence before touching isinstance. The reuse check used
    # to evaluate isinstance() unconditionally against the module-global symbol,
    # so it raised TypeError whenever nothing was registered AND that symbol had
    # been rebound to a factory (as dependency-injection and test harnesses do)
    # — turning "construct the first instance" into a hard failure.
    if existing is not None:
        # Only a real class can be an isinstance() argument; if the symbol has
        # been substituted, fall back to identifying the singleton by the
        # interface callers actually rely on.
        if isinstance(CausalWorldModel, type):
            if isinstance(existing, CausalWorldModel):
                return existing
        elif hasattr(existing, "simulate") or hasattr(existing, "name"):
            return existing
    model = CausalWorldModel()
    register_runtime_service(
        "causal_world_model",
        model,
        owner="core/brain/causal_world_model.py",
        registered_by="register_causal_world_model",
    )
    return model
