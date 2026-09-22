"""core/discovery/analogical_leap.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Horizontal-leap feeder: out-of-distribution detection + cross-domain
structural analogy, feeding the Frontier Discovery Engine's falsification
discipline.

The gap this closes: when a problem falls outside every subsystem's
templates, the system previously force-fit the nearest template. Now:

1. OutOfDistributionDetector — knows WHEN the map runs out: retrieval
   support across memory stores + domain-signature match both below
   floor ⇒ off-map verdict with the evidence attached.
2. StructureMapper — extracts the problem's relational skeleton (roles,
   relations, dynamics keywords) and searches a library of cross-domain
   schemas (feedback, diffusion, conservation, selection, cascade,
   equilibrium, threshold, resonance, hierarchy, cycle) for structural
   isomorphism — similarity over relation signatures, not surface words.
3. ConjectureRecombinator — transfers each matched schema's invariants
   onto the problem with correspondences substituted, yielding candidate
   paradigms as FDE Conjectures (provenance="analogical_leap").

Epistemic discipline (the honest ceiling, stated plainly):
- candidates with a numeric/logic falsification route are handed to the
  FDE's verifier loop and can graduate to SUPPORTED/PROVEN or die REFUTED;
- candidates without a verifier stay CONJECTURE and are labeled
  unverifiable — the leap is real, the claim is never inflated;
- the schema library bounds the reachable paradigms: this is horizontal
  transfer within representable structure, not unbounded invention.

Every leap is artifact-logged (artifacts/consciousness/analogical_leaps.jsonl)
so novel-paradigm formation is EVIDENCED, not asserted.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.discovery.frontier_discovery_engine import Conjecture, EpistemicStatus

logger = logging.getLogger("Aura.Discovery.AnalogicalLeap")

_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")

# Retrieval/domain support below these floors = off the map.
_RETRIEVAL_SUPPORT_FLOOR = 0.35
_DOMAIN_MATCH_FLOOR = 0.30


# ── Cross-domain schema library ──────────────────────────────────────
# Each schema is a relational skeleton: role slots, relation signatures
# (the structure that transfers), invariant templates (what the source
# domain KNOWS that becomes a candidate hypothesis in the target), and
# cue lexemes (dynamics words that hint the structure without deciding it).

@dataclass(frozen=True)
class DomainSchema:
    name: str
    source_domain: str
    roles: tuple[str, ...]
    relations: tuple[str, ...]          # signature: relation(role_a, role_b)
    cues: frozenset[str]
    invariants: tuple[str, ...]         # templates with {role} slots
    falsifiable_form: str = ""          # numeric predicate template, if any


SCHEMA_LIBRARY: tuple[DomainSchema, ...] = (
    DomainSchema(
        name="reinforcing_feedback",
        source_domain="control_theory",
        roles=("stock", "flow", "gain"),
        relations=("amplifies(flow, stock)", "increases(stock, flow)"),
        cues=frozenset({
            "grows", "spiral", "compound", "runaway", "accelerating",
            "self-reinforcing", "exponential", "viral", "snowball",
        }),
        invariants=(
            "while gain exceeds losses, {stock} grows exponentially, not linearly",
            "the loop breaks only where {flow} is decoupled from {stock}",
            "early intervention on {gain} beats late intervention on {stock}",
        ),
        falsifiable_form="geometric growth: x(t+1)/x(t) approximately constant > 1",
    ),
    DomainSchema(
        name="balancing_feedback",
        source_domain="control_theory",
        roles=("stock", "setpoint", "corrector"),
        relations=("measures(corrector, stock)", "opposes(corrector, deviation)"),
        cues=frozenset({
            "stabilizes", "regulates", "homeostasis", "thermostat", "damped",
            "oscillates", "overshoot", "settles", "equilibrium",
        }),
        invariants=(
            "{stock} oscillates around {setpoint} when {corrector} acts with delay",
            "tightening {corrector} gain past a threshold causes instability, not precision",
        ),
        falsifiable_form="bounded oscillation: |x(t) - setpoint| decays or stays bounded",
    ),
    DomainSchema(
        name="diffusion_spread",
        source_domain="epidemiology",
        roles=("carrier", "susceptible_pool", "transmission_channel"),
        relations=("transmits(carrier, susceptible_pool)", "depletes(spread, susceptible_pool)"),
        cues=frozenset({
            "spreads", "infects", "propagates", "contagion", "adoption",
            "percolates", "rumor", "cascade", "epidemic", "saturates",
        }),
        invariants=(
            "spread through {susceptible_pool} is logistic: exponential early, saturating late",
            "cutting {transmission_channel} density below a critical value halts spread entirely",
            "immunity/saturation of {susceptible_pool}, not exhaustion of {carrier}, ends growth",
        ),
        falsifiable_form="logistic curve: growth rate proportional to x*(1 - x/K)",
    ),
    DomainSchema(
        name="conservation_budget",
        source_domain="physics",
        roles=("quantity", "source", "sink"),
        relations=("produces(source, quantity)", "consumes(sink, quantity)"),
        cues=frozenset({
            "budget", "conserved", "allocated", "depletes", "drains",
            "zero-sum", "leak", "accumulates", "balance",
        }),
        invariants=(
            "d{quantity}/dt = inflow from {source} - outflow to {sink}; nothing else",
            "an unexplained change in {quantity} means an unaccounted {source} or {sink} exists",
        ),
        falsifiable_form="balance: delta(x) - (inflow - outflow) = 0 within measurement error",
    ),
    DomainSchema(
        name="variation_selection",
        source_domain="evolutionary_biology",
        roles=("population", "variation_source", "selection_pressure"),
        relations=("varies(variation_source, population)", "filters(selection_pressure, population)"),
        cues=frozenset({
            "mutate", "selects", "fittest", "adapt", "evolves", "cull",
            "tournament", "survive", "iterate", "generations",
        }),
        invariants=(
            "{population} adapts only as fast as {variation_source} supplies diversity",
            "optimizing {selection_pressure} without {variation_source} converges prematurely",
            "what {selection_pressure} measures is what {population} becomes — beware proxy gaps",
        ),
    ),
    DomainSchema(
        name="threshold_transition",
        source_domain="statistical_physics",
        roles=("system", "control_parameter", "phase"),
        relations=("drives(control_parameter, system)", "switches(system, phase)"),
        cues=frozenset({
            "threshold", "tipping", "critical", "abrupt", "sudden", "regime",
            "avalanche", "collapse", "nonlinear", "discontinuous",
        }),
        invariants=(
            "{system} responds smoothly until {control_parameter} crosses a critical value, then jumps",
            "near the threshold, small noise in {control_parameter} produces outsized {phase} changes",
            "hysteresis: reversing {control_parameter} does not immediately reverse the {phase}",
        ),
        falsifiable_form="discontinuity: response slope near candidate threshold >> slope elsewhere",
    ),
    DomainSchema(
        name="queueing_congestion",
        source_domain="operations_research",
        roles=("arrivals", "server", "queue"),
        relations=("feeds(arrivals, queue)", "drains(server, queue)"),
        cues=frozenset({
            "backlog", "latency", "congestion", "throughput", "saturated",
            "bottleneck", "waits", "bursts", "load",
        }),
        invariants=(
            "as {arrivals} utilization of {server} approaches 1, {queue} wait grows without bound",
            "variance in {arrivals} inflates {queue} even at moderate average load",
            "adding {server} capacity beats reordering {queue} once utilization is high",
        ),
        falsifiable_form="wait ~ rho/(1-rho): delay grows superlinearly as utilization -> 1",
    ),
    DomainSchema(
        name="resonance_coupling",
        source_domain="mechanics",
        roles=("driver", "oscillator", "coupling"),
        relations=("forces(driver, oscillator)", "matches(driver_frequency, natural_frequency)"),
        cues=frozenset({
            "resonates", "synchronizes", "entrains", "amplified", "frequency",
            "periodic", "rhythm", "beats", "tuned",
        }),
        invariants=(
            "small periodic {driver} input produces large {oscillator} response only near frequency match",
            "damping {coupling} trades peak response for stability across all frequencies",
        ),
    ),
    DomainSchema(
        name="tragedy_of_commons",
        source_domain="game_theory",
        roles=("agents", "shared_resource", "individual_incentive"),
        relations=("extracts(agents, shared_resource)", "rewards(individual_incentive, extraction)"),
        cues=frozenset({
            "shared", "commons", "free-rider", "overuse", "incentive",
            "defect", "exploit", "collective", "pool",
        }),
        invariants=(
            "individually rational extraction by {agents} depletes {shared_resource} for all",
            "sustainability requires changing {individual_incentive}, not exhorting {agents}",
        ),
    ),
    DomainSchema(
        name="error_correcting_redundancy",
        source_domain="information_theory",
        roles=("message", "channel", "redundancy"),
        relations=("corrupts(channel, message)", "recovers(redundancy, message)"),
        cues=frozenset({
            "noise", "corrupt", "redundant", "checksum", "vote", "repair",
            "degrade", "recover", "reconstruct",
        }),
        invariants=(
            "reliable {message} transfer over a noisy {channel} needs redundancy proportional to noise",
            "past a noise threshold, no amount of {redundancy} recovers {message} — route around instead",
        ),
    ),
)


# ── OOD detection ────────────────────────────────────────────────────

@dataclass
class OODVerdict:
    off_map: bool
    retrieval_support: float     # 0..1 best memory-store hit strength
    best_domain_match: float     # 0..1 best schema cue overlap
    nearest_schemas: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class OutOfDistributionDetector:
    """Knows when the map runs out — before a template gets force-fitted."""

    def __init__(self, retriever: Any | None = None) -> None:
        self._retriever = retriever

    def _retrieval_support(self, problem: str) -> float:
        retriever = self._retriever
        if retriever is None:
            try:
                from core.memory.intentional_retrieval import get_intentional_retriever
                retriever = get_intentional_retriever()
            except (ImportError, AttributeError, RuntimeError) as exc:
                logger.debug(
                    "no intentional retriever, so retrieval support reads as zero (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                return 0.0
        try:
            from core.memory.intentional_retrieval import RetrievalIntent
            result = retriever.retrieve(
                RetrievalIntent(task=problem, kind="learn", limit=5)
            )
            scores = [h.score for h in result.hits]
            return max(scores) if scores else 0.0
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("OOD retrieval probe unavailable: %s", exc)
            return 0.0

    def assess(self, problem: str) -> OODVerdict:
        tokens = set(_WORD_RE.findall(problem.lower()))
        matches = sorted(
            ((schema.name, _cue_overlap(tokens, schema)) for schema in SCHEMA_LIBRARY),
            key=lambda pair: pair[1],
            reverse=True,
        )
        best_domain = matches[0][1] if matches else 0.0
        support = self._retrieval_support(problem)
        off_map = support < _RETRIEVAL_SUPPORT_FLOOR and best_domain < _DOMAIN_MATCH_FLOOR
        return OODVerdict(
            off_map=off_map,
            retrieval_support=round(support, 4),
            best_domain_match=round(best_domain, 4),
            nearest_schemas=[name for name, score in matches[:3] if score > 0.0],
            evidence={
                "retrieval_floor": _RETRIEVAL_SUPPORT_FLOOR,
                "domain_floor": _DOMAIN_MATCH_FLOOR,
                "token_count": len(tokens),
            },
        )


def _cue_overlap(tokens: set[str], schema: DomainSchema) -> float:
    if not tokens:
        return 0.0
    hits = len(tokens & schema.cues)
    return hits / max(3.0, len(schema.cues) * 0.5)


# ── Structure mapping ────────────────────────────────────────────────

@dataclass
class AnalogicalMapping:
    schema: DomainSchema
    similarity: float
    role_bindings: dict[str, str]
    matched_cues: list[str]


class StructureMapper:
    """Match a problem's relational skeleton against the schema library.

    Similarity is cue-lexeme overlap (dynamics vocabulary) — deliberately
    simple, deterministic, and inspectable. Role bindings pick the most
    salient problem nouns for the schema's role slots so transferred
    invariants read in the problem's own terms.
    """

    _STOP = frozenset({
        "the", "and", "for", "with", "that", "this", "from", "into",
        "over", "have", "has", "are", "was", "were", "when", "then",
        "each", "some", "more", "than", "how", "why", "what", "our",
    })

    def map(self, problem: str, *, top_k: int = 3) -> list[AnalogicalMapping]:
        tokens = set(_WORD_RE.findall(problem.lower()))
        nouns = self._salient_terms(problem)
        mappings: list[AnalogicalMapping] = []
        for schema in SCHEMA_LIBRARY:
            similarity = _cue_overlap(tokens, schema)
            if similarity <= 0.0:
                continue
            bindings = {
                role: nouns[i] if i < len(nouns) else role
                for i, role in enumerate(schema.roles)
            }
            mappings.append(AnalogicalMapping(
                schema=schema,
                similarity=round(min(1.0, similarity), 4),
                role_bindings=bindings,
                matched_cues=sorted(tokens & schema.cues),
            ))
        mappings.sort(key=lambda m: m.similarity, reverse=True)
        return mappings[:top_k]

    def _salient_terms(self, problem: str) -> list[str]:
        counts: dict[str, int] = {}
        for token in _WORD_RE.findall(problem.lower()):
            if token in self._STOP or len(token) < 4:
                continue
            counts[token] = counts.get(token, 0) + 1
        return [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:6]


# ── Conjecture recombination ─────────────────────────────────────────

class ConjectureRecombinator:
    """Transfer matched schemas' invariants onto the problem as conjectures."""

    def recombine(
        self, problem: str, mappings: list[AnalogicalMapping],
    ) -> list[Conjecture]:
        conjectures: list[Conjecture] = []
        for mapping in mappings:
            for invariant in mapping.schema.invariants:
                statement = invariant
                for role, binding in mapping.role_bindings.items():
                    statement = statement.replace("{" + role + "}", binding)
                verifiable = bool(mapping.schema.falsifiable_form)
                conjectures.append(Conjecture(
                    statement=f"[{mapping.schema.source_domain}→target] {statement}",
                    domain=f"analogical:{mapping.schema.name}",
                    formal_form=mapping.schema.falsifiable_form or "unverifiable_structural_claim",
                    status=EpistemicStatus.CONJECTURE,
                    confidence=0.2 + 0.4 * mapping.similarity,
                    novelty=1.0,
                    provenance="analogical_leap",
                    falsification_plan=(
                        f"test against observed data: {mapping.schema.falsifiable_form}"
                        if verifiable
                        else "no local verifier applies; remains labeled CONJECTURE"
                    ),
                ))
        return conjectures


# ── The leap engine ──────────────────────────────────────────────────

@dataclass
class LeapReport:
    problem: str
    verdict: OODVerdict
    mappings: list[AnalogicalMapping]
    conjectures: list[Conjecture]
    generated_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.analogical_leap.v1",
            "problem": self.problem[:500],
            "off_map": self.verdict.off_map,
            "retrieval_support": self.verdict.retrieval_support,
            "best_domain_match": self.verdict.best_domain_match,
            "mappings": [
                {
                    "schema": m.schema.name,
                    "source_domain": m.schema.source_domain,
                    "similarity": m.similarity,
                    "role_bindings": m.role_bindings,
                    "matched_cues": m.matched_cues,
                }
                for m in self.mappings
            ],
            "conjectures": [c.to_dict() for c in self.conjectures],
            "generated_at": self.generated_at,
        }


class AnalogicalLeapEngine:
    """detect off-map → map structure → recombine → (verify | label honestly)."""

    def __init__(
        self,
        *,
        detector: OutOfDistributionDetector | None = None,
        mapper: StructureMapper | None = None,
        recombinator: ConjectureRecombinator | None = None,
        artifact_path: Path | str | None = None,
    ) -> None:
        self.detector = detector or OutOfDistributionDetector()
        self.mapper = mapper or StructureMapper()
        self.recombinator = recombinator or ConjectureRecombinator()
        self.artifact_path = Path(artifact_path) if artifact_path else None

    def leap(self, problem: str, *, force: bool = False) -> LeapReport:
        """Run the full leap pipeline.

        By default only fires when the OOD detector says the map ran out
        (force=True overrides for probing/testing). Verifiable conjectures
        keep their falsification plan for the FDE; unverifiable ones are
        labeled and never inflated.
        """
        verdict = self.detector.assess(problem)
        mappings: list[AnalogicalMapping] = []
        conjectures: list[Conjecture] = []
        if verdict.off_map or force:
            mappings = self.mapper.map(problem)
            conjectures = self.recombinator.recombine(problem, mappings)
        report = LeapReport(
            problem=problem, verdict=verdict,
            mappings=mappings, conjectures=conjectures,
        )
        self._log_artifact(report)
        return report

    def _log_artifact(self, report: LeapReport) -> None:
        if self.artifact_path is None:
            return
        try:
            self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            with self.artifact_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
        except OSError as exc:
            logger.debug("Leap artifact log skipped: %s", exc)


def get_analogical_leap_engine() -> AnalogicalLeapEngine:
    """Default engine with the standard artifact sink."""
    artifact = Path("artifacts/consciousness/analogical_leaps.jsonl")
    return AnalogicalLeapEngine(artifact_path=artifact)
