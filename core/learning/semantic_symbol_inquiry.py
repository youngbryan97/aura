"""Acquire proposed symbol meanings from source-identified demonstrations.

The hypotheses are caller-proposed contextual frame dictionaries. Experiments
compare their executable predictions; only independently supplied observations
revise the dictionary. Remaining hypotheses are not automatically true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from types import MappingProxyType

from scipy.special import logsumexp

from core.cognition.the_experiment_that_settles_it import what_it_ruled_out, what_to_try
from core.learning.procedure_induction import PRIMITIVES_BY_NAME
from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_program_ir import normalize_semantic_value
from core.learning.semantic_semasiographic import (
    MeaningDiagram,
    MeaningFrame,
    digest,
    execute_meaning_diagram,
    source_key,
)
from core.runtime.gateways import StateGateway


@dataclass(frozen=True)
class SymbolHypothesis:
    identity: str
    frames: tuple[MeaningFrame, ...]
    context: str

    def __post_init__(self):
        frames = tuple(self.frames)
        if (not isinstance(self.identity, str) or not self.identity or not self.context
                or not frames or any(not isinstance(f, MeaningFrame) for f in frames)
                or len({(f.world, f.symbol) for f in frames}) != len(frames)):
            raise ValueError("symbol hypothesis needs distinct contextual frame meanings")
        object.__setattr__(self, "frames", frames)

    @property
    def interpretations(self):
        return {(f.world, f.symbol): f for f in self.frames}

    def to_dict(self):
        return {"id": self.identity, "context": self.context,
            "frames": [{"symbol": f.symbol, "roles": f.roles, "operation": f.operation,
                        "output_type": f.output_type, "world": f.world} for f in self.frames]}

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["id"], tuple(MeaningFrame(**{**f,
            "roles": tuple(tuple(r) for r in f["roles"])}) for f in payload["frames"]), payload["context"])


def propose_floor_symbol_meanings(symbol, ports, output_type, context, *, world="actual", limit=256):
    """Enumerate typed denotations and port permutations from the existing floor.

    This is an explicit finite ontology, not an inference from a glyph's shape.
    Exhausted capacity refuses a truncated hypothesis space.
    """
    ports = tuple(tuple(p) for p in ports)
    if (not ports or len(ports) > 8
            or any(len(p) != 2 or any(not isinstance(v, str) or not v for v in p) for p in ports)
            or len({p[0] for p in ports}) != len(ports)
            or type(limit) is not int or not 0 < limit <= 256):
        raise ValueError("invalid bounded symbol port schema")
    proposals = []
    for operation in sorted(PRIMITIVES_BY_NAME):
        signature = semantic_primitive_type_signature(operation)
        if signature is None or signature[1] != output_type or len(signature[0]) != len(ports):
            continue
        for roles in permutations(ports):
            if tuple(t for _, t in roles) == signature[0]:
                frame = MeaningFrame(symbol, roles, output_type, operation, world)
                proposals.append(SymbolHypothesis(digest((context, world, symbol, operation, roles)),
                                                  (frame,), context))
                if len(proposals) > limit:
                    raise ValueError("symbol proposal space exceeds its declared exact bound")
    return tuple(proposals)


@dataclass(frozen=True)
class DemonstrationRequest:
    diagram: MeaningDiagram
    inputs: tuple
    output: tuple[str, str]
    cutoff: int
    context: str
    world: str = "actual"

    def __post_init__(self):
        source_key(self.output)
        inputs = tuple((source_key(k), normalize_semantic_value(v)) for k, v in self.inputs)
        if (not isinstance(self.diagram, MeaningDiagram)
                or any(not isinstance(v, str) or not v for v in (self.context, self.world))
                or type(self.cutoff) is not int or self.cutoff < 0
                or len({k for k, _ in inputs}) != len(inputs)):
            raise ValueError("invalid pending symbol demonstration")
        object.__setattr__(self, "inputs", tuple(sorted(inputs)))

    @property
    def identity(self):
        return digest((self.diagram.custody_sha256, self.inputs, self.output,
                       self.cutoff, self.context, self.world))

    def to_dict(self):
        return {"diagram": self.diagram.to_dict(), "inputs": self.inputs,
                "output": self.output, "cutoff": self.cutoff, "context": self.context, "world": self.world}

    @classmethod
    def from_dict(cls, payload):
        return cls(MeaningDiagram.from_dict(payload["diagram"]),
                   tuple((tuple(k), v) for k, v in payload["inputs"]), tuple(payload["output"]),
                   payload["cutoff"], payload["context"], payload["world"])


@dataclass(frozen=True)
class ObservedDemonstration:
    request: DemonstrationRequest
    outcome: object
    origin: str
    ref: str
    observed_at: int

    def __post_init__(self):
        if (not isinstance(self.request, DemonstrationRequest)
                or any(not isinstance(v, str) or not v for v in (self.origin, self.ref))
                or type(self.observed_at) is not int or self.observed_at < self.request.cutoff):
            raise ValueError("symbol feedback needs identified independent observation custody")
        object.__setattr__(self, "outcome", normalize_semantic_value(self.outcome))

    @property
    def identity(self):
        return digest((self.request.identity, self.outcome, self.origin, self.ref, self.observed_at))

    def to_dict(self):
        return {"request": self.request.to_dict(), "outcome": self.outcome,
                "origin": self.origin, "ref": self.ref, "observed_at": self.observed_at}

    @classmethod
    def from_dict(cls, payload):
        return cls(DemonstrationRequest.from_dict(payload["request"]), payload["outcome"],
                   payload["origin"], payload["ref"], payload["observed_at"])


def predict_symbol_hypothesis(hypothesis, request):
    if hypothesis.context != request.context:
        return None
    try:
        return execute_meaning_diagram(request.diagram, hypothesis.interpretations,
            dict(request.inputs), output=request.output, cutoff=request.cutoff,
            world=request.world).result
    except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
        # not a failure: undefined or unsupported denotation is silence, not a wrong answer.
        return None


class SymbolInquiry:
    def __init__(self, hypotheses, *, prior_weights=None, mismatch_likelihood=0.):
        items = tuple(hypotheses)
        if (not items or len(items) > 256 or any(not isinstance(h, SymbolHypothesis) for h in items)
                or len({h.identity for h in items}) != len(items)):
            raise ValueError("symbol inquiry needs a bounded distinct hypothesis set")
        self.initial = MappingProxyType({h.identity: h for h in sorted(items, key=lambda h: h.identity)})
        self.observations = ()
        weights = {h.identity: 1. for h in items} if prior_weights is None else dict(prior_weights)
        if (set(weights) != set(self.initial) or any(not math.isfinite(v) or v <= 0 for v in weights.values())
                or not math.isfinite(mismatch_likelihood) or not 0 <= mismatch_likelihood < 1):
            raise ValueError("symbol uncertainty needs explicit positive priors and an observation model")
        self.prior_weights = MappingProxyType({k: float(v) for k, v in weights.items()})
        self.mismatch_likelihood = float(mismatch_likelihood)
        self.replay(cutoff=0)

    def next_experiment(self, requests, *, costs=None):
        requests = tuple(requests)
        if len(requests) > 256 or any(not isinstance(r, DemonstrationRequest) for r in requests):
            raise ValueError("symbol inquiry requests exceed the admitted act space")
        if any(r.cutoff > self.cutoff for r in requests):
            raise ValueError("symbol experiment selection cannot read future evidence")
        if costs is not None:
            import math

            prices = {r.identity: float(costs(r)) for r in requests}
            if any(not math.isfinite(v) or v <= 0 for v in prices.values()):
                raise ValueError("symbol inquiry costs must be positive and finite")
            def costs(r):
                return prices[r.identity]
        return what_to_try(self.remaining, requests, predicts=predict_symbol_hypothesis,
            plausibility=lambda name, _hypothesis: self.posterior[name], costs=costs)

    def observe(self, observation, *, cutoff):
        if not isinstance(observation, ObservedDemonstration):
            raise ValueError("predictions cannot be recorded as observed symbol evidence")
        if type(cutoff) is not int or cutoff < max(self.cutoff, observation.observed_at):
            raise ValueError("symbol evidence cannot arrive from a future clock")
        for old in self.observations:
            if (old.origin, old.ref) == (observation.origin, observation.ref):
                if old.identity != observation.identity:
                    raise ValueError("one symbol observation reference cannot contradict itself")
                return self.replay(cutoff=cutoff, retracted=self.retracted)
        self.observations += (observation,)
        return self.replay(cutoff=cutoff, retracted=self.retracted)

    def replay(self, *, cutoff, retracted=()):
        if type(cutoff) is not int or cutoff < 0 or not set(retracted) <= {o.identity for o in self.observations}:
            raise ValueError("invalid symbol evidence replay or retraction")
        remaining = dict(self.initial)
        log_weights = {k: math.log(v) for k, v in self.prior_weights.items()}
        used = []
        for observation in self.observations:
            if observation.observed_at > cutoff or observation.identity in retracted:
                continue
            if self.mismatch_likelihood == 0.:
                remaining = what_it_ruled_out(remaining, observation.request, observation.outcome,
                    predicts=predict_symbol_hypothesis)
            else:
                for identity, hypothesis in remaining.items():
                    prediction = predict_symbol_hypothesis(hypothesis, observation.request)
                    if prediction is not None and prediction != observation.outcome:
                        log_weights[identity] += math.log(self.mismatch_likelihood)
            used.append(observation.identity)
        self.remaining, self.cutoff = MappingProxyType(remaining), cutoff
        self.used, self.retracted = tuple(used), tuple(sorted(retracted))
        normalizer = float(logsumexp([log_weights[k] for k in remaining])) if remaining else 0.
        self.posterior = MappingProxyType({k: math.exp(log_weights[k] - normalizer) for k in remaining})
        return self.receipt()

    def extend(self, hypotheses, *, prior_weight=1.):
        """Propose additional meanings, then replay the same retained observations."""
        additions = tuple(hypotheses)
        if (not additions or any(not isinstance(h, SymbolHypothesis) for h in additions)
                or len({h.identity for h in additions}) != len(additions)
                or set(h.identity for h in additions) & set(self.initial)
                or len(self.initial) + len(additions) > 256
                or not math.isfinite(prior_weight) or prior_weight <= 0):
            raise ValueError("symbol expansion needs fresh bounded proposals and a declared prior")
        combined = {**self.initial, **{h.identity: h for h in additions}}
        self.initial = MappingProxyType(dict(sorted(combined.items())))
        self.prior_weights = MappingProxyType({**self.prior_weights,
            **{h.identity: float(prior_weight) for h in additions}})
        return self.replay(cutoff=self.cutoff, retracted=self.retracted)

    def receipt(self):
        support = {identity: sum(predict_symbol_hypothesis(hypothesis, observation.request) == observation.outcome
                    for observation in self.observations if observation.identity in getattr(self, "used", ()))
                   for identity, hypothesis in self.remaining.items()}
        body = {"schema": "aura.symbol_meaning_inquiry.v2", "cutoff": self.cutoff,
            "hypotheses": [self.initial[k].to_dict() for k in sorted(self.initial)],
            "observations": [o.to_dict() for o in self.observations],
            "prior_weights": dict(self.prior_weights), "posterior": dict(self.posterior),
            "mismatch_likelihood": self.mismatch_likelihood, "posterior_is_calibrated_accuracy": False,
            "observation_model": "declared_mismatch_to_agreement_likelihood_ratio",
            "retained": sorted(self.remaining), "evidence": getattr(self, "used", ()),
            "observed_agreements": support,
            "retracted": getattr(self, "retracted", ()),
            "status": "contradicted" if not self.remaining else
                      "unsupported_survivors" if self.remaining and not any(support.values()) else
                      "one_surviving_proposal" if len(self.remaining) == 1 else "ambiguous",
            "hypothesis_space_complete": False, "serving_authority": False,
            "future_perception_claimed": False}
        return {**body, "receipt_sha256": digest(body)}

    @classmethod
    def restore(cls, payload):
        """Verify structured custody and independently replay all stored evidence."""
        body = dict(payload)
        expected = body.pop("receipt_sha256", None)
        if body.get("schema") != "aura.symbol_meaning_inquiry.v2" or digest(body) != expected:
            raise ValueError("symbol inquiry storage custody differs")
        inquiry = cls(tuple(SymbolHypothesis.from_dict(h) for h in body["hypotheses"]),
                      prior_weights=body["prior_weights"], mismatch_likelihood=body["mismatch_likelihood"])
        observations = tuple(ObservedDemonstration.from_dict(o) for o in body["observations"])
        if len({(o.origin, o.ref) for o in observations}) != len(observations):
            raise ValueError("symbol inquiry repeats an observation source")
        inquiry.observations = observations
        if digest(inquiry.replay(cutoff=body["cutoff"], retracted=body["retracted"])) != digest(payload):
            raise ValueError("stored symbol conclusions differ from evidence replay")
        return inquiry

    async def retain(self, gateway: StateGateway):
        from core.runtime.gateways import StateMutationRequest

        receipt = self.receipt()
        return await gateway.mutate(StateMutationRequest(key=receipt["receipt_sha256"],
            new_value=receipt, domain="semantic_symbol_meaning",
            cause="retain observed contextual symbol hypotheses without serving authority"))

    @classmethod
    async def load(cls, gateway: StateGateway, key):
        payload = await gateway.read(key, domain="semantic_symbol_meaning", fresh=True)
        if payload is None:
            raise ValueError("retained symbol inquiry is absent")
        if payload.get("receipt_sha256") != key:
            raise ValueError("retained symbol inquiry belongs to another storage key")
        return cls.restore(payload)
