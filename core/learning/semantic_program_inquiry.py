"""Turn executable interpretation conflicts into discriminating observations.

Simulated predictions choose an observation, never its actual outcome. The
existing perception information-gain algebra ranks the available questions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.learning.procedure_induction import Program
from core.learning.semantic_graph_counterexamples import ProgramObservationCache
from core.perception.expected_information_gain import Observation, choose
from core.runtime.gateways import StateGateway, StateMutationRequest
from core.evidence.packet import observe


@dataclass(frozen=True)
class ProgramInquiry:
    identity: str
    inputs: tuple
    predictions: tuple[tuple[str, str], ...]
    expected_bits: float
    program_shas: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        names = [name for name, _ in self.predictions]
        identities = dict(self.program_shas)
        if (not names or len(names) != len(set(names))
                or set(names) != set(identities)
                or len(identities) != len(self.program_shas)
                or any(not isinstance(name, str) or not name for name in names)
                or any(not isinstance(sha, str) or not sha.startswith("sha256:")
                       or len(sha) != 71
                       or any(c not in "0123456789abcdef" for c in sha[7:])
                       for sha in identities.values())
                or type(self.expected_bits) not in (float, int)
                or not math.isfinite(self.expected_bits) or self.expected_bits <= 0):
            raise ValueError("invalid inquiry hypothesis bindings")
        for value in self.inputs:
            _outcome(value)
        outcomes = {}
        for name, encoded in self.predictions:
            kind, value = json.loads(encoded)
            if kind not in {"integer", "integer_sequence"} or _outcome(value) != encoded:
                raise ValueError("invalid inquiry prediction")
            sha = identities[name]
            if sha in outcomes and outcomes[sha] != encoded:
                raise ValueError("one program has conflicting predictions")
            outcomes[sha] = encoded
        if len(set(outcomes.values())) < 2 or self.expected_bits > math.log2(len(outcomes)) + 1e-6:
            raise ValueError("inquiry has no valid discriminating information")
        if self.identity != _identity(self.inputs, identities.values()):
            raise ValueError("inquiry identity does not match its probe and programs")

    def to_dict(self) -> dict:
        payload = {
            "schema": "aura.semantic_program_inquiry.v1",
            "identity": self.identity,
            "inputs": self.inputs,
            "predictions": dict(self.predictions),
            "program_shas": dict(self.program_shas),
            "expected_bits": self.expected_bits,
            "observed_result": None,
            "observation_required": True,
            "correctness_authority": False,
        }
        payload["content_sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, allow_nan=False
        ).encode()).hexdigest()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping):
        content = dict(payload)
        digest = content.pop("content_sha256", None)
        if digest != hashlib.sha256(json.dumps(
            content, sort_keys=True, allow_nan=False
        ).encode()).hexdigest():
            raise ValueError("inquiry content digest mismatch")
        if (payload.get("schema") != "aura.semantic_program_inquiry.v1"
                or payload.get("observed_result") is not None
                or payload.get("observation_required") is not True
                or payload.get("correctness_authority") is not False):
            raise ValueError("invalid pending inquiry record")
        return cls(
            payload["identity"],
            tuple(tuple(v) if isinstance(v, list) else v for v in payload["inputs"]),
            tuple(payload["predictions"].items()), payload["expected_bits"],
            tuple(payload["program_shas"].items()),
        )

    async def retain(self, gateway: StateGateway):
        """Persist the pending inquiry through the canonical state owner."""
        return await gateway.mutate(StateMutationRequest(
            key=self.identity, new_value=self.to_dict(), domain="semantic_inquiries",
            cause="retain executable interpretation distinction",
        ))

    @classmethod
    async def restore(cls, gateway: StateGateway, identity: str):
        payload = await gateway.read(identity, domain="semantic_inquiries", fresh=True)
        if payload is None:
            return None
        inquiry = cls.from_dict(payload)
        if inquiry.identity != identity:
            raise ValueError("stored inquiry belongs to another request")
        return inquiry

    def compatible_methods(self, *, observed_result: object) -> tuple[str, ...]:
        """Report compatibility, not proof, after an independently obtained value."""
        outcome = _outcome(observed_result)
        return tuple(name for name, predicted in self.predictions if predicted == outcome)

    def evidence_for(self, *, observed_result: object, origin: str, ref: str):
        """Bind measured probe agreement to the existing evidence algebra.

        The caller supplies the independent observation identity. These packets
        support agreement on this probe, not universal program correctness.
        """
        if not isinstance(origin, str) or not origin.strip() or not isinstance(ref, str) or not ref.strip():
            raise ValueError("observed inquiry evidence requires a source and reference")
        outcome = _outcome(observed_result)
        identities = dict(self.program_shas)
        return {
            name: observe(
                float(predicted == outcome), origin=origin, ref=ref,
                subject=f"inquiry_agreement:{self.identity}:{identities[name]}",
                produced_by="semantic_program_inquiry",
            )
            for name, predicted in self.predictions
        }


def _outcome(value: object) -> str:
    if type(value) is int:
        return json.dumps(["integer", value])
    if type(value) in (tuple, list) and all(type(x) is int for x in value):
        return json.dumps(["integer_sequence", list(value)])
    raise ValueError("inquiry outcomes require a measured integer or integer sequence")


def _identity(inputs, programs):
    return hashlib.sha256(json.dumps(
        {"inputs": inputs, "programs": sorted(set(programs))},
        sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()


def plan_program_inquiries(
    proposals: Mapping[str, Program], probes: Sequence[tuple], *, fuel: int
) -> tuple[ProgramInquiry, ...]:
    """Rank finite supplied probes without an answer key or assumed correct method.

    Equal mass is assigned to distinct programs, not duplicate method entries.
    Only probes whose predictions all execute successfully are admitted. Thus an
    infrastructure failure never becomes a distinguishing semantic outcome.
    """
    if not proposals:
        return ()
    if any(not isinstance(p, Program) for p in proposals.values()):
        raise ValueError("inquiries require executable program proposals")
    programs = {p.sha(): p for p in proposals.values()}
    if len(programs) < 2:
        return ()
    cache = ProgramObservationCache()
    observations, retained = [], {}
    for probe in probes:
        values = tuple(tuple(v) if isinstance(v, list) else v for v in probe)
        identity = _identity(values, programs)
        if identity in retained:
            continue
        predictions = {}
        for key, program in programs.items():
            result = cache.observe(program, values, fuel=fuel)
            if result["status"] != "value":
                break
            predictions[key] = _outcome(result["result"])
        if len(predictions) != len(programs) or len(set(predictions.values())) < 2:
            continue
        likelihoods = {
            outcome: {key: float(value == outcome) for key, value in predictions.items()}
            for outcome in sorted(set(predictions.values()))
        }
        observations.append(Observation(identity, likelihoods))
        retained[identity] = (
            values,
            tuple((name, predictions[program.sha()]) for name, program in proposals.items()),
        )
    ranked = choose(dict.fromkeys(programs, 1.0), observations)
    return tuple(
        ProgramInquiry(row.observation, *retained[row.observation], row.expected_bits,
                       tuple((name, program.sha()) for name, program in proposals.items()))
        for row in ranked
        if row.take
    )
