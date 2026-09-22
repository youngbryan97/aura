"""Turn executable interpretation conflicts into discriminating observations.

Simulated predictions choose an observation, never its actual outcome. The
existing perception information-gain algebra ranks the available questions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.learning.procedure_induction import Program
from core.learning.semantic_graph_counterexamples import ProgramObservationCache
from core.perception.expected_information_gain import Observation, choose


@dataclass(frozen=True)
class ProgramInquiry:
    identity: str
    inputs: tuple
    predictions: tuple[tuple[str, str], ...]
    expected_bits: float

    def to_dict(self) -> dict:
        return {
            "schema": "aura.semantic_program_inquiry.v1",
            "identity": self.identity,
            "inputs": self.inputs,
            "predictions": dict(self.predictions),
            "expected_bits": self.expected_bits,
            "observed_result": None,
            "observation_required": True,
            "correctness_authority": False,
        }

    def compatible_methods(self, *, observed_result: object) -> tuple[str, ...]:
        """Report compatibility, not proof, after an independently obtained value."""
        outcome = _outcome(observed_result)
        return tuple(name for name, predicted in self.predictions if predicted == outcome)


def _outcome(value: object) -> str:
    if type(value) is int:
        return json.dumps(["integer", value])
    if type(value) in (tuple, list) and all(type(x) is int for x in value):
        return json.dumps(["integer_sequence", list(value)])
    raise ValueError("inquiry outcomes require a measured integer or integer sequence")


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
        identity = hashlib.sha256(
            json.dumps(
                {"inputs": values, "programs": sorted(programs)}, sort_keys=True, allow_nan=False
            ).encode()
        ).hexdigest()
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
        ProgramInquiry(row.observation, *retained[row.observation], row.expected_bits)
        for row in ranked
        if row.take
    )
