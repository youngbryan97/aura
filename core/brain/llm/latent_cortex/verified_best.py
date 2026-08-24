"""Confidence-bound authority for preserving verified recurrent states."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.runtime.tensor_identity import tensor_identity_sha256

VERIFIER_OBSERVATION_SCHEMA = "aura.rlc.verifier_observation.v1"
VERIFIED_BEST_SCHEMA = "aura.rlc.verified_best_state.v1"
MIN_CALIBRATED_SAMPLES = 8


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _probability(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite probability")
    return float(value)


def tensor_sha256(value: Any) -> str:
    return tensor_identity_sha256(value)


@dataclass(frozen=True, slots=True)
class VerifierObservation:
    score: float
    lower_bound: float
    upper_bound: float
    sample_count: int
    basis: str
    independent: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        score = _probability(self.score, name="verifier score")
        lower = _probability(self.lower_bound, name="verifier lower bound")
        upper = _probability(self.upper_bound, name="verifier upper bound")
        if lower > score or score > upper:
            raise ValueError("verifier score is outside its confidence interval")
        if type(self.sample_count) is not int or not 0 <= self.sample_count <= 1_000_000:
            raise ValueError("verifier sample count is invalid")
        if self.basis not in {
            "uncalibrated_scalar",
            "deterministic_exact",
            "calibrated_interval",
        }:
            raise ValueError("verifier observation basis is invalid")
        if type(self.independent) is not bool:
            raise ValueError("verifier independence flag must be boolean")
        if self.basis == "uncalibrated_scalar":
            if (
                self.independent
                or self.sample_count != 0
                or lower != 0.0
                or upper != 1.0
                or self.evidence_sha256
            ):
                raise ValueError("uncalibrated scalar cannot carry authority")
        else:
            if not self.independent or not _is_sha256(self.evidence_sha256):
                raise ValueError("authoritative verifier evidence is absent")
            if self.basis == "deterministic_exact" and (
                self.sample_count < 1 or lower != score or upper != score
            ):
                raise ValueError("deterministic verifier evidence is not exact")
            if self.basis == "calibrated_interval" and self.sample_count < MIN_CALIBRATED_SAMPLES:
                raise ValueError("calibrated verifier evidence is underpowered")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)

    @property
    def authoritative(self) -> bool:
        return self.basis != "uncalibrated_scalar"

    @classmethod
    def from_value(cls, value: Any) -> VerifierObservation:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(
                score=float(value),
                lower_bound=0.0,
                upper_bound=1.0,
                sample_count=0,
                basis="uncalibrated_scalar",
                independent=False,
                evidence_sha256="",
            )
        fields = {
            "schema",
            "score",
            "lower_bound",
            "upper_bound",
            "sample_count",
            "basis",
            "independent",
            "evidence_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("verifier observation fields differ")
        if value.get("schema") != VERIFIER_OBSERVATION_SCHEMA:
            raise ValueError("verifier observation schema is invalid")
        return cls(
            score=value["score"],
            lower_bound=value["lower_bound"],
            upper_bound=value["upper_bound"],
            sample_count=value["sample_count"],
            basis=value["basis"],
            independent=value["independent"],
            evidence_sha256=value["evidence_sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": VERIFIER_OBSERVATION_SCHEMA,
            "score": round(self.score, 10),
            "lower_bound": round(self.lower_bound, 10),
            "upper_bound": round(self.upper_bound, 10),
            "sample_count": self.sample_count,
            "basis": self.basis,
            "independent": self.independent,
            "evidence_sha256": self.evidence_sha256,
            "authoritative": self.authoritative,
        }
        return {**payload, "observation_sha256": canonical_sha256(payload)}


def validate_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("verifier observation must be an object")
    raw = dict(value)
    observation_sha256 = raw.pop("observation_sha256", None)
    authoritative = raw.pop("authoritative", None)
    observation = VerifierObservation.from_value(raw)
    expected = observation.to_dict()
    if (
        authoritative is not observation.authoritative
        or observation_sha256 != expected["observation_sha256"]
        or dict(value) != expected
    ):
        raise ValueError("verifier observation commitment is invalid")
    return expected


def build_verified_best_receipt(
    *,
    branches: list[Any],
    cognitive_action_trace: list[dict[str, Any]],
    loop_stability: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(loop_stability, Mapping) or not _is_sha256(
        loop_stability.get("receipt_sha256")
    ):
        raise ValueError("verified-best loop-stability source is invalid")
    rows = []
    for branch in branches:
        trace = [dict(row) for row in branch.verified_best_trace]
        rows.append(
            {
                "branch_index": int(branch.index),
                "decisions": trace,
                "decision_count": len(trace),
                "promotion_count": sum(row["decision"] == "promote" for row in trace),
                "preservation_count": sum(row["decision"] == "preserve_verified" for row in trace),
                "final_best_step": int(branch.verified_best_step),
                "final_best_state_sha256": str(branch.verified_best_state_sha256),
                "final_best_observation": dict(branch.verified_best_observation),
                "finalization": dict(branch.verified_finalization),
            }
        )
    payload = {
        "schema": VERIFIED_BEST_SCHEMA,
        "cognitive_action_trace_sha256": canonical_sha256(cognitive_action_trace),
        "loop_stability_sha256": loop_stability.get("receipt_sha256"),
        "branches": rows,
        "authoritative_promotions": sum(row["promotion_count"] for row in rows),
        "verified_preservations": sum(row["preservation_count"] for row in rows),
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_verified_best_receipt(
        receipt,
        cognitive_action_trace=cognitive_action_trace,
        loop_stability=loop_stability,
        expected_n_branches=len(branches),
    )


def validate_verified_best_receipt(
    value: Any,
    *,
    cognitive_action_trace: list[dict[str, Any]],
    loop_stability: dict[str, Any],
    expected_n_branches: int,
) -> dict[str, Any]:
    if not isinstance(cognitive_action_trace, list):
        raise ValueError("verified-best action source must be a list")
    if not isinstance(loop_stability, Mapping) or not _is_sha256(
        loop_stability.get("receipt_sha256")
    ):
        raise ValueError("verified-best loop-stability source is invalid")
    fields = {
        "schema",
        "cognitive_action_trace_sha256",
        "loop_stability_sha256",
        "branches",
        "authoritative_promotions",
        "verified_preservations",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("verified-best receipt fields differ")
    receipt = dict(value)
    payload = {key: receipt[key] for key in fields - {"receipt_sha256"}}
    if (
        receipt["schema"] != VERIFIED_BEST_SCHEMA
        or receipt["receipt_sha256"] != canonical_sha256(payload)
        or receipt["cognitive_action_trace_sha256"] != canonical_sha256(cognitive_action_trace)
        or receipt["loop_stability_sha256"] != loop_stability.get("receipt_sha256")
        or type(expected_n_branches) is not int
        or expected_n_branches < 1
        or not isinstance(receipt["branches"], list)
        or len(receipt["branches"]) != expected_n_branches
        or type(receipt["authoritative_promotions"]) is not int
        or type(receipt["verified_preservations"]) is not int
    ):
        raise ValueError("verified-best receipt identity is invalid")
    action_verification: dict[int, dict[str, Any]] = {}
    for row in cognitive_action_trace:
        decision = row.get("decision") if isinstance(row, Mapping) else None
        verification = row.get("verification") if isinstance(row, Mapping) else None
        step = decision.get("step_index") if isinstance(decision, Mapping) else None
        if type(step) is not int or not isinstance(verification, Mapping):
            raise ValueError("verified-best action source is invalid")
        if step in action_verification:
            raise ValueError("verified-best action source steps overlap")
        action_verification[step] = dict(verification)
    branch_fields = {
        "branch_index",
        "decisions",
        "decision_count",
        "promotion_count",
        "preservation_count",
        "final_best_step",
        "final_best_state_sha256",
        "final_best_observation",
        "finalization",
    }
    decision_fields = {
        "ordinal",
        "action_step",
        "branch_step",
        "candidate_state_sha256",
        "prior_best_state_sha256",
        "restore_target_state_sha256",
        "observation",
        "decision",
        "restored",
        "resulting_state_sha256",
    }
    promotions = preservations = 0
    for branch_index, branch in enumerate(receipt["branches"]):
        if (
            not isinstance(branch, Mapping)
            or set(branch) != branch_fields
            or branch["branch_index"] != branch_index
            or not isinstance(branch["decisions"], list)
            or branch["decision_count"] != len(branch["decisions"])
        ):
            raise ValueError("verified-best branch summary is invalid")
        best_sha256 = ""
        best_step = -1
        best_observation: dict[str, Any] = {}
        branch_promotions = branch_preservations = 0
        for ordinal, row in enumerate(branch["decisions"]):
            if (
                not isinstance(row, Mapping)
                or set(row) != decision_fields
                or row["ordinal"] != ordinal
                or type(row["action_step"]) is not int
                or type(row["branch_step"]) is not int
                or row["branch_step"] < 0
                or not _is_sha256(row["candidate_state_sha256"])
                or not _is_sha256(row["resulting_state_sha256"])
                or (
                    row["prior_best_state_sha256"]
                    and not _is_sha256(row["prior_best_state_sha256"])
                )
                or (
                    row["restore_target_state_sha256"]
                    and not _is_sha256(row["restore_target_state_sha256"])
                )
                or row["prior_best_state_sha256"] != best_sha256
                or type(row["restored"]) is not bool
            ):
                raise ValueError("verified-best decision evidence is invalid")
            observation = validate_observation(row["observation"])
            source = action_verification.get(row["action_step"])
            if (
                not isinstance(source, dict)
                or source.get("target_branch") != branch_index
                or source.get("observation") != observation
                or source.get("decision") != row["decision"]
                or source.get("restored") is not row["restored"]
                or (
                    source.get("candidate_state_sha256") is not None
                    and source.get("candidate_state_sha256") != row["candidate_state_sha256"]
                )
                or (
                    source.get("restore_target_state_sha256") is not None
                    and source.get("restore_target_state_sha256")
                    != row["restore_target_state_sha256"]
                )
            ):
                raise ValueError("verified-best decision source differs")
            if observation["authoritative"] and float(observation["upper_bound"]) <= 1e-9:
                expected_decision = "reject_verified_failure"
            elif not observation["authoritative"]:
                expected_decision = "ranking_only"
            elif not best_sha256:
                expected_decision = "promote"
            elif float(observation["lower_bound"]) > float(best_observation["upper_bound"]) + 1e-9:
                expected_decision = "promote"
            else:
                expected_decision = "preserve_verified"
            if row["decision"] != expected_decision:
                raise ValueError("verified-best interval decision is invalid")
            if expected_decision == "promote":
                best_sha256 = row["candidate_state_sha256"]
                best_step = row["branch_step"]
                best_observation = observation
                branch_promotions += 1
                expected_result = best_sha256
                expected_restored = False
            elif expected_decision == "preserve_verified":
                branch_preservations += 1
                expected_result = best_sha256
                expected_restored = row["candidate_state_sha256"] != best_sha256
            elif expected_decision == "reject_verified_failure":
                expected_result = row["restore_target_state_sha256"]
                expected_restored = True
            else:
                expected_result = row["candidate_state_sha256"]
                expected_restored = False
            if (
                row["resulting_state_sha256"] != expected_result
                or row["restored"] is not expected_restored
                or (
                    expected_decision == "reject_verified_failure"
                    and not _is_sha256(row["restore_target_state_sha256"])
                )
                or (
                    expected_decision != "reject_verified_failure"
                    and row["restore_target_state_sha256"]
                )
            ):
                raise ValueError("verified-best state disposition is invalid")
        finalization = branch["finalization"]
        finalization_fields = {
            "source",
            "pre_state_sha256",
            "post_state_sha256",
            "reverted",
            "fixed_depth",
        }
        if (
            not isinstance(finalization, Mapping)
            or set(finalization) != finalization_fields
            or finalization["source"] not in {"verified", "proxy", "current"}
            or not _is_sha256(finalization["pre_state_sha256"])
            or not _is_sha256(finalization["post_state_sha256"])
            or type(finalization["reverted"]) is not bool
            or type(finalization["fixed_depth"]) is not bool
            or (
                finalization["source"] == "current"
                and (
                    finalization["reverted"]
                    or finalization["pre_state_sha256"] != finalization["post_state_sha256"]
                )
            )
            or (finalization["source"] == "proxy" and not finalization["reverted"])
            or (
                finalization["source"] == "verified"
                and (
                    finalization["fixed_depth"]
                    or not best_sha256
                    or finalization["post_state_sha256"] != best_sha256
                    or finalization["reverted"]
                    is not (finalization["pre_state_sha256"] != best_sha256)
                )
            )
            or (finalization["fixed_depth"] and finalization["source"] == "verified")
        ):
            raise ValueError("verified-best finalization evidence is invalid")
        if (
            branch["promotion_count"] != branch_promotions
            or branch["preservation_count"] != branch_preservations
            or branch["final_best_step"] != best_step
            or branch["final_best_state_sha256"] != best_sha256
            or branch["final_best_observation"] != best_observation
        ):
            raise ValueError("verified-best final branch evidence is invalid")
        promotions += branch_promotions
        preservations += branch_preservations
    if (
        receipt["authoritative_promotions"] != promotions
        or receipt["verified_preservations"] != preservations
    ):
        raise ValueError("verified-best aggregate evidence is invalid")
    return receipt


__all__ = [
    "MIN_CALIBRATED_SAMPLES",
    "VERIFIED_BEST_SCHEMA",
    "VERIFIER_OBSERVATION_SCHEMA",
    "VerifierObservation",
    "build_verified_best_receipt",
    "tensor_sha256",
    "validate_observation",
    "validate_verified_best_receipt",
]
