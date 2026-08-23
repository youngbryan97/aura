"""Mechanical admission for candidate-cortex checkpoint evidence.

Model-heavy measurement happens in an exclusive model lane. This module is the
separate model-free trust boundary: it accepts only exact-bound loss aggregates
and deterministic behavioral outcomes, then reduces them to the adaptive
training admission schema. No generated prose or model judge participates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.learning.candidate_cortex_training import (
    ADMISSION_SCHEMA,
    StagePolicy,
    document_sha256,
)

EVIDENCE_SCHEMA: Final = "aura.candidate_cortex_training.checkpoint_evidence.v1"


class CandidateCortexAdmissionError(ValueError):
    """Stable rejection at the candidate-checkpoint evidence boundary."""


def _fail(code: str) -> None:
    raise CandidateCortexAdmissionError(code)


def _finite_nonnegative(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code)
    result = float(value)
    if not math.isfinite(result) or result < 0:
        _fail(code)
    return result


def _positive_integer(value: object, *, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(code)
    return value


def _loss_surface(raw: object, *, role: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "baseline_loss",
        "candidate_loss",
        "samples",
        "tokens",
        "domain_losses",
    }:
        _fail(f"{role}_loss_schema_invalid")
    baseline = _finite_nonnegative(raw.get("baseline_loss"), code=f"{role}_loss_invalid")
    candidate = _finite_nonnegative(raw.get("candidate_loss"), code=f"{role}_loss_invalid")
    samples = _positive_integer(raw.get("samples"), code=f"{role}_sample_count_invalid")
    tokens = _positive_integer(raw.get("tokens"), code=f"{role}_token_count_invalid")
    domains = raw.get("domain_losses")
    if not isinstance(domains, Mapping) or not domains:
        _fail(f"{role}_domain_losses_invalid")
    normalized_domains: dict[str, dict[str, float | int]] = {}
    domain_samples = 0
    domain_tokens = 0
    for domain, value in sorted(domains.items()):
        if not isinstance(domain, str) or not domain or not isinstance(value, Mapping):
            _fail(f"{role}_domain_losses_invalid")
        if set(value) != {"baseline_loss", "candidate_loss", "samples", "tokens"}:
            _fail(f"{role}_domain_losses_invalid")
        row_samples = _positive_integer(
            value.get("samples"), code=f"{role}_domain_sample_count_invalid"
        )
        row_tokens = _positive_integer(
            value.get("tokens"), code=f"{role}_domain_token_count_invalid"
        )
        normalized_domains[domain] = {
            "baseline_loss": _finite_nonnegative(
                value.get("baseline_loss"), code=f"{role}_domain_loss_invalid"
            ),
            "candidate_loss": _finite_nonnegative(
                value.get("candidate_loss"), code=f"{role}_domain_loss_invalid"
            ),
            "samples": row_samples,
            "tokens": row_tokens,
        }
        domain_samples += row_samples
        domain_tokens += row_tokens
    if domain_samples != samples or domain_tokens != tokens:
        _fail(f"{role}_domain_totals_mismatch")
    return {
        "baseline_loss": baseline,
        "candidate_loss": candidate,
        "samples": samples,
        "tokens": tokens,
        "domain_losses": normalized_domains,
    }


def _behavior_rows(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        _fail("behavior_evidence_invalid")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, Mapping) or set(value) != {
            "probe_id",
            "family",
            "baseline_passed",
            "candidate_passed",
            "evaluator_sha256",
        }:
            _fail("behavior_evidence_invalid")
        probe_id = value.get("probe_id")
        family = value.get("family")
        evaluator_sha = value.get("evaluator_sha256")
        if (
            not isinstance(probe_id, str)
            or not probe_id
            or probe_id in seen
            or not isinstance(family, str)
            or not family
            or not isinstance(evaluator_sha, str)
            or len(evaluator_sha) != 64
            or any(character not in "0123456789abcdef" for character in evaluator_sha)
            or not isinstance(value.get("baseline_passed"), bool)
            or not isinstance(value.get("candidate_passed"), bool)
        ):
            _fail("behavior_evidence_invalid")
        seen.add(probe_id)
        rows.append(dict(value))
    return sorted(rows, key=lambda row: row["probe_id"])


def _loss_retention_score(*, baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else 0.0
    return min(1.0, baseline / max(candidate, 1e-12))


def adjudicate_checkpoint_evidence(
    raw: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    stage_index: int,
) -> dict[str, Any]:
    """Return the canonical model-free stage admission.

    Persona and retention scores are ratios of candidate to frozen-baseline
    teacher-forced loss. Exact behavioral non-regression is stricter: every
    probe the baseline solved must remain solved by the candidate.
    """

    required = {
        "schema",
        "stage_index",
        "plan_sha256",
        "model_descriptor_sha256",
        "dataset_receipt_sha256",
        "checkpoint_sha256",
        "persona",
        "retention",
        "behavior",
        "measurement_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        _fail("checkpoint_evidence_schema_invalid")
    if raw.get("schema") != EVIDENCE_SCHEMA or raw.get("stage_index") != stage_index:
        _fail("checkpoint_evidence_identity_invalid")
    if (
        raw.get("plan_sha256") != plan.get("plan_sha256")
        or raw.get("model_descriptor_sha256")
        != plan.get("model", {}).get("descriptor_sha256")
        or raw.get("dataset_receipt_sha256")
        != plan.get("dataset", {}).get("receipt_sha256")
    ):
        _fail("checkpoint_evidence_binding_mismatch")
    for field in ("checkpoint_sha256", "measurement_sha256"):
        value = raw.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            _fail("checkpoint_evidence_digest_invalid")
    material = dict(raw)
    claimed_measurement = material.pop("measurement_sha256")
    if claimed_measurement != document_sha256(material):
        _fail("checkpoint_evidence_digest_invalid")

    persona = _loss_surface(raw.get("persona"), role="persona")
    retention = _loss_surface(raw.get("retention"), role="retention")
    behavior = _behavior_rows(raw.get("behavior"))
    baseline_successes = [row for row in behavior if row["baseline_passed"]]
    regressions = [
        row["probe_id"]
        for row in baseline_successes
        if not row["candidate_passed"]
    ]
    preserved = len(baseline_successes) - len(regressions)
    no_regression_score = (
        preserved / len(baseline_successes) if baseline_successes else 1.0
    )
    checks = persona["samples"] + retention["samples"] + len(behavior)
    policy = StagePolicy(**dict(plan["stages"]))
    if checks < policy.min_eval_samples:
        _fail("checkpoint_evidence_underpowered")

    admission = {
        "schema": ADMISSION_SCHEMA,
        "stage_index": stage_index,
        "model_free": True,
        "persona_score": _loss_retention_score(
            baseline=persona["baseline_loss"], candidate=persona["candidate_loss"]
        ),
        "retention_score": _loss_retention_score(
            baseline=retention["baseline_loss"],
            candidate=retention["candidate_loss"],
        ),
        "no_regression_score": no_regression_score,
        "regressions": len(regressions),
        "checks": checks,
        "evidence_sha256": claimed_measurement,
    }
    return admission


__all__ = [
    "EVIDENCE_SCHEMA",
    "CandidateCortexAdmissionError",
    "adjudicate_checkpoint_evidence",
]
