"""Compile exact candidate-cortex measurements for model-free admission.

The model-facing evaluator emits additive negative-log-likelihood rows and
deterministic behavior grades.  This module performs no model inference.  It
canonicalizes those rows, rejects ambiguous or duplicated evidence, and emits
the checkpoint evidence document consumed by
``candidate_cortex_admission``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.learning.candidate_cortex_admission import EVIDENCE_SCHEMA
from core.learning.candidate_cortex_training import document_sha256

LOSS_ROW_SCHEMA: Final = "aura.candidate_cortex_training.loss_row.v1"


class CandidateCortexMeasurementError(ValueError):
    """The model-heavy measurement rows cannot support admission."""


def _fail(code: str) -> None:
    raise CandidateCortexMeasurementError(code)


def _sha256(value: object, *, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(code)
    return value


def _finite_nonnegative(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code)
    result = float(value)
    if not math.isfinite(result) or result < 0:
        _fail(code)
    return result


def _loss_surface(raw: object, *, role: str) -> dict[str, Any]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        _fail(f"{role}_rows_invalid")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for value in raw:
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "sample_id",
            "domain",
            "baseline_nll_sum",
            "candidate_nll_sum",
            "tokens",
        }:
            _fail(f"{role}_row_schema_invalid")
        sample_id = value.get("sample_id")
        domain = value.get("domain")
        tokens = value.get("tokens")
        if (
            value.get("schema") != LOSS_ROW_SCHEMA
            or not isinstance(sample_id, str)
            or not sample_id
            or sample_id in seen
            or not isinstance(domain, str)
            or not domain
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
        ):
            _fail(f"{role}_row_invalid")
        seen.add(sample_id)
        rows.append(
            {
                **dict(value),
                "baseline_nll_sum": _finite_nonnegative(
                    value.get("baseline_nll_sum"), code=f"{role}_loss_invalid"
                ),
                "candidate_nll_sum": _finite_nonnegative(
                    value.get("candidate_nll_sum"), code=f"{role}_loss_invalid"
                ),
            }
        )
    rows.sort(key=lambda row: row["sample_id"])
    total_tokens = sum(int(row["tokens"]) for row in rows)
    baseline_sum = sum(float(row["baseline_nll_sum"]) for row in rows)
    candidate_sum = sum(float(row["candidate_nll_sum"]) for row in rows)
    domains: dict[str, dict[str, float | int]] = {}
    for row in rows:
        domain = str(row["domain"])
        aggregate = domains.setdefault(
            domain,
            {
                "baseline_nll_sum": 0.0,
                "candidate_nll_sum": 0.0,
                "samples": 0,
                "tokens": 0,
            },
        )
        aggregate["baseline_nll_sum"] = float(aggregate["baseline_nll_sum"]) + float(
            row["baseline_nll_sum"]
        )
        aggregate["candidate_nll_sum"] = float(aggregate["candidate_nll_sum"]) + float(
            row["candidate_nll_sum"]
        )
        aggregate["samples"] = int(aggregate["samples"]) + 1
        aggregate["tokens"] = int(aggregate["tokens"]) + int(row["tokens"])
    domain_losses = {
        domain: {
            "baseline_loss": float(value["baseline_nll_sum"]) / int(value["tokens"]),
            "candidate_loss": float(value["candidate_nll_sum"]) / int(value["tokens"]),
            "samples": int(value["samples"]),
            "tokens": int(value["tokens"]),
        }
        for domain, value in sorted(domains.items())
    }
    return {
        "baseline_loss": baseline_sum / total_tokens,
        "candidate_loss": candidate_sum / total_tokens,
        "samples": len(rows),
        "tokens": total_tokens,
        "domain_losses": domain_losses,
    }


def _behavior_rows(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        _fail("behavior_rows_invalid")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for value in raw:
        if not isinstance(value, Mapping) or set(value) != {
            "probe_id",
            "family",
            "baseline_passed",
            "candidate_passed",
            "evaluator_sha256",
        }:
            _fail("behavior_row_schema_invalid")
        probe_id = value.get("probe_id")
        family = value.get("family")
        if (
            not isinstance(probe_id, str)
            or not probe_id
            or probe_id in seen
            or not isinstance(family, str)
            or not family
            or not isinstance(value.get("baseline_passed"), bool)
            or not isinstance(value.get("candidate_passed"), bool)
        ):
            _fail("behavior_row_invalid")
        _sha256(value.get("evaluator_sha256"), code="behavior_evaluator_digest_invalid")
        seen.add(probe_id)
        rows.append(dict(value))
    return sorted(rows, key=lambda row: row["probe_id"])


def compile_checkpoint_evidence(
    *,
    plan: Mapping[str, Any],
    stage_index: int,
    checkpoint_sha256: str,
    persona_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    behavior_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile additive measurements into one canonical evidence document."""

    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0:
        _fail("stage_index_invalid")
    plan_sha = _sha256(plan.get("plan_sha256"), code="plan_digest_invalid")
    model = plan.get("model")
    dataset = plan.get("dataset")
    if not isinstance(model, Mapping) or not isinstance(dataset, Mapping):
        _fail("plan_identity_invalid")
    material = {
        "schema": EVIDENCE_SCHEMA,
        "stage_index": stage_index,
        "plan_sha256": plan_sha,
        "model_descriptor_sha256": _sha256(
            model.get("descriptor_sha256"), code="model_digest_invalid"
        ),
        "dataset_receipt_sha256": _sha256(
            dataset.get("receipt_sha256"), code="dataset_digest_invalid"
        ),
        "checkpoint_sha256": _sha256(
            checkpoint_sha256, code="checkpoint_digest_invalid"
        ),
        "persona": _loss_surface(persona_rows, role="persona"),
        "retention": _loss_surface(retention_rows, role="retention"),
        "behavior": _behavior_rows(behavior_rows),
    }
    return {**material, "measurement_sha256": document_sha256(material)}


__all__ = [
    "LOSS_ROW_SCHEMA",
    "CandidateCortexMeasurementError",
    "compile_checkpoint_evidence",
]
