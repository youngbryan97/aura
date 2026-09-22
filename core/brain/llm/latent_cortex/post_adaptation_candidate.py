"""Bind the candidate decoded from the state that adaptation actually left.

Branch probes are decoded before latent optimization.  Treating that snapshot
as the recurrent answer after an accepted latent or fast-weight update measures
the wrong computational object.  This module advances one selected candidate
through those adaptation boundaries while keeping representation repair
separate from correctness and public-answer authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.contract_repair import (
    parse_contract_repair_generation,
)

SCHEMA = "aura.rlc.post_adaptation_candidate.v2"
AUTHORITY = "candidate_pool_freshness_only"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_STAGES = {"post_final_adaptation"}
_DISPOSITIONS = {
    "strict_candidate_admitted",
    "serialization_coercion_admitted",
    "plain_candidate_admitted",
    "candidate_rejected",
}
_EFFECTS = {"added", "replaced", "retained", "removed"}


def _sha_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def advance_post_adaptation_candidate(
    *,
    selected_branch: int,
    prior_candidate: str | None,
    observed_candidate: str,
    stage: str,
    strict_answer_contract: bool,
    response_contract: str = "",
    adaptation_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Validate one post-adaptation probe and return its public commitment.

    The returned candidate remains private to the engine.  The receipt exposes
    only hashes, the deterministic representation disposition, and whether the
    candidate pool was replaced or removed.  It grants no correctness or
    serving authority.
    """

    if type(selected_branch) is not int or selected_branch < 0:
        raise ValueError("post-adaptation branch is invalid")
    if prior_candidate is not None and (
        not isinstance(prior_candidate, str) or not prior_candidate
    ):
        raise ValueError("post-adaptation prior candidate is invalid")
    if not isinstance(observed_candidate, str):
        raise TypeError("post-adaptation observation must be text")
    if stage not in _STAGES:
        raise ValueError("post-adaptation stage is invalid")
    if type(strict_answer_contract) is not bool:
        raise TypeError("strict answer-contract policy must be boolean")
    if not isinstance(response_contract, str):
        raise TypeError("response contract must be text")
    if not isinstance(adaptation_evidence, Mapping) or not adaptation_evidence:
        raise ValueError("post-adaptation evidence is missing")

    admitted: str | None = None
    disposition = "candidate_rejected"
    normalized = observed_candidate.strip()
    if strict_answer_contract:
        try:
            admitted = parse_contract_repair_generation(
                observed_candidate,
                response_contract=response_contract,
            )
        # not a failure: a candidate that will not parse against its contract is not
        # admitted by it.
        except (TypeError, ValueError):
            admitted = None
        else:
            disposition = (
                "strict_candidate_admitted"
                if admitted == normalized
                else "serialization_coercion_admitted"
            )
    elif normalized:
        admitted = normalized
        disposition = "plain_candidate_admitted"

    prior_available = prior_candidate is not None
    if admitted is None:
        effect = "removed"
        admitted_sha256 = ""
    elif not prior_available:
        effect = "added"
        admitted_sha256 = _text_sha(admitted)
    else:
        effect = "retained" if admitted == prior_candidate else "replaced"
        admitted_sha256 = _text_sha(admitted)

    evidence = dict(adaptation_evidence)
    payload = {
        "schema": SCHEMA,
        "selected_branch": selected_branch,
        "stage": stage,
        "strict_answer_contract": strict_answer_contract,
        "response_contract_sha256": _text_sha(response_contract),
        "prior_candidate_available": prior_available,
        "prior_candidate_sha256": _text_sha(prior_candidate or ""),
        "observed_candidate_sha256": _text_sha(observed_candidate),
        "admitted_candidate_sha256": admitted_sha256,
        "observation_changed": (
            not prior_available or observed_candidate != prior_candidate
        ),
        "disposition": disposition,
        "candidate_pool_effect": effect,
        "adaptation_evidence": evidence,
        "correctness_authority": "none",
        "answer_selection_authority": "none",
        "authority": AUTHORITY,
    }
    receipt = {**payload, "receipt_sha256": _sha_payload(payload)}
    validate_post_adaptation_transition(receipt)
    return receipt, admitted


def build_post_adaptation_candidate_receipt(
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(transitions, Sequence) or isinstance(
        transitions, (str, bytes)
    ):
        raise TypeError("post-adaptation transitions must be a sequence")
    rows = [validate_post_adaptation_transition(row) for row in transitions]
    if not rows:
        return {}
    branch = rows[0]["selected_branch"]
    if any(row["selected_branch"] != branch for row in rows):
        raise ValueError("post-adaptation transitions cross branches")
    for previous, current in zip(rows, rows[1:]):
        if (
            current["prior_candidate_available"] is not True
            or previous["admitted_candidate_sha256"]
            != current["prior_candidate_sha256"]
        ):
            raise ValueError("post-adaptation transition chain is discontinuous")
        if previous["candidate_pool_effect"] == "removed":
            raise ValueError("removed post-adaptation candidate cannot advance")
    payload = {
        "schema": SCHEMA,
        "selected_branch": branch,
        "transitions": rows,
        "transition_count": len(rows),
        "final_candidate_sha256": rows[-1]["admitted_candidate_sha256"],
        "final_candidate_available": bool(rows[-1]["admitted_candidate_sha256"]),
        "candidate_pool_effect": rows[-1]["candidate_pool_effect"],
        "correctness_authority": "none",
        "answer_selection_authority": "none",
        "authority": AUTHORITY,
    }
    receipt = {**payload, "receipt_sha256": _sha_payload(payload)}
    validate_post_adaptation_candidate_receipt(receipt)
    return receipt


def validate_post_adaptation_transition(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("post-adaptation transition is missing")
    fields = {
        "schema",
        "selected_branch",
        "stage",
        "strict_answer_contract",
        "response_contract_sha256",
        "prior_candidate_available",
        "prior_candidate_sha256",
        "observed_candidate_sha256",
        "admitted_candidate_sha256",
        "observation_changed",
        "disposition",
        "candidate_pool_effect",
        "adaptation_evidence",
        "correctness_authority",
        "answer_selection_authority",
        "authority",
        "receipt_sha256",
    }
    if set(value) != fields:
        raise ValueError("post-adaptation transition fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    admitted = value["admitted_candidate_sha256"]
    if (
        value["schema"] != SCHEMA
        or type(value["selected_branch"]) is not int
        or value["selected_branch"] < 0
        or value["stage"] not in _STAGES
        or type(value["strict_answer_contract"]) is not bool
        or type(value["prior_candidate_available"]) is not bool
        or any(
            _SHA256_RE.fullmatch(str(value[key])) is None
            for key in (
                "response_contract_sha256",
                "prior_candidate_sha256",
                "observed_candidate_sha256",
            )
        )
        or (admitted != "" and _SHA256_RE.fullmatch(str(admitted)) is None)
        or type(value["observation_changed"]) is not bool
        or value["disposition"] not in _DISPOSITIONS
        or value["candidate_pool_effect"] not in _EFFECTS
        or not isinstance(value["adaptation_evidence"], dict)
        or not value["adaptation_evidence"]
        or value["correctness_authority"] != "none"
        or value["answer_selection_authority"] != "none"
        or value["authority"] != AUTHORITY
        or value["receipt_sha256"] != _sha_payload(payload)
    ):
        raise ValueError("post-adaptation transition is invalid")
    if (value["candidate_pool_effect"] == "removed") != (admitted == ""):
        raise ValueError("post-adaptation removal binding differs")
    if (value["disposition"] == "candidate_rejected") != (admitted == ""):
        raise ValueError("post-adaptation disposition differs")
    if value["prior_candidate_available"] is False:
        if (
            value["prior_candidate_sha256"] != _text_sha("")
            or value["candidate_pool_effect"] not in {"added", "removed"}
        ):
            raise ValueError("post-adaptation absent-prior binding differs")
    elif value["candidate_pool_effect"] == "added":
        raise ValueError("post-adaptation addition had a prior candidate")
    return dict(value)


def validate_post_adaptation_candidate_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("post-adaptation candidate receipt is missing")
    fields = {
        "schema",
        "selected_branch",
        "transitions",
        "transition_count",
        "final_candidate_sha256",
        "final_candidate_available",
        "candidate_pool_effect",
        "correctness_authority",
        "answer_selection_authority",
        "authority",
        "receipt_sha256",
    }
    if set(value) != fields:
        raise ValueError("post-adaptation candidate receipt fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    rows = value["transitions"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("post-adaptation candidate transitions are missing")
    validated = [validate_post_adaptation_transition(row) for row in rows]
    if (
        value["schema"] != SCHEMA
        or type(value["selected_branch"]) is not int
        or value["selected_branch"] < 0
        or value["transition_count"] != len(validated)
        or any(row["selected_branch"] != value["selected_branch"] for row in validated)
        or value["final_candidate_sha256"]
        != validated[-1]["admitted_candidate_sha256"]
        or value["final_candidate_available"]
        is not bool(value["final_candidate_sha256"])
        or value["candidate_pool_effect"] != validated[-1]["candidate_pool_effect"]
        or value["correctness_authority"] != "none"
        or value["answer_selection_authority"] != "none"
        or value["authority"] != AUTHORITY
        or value["receipt_sha256"] != _sha_payload(payload)
    ):
        raise ValueError("post-adaptation candidate receipt is invalid")
    for previous, current in zip(validated, validated[1:]):
        if (
            current["prior_candidate_available"] is not True
            or previous["admitted_candidate_sha256"]
            != current["prior_candidate_sha256"]
        ):
            raise ValueError("post-adaptation candidate chain differs")
    return dict(value)


__all__ = [
    "AUTHORITY",
    "SCHEMA",
    "advance_post_adaptation_candidate",
    "build_post_adaptation_candidate_receipt",
    "validate_post_adaptation_candidate_receipt",
    "validate_post_adaptation_transition",
]
