"""Fresh-prefix conclusion recurrence measurement.

The selected candidate is decomposed into an independently verified prefix and
a conclusion.  Several seed-isolated continuations are then generated from
fresh, zero-offset KV contexts.  The resulting statistic measures conclusion
recurrence only: it has no branch-selection or correctness authority.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
    validate_atomic_decomposition,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    RouteOutcome,
    build_deterministic_router_receipt,
    validate_deterministic_router_envelope,
)
from core.learning.prefix_stability import (
    CALIBRATION_TARGET,
    PrefixStabilityCalibrator,
)
from core.runtime.file_read_gateway import read_stable_bytes

logger = logging.getLogger(__name__)

PREFIX_STABILITY_SCHEMA = "aura.rlc.prefix_stability_verifier.v1"
PREFIX_STABILITY_CONTEXT_SCHEMA = "aura.rlc.fresh_prefix_context.v1"
_RESULT_RE = re.compile(r"FINAL_ANSWER\s*:\s*(\{.*\})\s*$", re.DOTALL)
_ARITH_RE = re.compile(
    r"(?<![\d.])(-?\d{1,12})\s*([+\-*/x\u00d7])\s*(-?\d{1,12})"
    r"\s*=\s*(-?\d{1,12})(?!\d)(?!\.\d)"
)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SAMPLE_STATUSES = {"complete", "contract_refused", "generation_failed"}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _implementation_sha256() -> str:
    return hashlib.sha256(
        read_stable_bytes(Path(__file__), max_bytes=2 * 1024 * 1024)
    ).hexdigest()


def unavailable_prefix_stability(reason: str) -> dict[str, Any]:
    normalized = str(reason or "").strip()[:240]
    if not normalized:
        raise ValueError("prefix-stability unavailable reason is required")
    return {
        "requested": True,
        "available": False,
        "reason": normalized,
        "selection_effect": "none",
        "correctness_effect": "none",
    }


def _signature(conclusion: str) -> dict[str, str]:
    text = str(conclusion or "").strip()
    if not text or len(text) > 2048:
        raise ValueError("conclusion is empty or exceeds 2048 characters")
    try:
        parsed = json.loads(text)
    # not a failure: a conclusion that is not JSON is signed as the text it is.
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        canonical = json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        mode = "canonical_json"
    else:
        arithmetic = [
            [
                int(left),
                "*" if operator in {"x", "\u00d7"} else operator,
                int(right),
                int(claimed),
            ]
            for left, operator, right, claimed in _ARITH_RE.findall(text)
        ]
        if arithmetic:
            canonical = json.dumps(arithmetic, separators=(",", ":"), ensure_ascii=True)
            mode = "arithmetic_claim_sequence"
        else:
            tokens = [token.casefold() for token in _TOKEN_RE.findall(text)]
            if not tokens:
                raise ValueError("conclusion has no stable signature")
            canonical = " ".join(tokens)
            mode = "normalized_lexical_surface"
    return {
        "mode": mode,
        "canonical": canonical,
        "signature_sha256": _text_sha(f"{mode}\0{canonical}"),
    }


def _conclusion_boundary(
    candidate: str,
    *,
    objective: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str] | None:
    atomic = build_atomic_decomposition(candidate, objective=objective)
    visible = validate_atomic_decomposition(
        atomic,
        candidate=candidate,
        objective=objective,
    )
    atoms = visible["atoms"]
    if len(atoms) < 2:
        return None
    conclusion_index = next(
        (
            index
            for index, atom in enumerate(atoms)
            if index > 0 and atom["kind"] == "conclusion"
        ),
        len(atoms) - 1,
    )
    conclusion_atom = atoms[conclusion_index]
    prefix = candidate[: int(conclusion_atom["start"])].rstrip()
    conclusion = candidate[
        int(conclusion_atom["start"]) : int(conclusion_atom["end"])
    ].strip()
    if not prefix or not conclusion:
        return None
    prefix_atomic = build_atomic_decomposition(prefix, objective=objective)
    prefix_router = build_deterministic_router_receipt(
        prefix,
        objective=objective,
        atomic_receipt=prefix_atomic,
    )
    verified = (
        prefix_router["hard_pass"] is True
        and prefix_router["checked"] is True
        and prefix_router["counts"][RouteOutcome.VERIFIED.value]
        == len(prefix_router["routes"])
        and len(prefix_router["routes"]) >= 1
    )
    if not verified:
        return None
    return atomic, prefix_atomic, prefix_router, prefix, conclusion


def build_prefix_prompt(
    *,
    objective: str,
    candidate_sha256: str,
    prefix: str,
    prefix_sha256: str,
) -> str:
    """Build a continuation prompt that never reveals the source conclusion."""

    if not _SHA_RE.fullmatch(candidate_sha256) or not _SHA_RE.fullmatch(prefix_sha256):
        raise ValueError("prefix prompt bindings must be SHA-256 values")
    return (
        "You are a fresh continuation lane with no solver KV state and no access "
        "to the source conclusion. Continue only from the problem and verified "
        "prefix. Return the conclusion you independently reach.\n"
        "Return exactly FINAL_ANSWER followed by one JSON object with string keys "
        '"candidate_sha256", "prefix_sha256", and "conclusion". Do not add prose '
        "outside that object. The conclusion must be at most 2048 characters.\n\n"
        f"PROBLEM:\n{str(objective or '')[:8192]}\n\n"
        f"CANDIDATE_SHA256: {candidate_sha256}\n"
        f"VERIFIED_PREFIX_SHA256: {prefix_sha256}\n"
        f"VERIFIED_PREFIX:\n{prefix[:8192]}\n"
    )


def parse_prefix_result(
    text: str,
    *,
    candidate_sha256: str,
    prefix_sha256: str,
) -> dict[str, str]:
    match = _RESULT_RE.fullmatch(str(text or "").strip())
    if match is None:
        raise ValueError("prefix generation did not satisfy FINAL_ANSWER JSON contract")
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("prefix generation result is not valid JSON") from exc
    fields = {"candidate_sha256", "prefix_sha256", "conclusion"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("prefix generation result fields do not match contract")
    if any(not isinstance(value[field], str) for field in fields):
        raise ValueError("prefix generation result values must be strings")
    if value["candidate_sha256"] != candidate_sha256:
        raise ValueError("prefix generation is not bound to the candidate")
    if value["prefix_sha256"] != prefix_sha256:
        raise ValueError("prefix generation is not bound to the verified prefix")
    _signature(value["conclusion"])
    return {field: value[field] for field in ("candidate_sha256", "prefix_sha256", "conclusion")}


def _sample_seed(
    *,
    seed_root: int,
    objective_sha256: str,
    candidate_sha256: str,
    index: int,
) -> int:
    digest = hashlib.sha256(
        f"{seed_root}:{objective_sha256}:{candidate_sha256}:{index}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _validate_context(
    value: Any,
    *,
    seed: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    fields = {
        "schema",
        "prompt_token_count",
        "generated_token_count",
        "termination",
        "initial_cache_offsets",
        "final_cache_offsets",
        "all_initial_offsets_zero",
        "solver_context_imported",
        "parameter_relation",
        "sample_seed",
        "temperature",
        "top_p",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("prefix-stability context fields do not match schema")
    initial = value["initial_cache_offsets"]
    final = value["final_cache_offsets"]
    if (
        value["schema"] != PREFIX_STABILITY_CONTEXT_SCHEMA
        or value["sample_seed"] != seed
        or value["temperature"] != temperature
        or value["top_p"] != top_p
        or not isinstance(initial, list)
        or not initial
        or any(type(offset) is not int or offset != 0 for offset in initial)
        or not isinstance(final, list)
        or len(final) != len(initial)
        or any(type(offset) is not int or offset < 0 for offset in final)
        or len(set(final)) != 1
        or value["all_initial_offsets_zero"] is not True
        or value["solver_context_imported"] is not False
        or value["parameter_relation"] != "shared_resident_checkpoint"
        or type(value["prompt_token_count"]) is not int
        or value["prompt_token_count"] < 1
        or type(value["generated_token_count"]) is not int
        or value["generated_token_count"] < 1
        or final[0] < value["prompt_token_count"]
        or not isinstance(value["termination"], str)
        or not value["termination"]
    ):
        raise ValueError("prefix-stability context isolation is invalid")
    return dict(value)


def _calibration(
    raw_stability: float | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "target": CALIBRATION_TARGET,
        "selection_authority_admitted": False,
        "correctness_authority_admitted": False,
    }
    if config is None:
        return {
            **base,
            "mode": "bootstrap_uncalibrated",
            "artifact_sha256": "",
            "future_recurrence_probability": None,
            "calibrated": False,
            "reason": (
                "measurement_unavailable"
                if raw_stability is None
                else "calibrator_not_configured"
            ),
        }
    fields = {"mode", "artifact_path", "artifact_sha256"}
    if not isinstance(config, Mapping) or set(config) != fields:
        raise ValueError("prefix-stability calibrator config fields differ")
    if (
        config["mode"] != "learned"
        or not isinstance(config["artifact_path"], str)
        or not config["artifact_path"]
        or len(config["artifact_path"]) > 4096
        or not _SHA_RE.fullmatch(str(config["artifact_sha256"]))
    ):
        raise ValueError("prefix-stability calibrator config is invalid")
    if raw_stability is None:
        return {
            **base,
            "mode": "learned",
            "artifact_sha256": config["artifact_sha256"],
            "future_recurrence_probability": None,
            "calibrated": False,
            "reason": "measurement_unavailable",
        }
    calibrator = PrefixStabilityCalibrator.load(
        config["artifact_path"],
        expected_sha256=config["artifact_sha256"],
    )
    estimate = calibrator.estimate(raw_stability)
    return {
        **base,
        "mode": "learned",
        "artifact_sha256": config["artifact_sha256"],
        "future_recurrence_probability": estimate["future_recurrence_probability"],
        "calibrated": True,
        "reason": "",
    }


def _metrics(
    signatures: list[dict[str, str]],
    *,
    reference: Mapping[str, str],
) -> dict[str, Any]:
    sample_count = len(signatures)
    hashes = [signature["signature_sha256"] for signature in signatures]
    counts = Counter(hashes)
    reference_matches = sum(
        signature["mode"] == reference["mode"]
        and signature["signature_sha256"] == reference["signature_sha256"]
        for signature in signatures
    )
    pair_total = sample_count * (sample_count - 1) // 2
    pair_matches = sum(count * (count - 1) // 2 for count in counts.values())
    entropy = -sum(
        (count / sample_count) * math.log2(count / sample_count)
        for count in counts.values()
    )
    normalized_entropy = entropy / math.log2(sample_count) if sample_count > 1 else 0.0
    reference_agreement = reference_matches / sample_count
    pairwise_agreement = pair_matches / pair_total if pair_total else 1.0
    modal_fraction = max(counts.values()) / sample_count
    raw_stability = min(reference_agreement, pairwise_agreement, modal_fraction)
    return {
        "sample_count": sample_count,
        "reference_matches": reference_matches,
        "pair_total": pair_total,
        "pair_matches": pair_matches,
        "distinct_signatures": len(counts),
        "reference_agreement": round(reference_agreement, 10),
        "pairwise_agreement": round(pairwise_agreement, 10),
        "modal_fraction": round(modal_fraction, 10),
        "normalized_entropy": round(normalized_entropy, 10),
        "raw_stability": round(raw_stability, 10),
    }


def run_prefix_stability_verifier(
    candidate: str,
    *,
    objective: str,
    generate: Callable[[str, int, float, float], Mapping[str, Any]],
    samples: int = 3,
    temperature: float = 0.35,
    top_p: float = 0.9,
    seed_root: int = 104_729,
    calibrator_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Regenerate a conclusion from one machine-verified prefix."""

    if not isinstance(candidate, str) or len(candidate) > 16_384:
        raise ValueError("prefix-stability candidate must be a bounded string")
    if not isinstance(objective, str):
        raise TypeError("prefix-stability objective must be a string")
    if type(samples) is not int or not 3 <= samples <= 8:
        raise ValueError("prefix-stability samples must be inside [3, 8]")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0.05 <= float(temperature) <= 1.5
        or isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not 0.1 <= float(top_p) <= 1.0
        or type(seed_root) is not int
        or not -(2**63) <= seed_root <= 2**63 - 1
    ):
        raise ValueError("prefix-stability sampling config is invalid")
    normalized_objective = objective[:8192]
    boundary = _conclusion_boundary(candidate, objective=normalized_objective)
    if boundary is None:
        return unavailable_prefix_stability("verified_prefix_unavailable")
    atomic, prefix_atomic, prefix_router, prefix, reference_conclusion = boundary
    candidate_sha = _text_sha(candidate)
    objective_sha = _text_sha(normalized_objective)
    prefix_sha = _text_sha(prefix)
    prompt = build_prefix_prompt(
        objective=normalized_objective,
        candidate_sha256=candidate_sha,
        prefix=prefix,
        prefix_sha256=prefix_sha,
    )
    reference_signature = _signature(reference_conclusion)
    rows: list[dict[str, Any]] = []
    for index in range(samples):
        seed = _sample_seed(
            seed_root=seed_root,
            objective_sha256=objective_sha,
            candidate_sha256=candidate_sha,
            index=index,
        )
        base = {
            "sample_index": index,
            "sample_seed": seed,
            "prompt_sha256": _text_sha(prompt),
        }
        try:
            generated = generate(prompt, seed, float(temperature), float(top_p))
            if not isinstance(generated, Mapping):
                raise TypeError("prefix-stability generator result must be a mapping")
            generated_text = str(generated.get("text") or "")
            if not 1 <= len(generated_text) <= 4096:
                raise ValueError("prefix-stability generated text is empty or too large")
            context = _validate_context(
                generated.get("context"),
                seed=seed,
                temperature=float(temperature),
                top_p=float(top_p),
            )
        except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
            rows.append(
                {
                    **base,
                    "status": "generation_failed",
                    "generated_text": "",
                    "generated_text_sha256": "",
                    "conclusion": "",
                    "conclusion_sha256": "",
                    "signature": {},
                    "context": {},
                    "reason": f"{type(exc).__name__}:{exc}"[:240],
                }
            )
        else:
            try:
                parsed = parse_prefix_result(
                    generated_text,
                    candidate_sha256=candidate_sha,
                    prefix_sha256=prefix_sha,
                )
                signature = _signature(parsed["conclusion"])
            except (TypeError, ValueError) as exc:
                rows.append(
                    {
                        **base,
                        "status": "contract_refused",
                        "generated_text": generated_text,
                        "generated_text_sha256": _text_sha(generated_text),
                        "conclusion": "",
                        "conclusion_sha256": "",
                        "signature": {},
                        "context": context,
                        "reason": f"{type(exc).__name__}:{exc}"[:240],
                    }
                )
            else:
                rows.append(
                    {
                        **base,
                        "status": "complete",
                        "generated_text": generated_text,
                        "generated_text_sha256": _text_sha(generated_text),
                        "conclusion": parsed["conclusion"],
                        "conclusion_sha256": _text_sha(parsed["conclusion"]),
                        "signature": signature,
                        "context": context,
                        "reason": "",
                    }
                )
    complete = [row for row in rows if row["status"] == "complete"]
    admitted = len(complete) == samples
    metrics = (
        _metrics(
            [dict(row["signature"]) for row in complete],
            reference=reference_signature,
        )
        if admitted
        else {
            "sample_count": len(complete),
            "reference_matches": 0,
            "pair_total": 0,
            "pair_matches": 0,
            "distinct_signatures": 0,
            "reference_agreement": None,
            "pairwise_agreement": None,
            "modal_fraction": None,
            "normalized_entropy": None,
            "raw_stability": None,
        }
    )
    raw_stability = metrics["raw_stability"] if admitted else None
    payload = {
        "schema": PREFIX_STABILITY_SCHEMA,
        "implementation_sha256": _implementation_sha256(),
        "objective_text": normalized_objective,
        "objective_sha256": objective_sha,
        "candidate_text": candidate,
        "candidate_sha256": candidate_sha,
        "atomic_decomposition": atomic,
        "prefix_atomic_decomposition": prefix_atomic,
        "prefix_deterministic_router": prefix_router,
        "prefix_text": prefix,
        "prefix_sha256": prefix_sha,
        "reference_conclusion": reference_conclusion,
        "reference_conclusion_sha256": _text_sha(reference_conclusion),
        "reference_signature": reference_signature,
        "samples_requested": samples,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed_root": seed_root,
        "samples": rows,
        "measurement_admitted": admitted,
        "metrics": metrics,
        "calibration": _calibration(raw_stability, calibrator_config),
        "parameter_independence": False,
        "context_independence": True,
        "authority_mode": "diagnostic_conclusion_recurrence_only",
        "selection_authority_admitted": False,
        "correctness_authority_admitted": False,
        "selection_effect": "none",
        "correctness_effect": "none",
    }
    return validate_prefix_stability_envelope(
        {**payload, "receipt_sha256": _sha(payload)},
        expected_calibrator_config=calibrator_config,
    )


def validate_prefix_stability_envelope(
    value: Any,
    *,
    expected_calibrator_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct the prefix boundary, samples, metric, and calibration."""

    fields = {
        "schema",
        "implementation_sha256",
        "objective_text",
        "objective_sha256",
        "candidate_text",
        "candidate_sha256",
        "atomic_decomposition",
        "prefix_atomic_decomposition",
        "prefix_deterministic_router",
        "prefix_text",
        "prefix_sha256",
        "reference_conclusion",
        "reference_conclusion_sha256",
        "reference_signature",
        "samples_requested",
        "temperature",
        "top_p",
        "seed_root",
        "samples",
        "measurement_admitted",
        "metrics",
        "calibration",
        "parameter_independence",
        "context_independence",
        "authority_mode",
        "selection_authority_admitted",
        "correctness_authority_admitted",
        "selection_effect",
        "correctness_effect",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("prefix-stability verifier fields do not match schema")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _sha(payload):
        raise ValueError("prefix-stability verifier commitment mismatch")
    if (
        value["schema"] != PREFIX_STABILITY_SCHEMA
        or value["implementation_sha256"] != _implementation_sha256()
        or value["parameter_independence"] is not False
        or value["context_independence"] is not True
        or value["authority_mode"] != "diagnostic_conclusion_recurrence_only"
        or value["selection_authority_admitted"] is not False
        or value["correctness_authority_admitted"] is not False
        or value["selection_effect"] != "none"
        or value["correctness_effect"] != "none"
    ):
        raise ValueError("prefix-stability identity or authority boundary is invalid")
    objective = value["objective_text"]
    candidate = value["candidate_text"]
    if (
        not isinstance(objective, str)
        or len(objective) > 8192
        or not isinstance(candidate, str)
        or len(candidate) > 16_384
        or value["objective_sha256"] != _text_sha(objective)
        or value["candidate_sha256"] != _text_sha(candidate)
    ):
        raise ValueError("prefix-stability source binding is invalid")
    boundary = _conclusion_boundary(candidate, objective=objective)
    if boundary is None:
        raise ValueError("prefix-stability verified prefix no longer reconstructs")
    atomic, prefix_atomic, prefix_router, prefix, conclusion = boundary
    validate_atomic_decomposition(
        value["atomic_decomposition"],
        candidate=candidate,
        objective=objective,
    )
    validate_atomic_decomposition(
        value["prefix_atomic_decomposition"],
        candidate=prefix,
        objective=objective,
    )
    validate_deterministic_router_envelope(
        value["prefix_deterministic_router"],
        atomic_receipt=value["prefix_atomic_decomposition"],
    )
    if (
        value["atomic_decomposition"] != atomic
        or value["prefix_atomic_decomposition"] != prefix_atomic
        or value["prefix_deterministic_router"] != prefix_router
        or value["prefix_text"] != prefix
        or value["prefix_sha256"] != _text_sha(prefix)
        or value["reference_conclusion"] != conclusion
        or value["reference_conclusion_sha256"] != _text_sha(conclusion)
        or value["reference_signature"] != _signature(conclusion)
    ):
        raise ValueError("prefix-stability boundary evidence differs")
    samples = value["samples_requested"]
    temperature = value["temperature"]
    top_p = value["top_p"]
    seed_root = value["seed_root"]
    if (
        type(samples) is not int
        or not 3 <= samples <= 8
        or isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0.05 <= float(temperature) <= 1.5
        or isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not 0.1 <= float(top_p) <= 1.0
        or type(seed_root) is not int
        or not -(2**63) <= seed_root <= 2**63 - 1
    ):
        raise ValueError("prefix-stability sampling configuration is invalid")
    rows = value["samples"]
    if not isinstance(rows, list) or len(rows) != samples:
        raise ValueError("prefix-stability sample inventory differs")
    prompt = build_prefix_prompt(
        objective=objective,
        candidate_sha256=value["candidate_sha256"],
        prefix=prefix,
        prefix_sha256=value["prefix_sha256"],
    )
    complete_signatures: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        fields = {
            "sample_index",
            "sample_seed",
            "prompt_sha256",
            "status",
            "generated_text",
            "generated_text_sha256",
            "conclusion",
            "conclusion_sha256",
            "signature",
            "context",
            "reason",
        }
        seed = _sample_seed(
            seed_root=seed_root,
            objective_sha256=value["objective_sha256"],
            candidate_sha256=value["candidate_sha256"],
            index=index,
        )
        if (
            not isinstance(row, Mapping)
            or set(row) != fields
            or row["sample_index"] != index
            or row["sample_seed"] != seed
            or row["prompt_sha256"] != _text_sha(prompt)
            or row["status"] not in _SAMPLE_STATUSES
        ):
            raise ValueError("prefix-stability sample identity is invalid")
        if row["status"] == "complete":
            generated_text = row["generated_text"]
            if (
                not isinstance(generated_text, str)
                or not 1 <= len(generated_text) <= 4096
                or row["generated_text_sha256"] != _text_sha(generated_text)
                or row["reason"] != ""
            ):
                raise ValueError("prefix-stability complete sample evidence is invalid")
            parsed = parse_prefix_result(
                generated_text,
                candidate_sha256=value["candidate_sha256"],
                prefix_sha256=value["prefix_sha256"],
            )
            signature = _signature(parsed["conclusion"])
            if (
                row["conclusion"] != parsed["conclusion"]
                or row["conclusion_sha256"] != _text_sha(parsed["conclusion"])
                or row["signature"] != signature
            ):
                raise ValueError("prefix-stability conclusion evidence differs")
            _validate_context(
                row["context"],
                seed=seed,
                temperature=float(temperature),
                top_p=float(top_p),
            )
            complete_signatures.append(signature)
        elif row["status"] == "contract_refused":
            generated_text = row["generated_text"]
            if (
                not isinstance(generated_text, str)
                or not 1 <= len(generated_text) <= 4096
                or row["generated_text_sha256"] != _text_sha(generated_text)
                or row["conclusion"] != ""
                or row["conclusion_sha256"] != ""
                or row["signature"] != {}
                or not isinstance(row["reason"], str)
                or not row["reason"]
                or len(row["reason"]) > 240
            ):
                raise ValueError("prefix-stability refused-contract evidence is invalid")
            _validate_context(
                row["context"],
                seed=seed,
                temperature=float(temperature),
                top_p=float(top_p),
            )
            try:
                parse_prefix_result(
                    generated_text,
                    candidate_sha256=value["candidate_sha256"],
                    prefix_sha256=value["prefix_sha256"],
                )
            except (TypeError, ValueError) as exc:
                logger.debug(
                    "%s unavailable (%s: %s); a prefix result went unparsed, so this sample proves nothing about stability",
                    "it",
                    type(exc).__name__,
                    exc,
                )
            else:
                raise ValueError("prefix-stability valid contract was marked refused")
        elif (
            row["generated_text"] != ""
            or row["generated_text_sha256"] != ""
            or row["conclusion"] != ""
            or row["conclusion_sha256"] != ""
            or row["signature"] != {}
            or row["context"] != {}
            or not isinstance(row["reason"], str)
            or not row["reason"]
            or len(row["reason"]) > 240
        ):
            raise ValueError("prefix-stability generation-failure evidence is invalid")
    admitted = len(complete_signatures) == samples
    expected_metrics = (
        _metrics(complete_signatures, reference=value["reference_signature"])
        if admitted
        else {
            "sample_count": len(complete_signatures),
            "reference_matches": 0,
            "pair_total": 0,
            "pair_matches": 0,
            "distinct_signatures": 0,
            "reference_agreement": None,
            "pairwise_agreement": None,
            "modal_fraction": None,
            "normalized_entropy": None,
            "raw_stability": None,
        }
    )
    if (
        value["measurement_admitted"] is not admitted
        or value["metrics"] != expected_metrics
    ):
        raise ValueError("prefix-stability metric does not reconstruct")
    expected_calibration = _calibration(
        expected_metrics["raw_stability"] if admitted else None,
        expected_calibrator_config,
    )
    if value["calibration"] != expected_calibration:
        raise ValueError("prefix-stability calibration does not reconstruct")
    return dict(value)


__all__ = [
    "PREFIX_STABILITY_CONTEXT_SCHEMA",
    "PREFIX_STABILITY_SCHEMA",
    "build_prefix_prompt",
    "parse_prefix_result",
    "run_prefix_stability_verifier",
    "unavailable_prefix_stability",
    "validate_prefix_stability_envelope",
]
