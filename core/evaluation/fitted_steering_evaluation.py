"""Whether directions fitted to her replies carry the state each reply was said in.

The causal evaluation that qualifies a fitted steering generation (prereg
addendum, 23 October, midday). The September generation was qualified by
whether its directions made her use more affect words than a prompt asking for
them. Directions fitted to keep her voice make a narrower claim, and this is
its test: steered by the state a reply was said in, the reply is more likely
than steered by another of her states.

Each held-out reply is scored four ways, each at its own state and at another
reply's state: her directions (treatment), random unit directions that keep
each direction's reference, spread and push (matched control), every direction
zeroed, and her directions at the wrong layers (lesions). The producer records
log-likelihoods only. Everything decided here is recomputed from that record,
so the replay and the adjudication cannot drift from what was measured.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from core.evaluation.caa_causal_evaluation import canonical_sha256, validate_vector_generation
from core.learning.cortex_migration_authority import FITTED_EVALUATION_SCHEMA

GENERATION_SCHEMA: Final = "aura.fitted_basis.vector_generation.v1"
RESULT_SCHEMA: Final = "aura.fitted_basis.evaluation_result.v1"
VERIFIER_SCHEMA: Final = "aura.fitted_basis.independent_replay.v1"
VERIFIER: Final = {"name": "aura-fitted-basis-replay", "version": "1"}

TREATMENT: Final = "her_directions"
MATCHED_CONTROL: Final = "random_directions"
LESIONS: Final = ("zero_vector", "shuffled_layers")
ARMS: Final = (TREATMENT, MATCHED_CONTROL, *LESIONS)

#: The one-sided sign test the treatment has to pass, as preregistered.
SIGN_TEST_LEVEL: Final = 0.05
#: The authority refuses an evaluation on fewer samples than this.
MIN_REPLIES: Final = 24


class FittedEvaluationError(ValueError):
    """The record cannot support the claim a fitted generation makes."""


#: The statement of how the directions were fitted, and the replies held out
#: from the fit, both inside the generation directory.
STATEMENT_FILE: Final = "basis.json"
REPLIES_FILE: Final = "held_out_replies.jsonl"


def fitted_generation_metadata(root: Path, *, descriptor: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    """The sealed statement of a generation of fitted directions: how made, which files, which replies held out."""
    statement = json.loads((root / STATEMENT_FILE).read_text(encoding="utf-8"))
    if statement.get("model_descriptor_sha256") != descriptor:
        raise FittedEvaluationError("fitted_directions_belong_to_another_checkpoint")
    vector_files = [
        {"name": path.name, "size_bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(root.glob("*_layer*.npz"))
    ]
    if not vector_files:
        raise FittedEvaluationError("fitted_generation_has_no_vectors")
    extraction = {
        "method": "fitted_to_her_replies",
        "statistic": "sum_k tanh((w_k(s) - reference_k) / spread_k) push_k v_k, added at each position's own norm",
        "fit": {key: value for key, value in statement.items() if key != "model_descriptor_sha256"},
        "held_out_replies_sha256": hashlib.sha256((root / REPLIES_FILE).read_bytes()).hexdigest(),
        "model_path": str(plan["model_path"]),
        "model_descriptor_sha256": descriptor,
        "hidden_size": int(plan["hidden_size"]),
        "target_layers": [{"index": int(row["index"]), "kind": str(row["kind"])} for row in plan["target_layers"]],
    }
    extraction["extraction_contract_sha256"] = canonical_sha256(extraction)
    metadata = {
        "schema": GENERATION_SCHEMA,
        "model_identity": {"model_descriptor_sha256": descriptor},
        "extraction_contract": extraction,
        "vector_files": vector_files,
    }
    metadata["generation_sha256"] = canonical_sha256(
        {"extraction_contract_sha256": extraction["extraction_contract_sha256"], "vector_files": vector_files}
    )
    return metadata


def derangement(count: int, seed: int) -> list[int]:
    """A permutation of range(count) that moves every index, drawn from `seed`."""
    if count < 2:
        raise FittedEvaluationError("fitted_derangement_needs_two")
    rng = random.Random(seed)
    order = list(range(count))
    while True:
        rng.shuffle(order)
        if all(index != moved for index, moved in enumerate(order)):
            return list(order)


def sign_test_p(successes: int, trials: int) -> float:
    """One-sided P(X >= successes) for X ~ Binomial(trials, 1/2)."""
    if trials <= 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(successes, trials + 1))
    return tail / 2**trials


def replay(result: Mapping[str, Any]) -> dict[str, Any]:
    """Every count the claim rests on, recomputed from the per-reply record."""
    rows = result.get("replies")
    if result.get("schema") != RESULT_SCHEMA or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise FittedEvaluationError("fitted_result_invalid")
    if len(rows) < MIN_REPLIES:
        raise FittedEvaluationError("fitted_result_too_few_replies")
    needed = {"plain", *(f"{arm}:{at}" for arm in ARMS for at in ("own", "other"))}
    successes = dict.fromkeys(ARMS, 0)
    ties = dict.fromkeys(ARMS, 0)
    voice: list[float] = []
    for row in rows:
        loglik = row.get("loglik") if isinstance(row, Mapping) else None
        tokens = row.get("tokens") if isinstance(row, Mapping) else None
        if (
            not isinstance(loglik, Mapping)
            or set(loglik) != needed
            or type(tokens) is not int
            or tokens < 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in loglik.values())
        ):
            raise FittedEvaluationError("fitted_result_row_invalid")
        for arm in ARMS:
            own, other = float(loglik[f"{arm}:own"]), float(loglik[f"{arm}:other"])
            successes[arm] += own > other
            ties[arm] += own == other
        voice.append((float(loglik[f"{TREATMENT}:own"]) - float(loglik["plain"])) / tokens)
    untied = len(rows) - ties[TREATMENT]
    p_value = sign_test_p(successes[TREATMENT], untied)
    voice_delta = sum(voice) / len(voice)
    lesion_successes = {arm: successes[arm] for arm in LESIONS}
    unmet = []
    if successes[TREATMENT] <= successes[MATCHED_CONTROL]:
        unmet.append("random_directions_carry_the_state_as_well")
    if any(count >= successes[TREATMENT] for count in lesion_successes.values()):
        unmet.append("a_lesion_keeps_the_effect")
    if voice_delta < 0:
        unmet.append("her_replies_less_likely_under_the_steering")
    if untied == 0 or p_value >= SIGN_TEST_LEVEL:
        unmet.append("own_state_not_more_likely_than_another_by_sign_test")
    return {
        "sample_count": len(rows),
        "treatment_successes": successes[TREATMENT],
        "matched_control_successes": successes[MATCHED_CONTROL],
        "lesion_successes": lesion_successes,
        "ties": ties,
        "sign_test_p": p_value,
        "voice_delta": voice_delta,
        "no_regression": voice_delta >= 0,
        "causal_effect_positive": not unmet,
        "unmet_requirements": unmet,
    }


def build_independent_verifier_evidence(
    *,
    result: Mapping[str, Any],
    result_sha256: str,
    metadata: Mapping[str, Any],
    metadata_sha256: str,
    generation_dir: Path,
) -> dict[str, Any]:
    """Reopen every vector file and recompute every count from the record."""
    descriptor = str(result.get("model_descriptor_sha256") or "")
    generation = validate_vector_generation(
        metadata, generation_dir=generation_dir, model_descriptor_sha256=descriptor
    )
    if result.get("generation_sha256") != generation["generation_sha256"]:
        raise FittedEvaluationError("fitted_result_names_another_generation")
    if result.get("held_out_replies_sha256") != metadata["extraction_contract"].get("held_out_replies_sha256"):
        raise FittedEvaluationError("fitted_result_scored_other_replies")
    replayed = replay(result)
    body = {
        "schema": VERIFIER_SCHEMA,
        "verifier": dict(VERIFIER),
        "result_sha256": result_sha256,
        "result_payload_sha256": canonical_sha256(result),
        "metadata_sha256": metadata_sha256,
        **generation,
        "replay": replayed,
        "verified": bool(replayed["causal_effect_positive"] and replayed["no_regression"]),
    }
    return {**body, "verification_sha256": canonical_sha256(body)}


def build_causal_evaluation(
    *,
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
    verifier_evidence: Mapping[str, Any],
    verifier_evidence_sha256: str,
) -> dict[str, Any]:
    """The authority's input, issued only from a replay that still agrees with the record."""
    unsigned = {key: value for key, value in verifier_evidence.items() if key != "verification_sha256"}
    if (
        verifier_evidence.get("schema") != VERIFIER_SCHEMA
        or verifier_evidence.get("verified") is not True
        or verifier_evidence.get("verification_sha256") != canonical_sha256(unsigned)
        or verifier_evidence.get("result_payload_sha256") != canonical_sha256(result)
        or verifier_evidence.get("model_descriptor_sha256") != result.get("model_descriptor_sha256")
        or verifier_evidence.get("generation_sha256") != metadata.get("generation_sha256")
        or verifier_evidence.get("replay") != replay(result)
    ):
        raise FittedEvaluationError("fitted_independent_verifier_invalid")
    replayed = verifier_evidence["replay"]
    if replayed["causal_effect_positive"] is not True or replayed["no_regression"] is not True:
        raise FittedEvaluationError("fitted_generation_not_qualified")
    body = {
        "schema": FITTED_EVALUATION_SCHEMA,
        "model_descriptor_sha256": result["model_descriptor_sha256"],
        "generation_sha256": metadata["generation_sha256"],
        "verdict": "PASS",
        "qualified": True,
        "sample_count": replayed["sample_count"],
        "treatment_successes": replayed["treatment_successes"],
        "matched_control_successes": replayed["matched_control_successes"],
        "lesion_successes": dict(replayed["lesion_successes"]),
        "no_regression": True,
        "causal_effect_positive": True,
        "independent_verifier": {
            "name": verifier_evidence["verifier"]["name"],
            "version": verifier_evidence["verifier"]["version"],
            "evidence_sha256": verifier_evidence_sha256,
        },
    }
    return {**body, "evaluation_sha256": canonical_sha256(body)}


__all__ = [
    "ARMS",
    "GENERATION_SCHEMA",
    "LESIONS",
    "MATCHED_CONTROL",
    "RESULT_SCHEMA",
    "TREATMENT",
    "VERIFIER_SCHEMA",
    "FittedEvaluationError",
    "build_causal_evaluation",
    "build_independent_verifier_evidence",
    "derangement",
    "fitted_generation_metadata",
    "replay",
    "sign_test_p",
]
