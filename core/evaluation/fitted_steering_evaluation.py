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
PAIRED_PROTOCOL: Final = "aura.fitted_basis.paired_controls.v2"


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


def _replay_legacy(result: Mapping[str, Any]) -> dict[str, Any]:
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


def replay_control_contrasts(result: Mapping[str, Any]) -> dict[str, Any]:
    """Paired control diagnostics; this does not issue a steering authority.

    The older protocol compares total win counts. Here each reply supplies
    one matched pair, and Holm correction covers the three control contrasts.
    This is a post-hoc diagnostic until fixed in a new experiment's protocol;
    repeated replies and fitting custody still need independent review.
    """
    legacy = _replay_legacy(result)
    rows = result["replies"]
    indices = [row.get("index") for row in rows]
    if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(rows):
        raise FittedEvaluationError("paired_controls_need_unique_reply_indices")
    if any(type(value) not in (int, float) for row in rows for value in row["loglik"].values()):
        raise FittedEvaluationError("paired_controls_need_numeric_likelihoods")
    contrasts = {}
    for control in (MATCHED_CONTROL, *LESIONS):
        treatment_only = control_only = both = neither = 0
        for row in rows:
            values = row["loglik"]
            treatment_wins = values[f"{TREATMENT}:own"] > values[f"{TREATMENT}:other"]
            control_wins = values[f"{control}:own"] > values[f"{control}:other"]
            treatment_only += treatment_wins and not control_wins
            control_only += control_wins and not treatment_wins
            both += treatment_wins and control_wins
            neither += not treatment_wins and not control_wins
        contrasts[control] = {
            "treatment_only": treatment_only, "control_only": control_only,
            "both": both, "neither": neither,
            "one_sided_exact_p": sign_test_p(treatment_only, treatment_only + control_only),
        }
    prior = 0.0
    ordered = sorted(contrasts, key=lambda arm: contrasts[arm]["one_sided_exact_p"])
    for rank, arm in enumerate(ordered):
        adjusted = min(1.0, max(prior, contrasts[arm]["one_sided_exact_p"] * (len(ordered) - rank)))
        contrasts[arm]["holm_adjusted_p"] = adjusted
        contrasts[arm]["rejects"] = adjusted < SIGN_TEST_LEVEL
        prior = adjusted
    return {
        "schema": "aura.fitted_basis.paired_control_diagnostic.v1",
        "result_sha256": canonical_sha256(result),
        "scope": "post_hoc_paired_binary_win_diagnostic",
        "sample_count": len(rows), "alpha": SIGN_TEST_LEVEL,
        "contrasts": contrasts,
        "all_control_contrasts_significant": all(row["rejects"] for row in contrasts.values()),
        "legacy_protocol_passes": legacy["causal_effect_positive"],
        "serving_authority": False,
        "general_task_quality_measured": False,
    }


def paired_qualification_plan(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the analysis and input identity before a new producer scores replies.

    A matching hash proves consistency of a record, not independent custody or
    when it was authored. Those require an external experiment review.
    """
    return {
        "protocol": PAIRED_PROTOCOL,
        "inputs": {key: identity[key] for key in (
            "model_descriptor_sha256", "generation_sha256", "held_out_replies_sha256", "alpha", "seed")},
        "min_replies": MIN_REPLIES,
        "alpha": SIGN_TEST_LEVEL,
        "test": "one_sided_exact_binomial_on_discordant_paired_binary_wins",
        "family": [MATCHED_CONTROL, *LESIONS],
        "correction": "Holm",
        "acceptance": "all_corrected_p_strictly_below_alpha_and_legacy_requirements",
    }


def replay(result: Mapping[str, Any]) -> dict[str, Any]:
    """Replay historical records faithfully and require paired tests for v2."""
    protocol = result.get("qualification_protocol")
    if protocol is None:
        return _replay_legacy(result)
    if protocol != PAIRED_PROTOCOL:
        raise FittedEvaluationError("fitted_qualification_protocol_unknown")
    try:
        expected = paired_qualification_plan(result)
        if result.get("analysis_plan") != expected or result.get("analysis_plan_sha256") != canonical_sha256(expected):
            raise FittedEvaluationError("fitted_analysis_plan_mismatch")
    except (KeyError, TypeError) as exc:
        raise FittedEvaluationError("fitted_analysis_plan_invalid") from exc
    legacy = _replay_legacy(result)
    paired = replay_control_contrasts(result)
    unmet = list(legacy["unmet_requirements"])
    if not paired["all_control_contrasts_significant"]:
        unmet.append("paired_control_superiority_not_established")
    return {**legacy, "qualification_protocol": protocol, "paired_controls": paired["contrasts"],
            "causal_effect_positive": not unmet, "unmet_requirements": unmet}


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
    if result.get("qualification_protocol") == PAIRED_PROTOCOL:
        body.update(qualification_protocol=PAIRED_PROTOCOL,
                    analysis_plan_sha256=result["analysis_plan_sha256"],
                    paired_controls=replayed["paired_controls"])
    return {**body, "evaluation_sha256": canonical_sha256(body)}


__all__ = [
    "ARMS",
    "GENERATION_SCHEMA",
    "LESIONS",
    "MATCHED_CONTROL",
    "PAIRED_PROTOCOL",
    "RESULT_SCHEMA",
    "TREATMENT",
    "VERIFIER_SCHEMA",
    "FittedEvaluationError",
    "build_causal_evaluation",
    "build_independent_verifier_evidence",
    "derangement",
    "fitted_generation_metadata",
    "replay",
    "replay_control_contrasts",
    "paired_qualification_plan",
    "sign_test_p",
]
