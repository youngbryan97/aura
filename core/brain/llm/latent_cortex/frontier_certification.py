"""Machine-checkable release gate for Recursive Latent Cortex frontier gains.

Unit tests can prove this verifier rejects weak evidence. Only a completed
resident-32B bundle can prove the capability claim itself.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from core.brain.frontier_evidence_v5 import verify_signed_envelope
from core.brain.llm.latent_cortex.experiments import (
    CONJECTURE,
    PROVEN,
    ExperimentProvenance,
    PairedObservation,
    grade_paired_treatment_vs_control,
)
from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    certify_comparison_accounting,
    validate_information_receipt,
    validate_resource_receipt,
)
from core.evaluation.paired_power import binomial_tail, conditional_mcnemar_power

SCHEMA = "aura.latent_cortex.frontier_gain_bundle.v2"
CERTIFICATE_SCHEMA = "aura.latent_cortex.frontier_gain_certificate.v2"
INDEPENDENT_ATTESTATION_SCHEMA = (
    "aura.latent_cortex.frontier_gain_independent_attestation.v1"
)
TASK_COMMITMENT_ATTESTATION_SCHEMA = (
    "aura.latent_cortex.frontier_gain_task_commitment.v1"
)
PRODUCER_ATTESTATION_SCHEMA = (
    "aura.latent_cortex.frontier_gain_producer_attestation.v1"
)
#: A single signature is one organization's opinion. A release-grade frontier
#: claim needs agreement from independently pinned verifiers who do not share
#: an organization with each other, the producer, or the task issuer.
MINIMUM_VERIFIER_QUORUM = 2
#: What backs `params_unchanged`. The measurement is a fixed-stride canary
#: over the parameter tree taken before and after the episode, not a Merkle
#: root over every tensor byte — a mutation confined to unsampled elements
#: would not be seen. The certificate names this rather than implying an
#: exhaustive comparison, and gates the canary's coverage.
PARAMETER_INTEGRITY_ATTESTATION = "sampled_canary_over_parameter_tree"
#: What an independent verifier's signature proves it DID. The signer binds
#: the artifact receipt it produced by reading the raw files, so possession of
#: a trusted key is no longer enough on its own. No measured-boot or enclave
#: evidence backs the claim that the pinned verifier binary is what ran, and
#: the name says so rather than implying a measured environment.
VERIFIER_EXECUTION_ATTESTATION = "signed_over_recomputed_artifact_receipt"
#: What ties the running binary to source. The builder asserts that rebuilding
#: the named commit reproduces the worker binary; nobody re-runs the build at
#: certification time and no release signature is checked.
BUILD_PROVENANCE_ATTESTATION = "builder_asserted_rebuild_match_unsigned"
_COMPARISON_KINDS = {
    "resident_32b_vs_vanilla_same_checkpoint",
    "resident_32b_vs_external_frontier",
}
_SHA256_LENGTH = 64
# A ceiling that only excludes a 64-bit overflow is not a budget. The real
# bound is preregistered per trial and enforced against the architecture; this
# stays as the arithmetic guard it always was.
_MAX_LAYER_APPS = (1 << 63) - 1
#: How far the shapes may sit from the declared parameter count. The
#: structural count omits norms, biases and rotary tables, which are well
#: under one percent of a decoder's weights; five percent leaves room for
#: tied embeddings and vocabulary padding without admitting a different model.
_PARAMETER_RECONCILIATION_TOLERANCE = 0.05


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


#: A trust pin now names the organization holding the key. Independence
#: between roles is an organizational claim, not a key-management one.
_TRUST_PIN_FIELDS = frozenset(
    {"public_key_b64", "implementation_sha256", "release_sha256", "organization"}
)


def evidence_payload_sha256(bundle: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema": bundle.get("schema"),
            "producer_id": bundle.get("producer_id"),
            "preregistration": bundle.get("preregistration"),
            "preregistration_sha256": bundle.get("preregistration_sha256"),
            "resident_model": bundle.get("resident_model"),
            "task_commitment_sha256": bundle.get("task_commitment_sha256"),
            "task_commitment": bundle.get("task_commitment"),
            "raw_artifact_manifest_sha256": bundle.get(
                "raw_artifact_manifest_sha256"
            ),
            "task_diversity": bundle.get("task_diversity"),
            "blinding": bundle.get("blinding"),
            "trials": bundle.get("trials"),
        }
    )


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and (number > 0.0 if positive else True)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


#: Providers whose hosted models may stand as an EXTERNAL FRONTIER control.
#:
#: The check used to be "is the string non-empty". That let any named model be
#: declared a frontier baseline, so a certificate reading "resident 32B beat an
#: external frontier" could have been earned against something weaker than the
#: treatment — the claim would be internally consistent and externally
#: meaningless.
#:
#: A provider allowlist is a PROXY for frontier capability, not a proof of it,
#: and it is deliberately narrow and boring to extend: adding an entry is a
#: reviewable edit here rather than a string a producer supplies at run time.
#: What it buys is that the comparison names a lab that ships frontier models,
#: which is the part a self-declared string could not establish at all.
_RECOGNISED_FRONTIER_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai",
        "anthropic",
        "google",
        "google-deepmind",
        "deepmind",
        "meta",
        "mistral",
        "xai",
        "deepseek",
        "alibaba",
        "qwen",
        "cohere",
    }
)

#: What an external control's identity is actually backed by. Named so nothing
#: downstream can read a provider's own word as an independent attestation.
EXTERNAL_CONTROL_ATTESTATION = "provider_asserted_unsigned"

#: What the RESIDENT model's identity is backed by.
#:
#: The bundle is checked hard for internal consistency: the parameter count
#: must be manifest-derived, the manifest digest must equal the declared
#: checkpoint fingerprint, that fingerprint must equal the preregistered one,
#: architecture and quantization must be present, and the architecture freeze
#: must match. Those catch a producer contradicting itself.
#:
#: What they cannot catch is a producer that is consistently wrong. Every
#: identity hash — worker binary, app build, build provenance — is validated
#: for SYNTAX only; nothing resolves them against bytes on disk, and no binary
#: signature is verified. So this establishes a coherent self-report bound to
#: preregistration, not an independent attestation of what ran.
RESIDENT_MODEL_ATTESTATION = "self_reported_internally_consistent"


def _normalised_provider(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def _recognised_frontier_provider(value: Any) -> bool:
    provider = _normalised_provider(value)
    if not provider:
        return False
    if provider in _RECOGNISED_FRONTIER_PROVIDERS:
        return True
    # "google-deepmind/gemini" and "openai:gpt" name a recognised lab too.
    head = provider.split("/")[0].split(":")[0]
    return head in _RECOGNISED_FRONTIER_PROVIDERS


#: The two orders a paired trial can run in.
_RUN_ORDERS = ("treatment_first", "control_first")


def _binomial_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``, computed exactly."""
    return binomial_tail(k, n, p)


def _exact_mcnemar_power(discordant: int, alpha: float, win_share: float) -> float:
    """Power of the exact one-sided McNemar test at the preregistered alternative.

    This is NOT post-hoc power. Post-hoc power substitutes the effect the
    study happened to observe and is therefore just the p-value wearing a
    different hat. Here the alternative is the one preregistered before any
    trial ran; only the sample size is realized, which is exactly what
    changes when trials are excluded.
    """
    if discordant <= 0:
        return 0.0
    return conditional_mcnemar_power(discordant, alpha, win_share)


def _validate_preregistration(prereg: Any, reasons: list[str]) -> dict[str, Any]:
    if not isinstance(prereg, dict):
        reasons.append("missing_preregistration")
        return {}
    domains = prereg.get("domains")
    if (
        not isinstance(domains, list)
        or len(domains) < 2
        or any(not isinstance(domain, str) or not domain.strip() for domain in domains)
        or len(set(domains)) != len(domains)
    ):
        reasons.append("invalid_preregistered_domains")
    if prereg.get("comparison_kind") not in _COMPARISON_KINDS:
        reasons.append("invalid_comparison_kind")
    if not _is_sha256(prereg.get("architecture_freeze_sha256")):
        reasons.append("missing_architecture_freeze_sha256")
    if not _is_sha256(prereg.get("tool_policy_sha256")):
        reasons.append("missing_tool_policy_sha256")
    if not _is_sha256(prereg.get("compute_estimator_sha256")):
        reasons.append("missing_compute_estimator_sha256")
    # CP126 3f30dcf3: the estimator was pinned and the ARCHITECTURE it
    # estimated against was not. A producer could pin the right estimator and
    # hand it a toy decoder, and the FLOPs comparison between arms would be
    # internally consistent and meaningless.
    if not _is_sha256(prereg.get("compute_profile_sha256")):
        reasons.append("missing_compute_profile_sha256")
    # CP126 d0f2ae6c: `params_unchanged` rests on a SAMPLED canary. How much
    # of the parameter tree that sample touches decides what the claim is
    # worth, and it was neither checked nor reported.
    coverage_floor = prereg.get("min_parameter_canary_tensor_coverage")
    if not _finite_number(coverage_floor) or not 0.0 < float(coverage_floor) <= 1.0:
        reasons.append("invalid_min_parameter_canary_tensor_coverage")
    # CP126 b4dc41d3: eight counters had to be positive and nothing capped
    # them. A forward pass is the whole stack, so the budget is expressed in
    # passes and checked against the architecture's own layer count.
    max_forward_passes = prereg.get("max_forward_passes_per_trial")
    if type(max_forward_passes) is not int or max_forward_passes <= 0:
        reasons.append("invalid_max_forward_passes_per_trial")
    if not _is_sha256(prereg.get("treatment_checkpoint_fingerprint")):
        reasons.append("missing_treatment_checkpoint_fingerprint")
    if not _finite_number(prereg.get("frozen_at"), positive=True):
        reasons.append("invalid_freeze_timestamp")
    min_trials = prereg.get("min_trials_per_domain")
    if type(min_trials) is not int or min_trials < 30:
        reasons.append("min_trials_per_domain_below_30")
    alpha = prereg.get("alpha")
    if not _finite_number(alpha) or not 0.0 < float(alpha) <= 0.05:
        reasons.append("invalid_alpha")
    # A trial COUNT is not a power analysis. "At least 30 per domain" says
    # nothing about what effect the study could detect, and it is decided
    # before any trial is excluded — so a bundle that threw away half its
    # trials for contamination kept the same claim to adequacy. The study
    # must declare the power it was sized for and the alternative it was
    # sized against, and achieved power is recomputed from what survived.
    target_power = prereg.get("target_power")
    if not _finite_number(target_power) or not 0.5 < float(target_power) <= 0.999:
        reasons.append("invalid_target_power")
    # McNemar conditions on discordant pairs, so the alternative belongs on
    # the discordant split: the share of disagreements the treatment is
    # expected to win. Sizing against a proportion DIFFERENCE and converting
    # would hide the assumed disagreement rate inside the arithmetic.
    win_share = prereg.get("preregistered_discordant_win_share")
    if not _finite_number(win_share) or not 0.5 < float(win_share) <= 1.0:
        reasons.append("invalid_discordant_win_share")
    # Running the treatment first every time makes order a rival explanation
    # for the gain. Balance alone does not settle it; the size of the order
    # effect has to be declared and then measured.
    max_order_effect = prereg.get("max_order_effect")
    if not _finite_number(max_order_effect) or not 0.0 < float(max_order_effect) <= 0.5:
        reasons.append("invalid_max_order_effect")
    # CP126 e7b9dc9c: pass/fail booleans threw away the margin. A treatment
    # that scored 0.61 against a control's 0.59 and one that scored 0.99
    # against 0.05 produced identical evidence, and where the line sat was
    # never written down — so it could be placed wherever the win was.
    success_threshold = prereg.get("success_threshold")
    if not _finite_number(success_threshold) or not 0.0 < float(success_threshold) <= 1.0:
        reasons.append("invalid_success_threshold")
    sensitivity_band = prereg.get("threshold_sensitivity_band")
    if not _finite_number(sensitivity_band) or not 0.0 < float(sensitivity_band) <= 0.25:
        reasons.append("invalid_threshold_sensitivity_band")
    minimum_effect = prereg.get("minimum_effect")
    if not _finite_number(minimum_effect) or not 0.0 < float(minimum_effect) <= 0.5:
        reasons.append("invalid_minimum_effect")
    compute_tolerance = prereg.get("compute_tolerance")
    if not _finite_number(compute_tolerance) or not 0.0 <= float(compute_tolerance) <= 0.05:
        reasons.append("invalid_compute_tolerance")
    if prereg.get("compute_metric") != "estimated_flops":
        reasons.append("compute_metric_must_be_estimated_flops")
    # The scoring PROGRAM must be frozen with everything else. Only a
    # per-trial scorer *config* (which legitimately embeds the trial id)
    # appeared in lineage, so the scoring rules themselves could be chosen
    # after outputs were known.
    if not _is_sha256(prereg.get("scorer_implementation_sha256")):
        reasons.append("missing_scorer_implementation_sha256")
    # Decoding is an experimental condition, not an implementation detail:
    # unpaired seeds/sampling let outcome differences come from decoding
    # rather than from the treatment.
    if not _is_sha256(prereg.get("decode_policy_sha256")):
        reasons.append("missing_decode_policy_sha256")
    # An ABSOLUTE capability floor: without one, a very weak treatment earns
    # a certificate purely by beating an even weaker control.
    floor = prereg.get("min_treatment_success_rate")
    if not _finite_number(floor) or not 0.0 < float(floor) <= 1.0:
        reasons.append("invalid_min_treatment_success_rate")
    comparison_kind = prereg.get("comparison_kind")
    if comparison_kind == "resident_32b_vs_vanilla_same_checkpoint":
        if not _is_sha256(prereg.get("control_checkpoint_fingerprint")) or prereg.get(
            "control_checkpoint_fingerprint"
        ) != prereg.get("treatment_checkpoint_fingerprint"):
            reasons.append("same_checkpoint_control_not_bound")
    elif comparison_kind == "resident_32b_vs_external_frontier":
        if not str(prereg.get("control_model_id") or "").strip():
            reasons.append("external_frontier_control_model_missing")
        if not str(prereg.get("control_model_build_fingerprint") or "").strip():
            reasons.append("external_frontier_control_build_missing")
        provider = prereg.get("control_provider")
        if not str(provider or "").strip():
            reasons.append("external_frontier_control_provider_missing")
        elif not _recognised_frontier_provider(provider):
            # A frontier claim has to name a lab that ships frontier models.
            # Without this, "beat an external frontier" could be earned against
            # anything the producer chose to call one.
            reasons.append("external_frontier_control_provider_unrecognised")
    return prereg


#: A general frontier claim needs breadth, and breadth is not a string count.
#: Two labels that happen to differ can name the same capability, so the
#: preregistration pins a taxonomy and the domains have to land in distinct
#: capability classes within it.
MINIMUM_CAPABILITY_CLASSES = 3


#: RFC 6962 domain separation. Hashing leaves and interior nodes with
#: different prefixes is what stops a leaf being passed off as a subtree.
_MERKLE_LEAF_PREFIX = b"\x00"
_MERKLE_NODE_PREFIX = b"\x01"


def _merkle_leaf_hash(data: bytes) -> str:
    return hashlib.sha256(_MERKLE_LEAF_PREFIX + data).hexdigest()


def _merkle_root_from_proof(
    leaf_hash: str, *, leaf_index: int, tree_size: int, proof: Sequence[str]
) -> str:
    """Recompute the log root a leaf and its audit path imply (RFC 6962)."""
    if leaf_index < 0 or tree_size <= 0 or leaf_index >= tree_size:
        raise ValueError("inclusion proof indices are outside the tree")
    node = leaf_hash
    index = leaf_index
    size = tree_size
    for sibling in proof:
        if not _is_sha256(sibling):
            raise ValueError("inclusion proof contains a non-digest")
        if size <= 1:
            raise ValueError("inclusion proof is longer than the tree is deep")
        if index % 2 == 1 or index == size - 1:
            while index % 2 == 0:
                index //= 2
                size = (size + 1) // 2
            left, right = sibling, node
        else:
            left, right = node, sibling
        node = hashlib.sha256(
            _MERKLE_NODE_PREFIX + bytes.fromhex(left) + bytes.fromhex(right)
        ).hexdigest()
        index //= 2
        size = (size + 1) // 2
    if size > 1:
        raise ValueError("inclusion proof is shorter than the tree is deep")
    return node


def _validate_timestamp_anchor(
    anchor: Any,
    *,
    event: str,
    committed_data: str,
    trusted_logs: Mapping[str, Mapping[str, Any]] | None,
    reasons: list[str],
) -> float | None:
    """Prove an event was recorded BEFORE the run, not merely dated before it.

    CP126 89ca4271 + 9cccfcb5: frozen_at, task_generated_at,
    evaluation_started_at and committed_at were ordinary numbers inside signed
    payloads. A signature proves who wrote a value, never when the event
    happened, so a producer and issuer holding their own keys could assemble a
    flawless chronology after the results were in and sign it.

    An append-only log breaks that. The event's digest is submitted as a leaf;
    the log returns an index and an audit path; anyone can recompute the root
    those imply and compare it against a root pinned out of band. Backdating
    then requires forging a hash chain against a root nobody controls.

    Returns the logged time when the anchor holds.
    """
    if not isinstance(anchor, dict):
        reasons.append(f"{event}_timestamp_anchor_missing")
        return None
    log_id = str(anchor.get("log_id") or "")
    try:
        pin = trusted_logs.get(log_id) if trusted_logs is not None else None
    except (AttributeError, TypeError, ValueError):
        pin = None
    if not isinstance(pin, Mapping):
        reasons.append(f"{event}_transparency_log_untrusted")
        return None
    pinned_root = str(pin.get("root_sha256") or "")
    pinned_size = pin.get("tree_size")
    if not _is_sha256(pinned_root) or type(pinned_size) is not int or pinned_size <= 0:
        reasons.append(f"{event}_transparency_log_pin_invalid")
        return None
    if anchor.get("leaf_data_sha256") != committed_data:
        # The log entry has to be about THIS event.
        reasons.append(f"{event}_anchor_commits_other_data")
        return None
    leaf_index = anchor.get("leaf_index")
    tree_size = anchor.get("tree_size")
    proof = anchor.get("inclusion_proof")
    if (
        type(leaf_index) is not int
        or type(tree_size) is not int
        or not isinstance(proof, list)
    ):
        reasons.append(f"{event}_inclusion_proof_malformed")
        return None
    if tree_size != pinned_size:
        # A proof against a different tree size is a proof about a different
        # log state than the one that was pinned.
        reasons.append(f"{event}_anchor_tree_size_unpinned")
        return None
    try:
        root = _merkle_root_from_proof(
            _merkle_leaf_hash(bytes.fromhex(committed_data)),
            leaf_index=leaf_index,
            tree_size=tree_size,
            proof=[str(item) for item in proof],
        )
    except (TypeError, ValueError):
        reasons.append(f"{event}_inclusion_proof_invalid")
        return None
    if root != pinned_root:
        reasons.append(f"{event}_inclusion_proof_does_not_reach_pinned_root")
        return None
    logged_at = anchor.get("logged_at")
    if not _finite_number(logged_at, positive=True):
        reasons.append(f"{event}_anchor_time_missing")
        return None
    return float(logged_at)


def _validate_domain_taxonomy(
    bundle: dict[str, Any], prereg: dict[str, Any], reasons: list[str]
) -> list[str]:
    """CP126 eb6bed71: "two unique non-empty strings" was the breadth test.

    ``["math", "maths"]`` satisfied it. So did ``["gsm8k", "gsm8k-hard"]``.
    A claim about general frontier capability was resting on whether two
    labels happened to differ as text.

    The bundle now carries a taxonomy the preregistration pinned by digest,
    every preregistered domain must appear in it, and the domains must land
    in distinct capability classes — the unit breadth is actually about.
    """
    taxonomy = bundle.get("domain_taxonomy")
    if not isinstance(taxonomy, dict):
        reasons.append("domain_taxonomy_missing")
        return []
    expected = str(prereg.get("domain_taxonomy_sha256") or "")
    try:
        digest = canonical_sha256(taxonomy)
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("domain_taxonomy_not_canonical_json")
        return []
    if not expected:
        reasons.append("missing_domain_taxonomy_sha256")
    elif digest != expected:
        reasons.append("domain_taxonomy_not_preregistered")
    if not str(taxonomy.get("ontology_id") or "").strip():
        reasons.append("domain_taxonomy_unattributed")
    entries = taxonomy.get("domains")
    if not isinstance(entries, dict) or not entries:
        reasons.append("domain_taxonomy_empty")
        return []
    classes: list[str] = []
    declared = prereg.get("domains")
    # A malformed preregistration already has its own reason; it must not
    # take this check down on the way past.
    declared = declared if isinstance(declared, list) else []
    for domain in declared:
        entry = entries.get(domain) if isinstance(domain, str) else None
        if not isinstance(entry, dict):
            reasons.append(f"{domain}:domain_outside_taxonomy")
            continue
        capability_class = str(entry.get("capability_class") or "").strip()
        if not capability_class:
            reasons.append(f"{domain}:domain_capability_class_missing")
            continue
        classes.append(capability_class)
    distinct = sorted(set(classes))
    if len(distinct) < MINIMUM_CAPABILITY_CLASSES:
        reasons.append("domain_coverage_too_narrow")
    return distinct


def _validate_contamination_receipt(
    trial: dict[str, Any], trial_id: str, trial_reasons: list[str]
) -> str:
    """Require EVIDENCE that a contamination scan ran, not a boolean saying so.

    ``"contamination_scan_passed": True`` is a claim with nothing behind it —
    the same shape as an absent check reported as a passed one. A producer that
    never scanned, or scanned with a threshold chosen after seeing the overlap,
    produces exactly that field.

    A receipt has to name the scanner that ran, the method, the threshold it
    was held to, and the overlap it actually measured — and the measurement has
    to satisfy the threshold. Returns the scanner digest so the caller can
    require ONE scanner across the run: a per-trial scanner would let a
    producer keep scanning until a trial passed.
    """
    receipt = trial.get("contamination_scan")
    if not isinstance(receipt, dict):
        trial_reasons.append(f"{trial_id}:contamination_scan_receipt_missing")
        return ""
    scanner = str(receipt.get("scanner_implementation_sha256") or "")
    if not _is_sha256(scanner):
        trial_reasons.append(f"{trial_id}:contamination_scanner_unproven")
    if not str(receipt.get("method") or "").strip():
        trial_reasons.append(f"{trial_id}:contamination_method_missing")
    threshold = receipt.get("max_overlap_threshold")
    measured = receipt.get("max_overlap_observed")
    if not _finite_number(threshold) or not 0.0 <= float(threshold) <= 1.0:
        trial_reasons.append(f"{trial_id}:contamination_threshold_invalid")
        return scanner
    if not _finite_number(measured) or not 0.0 <= float(measured) <= 1.0:
        trial_reasons.append(f"{trial_id}:contamination_overlap_unmeasured")
        return scanner
    if float(measured) > float(threshold):
        trial_reasons.append(f"{trial_id}:contamination_overlap_exceeds_threshold")
    return scanner


def _reconcile_compute_profile(
    resident: dict[str, Any], prereg: dict[str, Any], reasons: list[str]
) -> None:
    """Tie the declared parameter count to the architecture that was costed.

    The bundle declared a 32B-class model in one place and a compute profile
    in another, and nothing connected them. A producer could pin the right
    estimator, hand it a toy decoder, and every FLOPs comparison between the
    arms would come out internally consistent and mean nothing about the model
    the claim is about.
    """
    profile_receipt = resident.get("compute_profile")
    if not isinstance(profile_receipt, dict):
        reasons.append("resident_compute_profile_missing")
        return
    expected = str(prereg.get("compute_profile_sha256") or "")
    if expected and profile_receipt.get("profile_sha256") != expected:
        reasons.append("resident_compute_profile_not_preregistered")
    try:
        profile = ModelComputeProfile.from_receipt(profile_receipt)
    except (TypeError, ValueError):
        reasons.append("resident_compute_profile_invalid")
        return
    declared = resident.get("parameter_count")
    if type(declared) is not int or declared <= 0:
        return
    structural = profile.structural_parameter_count
    if abs(structural - declared) / declared > _PARAMETER_RECONCILIATION_TOLERANCE:
        reasons.append("resident_compute_profile_parameter_count_mismatch")


def _validate_resident_model(
    resident: Any, prereg: dict[str, Any], reasons: list[str]
) -> dict[str, Any]:
    if not isinstance(resident, dict):
        reasons.append("missing_resident_model_receipt")
        return {}
    parameter_count = resident.get("parameter_count")
    if type(parameter_count) is not int or not 30_000_000_000 <= parameter_count <= 40_000_000_000:
        reasons.append("resident_model_not_32b_class")
    # The class claim must be DERIVED from the hashed checkpoint, not merely
    # asserted alongside it: a bare integer in range is something any
    # producer can write. The parameter count is required to come from the
    # same manifest whose fingerprint is bound above, and the manifest's own
    # digest must match the declared checkpoint fingerprint.
    if resident.get("parameter_count_source") != "checkpoint_manifest":
        reasons.append("resident_parameter_count_not_manifest_derived")
    # CP126 8923b135. A count inside the 30-40B window, even a
    # manifest-derived one, does not establish WHICH 32B-class model this is:
    # quantization layout and architecture change what the weights compute,
    # and two runtimes can share a parameter count while serving materially
    # different functions. The class claim must carry that identity.
    for required_field in ("architecture", "quantization_bits", "quantization_group_size"):
        if not resident.get(required_field):
            reasons.append(f"resident_identity_incomplete:{required_field}")
    manifest_digest = str(resident.get("checkpoint_manifest_sha256") or "")
    if not _is_sha256(manifest_digest):
        reasons.append("resident_checkpoint_manifest_missing")
    elif manifest_digest != str(resident.get("checkpoint_fingerprint") or ""):
        reasons.append("resident_checkpoint_manifest_unbound")
    fingerprint = str(resident.get("checkpoint_fingerprint") or "")
    if not _is_sha256(fingerprint) or fingerprint != str(
        prereg.get("treatment_checkpoint_fingerprint") or ""
    ):
        reasons.append("resident_checkpoint_mismatch")
    if resident.get("checkpoint_fingerprint_method") != "sha256":
        reasons.append("resident_checkpoint_not_fully_hashed")
    if type(resident.get("checkpoint_file_count")) is not int or resident[
        "checkpoint_file_count"
    ] <= 0:
        reasons.append("resident_checkpoint_file_count_missing")
    if resident.get("latent_cortex_architecture_sha256") != prereg.get(
        "architecture_freeze_sha256"
    ):
        reasons.append("resident_architecture_freeze_mismatch")
    if not str(resident.get("worker_boot_id") or ""):
        reasons.append("missing_resident_worker_boot_id")
    if resident.get("loaded_in_installed_app") is not True:
        reasons.append("installed_app_residency_unproven")
    if not str(resident.get("installed_app_bundle_id") or ""):
        reasons.append("installed_app_bundle_id_missing")
    if not _is_sha256(resident.get("installed_app_build_sha256")):
        reasons.append("installed_app_build_unproven")
    if not _is_sha256(resident.get("worker_binary_sha256")):
        reasons.append("resident_worker_binary_unproven")
    if not _is_sha256(resident.get("build_provenance_sha256")):
        reasons.append("installed_app_provenance_unproven")
    if not _is_git_sha(resident.get("source_commit")):
        reasons.append("installed_app_source_commit_invalid")
    # CP126 f17aa1b8: the four identifiers above were accepted for LOOKING
    # like hashes. Four well-formed digests with no relationship between them
    # say nothing about which source produced the binary that ran, so any of
    # them could name a build that never existed. A release attestation ties
    # them together: one builder claiming that rebuilding this commit produces
    # this binary under this provenance record. Nobody re-runs the build here
    # and no release signature is checked — BUILD_PROVENANCE_ATTESTATION says
    # so, because "asserted by the builder" and "reproduced independently"
    # must not read alike.
    attestation = resident.get("release_attestation")
    if not isinstance(attestation, dict):
        reasons.append("release_attestation_missing")
        return resident
    if not str(attestation.get("builder_id") or "").strip():
        reasons.append("release_attestation_unattributed")
    for field, source in (
        ("source_commit", "source_commit"),
        ("rebuild_worker_binary_sha256", "worker_binary_sha256"),
        ("build_provenance_sha256", "build_provenance_sha256"),
        ("installed_app_build_sha256", "installed_app_build_sha256"),
    ):
        if attestation.get(field) != resident.get(source):
            reasons.append(f"release_attestation_{field}_unbound")
    if not _finite_number(attestation.get("attested_at"), positive=True):
        reasons.append("release_attestation_time_missing")
    return resident


def _trial_compute(
    trial: dict[str, Any],
    arm: str,
    expected_estimator: str,
    reasons: list[str],
    *,
    expected_profile: str = "",
    max_forward_passes: int = 0,
) -> tuple[float | None, int | None, dict[str, Any] | None]:
    compute = trial.get(f"{arm}_compute")
    trial_id = str(trial.get("trial_id") or "unknown")
    if not isinstance(compute, dict):
        reasons.append(f"{trial_id}:{arm}_compute_missing")
        return None, None, None
    flops = compute.get("estimated_flops")
    layer_apps = compute.get("layer_apps")
    if not _finite_number(flops, positive=True):
        reasons.append(f"{trial_id}:{arm}_flops_invalid")
        flops = None
    if (
        type(layer_apps) is not int
        or layer_apps <= 0
        or layer_apps > _MAX_LAYER_APPS
    ):
        reasons.append(f"{trial_id}:{arm}_layer_apps_invalid")
        layer_apps = None
    estimator = compute.get("estimator_sha256")
    if not _is_sha256(estimator):
        reasons.append(f"{trial_id}:{arm}_compute_estimator_unidentified")
    elif estimator != expected_estimator:
        reasons.append(f"{trial_id}:{arm}_compute_estimator_mismatch")
    resource: dict[str, Any] | None = None
    try:
        resource = validate_resource_receipt(compute.get("resource_accounting"))
    except (TypeError, ValueError):
        reasons.append(f"{trial_id}:{arm}_resource_accounting_invalid")
    if resource is not None:
        if resource["accounting_complete"] is not True:
            reasons.append(f"{trial_id}:{arm}_resource_accounting_incomplete")
        if flops is not None and float(resource["estimated_flops"]) != float(flops):
            reasons.append(f"{trial_id}:{arm}_compute_receipt_mismatch")
        profile = resource.get("model_profile")
        profile = profile if isinstance(profile, Mapping) else {}
        if expected_profile and profile.get("profile_sha256") != expected_profile:
            # The estimator was pinned; the architecture it estimated against
            # was not. Both have to be.
            reasons.append(f"{trial_id}:{arm}_compute_profile_not_preregistered")
        totals = resource.get("totals")
        totals = totals if isinstance(totals, Mapping) else {}
        accounted = totals.get("transformer_layer_apps")
        if layer_apps is not None and accounted != layer_apps:
            # layer_apps was a free-standing number beside a receipt that had
            # its own count of the same thing, and nobody compared them.
            reasons.append(f"{trial_id}:{arm}_layer_apps_not_accounted")
        layers = profile.get("num_hidden_layers")
        if layer_apps is not None and type(layers) is int and layers > 0:
            if layer_apps % layers:
                # A forward pass applies the whole stack. A count that is not
                # a multiple of the depth did not come from whole passes.
                reasons.append(f"{trial_id}:{arm}_layer_apps_not_whole_passes")
            elif max_forward_passes and layer_apps // layers > max_forward_passes:
                reasons.append(f"{trial_id}:{arm}_forward_passes_over_budget")
    return float(flops) if flops is not None else None, layer_apps, resource


def _trial_information(
    trial: dict[str, Any],
    arm: str,
    reasons: list[str],
    prereg: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    prereg = prereg if isinstance(prereg, Mapping) else {}
    trial_id = str(trial.get("trial_id") or "unknown")
    try:
        receipt = validate_information_receipt(trial.get(f"{arm}_information"))
    except (TypeError, ValueError):
        reasons.append(f"{trial_id}:{arm}_information_accounting_invalid")
        return None
    if receipt["accounting_complete"] is not True:
        reasons.append(f"{trial_id}:{arm}_information_accounting_incomplete")
    task_payload_sha256 = trial.get("task_payload_sha256")
    if not any(
        source.get("content_sha256") == task_payload_sha256
        and source.get("kind") == "task_prompt"
        for source in receipt["sources"]
    ):
        reasons.append(f"{trial_id}:{arm}_task_information_unbound")
    if trial.get(f"{arm}_information_sha256") != receipt["receipt_sha256"]:
        reasons.append(f"{trial_id}:{arm}_information_receipt_mismatch")
        return None
    # CP126 a70d12ad: the two arms' information receipts had to MATCH EACH
    # OTHER and nothing tied either to the preregistration. Both arms could
    # therefore be given the same undeclared retrieval context, decoded under
    # the same undeclared policy, and the parity check would pass while the
    # experiment ran under conditions nobody committed to in advance.
    policies = receipt.get("policies")
    policies = policies if isinstance(policies, Mapping) else {}
    for policy_name, prereg_key in (
        ("decode", "decode_policy_sha256"),
        ("tool", "tool_policy_sha256"),
        ("verifier", "scorer_implementation_sha256"),
    ):
        expected = str(prereg.get(prereg_key) or "")
        if not expected:
            continue
        if str(policies.get(policy_name) or "") != expected:
            reasons.append(
                f"{trial_id}:{arm}_{policy_name}_policy_not_preregistered"
            )
    return receipt


def _expected_arm_identifier(
    trial: Mapping[str, Any], arm: str, worker_boot_id: str
) -> str:
    """The identifier a trial's arm is entitled to, derived from its own record.

    CP126 2cbeadca: ``episode_id`` and ``request_id`` only had to be non-empty
    and globally unique. They were free strings, unrelated to the task they
    ran, the output they produced, or the worker that served them — so two
    trials could swap identifiers, or an identifier could be minted after the
    fact to make a lineage look clean, and uniqueness would still hold.

    Deriving the identifier from the canonical record makes it a checkable
    consequence of the work instead of a label attached to it.
    """
    return canonical_sha256(
        {
            "arm": arm,
            "trial_id": trial.get("trial_id"),
            "task_id": trial.get("task_id"),
            "task_payload_sha256": trial.get("task_payload_sha256"),
            "output_sha256": trial.get(f"{arm}_output_sha256"),
            "worker_boot_id": worker_boot_id,
            "evaluation_started_at": trial.get("evaluation_started_at"),
        }
    )


def _validate_treatment_receipt(
    trial: dict[str, Any],
    checkpoint: str,
    worker_boot_id: str,
    installed_app_build_sha256: str,
    reasons: list[str],
    unproven_integrity_claims: list[str] | None = None,
) -> str:
    trial_id = str(trial.get("trial_id") or "unknown")
    if unproven_integrity_claims is None:
        unproven_integrity_claims = []
    receipt = trial.get("treatment_receipt")
    if not isinstance(receipt, dict):
        reasons.append(f"{trial_id}:treatment_receipt_missing")
        return ""
    required_truths = {
        "params_unchanged": True,
        "latent_opt_applied": True,
        "fast_weights_applied": True,
        "fast_weights_erased": True,
    }
    for key, expected in required_truths.items():
        if receipt.get(key) is not expected:
            reasons.append(f"{trial_id}:{key}_unproven")
    # CP126 6090a5ae + 01e8b3c1. The control receipt already required DIGEST
    # evidence for params_unchanged; the treatment receipt — the arm whose
    # effect is being published — passed on the literal booleans above alone.
    #
    # The producer DOES emit this: EpisodeReceipt.to_dict() carries
    # weight_integrity digests and integrity_verdicts(). So an absent verdict
    # on a published capability claim means the receipt was hand-assembled or
    # came from a lane that measured nothing — neither is evidence, and this
    # is the exact fail-open the finding names. Both a REFUTED and an
    # UNPROVEN verdict now disqualify the trial; the distinction is preserved
    # in the reason so an operator can tell "measured false" from "never
    # measured".
    for claim in ("params_unchanged", "fast_weights_erased"):
        verdict = _receipt_integrity_verdict(receipt, claim)
        if verdict == "refuted":
            reasons.append(f"{trial_id}:treatment_{claim}_refuted_by_digest")
        elif verdict != "proven":
            reasons.append(f"{trial_id}:treatment_{claim}_unproven_no_digest")
            unproven_integrity_claims.append(f"treatment_{claim}")
    if str(receipt.get("checkpoint_fingerprint") or "") != checkpoint:
        reasons.append(f"{trial_id}:treatment_checkpoint_mismatch")
    if receipt.get("checkpoint_fingerprint_method") != "sha256" or type(
        receipt.get("checkpoint_file_count")
    ) is not int or receipt["checkpoint_file_count"] <= 0:
        reasons.append(f"{trial_id}:treatment_checkpoint_identity_incomplete")
    if str(receipt.get("worker_boot_id") or "") != worker_boot_id:
        reasons.append(f"{trial_id}:treatment_worker_boot_mismatch")
    if receipt.get("installed_app_build_sha256") != installed_app_build_sha256:
        reasons.append(f"{trial_id}:treatment_app_build_mismatch")
    episode_id = str(receipt.get("episode_id") or "")
    if not episode_id or not _is_sha256(receipt.get("schedule_hash")):
        reasons.append(f"{trial_id}:treatment_identity_incomplete")
    if receipt.get("latent_opt_mode") != "gradient":
        reasons.append(f"{trial_id}:latent_opt_mode_not_gradient")
    for metric in (
        "n_slots",
        "n_branches",
        "steps_taken",
        "latent_opt_attempts",
        "latent_opt_steps",
        "fast_weights_layers",
        "fast_weight_optimization_attempts",
        "fast_weight_optimized_steps",
    ):
        if type(receipt.get(metric)) is not int or receipt[metric] <= 0:
            reasons.append(f"{trial_id}:treatment_{metric}_unproven")
    for metric in ("latent_opt_rejected", "fast_weight_rejected_steps"):
        if type(receipt.get(metric)) is not int or receipt[metric] < 0:
            reasons.append(f"{trial_id}:treatment_{metric}_invalid")
    if (
        type(receipt.get("latent_opt_attempts")) is int
        and type(receipt.get("latent_opt_steps")) is int
        and type(receipt.get("latent_opt_rejected")) is int
        and receipt["latent_opt_attempts"]
        != receipt["latent_opt_steps"] + receipt["latent_opt_rejected"]
    ):
        reasons.append(f"{trial_id}:latent_opt_accounting_mismatch")
    if (
        type(receipt.get("fast_weight_optimization_attempts")) is int
        and type(receipt.get("fast_weight_optimized_steps")) is int
        and type(receipt.get("fast_weight_rejected_steps")) is int
        and receipt["fast_weight_optimization_attempts"]
        != receipt["fast_weight_optimized_steps"]
        + receipt["fast_weight_rejected_steps"]
    ):
        reasons.append(f"{trial_id}:fast_weight_accounting_mismatch")
    if receipt.get("latent_opt_budget_exhausted") is not False:
        reasons.append(f"{trial_id}:latent_opt_budget_exhausted")
    if receipt.get("fast_weight_budget_exhausted") is not False:
        reasons.append(f"{trial_id}:fast_weight_budget_exhausted")
    raw_flags = receipt.get("honest_flags")
    if not isinstance(raw_flags, list):
        reasons.append(f"{trial_id}:treatment_honest_flags_invalid")
        flags: list[str] = []
    else:
        flags = [str(flag) for flag in raw_flags]
    if flags:
        reasons.append(f"{trial_id}:treatment_honest_flags_present")
    return episode_id


# Components that change what a "vanilla" control actually computes. Each
# must be positively OFF in a control receipt: an absent field is treated as
# undeclared, not as disabled, because a control that never reported its
# configuration cannot be shown to have been unenhanced.
_CONTROL_MUST_BE_DISABLED: tuple[tuple[str, Any], ...] = (
    ("fast_weights_applied", False),
    ("latent_opt_applied", False),
    ("recurrence_adapter_applied", False),
    ("retrieval_applied", False),
    ("nonparametric_memory_applied", False),
    ("contrastive_decoding_applied", False),
    ("speculative_decoding_applied", False),
    ("expert_adapter_applied", False),
    ("affective_steering_active", False),
    ("prompt_cache_reused", False),
)

# Decoding parameters that must match the preregistered control spec. A
# control run hotter, wider, or with a different repetition penalty than the
# treatment is not a control of the same thing.
_CONTROL_DECODE_PARAMETERS: tuple[str, ...] = (
    "decode_temperature",
    "decode_top_p",
    "decode_repetition_penalty_applied",
)


def _parameter_canary_coverage(receipt: Any) -> float | None:
    """Fraction of parameter tensors the before/after canary actually hashed.

    ``params_unchanged`` is proven by hashing a fixed-stride sample of the
    parameter tree twice. The sample is what makes it affordable on a 32B
    model and also what makes it partial: a mutation living entirely in
    unsampled tensors leaves both digests identical. The certificate needs
    the number, not the boolean.
    """
    if not isinstance(receipt, dict):
        return None
    integrity = receipt.get("runtime_integrity")
    if not isinstance(integrity, Mapping):
        return None
    parameters = integrity.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    coverages: list[float] = []
    for side in ("before", "after"):
        snapshot = parameters.get(side)
        if not isinstance(snapshot, Mapping):
            return None
        leaves = snapshot.get("parameter_leaf_count")
        sampled = snapshot.get("sampled_tensor_count")
        if type(leaves) is not int or leaves <= 0:
            return None
        if type(sampled) is not int or sampled < 0 or sampled > leaves:
            return None
        coverages.append(sampled / leaves)
    return min(coverages)


def _receipt_integrity_verdict(receipt: Any, claim: str) -> str:
    """Verdict reconstructed from the complete worker-bound measurements."""
    if not isinstance(receipt, dict):
        return "unproven"
    try:
        from core.brain.llm.latent_cortex.runtime_integrity import (
            runtime_integrity_claim_verdict,
        )

        worker_identity = receipt.get("worker_identity")
        if not isinstance(worker_identity, dict):
            return "unproven"
        return runtime_integrity_claim_verdict(
            receipt.get("runtime_integrity"),
            claim,
            expected_episode_id=str(receipt.get("episode_id") or ""),
            expected_input_tokens_sha256=str(
                receipt.get("input_tokens_sha256") or ""
            ),
            expected_worker_identity=worker_identity,
            expected_fast_weights_applied=(
                receipt.get("fast_weights_applied") is True
            ),
            expected_fast_weights_attach_attempted=(
                receipt.get("fast_weights_attach_attempted") is True
            ),
            expected_checkpoint_fingerprint=str(
                receipt.get("checkpoint_fingerprint") or ""
            ),
            expected_checkpoint_method=str(
                receipt.get("checkpoint_fingerprint_method") or ""
            ),
            expected_checkpoint_file_count=receipt.get(
                "checkpoint_file_count"
            ),
        )
    except (ImportError, TypeError, ValueError):
        return "unproven"


#: Everything that can make two arms decode differently. CP126 8a56c486: the
#: arm-parity checks covered information hashes, tool policy, compute and run
#: order but NOT the generation manifest, so an outcome difference could come
#: from sampling rather than from the latent-cortex treatment.
_ARM_GENERATION_PARITY_FIELDS: tuple[str, ...] = (
    "decode_seed",
    "decode_temperature",
    "decode_top_p",
    "decode_repetition_penalty_applied",
    "decode_max_tokens",
    "decode_stop_sequences_sha256",
    "context_window_tokens",
    "tokenizer_sha256",
    "prompt_template_sha256",
)


def _validate_arm_generation_parity(
    trial_id: str,
    treatment_receipt: Any,
    control_receipt: Any,
    reasons: list[str],
) -> list[str]:
    """Both arms must have decoded the same way.

    A DECLARED difference is disqualifying: the trial's outcome gap can come
    from sampling rather than from the treatment, so the trial cannot feed the
    paired claim.

    A field neither arm declares is a real gap in the evidence chain, but the
    producers do not emit the full generation manifest yet. Rejecting on it
    would make certification permanently unachievable rather than more honest,
    so undeclared fields are RETURNED as certificate-visible gaps (the claim
    carries what it could not verify) instead of silently passing as equal.
    Closing them for real needs the worker-side manifest emission.
    """
    gaps: list[str] = []
    if not isinstance(treatment_receipt, dict) or not isinstance(control_receipt, dict):
        reasons.append(f"{trial_id}:generation_parity_receipts_missing")
        return gaps
    for field_name in _ARM_GENERATION_PARITY_FIELDS:
        in_treatment = field_name in treatment_receipt
        in_control = field_name in control_receipt
        if not in_treatment and not in_control:
            gaps.append(field_name)
            continue
        if in_treatment != in_control:
            # Only one arm declares it. That is an incomplete manifest, not
            # proof the arms diverged — the control-side decode manifest
            # landed before the treatment-side one exists. Report it.
            gaps.append(field_name)
            continue
        if treatment_receipt.get(field_name) != control_receipt.get(field_name):
            reasons.append(f"{trial_id}:generation_parity_mismatch:{field_name}")
    return gaps


def _validate_vanilla_control_manifest(
    trial_id: str,
    receipt: dict[str, Any],
    prereg: dict[str, Any],
    reasons: list[str],
) -> None:
    """Every enhancement is off, declared, and decoding matches the spec."""
    for field_name, required in _CONTROL_MUST_BE_DISABLED:
        if field_name not in receipt:
            reasons.append(f"{trial_id}:control_component_undeclared:{field_name}")
            continue
        if receipt.get(field_name) is not required:
            reasons.append(f"{trial_id}:control_component_enabled:{field_name}")

    # Counters betray activity a boolean might not: a control that took
    # optimization steps or wrapped layers was not vanilla whatever it
    # claims.
    for counter in (
        "fast_weights_layers",
        "fast_weight_optimization_attempts",
        "latent_opt_attempts",
        "steps_taken",
    ):
        value = receipt.get(counter)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            reasons.append(f"{trial_id}:control_activity_observed:{counter}")

    declared = prereg.get("control_decode_spec")
    if not isinstance(declared, dict):
        reasons.append(f"{trial_id}:control_decode_spec_missing")
        return
    for parameter in _CONTROL_DECODE_PARAMETERS:
        if parameter not in declared:
            reasons.append(f"{trial_id}:control_decode_spec_incomplete:{parameter}")
            continue
        if parameter not in receipt:
            reasons.append(f"{trial_id}:control_decode_undeclared:{parameter}")
            continue
        expected = declared.get(parameter)
        actual = receipt.get(parameter)
        if not _numbers_match(expected, actual):
            reasons.append(f"{trial_id}:control_decode_mismatch:{parameter}")


def _numbers_match(expected: Any, actual: Any) -> bool:
    """Exact for non-numerics; tolerant of float representation otherwise."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= 1e-9
    return expected == actual


def _validate_control_receipt(
    trial: dict[str, Any],
    prereg: dict[str, Any],
    worker_boot_id: str,
    installed_app_build_sha256: str,
    reasons: list[str],
) -> str:
    trial_id = str(trial.get("trial_id") or "unknown")
    receipt = trial.get("control_receipt")
    if not isinstance(receipt, dict):
        reasons.append(f"{trial_id}:control_receipt_missing")
        return ""
    request_id = str(receipt.get("request_id") or "")
    if not request_id:
        reasons.append(f"{trial_id}:control_request_id_missing")
    comparison_kind = prereg.get("comparison_kind")
    if comparison_kind == "resident_32b_vs_vanilla_same_checkpoint":
        if receipt.get("mode") != "vanilla" or receipt.get("latent_cortex_enabled") is not False:
            reasons.append(f"{trial_id}:control_not_vanilla")
        # A published capability claim is exactly where an asserted-but-
        # unmeasured integrity boolean must not pass: certification requires
        # the digest evidence, not the claim.
        if _receipt_integrity_verdict(receipt, "params_unchanged") != "proven":
            reasons.append(f"{trial_id}:control_params_unchanged_unproven")
        # CP126 869a0ce4. "Vanilla" was three fields: mode, the latent-cortex
        # switch, and params_unchanged. Nothing prohibited fast weights,
        # retrieval, extra reasoning passes, altered decoding, or a warm
        # cache carrying state from an earlier turn — so a control could be
        # materially enhanced and still certify as the baseline the
        # treatment is measured against. Every enhancement the runtime can
        # apply must be positively absent, not merely unmentioned.
        _validate_vanilla_control_manifest(trial_id, receipt, prereg, reasons)
        if str(receipt.get("checkpoint_fingerprint") or "") != str(
            prereg.get("control_checkpoint_fingerprint") or ""
        ):
            reasons.append(f"{trial_id}:control_checkpoint_mismatch")
        if str(receipt.get("worker_boot_id") or "") != worker_boot_id:
            reasons.append(f"{trial_id}:control_worker_boot_mismatch")
        if receipt.get("checkpoint_fingerprint_method") != "sha256" or type(
            receipt.get("checkpoint_file_count")
        ) is not int or receipt["checkpoint_file_count"] <= 0:
            reasons.append(f"{trial_id}:control_checkpoint_identity_incomplete")
        if receipt.get("installed_app_build_sha256") != installed_app_build_sha256:
            reasons.append(f"{trial_id}:control_app_build_mismatch")
    elif comparison_kind == "resident_32b_vs_external_frontier":
        if receipt.get("model_id") != prereg.get("control_model_id"):
            reasons.append(f"{trial_id}:external_control_model_mismatch")
        if receipt.get("model_build_fingerprint") != prereg.get(
            "control_model_build_fingerprint"
        ):
            reasons.append(f"{trial_id}:external_control_build_mismatch")
        if receipt.get("provider") != prereg.get("control_provider"):
            reasons.append(f"{trial_id}:external_control_provider_mismatch")
        if not _is_sha256(receipt.get("provider_receipt_sha256")):
            reasons.append(f"{trial_id}:external_control_receipt_unproven")
    return request_id


def _validate_trial_lineage(trial: dict[str, Any], reasons: list[str]) -> None:
    trial_id = str(trial.get("trial_id") or "unknown")
    for field in (
        "task_payload_sha256",
        "treatment_output_sha256",
        "control_output_sha256",
        "scorer_config_sha256",
        "verifier_receipt_sha256",
    ):
        if not _is_sha256(trial.get(field)):
            reasons.append(f"{trial_id}:{field}_invalid")


def _task_manifest_sha256(trials: list[Any]) -> str:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("task manifest contains a non-mapping trial")
        rows.append(
            {
                "trial_id": trial.get("trial_id"),
                "task_id": trial.get("task_id"),
                "domain": trial.get("domain"),
                "task_payload_sha256": trial.get("task_payload_sha256"),
                "task_generated_at": trial.get("task_generated_at"),
                # CP126 6b65ffd0: the manifest committed WHICH tasks would run
                # and nothing about HOW. Arm order, the scoring configuration,
                # the decoding seeds, and the tool policies are experimental
                # design, and every one of them could be chosen after outputs
                # were known. They are pre-evaluation choices, so the issuer
                # signs them alongside the tasks.
                "run_order": trial.get("run_order"),
                "scorer_config_sha256": trial.get("scorer_config_sha256"),
                "treatment_tool_policy_sha256": trial.get(
                    "treatment_tool_policy_sha256"
                ),
                "control_tool_policy_sha256": trial.get("control_tool_policy_sha256"),
                "treatment_decode_policy_sha256": trial.get(
                    "treatment_decode_policy_sha256"
                ),
                "control_decode_policy_sha256": trial.get(
                    "control_decode_policy_sha256"
                ),
            }
        )
    rows.sort(key=lambda row: str(row["trial_id"] or ""))
    return canonical_sha256(rows)


def _validate_task_diversity(
    bundle: dict[str, Any], reasons: list[str]
) -> tuple[dict[str, str], str]:
    """Distinct task IDs are not distinct tasks.

    The certificate deduplicated on ``task_payload_sha256``, which catches a
    copy-paste and nothing else. Forty paraphrases of one problem have forty
    distinct hashes, and each one counts as an independent trial toward the
    per-domain minimum and toward the paired test. Sample size inflates while
    the evidence stands still.

    The task issuer therefore clusters its own tasks and commits the result
    before evaluation: the method, the similarity ceiling it held tasks to,
    the highest similarity it actually found, and which family each task
    landed in. The families become the unit of independence downstream.

    Returns the task-to-family map and the receipt digest that the signed
    commitment has to carry.
    """
    receipt = bundle.get("task_diversity")
    if not isinstance(receipt, dict):
        reasons.append("task_diversity_receipt_missing")
        return {}, ""
    if not str(receipt.get("method") or "").strip():
        reasons.append("task_diversity_method_missing")
    threshold = receipt.get("similarity_threshold")
    observed = receipt.get("max_pairwise_similarity")
    threshold_valid = _finite_number(threshold) and 0.0 < float(threshold) <= 1.0
    if not threshold_valid:
        reasons.append("task_diversity_threshold_invalid")
    if not _finite_number(observed) or not 0.0 <= float(observed) <= 1.0:
        reasons.append("task_diversity_similarity_unmeasured")
    elif threshold_valid and float(observed) > float(threshold):
        reasons.append("task_similarity_exceeds_threshold")
    families_raw = receipt.get("task_families")
    families: dict[str, str] = {}
    if not isinstance(families_raw, dict) or not families_raw:
        reasons.append("task_families_missing")
    else:
        for task_id, family in families_raw.items():
            if not isinstance(task_id, str) or not isinstance(family, str):
                reasons.append("task_families_malformed")
                families = {}
                break
            if not task_id.strip() or not family.strip():
                reasons.append("task_families_malformed")
                families = {}
                break
            families[task_id] = family
    try:
        digest = canonical_sha256(receipt)
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("task_diversity_not_canonical_json")
        return families, ""
    return families, digest


def _validate_task_commitment(
    bundle: dict[str, Any],
    prereg: dict[str, Any],
    trials: list[Any],
    *,
    latest_task_generated_at: float,
    earliest_evaluation_started_at: float,
    task_diversity_sha256: str,
    blinding_map_sha256: str,
    trusted_task_issuers: Mapping[str, Mapping[str, str]] | None,
    reasons: list[str],
) -> tuple[str, str]:
    raw = bundle.get("task_commitment")
    producer_id = str(bundle.get("producer_id") or "")
    signer = raw.get("signer") if isinstance(raw, dict) else None
    signer_id = str(signer.get("signer_id") or "") if isinstance(signer, dict) else ""
    try:
        pin = (
            trusted_task_issuers.get(signer_id)
            if trusted_task_issuers is not None
            else None
        )
    except (AttributeError, TypeError, ValueError):
        pin = None
    if not isinstance(pin, Mapping) or set(pin) != _TRUST_PIN_FIELDS:
        reasons.append("task_issuer_trust_pin_missing")
        return "", ""
    public_key = pin.get("public_key_b64")
    implementation_sha256 = pin.get("implementation_sha256")
    release_sha256 = pin.get("release_sha256")
    if (
        not isinstance(public_key, str)
        or not _is_sha256(implementation_sha256)
        or not _is_sha256(release_sha256)
    ):
        reasons.append("task_issuer_trust_pin_invalid")
        return "", ""
    try:
        envelope, payload, verified_signer_id = verify_signed_envelope(
            raw,
            schema=TASK_COMMITMENT_ATTESTATION_SCHEMA,
            trusted_keys={signer_id: public_key},
            role="latent cortex task commitment issuer",
        )
        manifest_sha256 = _task_manifest_sha256(trials)
        expected_commitment = canonical_sha256(
            {
                "architecture_freeze_sha256": prereg.get(
                    "architecture_freeze_sha256"
                ),
                "preregistration_sha256": bundle.get("preregistration_sha256"),
                "task_count": len(trials),
                "task_manifest_sha256": manifest_sha256,
                # The clustering has to be committed before evaluation too.
                # Clustering afterwards is choosing how many independent
                # trials there were once the results are in.
                "task_diversity_sha256": task_diversity_sha256,
                # Which arm carried which label, fixed before anything ran.
                "blinding_map_sha256": blinding_map_sha256,
            }
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("task_commitment_signature_or_manifest_invalid")
        return "", ""
    expected_fields = {
        "architecture_freeze_sha256",
        "preregistration_sha256",
        "task_commitment_sha256",
        "task_manifest_sha256",
        "task_diversity_sha256",
        "blinding_map_sha256",
        "task_count",
        "issuer_implementation_sha256",
        "issuer_release_sha256",
        "committed_at",
    }
    committed_at = payload.get("committed_at")
    invalid = (
        set(payload) != expected_fields
        or verified_signer_id == producer_id
        or payload.get("architecture_freeze_sha256")
        != prereg.get("architecture_freeze_sha256")
        or payload.get("preregistration_sha256")
        != bundle.get("preregistration_sha256")
        or payload.get("task_manifest_sha256") != manifest_sha256
        or not task_diversity_sha256
        or payload.get("task_diversity_sha256") != task_diversity_sha256
        or not blinding_map_sha256
        or payload.get("blinding_map_sha256") != blinding_map_sha256
        or payload.get("task_commitment_sha256") != expected_commitment
        or bundle.get("task_commitment_sha256") != expected_commitment
        or payload.get("task_count") != len(trials)
        or payload.get("issuer_implementation_sha256") != implementation_sha256
        or payload.get("issuer_release_sha256") != release_sha256
        or not _finite_number(committed_at, positive=True)
        or float(committed_at or 0.0) <= latest_task_generated_at
        or not math.isfinite(earliest_evaluation_started_at)
        or float(committed_at or 0.0) >= earliest_evaluation_started_at
    )
    if invalid:
        reasons.append("task_commitment_invalid")
        return verified_signer_id, ""
    try:
        return verified_signer_id, canonical_sha256(envelope)
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("task_commitment_not_canonical_json")
        return verified_signer_id, ""


def _blinding_map_sha256(bundle: dict[str, Any]) -> str:
    """The committed arm-label map digest, read without judging it.

    The commitment binds this before ``_validate_blinding`` runs, because
    validating the reveal event needs the issuer identity that only the
    commitment can establish.
    """
    blinding = bundle.get("blinding")
    if not isinstance(blinding, dict):
        return ""
    digest = str(blinding.get("arm_label_map_sha256") or "")
    return digest if _is_sha256(digest) else ""


def _validate_blinding(
    bundle: dict[str, Any],
    *,
    latest_scoring_completed_at: float,
    task_issuer_id: str,
    reasons: list[str],
) -> None:
    """``verifier_blinded: True`` is the producer grading its own blinding.

    The certificate checked a literal boolean per trial. It never asked which
    arm carried which label, when the labels were revealed, or whether the
    text handed to the scorer said "treatment" on it. A producer that scored
    everything with the labels visible emits the same ``True``.

    A blinding claim needs three things the boolean cannot carry: an arm-label
    map committed before evaluation so it cannot be written to fit the result,
    a reveal event that happened after the last score and was performed by
    somebody other than the producer, and a scan showing the scorer's inputs
    carried no arm markers.
    """
    blinding = bundle.get("blinding")
    if not isinstance(blinding, dict):
        reasons.append("blinding_evidence_missing")
        return
    if not _is_sha256(blinding.get("arm_label_map_sha256")):
        reasons.append("blinding_map_uncommitted")
    if not str(blinding.get("method") or "").strip():
        reasons.append("blinding_method_missing")
    revealed_at = blinding.get("revealed_at")
    if not _finite_number(revealed_at, positive=True):
        reasons.append("blinding_reveal_time_missing")
    elif latest_scoring_completed_at and float(revealed_at) < latest_scoring_completed_at:
        # Unblinding mid-run means the last trials were not blind at all.
        reasons.append("blinding_revealed_before_scoring_completed")
    revealed_by = str(blinding.get("revealed_by") or "")
    producer_id = str(bundle.get("producer_id") or "")
    if not revealed_by:
        reasons.append("blinding_reveal_unattributed")
    elif revealed_by == producer_id:
        reasons.append("blinding_revealed_by_producer")
    elif task_issuer_id and revealed_by != task_issuer_id:
        # The role that held the assignment is the role that can reveal it.
        reasons.append("blinding_revealed_by_unknown_role")
    scan = blinding.get("marker_scan")
    if not isinstance(scan, dict):
        reasons.append("blinding_marker_scan_missing")
        return
    if not _is_sha256(scan.get("scanner_implementation_sha256")):
        reasons.append("blinding_marker_scanner_unproven")
    if not str(scan.get("method") or "").strip():
        reasons.append("blinding_marker_scan_method_missing")
    checked = scan.get("markers_checked")
    found = scan.get("markers_found")
    if type(checked) is not int or checked <= 0:
        reasons.append("blinding_marker_scan_checked_nothing")
    if type(found) is not int or found < 0:
        reasons.append("blinding_marker_count_unmeasured")
    elif found > 0:
        # The scorer could see which arm it was grading.
        reasons.append("blinding_markers_present_in_scorer_inputs")


def _validate_producer_identity(
    bundle: dict[str, Any],
    *,
    evidence_hash: str,
    trusted_producers: Mapping[str, Mapping[str, str]] | None,
    reasons: list[str],
) -> tuple[str, str]:
    """Independence was decided against an unauthenticated string.

    ``producer_id`` was whatever the bundle said it was, and a verifier
    counted as independent when its signer id differed from that string. One
    actor holding a trusted verifier key could therefore write any producer
    name into the bundle and independently verify itself.

    The producer signs its own bundle with a pinned key. Its identity becomes
    cryptographic, and independence becomes a comparison between two verified
    signers and the organizations that hold their keys.

    Returns the verified producer id and its organization.
    """
    claimed = str(bundle.get("producer_id") or "")
    if not claimed:
        reasons.append("producer_id_missing")
    raw = bundle.get("producer_attestation")
    signer = raw.get("signer") if isinstance(raw, dict) else None
    signer_id = str(signer.get("signer_id") or "") if isinstance(signer, dict) else ""
    try:
        pin = trusted_producers.get(signer_id) if trusted_producers is not None else None
    except (AttributeError, TypeError, ValueError):
        pin = None
    if not isinstance(pin, Mapping) or set(pin) != _TRUST_PIN_FIELDS:
        reasons.append("producer_trust_pin_missing")
        return "", ""
    public_key = pin.get("public_key_b64")
    if not isinstance(public_key, str) or not _is_sha256(
        pin.get("implementation_sha256")
    ):
        reasons.append("producer_trust_pin_invalid")
        return "", ""
    try:
        _envelope, payload, verified_producer_id = verify_signed_envelope(
            raw,
            schema=PRODUCER_ATTESTATION_SCHEMA,
            trusted_keys={signer_id: public_key},
            role="latent cortex evidence producer",
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("producer_signature_invalid")
        return "", ""
    expected_fields = {
        "producer_id",
        "evidence_payload_sha256",
        "preregistration_sha256",
        "task_commitment_sha256",
        "raw_artifact_manifest_sha256",
        "produced_at",
    }
    if (
        set(payload) != expected_fields
        or payload.get("producer_id") != verified_producer_id
        or verified_producer_id != claimed
        or payload.get("evidence_payload_sha256") != evidence_hash
        or payload.get("preregistration_sha256")
        != bundle.get("preregistration_sha256")
        or payload.get("task_commitment_sha256")
        != bundle.get("task_commitment_sha256")
        or payload.get("raw_artifact_manifest_sha256")
        != bundle.get("raw_artifact_manifest_sha256")
        or not _finite_number(payload.get("produced_at"), positive=True)
    ):
        reasons.append("producer_attestation_invalid")
        return "", ""
    return verified_producer_id, str(pin.get("organization") or "")


def _validate_independent_attestation(
    bundle: dict[str, Any],
    raw: Any,
    *,
    evidence_hash: str,
    artifact_receipt_sha256: str,
    latest_task_generated_at: float,
    producer_id: str,
    trusted_verifiers: Mapping[str, Mapping[str, str]] | None,
    reasons: list[str],
) -> tuple[str, str, str]:
    """Validate ONE attestation. The quorum is assembled by the caller."""
    signer = raw.get("signer") if isinstance(raw, dict) else None
    signer_id = str(signer.get("signer_id") or "") if isinstance(signer, dict) else ""
    try:
        pin = trusted_verifiers.get(signer_id) if trusted_verifiers is not None else None
    except (AttributeError, TypeError, ValueError):
        pin = None
    if not isinstance(pin, Mapping) or set(pin) != _TRUST_PIN_FIELDS:
        reasons.append("independent_verifier_trust_pin_missing")
        return "", "", ""
    public_key = pin.get("public_key_b64")
    implementation_sha256 = pin.get("implementation_sha256")
    release_sha256 = pin.get("release_sha256")
    if (
        not isinstance(public_key, str)
        or not _is_sha256(implementation_sha256)
        or not _is_sha256(release_sha256)
    ):
        reasons.append("independent_verifier_trust_pin_invalid")
        return "", "", ""
    try:
        envelope, payload, verified_signer_id = verify_signed_envelope(
            raw,
            schema=INDEPENDENT_ATTESTATION_SCHEMA,
            trusted_keys={signer_id: public_key},
            role="latent cortex independent verifier",
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("independent_verifier_signature_invalid")
        return "", "", ""
    expected_fields = {
        "accepted",
        "claim_tier",
        "producer_id",
        "evidence_payload_sha256",
        "preregistration_sha256",
        "implementation_sha256",
        "verifier_release_sha256",
        "raw_artifact_manifest_sha256",
        "task_commitment_sha256",
        # CP126 2528aa7e: the payload repeated values copied straight out of
        # the trust pin, so a signature proved key possession and nothing
        # about work. This digest comes from opening the raw artifact files
        # and recomputing their receipt, which a signer that never ran cannot
        # produce.
        "raw_artifact_receipt_sha256",
        "verified_at",
    }
    invalid = (
        set(payload) != expected_fields
        or payload.get("accepted") is not True
        or payload.get("claim_tier") != PROVEN
        or verified_signer_id == producer_id
        or payload.get("producer_id") != producer_id
        or payload.get("evidence_payload_sha256") != evidence_hash
        or payload.get("preregistration_sha256")
        != bundle.get("preregistration_sha256")
        or payload.get("implementation_sha256") != implementation_sha256
        or payload.get("verifier_release_sha256") != release_sha256
        or payload.get("raw_artifact_manifest_sha256")
        != bundle.get("raw_artifact_manifest_sha256")
        or payload.get("task_commitment_sha256")
        != bundle.get("task_commitment_sha256")
        or not artifact_receipt_sha256
        or payload.get("raw_artifact_receipt_sha256") != artifact_receipt_sha256
        or not _finite_number(payload.get("verified_at"), positive=True)
        or float(payload.get("verified_at") or 0.0) <= latest_task_generated_at
    )
    organization = str(pin.get("organization") or "")
    if invalid:
        reasons.append("independent_verification_invalid")
        return verified_signer_id, "", organization
    try:
        return verified_signer_id, canonical_sha256(envelope), organization
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("independent_attestation_not_canonical_json")
        return verified_signer_id, "", organization


def _validate_raw_artifact_receipt(
    receipt: Any,
    bundle: Mapping[str, Any],
    reasons: list[str],
) -> bool:
    """Require a real receipt from ``verify_raw_artifact_package``.

    Certification previously accepted a bundle whose raw artifacts were never
    opened: ``raw_artifact_manifest_sha256`` was checked for SHA-256 SYNTAX
    only. A syntactically valid but entirely fabricated hash therefore
    reached ``accepted=True`` / ``claim_tier=PROVEN``. The receipt must be
    produced by the artifact verifier, report acceptance, and bind the same
    manifest digest this bundle declares.
    """
    # Lazy import: frontier_artifacts imports canonical_sha256 from this
    # module, so a module-level import would be circular. Importing here
    # keeps ONE source of truth for the schema constant.
    from core.brain.llm.latent_cortex.frontier_artifacts import (
        ARTIFACT_VERIFICATION_SCHEMA,
    )

    if not isinstance(receipt, Mapping):
        reasons.append("raw_artifact_package_unverified")
        return False
    if receipt.get("schema") != ARTIFACT_VERIFICATION_SCHEMA:
        reasons.append("raw_artifact_receipt_schema_mismatch")
        return False
    if receipt.get("accepted") is not True:
        reasons.append("raw_artifact_receipt_not_accepted")
        return False
    manifest_sha256 = str(receipt.get("manifest_sha256") or "")
    if not _is_sha256(manifest_sha256) or manifest_sha256 != str(
        bundle.get("raw_artifact_manifest_sha256") or ""
    ):
        reasons.append("raw_artifact_receipt_manifest_mismatch")
        return False
    if not _is_sha256(receipt.get("receipt_sha256")):
        reasons.append("raw_artifact_receipt_unsigned")
        return False
    # The receipt must cover every trial the bundle grades on.
    trials = bundle.get("trials")
    declared_trials = len(trials) if isinstance(trials, list) else 0
    if type(receipt.get("trial_count")) is not int or receipt["trial_count"] != declared_trials:
        reasons.append("raw_artifact_receipt_trial_count_mismatch")
        return False
    return True


def _validate_release_readiness(
    bundle: dict[str, Any],
    prereg: dict[str, Any],
    *,
    treatment_success_rate: float,
    reasons: list[str],
) -> dict[str, Any]:
    """A win over the control is not a release.

    The certificate compared the treatment against the control in the same
    run and stopped there. Nothing asked whether this release was worse than
    the last one, whether it was miscalibrated, whether it was slower or more
    expensive, or whether it had started failing safety cases it used to
    pass. A model can beat its ablation on every domain while regressing
    against the version already shipped, and the certificate would have said
    PROVEN.

    Returns the measured regression summary for the certificate to carry.
    """
    summary: dict[str, Any] = {}
    readiness = bundle.get("release_readiness")
    if not isinstance(readiness, dict):
        reasons.append("release_readiness_missing")
        return summary

    # ── previous release ────────────────────────────────────────────────
    baseline = readiness.get("previous_release")
    first_release = readiness.get("first_release") is True
    if first_release:
        # A first release has nothing to regress against, but it has to SAY
        # so rather than leave the section out.
        if baseline is not None:
            reasons.append("first_release_carries_a_baseline")
        summary["baseline"] = "first_release"
    elif not isinstance(baseline, dict):
        reasons.append("release_baseline_missing")
    else:
        if not _is_sha256(baseline.get("certificate_sha256")):
            reasons.append("release_baseline_certificate_unidentified")
        previous_rate = baseline.get("treatment_success_rate")
        allowed = prereg.get("max_success_rate_regression")
        if not _finite_number(previous_rate) or not 0.0 <= float(previous_rate) <= 1.0:
            reasons.append("release_baseline_success_rate_invalid")
        elif not _finite_number(allowed) or not 0.0 <= float(allowed) <= 0.1:
            reasons.append("invalid_max_success_rate_regression")
        else:
            regression = float(previous_rate) - treatment_success_rate
            summary["success_rate_regression"] = round(regression, 6)
            if regression > float(allowed):
                reasons.append("release_regresses_against_previous_certificate")
        for metric in ("latency", "compute"):
            previous = baseline.get(f"median_{metric}")
            current = readiness.get(f"median_{metric}")
            ceiling = prereg.get(f"max_{metric}_regression_ratio")
            if not _finite_number(previous, positive=True) or not _finite_number(
                current, positive=True
            ):
                reasons.append(f"release_{metric}_unmeasured")
                continue
            if not _finite_number(ceiling) or float(ceiling) < 1.0:
                reasons.append(f"invalid_max_{metric}_regression_ratio")
                continue
            ratio = float(current) / float(previous)
            summary[f"{metric}_ratio"] = round(ratio, 6)
            if ratio > float(ceiling):
                reasons.append(f"release_{metric}_regressed")

    # ── safety ──────────────────────────────────────────────────────────
    safety = readiness.get("safety_suite")
    if not isinstance(safety, dict):
        reasons.append("release_safety_suite_missing")
    else:
        if safety.get("suite_sha256") != prereg.get("safety_suite_sha256"):
            reasons.append("release_safety_suite_not_preregistered")
        cases = safety.get("cases_run")
        violations = safety.get("violations")
        allowed = prereg.get("max_safety_violations")
        if type(cases) is not int or cases <= 0:
            reasons.append("release_safety_suite_ran_nothing")
        if type(violations) is not int or violations < 0:
            reasons.append("release_safety_violations_unmeasured")
        elif type(allowed) is not int or allowed < 0:
            reasons.append("invalid_max_safety_violations")
        else:
            summary["safety_violations"] = violations
            if violations > allowed:
                reasons.append("release_safety_violations_exceed_budget")

    # ── calibration ─────────────────────────────────────────────────────
    calibration = readiness.get("calibration")
    if not isinstance(calibration, dict):
        reasons.append("release_calibration_missing")
        return summary
    if not str(calibration.get("method") or "").strip():
        reasons.append("release_calibration_method_missing")
    bins = calibration.get("bins")
    if type(bins) is not int or bins < 2:
        reasons.append("release_calibration_bins_invalid")
    error = calibration.get("expected_calibration_error")
    ceiling = prereg.get("max_expected_calibration_error")
    if not _finite_number(error) or not 0.0 <= float(error) <= 1.0:
        reasons.append("release_calibration_unmeasured")
    elif not _finite_number(ceiling) or not 0.0 < float(ceiling) <= 1.0:
        reasons.append("invalid_max_expected_calibration_error")
    else:
        summary["expected_calibration_error"] = round(float(error), 6)
        if float(error) > float(ceiling):
            # A model whose confidence does not track its accuracy is not
            # releasable however well it scores.
            reasons.append("release_calibration_outside_budget")
    return summary


def _verify_frontier_gain_bundle_part_1(
    prereg: Any,
    seen_task_payloads: set[str],
    trial: Any,
    trial_id: str,
    trial_reasons: list[str],
) -> tuple[Any, Any]:
    _validate_trial_lineage(trial, trial_reasons)
    # Each trial must have been scored by the PREREGISTERED scoring
    # program (its per-trial config may differ; the program may not).
    if str(trial.get("scorer_implementation_sha256") or "") != str(
        prereg.get("scorer_implementation_sha256") or ""
    ):
        trial_reasons.append(f"{trial_id}:scorer_not_preregistered")
    task_payload_hash = str(trial.get("task_payload_sha256") or "")
    if task_payload_hash in seen_task_payloads:
        trial_reasons.append(f"{trial_id}:duplicate_task_payload")
    seen_task_payloads.add(task_payload_hash)
    if trial.get("verifier_blinded") is not True:
        trial_reasons.append(f"{trial_id}:blinded_verifier_unproven")
    treatment_information = _trial_information(
        trial,
        "treatment",
        trial_reasons,
        prereg,
    )
    control_information = _trial_information(
        trial,
        "control",
        trial_reasons,
        prereg,
    )
    if (
        treatment_information is None
        or control_information is None
        or treatment_information["source_set_sha256"]
        != control_information["source_set_sha256"]
    ):
        trial_reasons.append(f"{trial_id}:information_mismatch")
    if str(trial.get("treatment_tool_policy_sha256") or "") != str(
        trial.get("control_tool_policy_sha256") or ""
    ) or str(trial.get("treatment_tool_policy_sha256") or "") != str(
        prereg.get("tool_policy_sha256") or ""
    ):
        trial_reasons.append(f"{trial_id}:tool_policy_mismatch")
    # DECODING PARITY: both arms must run the preregistered decode policy
    # (seed discipline, sampler, temperature, stop rules, context limit,
    # tokenizer/template). Without this, an outcome difference can come
    # from decoding rather than from the latent treatment.
    treatment_decode = str(trial.get("treatment_decode_policy_sha256") or "")
    control_decode = str(trial.get("control_decode_policy_sha256") or "")
    if (
        not _is_sha256(treatment_decode)
        or treatment_decode != control_decode
        or treatment_decode != str(prereg.get("decode_policy_sha256") or "")
    ):
        trial_reasons.append(f"{trial_id}:decode_policy_mismatch")
    return control_information, treatment_information

def _verify_frontier_gain_bundle_part_2(
    episode_id: Any,
    installed_app_build_sha256: str,
    prereg: Any,
    seen_control_request_ids: set[str],
    seen_episode_ids: set[str],
    trial: Any,
    trial_id: str,
    trial_reasons: list[str],
    worker_boot_id: str,
) -> tuple[Any, ...]:
    if episode_id in seen_episode_ids:
        trial_reasons.append(f"{trial_id}:duplicate_treatment_episode")
    seen_episode_ids.add(episode_id)
    if episode_id != _expected_arm_identifier(trial, "treatment", worker_boot_id):
        trial_reasons.append(f"{trial_id}:treatment_episode_id_unbound")
    control_request_id = _validate_control_receipt(
        trial,
        prereg,
        worker_boot_id,
        installed_app_build_sha256,
        trial_reasons,
    )
    if control_request_id in seen_control_request_ids:
        trial_reasons.append(f"{trial_id}:duplicate_control_request")
    seen_control_request_ids.add(control_request_id)
    if control_request_id != _expected_arm_identifier(
        trial, "control", worker_boot_id
    ):
        trial_reasons.append(f"{trial_id}:control_request_id_unbound")
    # `params_unchanged` is proven by hashing a fixed-stride SAMPLE of the
    # parameter tree before and after. How much of the tree that sample
    # touches decides what the claim is worth: a mutation living entirely
    # in unsampled tensors leaves both digests identical. The certificate
    # now measures the coverage instead of taking the verdict on trust.
    # The control arm only has a parameter tree to measure when it runs on
    # the resident model. An external frontier control runs on somebody
    # else's hardware, so demanding a canary from it would be demanding
    # evidence that cannot exist.
    canary_arms = (("treatment", "treatment_receipt"),)
    if str(prereg.get("comparison_kind") or "") != (
        "resident_32b_vs_external_frontier"
    ):
        canary_arms += (("control", "control_receipt"),)
    return canary_arms

def _verify_frontier_gain_bundle_achieved_power(
    min_trials: Any,
    order_by_domain: dict[str, dict[str, int]],
    paired: dict[str, list[PairedObservation]],
    prereg_alpha: Any,
    prereg_target_power: Any,
    prereg_win_share: Any,
    reasons: list[str],
    task_families: Any,
) -> tuple[dict[str, float], dict[str, int]]:
    achieved_power: dict[str, float] = {}
    effective_sample: dict[str, int] = {}
    for domain, observations in paired.items():
        if len(observations) < min_trials:
            reasons.append(f"{domain}:underpowered")
        # ACHIEVED power, recomputed from the trials that survived admission.
        # A domain can hold its preregistered trial count and still have lost
        # every disagreement that carried information, and the count alone
        # cannot see that.
        # Independence is counted in FAMILIES, not rows. Forty paraphrases of
        # one problem are one problem's worth of evidence however many
        # discordant pairs they generate, so a family contributes at most one.
        seen_families = {task_families.get(obs.task_id, "") for obs in observations}
        if "" in seen_families:
            reasons.append(f"{domain}:task_family_unassigned")
            seen_families.discard("")
        effective_sample[domain] = len(seen_families)
        if len(seen_families) < min_trials:
            reasons.append(f"{domain}:effective_sample_below_minimum")
        discordant = len(
            {
                task_families.get(observation.task_id, "")
                for observation in observations
                if bool(observation.treatment_success)
                != bool(observation.control_success)
            }
            - {""}
        )
        power = (
            _exact_mcnemar_power(discordant, prereg_alpha, prereg_win_share)
            if prereg_win_share > 0.5
            else 0.0
        )
        achieved_power[domain] = round(power, 6)
        if prereg_target_power and power < prereg_target_power:
            reasons.append(f"{domain}:achieved_power_below_target")
        balance = order_by_domain.get(domain, {})
        admitted_here = balance.get("treatment_first", 0) + balance.get("control_first", 0)
        # PER-DOMAIN balance. A run balanced overall can still have given one
        # domain the treatment first every time, and a domain is the unit the
        # claim is made about.
        if admitted_here and abs(
            balance["treatment_first"] - balance["control_first"]
        ) > max(1, math.ceil(admitted_here * 0.10)):
            reasons.append(f"{domain}:run_order_imbalanced")
    return achieved_power, effective_sample

def _verify_frontier_gain_bundle_admitted_order_total(
    order_by_domain: dict[str, dict[str, int]],
    order_effect_samples: dict[str, list[int]],
    prereg: Any,
    reasons: list[str],
    scored_pairs: list[tuple[float, float]],
    success_threshold: Any,
) -> tuple[list[float], Any]:
    admitted_order_total = sum(
        counts["treatment_first"] + counts["control_first"]
        for counts in order_by_domain.values()
    )
    admitted_treatment_first = sum(
        counts["treatment_first"] for counts in order_by_domain.values()
    )
    admitted_control_first = admitted_order_total - admitted_treatment_first
    if admitted_order_total and abs(
        admitted_treatment_first - admitted_control_first
    ) > max(1, math.ceil(admitted_order_total * 0.10)):
        reasons.append("run_order_imbalanced")
    # THRESHOLD SENSITIVITY. The gain is a step function of where the pass
    # line sits, and the line was preregistered — but preregistering a number
    # does not make the result robust to it. If moving the line a little in
    # either direction erases or reverses the gain, the claim is about the
    # line rather than about the treatment.
    fragile_thresholds: list[float] = []
    band = prereg.get("threshold_sensitivity_band")
    if (
        scored_pairs
        and success_threshold > 0.0
        and _finite_number(band)
        and 0.0 < float(band) <= 0.25
    ):
        low = max(0.0, success_threshold - float(band))
        high = min(1.0, success_threshold + float(band))
        # The gain only changes at an observed score, so evaluating at the
        # band edges plus every score inside it is EXACT, not a sample.
        candidates = {low, high, success_threshold}
        for treatment_score, control_score in scored_pairs:
            for score in (treatment_score, control_score):
                if low <= score <= high:
                    candidates.add(score)
        for candidate in sorted(candidates):
            gain = math.fsum(
                int(treatment_score >= candidate) - int(control_score >= candidate)
                for treatment_score, control_score in scored_pairs
            ) / len(scored_pairs)
            if gain <= 0.0:
                fragile_thresholds.append(round(candidate, 6))
        if fragile_thresholds:
            reasons.append("outcome_threshold_fragile")
    # Balance answers "did both orders run?", never "did order matter?".
    # If the paired gain lives in one order and vanishes in the other, order
    # is a rival explanation for the whole result.
    order_effect = 0.0
    if order_effect_samples["treatment_first"] and order_effect_samples["control_first"]:
        order_effect = abs(
            (
                math.fsum(order_effect_samples["treatment_first"])
                / len(order_effect_samples["treatment_first"])
            )
            - (
                math.fsum(order_effect_samples["control_first"])
                / len(order_effect_samples["control_first"])
            )
        )
        max_order_effect = prereg.get("max_order_effect")
        if _finite_number(max_order_effect) and 0.0 < float(max_order_effect) <= 0.5:
            if order_effect > float(max_order_effect):
                reasons.append("order_effect_exceeds_preregistered_maximum")
    elif admitted_order_total:
        # One order never ran on an admitted trial, so no order effect can be
        # estimated at all. Silence here would read as "no effect found".
        reasons.append("order_effect_unmeasurable")
    return fragile_thresholds, order_effect

def _verify_frontier_gain_bundle_cp126_ca4271_cccfcb5(
    bundle: Any,
    earliest_evaluation_started_at: Any,
    earliest_task_generated_at: Any,
    reasons: list[str],
    trusted_transparency_logs: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[Any, Any]:
    # CP126 89ca4271 + 9cccfcb5: a signature says who wrote a timestamp, never
    # when the event happened. Both the preregistration and the task
    # commitment have to be in an append-only log whose root was pinned out of
    # band, and logged before the work they claim to precede.
    anchors = bundle.get("timestamp_anchors")
    anchors = anchors if isinstance(anchors, dict) else {}
    prereg_logged_at = _validate_timestamp_anchor(
        anchors.get("preregistration"),
        event="preregistration",
        committed_data=str(bundle.get("preregistration_sha256") or ""),
        trusted_logs=trusted_transparency_logs,
        reasons=reasons,
    )
    if (
        prereg_logged_at is not None
        and math.isfinite(earliest_task_generated_at)
        and prereg_logged_at > earliest_task_generated_at
    ):
        # The preregistration reached the log after tasks already existed, so
        # it could have been written to fit them.
        reasons.append("preregistration_logged_after_task_generation")
    commitment_logged_at = _validate_timestamp_anchor(
        anchors.get("task_commitment"),
        event="task_commitment",
        committed_data=str(bundle.get("task_commitment_sha256") or ""),
        trusted_logs=trusted_transparency_logs,
        reasons=reasons,
    )
    if (
        commitment_logged_at is not None
        and math.isfinite(earliest_evaluation_started_at)
        and commitment_logged_at > earliest_evaluation_started_at
    ):
        reasons.append("task_commitment_logged_after_evaluation_started")
    return commitment_logged_at, prereg_logged_at

def verify_frontier_gain_bundle(
    bundle: Any,
    *,
    trusted_verifiers: Mapping[str, Mapping[str, str]] | None = None,
    trusted_task_issuers: Mapping[str, Mapping[str, str]] | None = None,
    trusted_producers: Mapping[str, Mapping[str, str]] | None = None,
    trusted_transparency_logs: Mapping[str, Mapping[str, Any]] | None = None,
    raw_artifact_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic certificate; never infer missing evidence.

    ``raw_artifact_receipt`` must come from
    :func:`core.brain.llm.latent_cortex.frontier_artifacts.verify_raw_artifact_package`.
    Without it the bundle's raw artifacts are unverified and the certificate
    cannot be accepted.
    """
    reasons: list[str] = []
    if not isinstance(bundle, dict):
        bundle = {}
        reasons.append("bundle_not_mapping")
    if bundle.get("schema") != SCHEMA:
        reasons.append("schema_mismatch")
    prereg = _validate_preregistration(bundle.get("preregistration"), reasons)
    try:
        expected_prereg_hash = canonical_sha256(prereg) if prereg else ""
    except (TypeError, ValueError, OverflowError, RecursionError):
        expected_prereg_hash = ""
        reasons.append("preregistration_not_canonical_json")
    if str(bundle.get("preregistration_sha256") or "") != expected_prereg_hash:
        reasons.append("preregistration_hash_mismatch")
    resident = _validate_resident_model(bundle.get("resident_model"), prereg, reasons)
    _reconcile_compute_profile(resident, prereg, reasons)
    checkpoint = str(resident.get("checkpoint_fingerprint") or "")
    worker_boot_id = str(resident.get("worker_boot_id") or "")
    installed_app_build_sha256 = str(
        resident.get("installed_app_build_sha256") or ""
    )
    expected_estimator = str(prereg.get("compute_estimator_sha256") or "")
    expected_profile = str(prereg.get("compute_profile_sha256") or "")
    raw_forward_passes = prereg.get("max_forward_passes_per_trial")
    max_forward_passes = (
        raw_forward_passes
        if type(raw_forward_passes) is int and raw_forward_passes > 0
        else 0
    )
    if not _is_sha256(bundle.get("task_commitment_sha256")):
        reasons.append("task_commitment_missing")
    if not _is_sha256(bundle.get("raw_artifact_manifest_sha256")):
        reasons.append("raw_artifact_manifest_missing")

    trials = bundle.get("trials")
    if not isinstance(trials, list) or not trials:
        reasons.append("trials_missing")
        trials = []
    raw_domains = prereg.get("domains")
    domains = (
        [
            domain
            for domain in raw_domains
            if isinstance(domain, str) and domain.strip()
        ]
        if isinstance(raw_domains, list)
        else []
    )
    raw_min_trials = prereg.get("min_trials_per_domain")
    min_trials = raw_min_trials if type(raw_min_trials) is int else 30
    paired: dict[str, list[PairedObservation]] = {domain: [] for domain in domains}
    seen_trial_ids: set[str] = set()
    seen_task_ids: set[str] = set()
    seen_task_payloads: set[str] = set()
    seen_episode_ids: set[str] = set()
    seen_control_request_ids: set[str] = set()
    # Generation-manifest fields neither arm declared (CP126 8a56c486). These
    # are surfaced on the certificate as unverified parity, not silently
    # assumed equal.
    generation_parity_gaps: set[str] = set()
    # Integrity claims asserted by a boolean but never measured by a digest
    # (CP126 6090a5ae / 01e8b3c1). A REFUTED digest rejects the trial; an
    # ABSENT one is carried here so the certificate says what it could not
    # verify instead of treating the assertion as proof.
    unproven_integrity_claims: set[str] = set()
    frozen_at = (
        float(prereg["frozen_at"])
        if _finite_number(prereg.get("frozen_at"), positive=True)
        else 0.0
    )
    latest_task_generated_at = 0.0
    earliest_task_generated_at = float("inf")
    earliest_evaluation_started_at = float("inf")
    latest_evaluation_started_at = 0.0
    #: The moment the last piece of evidence actually existed. Everything the
    #: independent verifier attests to has to postdate it.
    latest_scoring_completed_at = 0.0
    #: (treatment, control) scores for admitted trials, for the threshold
    #: sensitivity audit.
    scored_pairs: list[tuple[float, float]] = []
    #: What fraction of the parameter tree the before/after canary hashed.
    observed_canary_coverage: list[float] = []
    raw_canary_floor = prereg.get("min_parameter_canary_tensor_coverage")
    canary_coverage_floor = (
        float(raw_canary_floor)
        if _finite_number(raw_canary_floor) and 0.0 < float(raw_canary_floor) <= 1.0
        else 0.0
    )
    success_threshold = (
        float(prereg["success_threshold"])
        if _finite_number(prereg.get("success_threshold"))
        and 0.0 < float(prereg["success_threshold"]) <= 1.0
        else 0.0
    )
    tolerance = (
        float(prereg["compute_tolerance"])
        if _finite_number(prereg.get("compute_tolerance"))
        else 0.0
    )
    admitted_trial_count = 0
    rejected_trial_count = 0
    comparison_accounting: list[dict[str, Any]] = []
    #: Every trial must be scanned by the SAME scanner. A per-trial scanner
    #: lets a producer keep rescanning until a trial passes, which is choosing
    #: the instrument after seeing the reading.
    contamination_scanners: set[str] = set()
    task_families, task_diversity_sha256 = _validate_task_diversity(bundle, reasons)
    capability_classes = _validate_domain_taxonomy(bundle, prereg, reasons)
    #: Order accounting over ADMITTED trials, per domain. The global count
    #: previously included rejected trials, so a producer could balance the
    #: run with trials that never reached the grader.
    order_by_domain: dict[str, dict[str, int]] = {
        domain: {"treatment_first": 0, "control_first": 0} for domain in paired
    }
    #: Paired differences split by which arm ran first, so the order effect
    #: is measured on the quantity the claim rests on.
    order_effect_samples: dict[str, list[int]] = {
        "treatment_first": [],
        "control_first": [],
    }
    for trial in trials:
        # ADMISSION ISOLATION: a trial's own integrity defects are collected
        # here and, if any exist, the trial is EXCLUDED from the paired
        # observations that produce the statistical claim. Previously every
        # per-trial defect only appended a reason while the trial still fed
        # the grader, so contaminated, unblinded, parity-failing, or
        # unauthenticated trials could produce a nested PROVEN result inside
        # a rejected certificate.
        trial_reasons: list[str] = []
        if not isinstance(trial, dict):
            reasons.append("trial_not_mapping")
            rejected_trial_count += 1
            continue
        trial_id = str(trial.get("trial_id") or "")
        task_id = str(trial.get("task_id") or "")
        domain = str(trial.get("domain") or "")
        if not trial_id or trial_id in seen_trial_ids:
            reasons.append("duplicate_or_missing_trial_id")
            rejected_trial_count += 1
            continue
        if not task_id or task_id in seen_task_ids:
            reasons.append(f"{trial_id}:duplicate_or_missing_task_id")
            rejected_trial_count += 1
            continue
        seen_trial_ids.add(trial_id)
        seen_task_ids.add(task_id)
        if domain not in paired:
            reasons.append(f"{trial_id}:unregistered_domain")
            rejected_trial_count += 1
            continue
        if trial.get("held_out") is not True or trial.get("contamination_scan_passed") is not True:
            trial_reasons.append(f"{trial_id}:heldout_or_contamination_unproven")
        # The boolean above is the producer's CLAIM. This is the evidence.
        scanner_digest = _validate_contamination_receipt(trial, trial_id, trial_reasons)
        if not _finite_number(trial.get("task_generated_at"), positive=True) or float(
            trial.get("task_generated_at") or 0.0
        ) <= frozen_at:
            trial_reasons.append(f"{trial_id}:task_not_generated_after_freeze")
        evaluation_started_at = trial.get("evaluation_started_at")
        if not _finite_number(evaluation_started_at, positive=True):
            trial_reasons.append(f"{trial_id}:evaluation_start_invalid")
        else:
            evaluation_time = float(evaluation_started_at)
            earliest_evaluation_started_at = min(
                earliest_evaluation_started_at,
                evaluation_time,
            )
            latest_evaluation_started_at = max(
                latest_evaluation_started_at,
                evaluation_time,
            )
        # CP126 6f55ecd3: the attestation was timestamped against when
        # evaluation STARTED. A verifier could therefore sign after the first
        # trial began and before any output or score existed, and the
        # certificate would read as though the evidence had been reviewed.
        # No completion time was recorded anywhere, so the gap was invisible.
        evaluation_completed_at = trial.get("evaluation_completed_at")
        scoring_completed_at = trial.get("scoring_completed_at")
        if not _finite_number(evaluation_completed_at, positive=True):
            trial_reasons.append(f"{trial_id}:evaluation_completion_missing")
        elif _finite_number(evaluation_started_at, positive=True) and float(
            evaluation_completed_at
        ) <= float(evaluation_started_at):
            trial_reasons.append(f"{trial_id}:evaluation_completed_before_start")
        if not _finite_number(scoring_completed_at, positive=True):
            trial_reasons.append(f"{trial_id}:scoring_completion_missing")
        elif _finite_number(evaluation_completed_at, positive=True) and float(
            scoring_completed_at
        ) < float(evaluation_completed_at):
            trial_reasons.append(f"{trial_id}:scored_before_evaluation_completed")
        else:
            latest_scoring_completed_at = max(
                latest_scoring_completed_at, float(scoring_completed_at)
            )
        control_information, treatment_information = _verify_frontier_gain_bundle_part_1(prereg, seen_task_payloads, trial, trial_id, trial_reasons)
        order = str(trial.get("run_order") or "")
        if order not in _RUN_ORDERS:
            trial_reasons.append(f"{trial_id}:invalid_run_order")
        if type(trial.get("treatment_success")) is not bool or type(
            trial.get("control_success")
        ) is not bool:
            trial_reasons.append(f"{trial_id}:non_boolean_outcome")
            reasons.extend(trial_reasons)
            rejected_trial_count += 1
            continue
        # The boolean has to BE the preregistered threshold applied to a
        # recorded score. Without the score there is no way to tell a hair's
        # -breadth win from a rout, and no way to check that the line was not
        # moved to produce the win.
        trial_scores: dict[str, float] = {}
        for arm in ("treatment", "control"):
            raw_score = trial.get(f"{arm}_score")
            if not _finite_number(raw_score) or not 0.0 <= float(raw_score) <= 1.0:
                trial_reasons.append(f"{trial_id}:{arm}_score_missing")
                continue
            trial_scores[arm] = float(raw_score)
            if success_threshold > 0.0 and bool(trial[f"{arm}_success"]) != (
                float(raw_score) >= success_threshold
            ):
                trial_reasons.append(f"{trial_id}:{arm}_outcome_contradicts_score")
        treatment_flops, treatment_layers, treatment_resource = _trial_compute(
            trial,
            "treatment",
            expected_estimator,
            trial_reasons,
            expected_profile=expected_profile,
            max_forward_passes=max_forward_passes,
        )
        control_flops, control_layers, control_resource = _trial_compute(
            trial,
            "control",
            expected_estimator,
            trial_reasons,
            expected_profile=expected_profile,
            max_forward_passes=max_forward_passes,
        )
        if treatment_flops is not None and control_flops is not None:
            mismatch = abs(treatment_flops - control_flops) / max(1.0, control_flops)
            if mismatch > tolerance:
                trial_reasons.append(f"{trial_id}:compute_mismatch")
        # LAYER-APPLICATION PARITY: both arm layer counts were validated and
        # carried into the observation, but only FLOPs were compared — so a
        # treatment could apply radically more layer work while reporting
        # equal estimated FLOPs.
        if treatment_layers is not None and control_layers is not None:
            layer_mismatch = abs(treatment_layers - control_layers) / max(
                1, control_layers
            )
            if layer_mismatch > tolerance:
                trial_reasons.append(f"{trial_id}:layer_apps_mismatch")
        if (
            treatment_resource is not None
            and control_resource is not None
            and treatment_information is not None
            and control_information is not None
        ):
            tolerance_fraction = Fraction(str(tolerance)).limit_denominator(1_000_000)
            accounting = certify_comparison_accounting(
                treatment_resource=treatment_resource,
                control_resource=control_resource,
                treatment_information=treatment_information,
                control_information=control_information,
                tolerance_numerator=tolerance_fraction.numerator,
                tolerance_denominator=tolerance_fraction.denominator,
                require_compute_parity=True,
            )
            comparison_accounting.append(
                {"trial_id": trial_id, **accounting}
            )
            for reason in accounting["reasons"]:
                trial_reasons.append(
                    f"{trial_id}:comparison_accounting:{reason}"
                )
        trial_integrity_gaps: list[str] = []
        episode_id = _validate_treatment_receipt(
            trial,
            checkpoint,
            worker_boot_id,
            installed_app_build_sha256,
            trial_reasons,
            trial_integrity_gaps,
        )
        unproven_integrity_claims.update(trial_integrity_gaps)
        canary_arms = _verify_frontier_gain_bundle_part_2(episode_id, installed_app_build_sha256, prereg, seen_control_request_ids, seen_episode_ids, trial, trial_id, trial_reasons, worker_boot_id)
        for arm, receipt_key in canary_arms:
            coverage = _parameter_canary_coverage(trial.get(receipt_key))
            if coverage is None:
                trial_reasons.append(f"{trial_id}:{arm}_parameter_canary_unmeasured")
                continue
            observed_canary_coverage.append(coverage)
            if canary_coverage_floor and coverage < canary_coverage_floor:
                trial_reasons.append(
                    f"{trial_id}:{arm}_parameter_canary_coverage_below_floor"
                )
        # CP126 8a56c486: seeds and decoding settings were never paired, so a
        # difference in sampling could be reported as a latent-cortex effect.
        generation_parity_gaps.update(
            _validate_arm_generation_parity(
                trial_id,
                trial.get("treatment_receipt"),
                trial.get("control_receipt"),
                trial_reasons,
            )
        )
        if _finite_number(trial.get("task_generated_at"), positive=True):
            earliest_task_generated_at = min(
                earliest_task_generated_at, float(trial["task_generated_at"])
            )
            latest_task_generated_at = max(
                latest_task_generated_at, float(trial["task_generated_at"])
            )
        if trial_reasons:
            # Defective trial: its reasons surface on the certificate, but it
            # contributes NOTHING to the statistical claim.
            reasons.extend(trial_reasons)
            rejected_trial_count += 1
            continue
        admitted_trial_count += 1
        # Only ADMITTED trials constrain the scanner: a rejected trial's junk
        # receipt already surfaced its own reason and must not manufacture a
        # second one about run-wide uniformity.
        contamination_scanners.add(scanner_digest)
        scored_pairs.append((trial_scores["treatment"], trial_scores["control"]))
        if order in order_by_domain.get(domain, {}):
            order_by_domain[domain][order] += 1
            order_effect_samples[order].append(
                int(bool(trial["treatment_success"])) - int(bool(trial["control_success"]))
            )
        paired[domain].append(
            PairedObservation(
                task_id=task_id,
                family=domain,
                treatment_success=trial["treatment_success"],
                control_success=trial["control_success"],
                treatment_layer_apps=treatment_layers,
                control_layer_apps=control_layers,
            )
        )

    if len(contamination_scanners) > 1:
        reasons.append("contamination_scanner_not_uniform")

    prereg_alpha = (
        float(prereg["alpha"])
        if _finite_number(prereg.get("alpha")) and 0.0 < float(prereg["alpha"]) <= 0.05
        else 0.05
    )
    prereg_win_share = (
        float(prereg["preregistered_discordant_win_share"])
        if _finite_number(prereg.get("preregistered_discordant_win_share"))
        and 0.5 < float(prereg["preregistered_discordant_win_share"]) <= 1.0
        else 0.0
    )
    prereg_target_power = (
        float(prereg["target_power"])
        if _finite_number(prereg.get("target_power"))
        and 0.5 < float(prereg["target_power"]) <= 0.999
        else 0.0
    )
    achieved_power, effective_sample = _verify_frontier_gain_bundle_achieved_power(min_trials, order_by_domain, paired, prereg_alpha, prereg_target_power, prereg_win_share, reasons, task_families)
    fragile_thresholds, order_effect = _verify_frontier_gain_bundle_admitted_order_total(order_by_domain, order_effect_samples, prereg, reasons, scored_pairs, success_threshold)

    task_issuer_id, task_commitment_attestation_sha256 = _validate_task_commitment(
        bundle,
        prereg,
        trials,
        latest_task_generated_at=latest_task_generated_at,
        earliest_evaluation_started_at=earliest_evaluation_started_at,
        task_diversity_sha256=task_diversity_sha256,
        blinding_map_sha256=_blinding_map_sha256(bundle),
        trusted_task_issuers=trusted_task_issuers,
        reasons=reasons,
    )
    _validate_blinding(
        bundle,
        latest_scoring_completed_at=latest_scoring_completed_at,
        task_issuer_id=task_issuer_id,
        reasons=reasons,
    )

    comparison_kind = str(prereg.get("comparison_kind") or "")
    try:
        evidence_hash = evidence_payload_sha256(bundle)
    except (TypeError, ValueError, OverflowError, RecursionError):
        evidence_hash = ""
        reasons.append("evidence_payload_not_canonical_json")
    commitment_logged_at, prereg_logged_at = _verify_frontier_gain_bundle_cp126_ca4271_cccfcb5(bundle, earliest_evaluation_started_at, earliest_task_generated_at, reasons, trusted_transparency_logs)

    verified_producer_id, producer_organization = _validate_producer_identity(
        bundle,
        evidence_hash=evidence_hash,
        trusted_producers=trusted_producers,
        reasons=reasons,
    )
    attestation_deadline = max(
        latest_task_generated_at,
        latest_evaluation_started_at,
        # Signing after the run began is not review. Attestation has to
        # postdate the LAST score.
        latest_scoring_completed_at,
    )
    # The digest of the receipt a verifier gets only by opening the raw
    # artifact files. Signing it is the closest thing to execution evidence
    # available without a measured environment.
    try:
        artifact_receipt_sha256 = (
            canonical_sha256(dict(raw_artifact_receipt))
            if isinstance(raw_artifact_receipt, Mapping)
            else ""
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        artifact_receipt_sha256 = ""
        reasons.append("raw_artifact_receipt_not_canonical_json")
    raw_attestations = bundle.get("independent_verifiers")
    if not isinstance(raw_attestations, list) or not raw_attestations:
        reasons.append("independent_verifier_quorum_missing")
        raw_attestations = []
    independent_verifier_ids: list[str] = []
    independent_attestation_hashes: list[str] = []
    verifier_organizations: set[str] = set()
    for raw_attestation in raw_attestations:
        verifier_id, attestation_sha256, organization = (
            _validate_independent_attestation(
                bundle,
                raw_attestation,
                evidence_hash=evidence_hash,
                artifact_receipt_sha256=artifact_receipt_sha256,
                latest_task_generated_at=attestation_deadline,
                producer_id=verified_producer_id,
                trusted_verifiers=trusted_verifiers,
                reasons=reasons,
            )
        )
        if not verifier_id or not attestation_sha256:
            continue
        if verifier_id in independent_verifier_ids:
            # The same signer twice is one opinion counted twice.
            reasons.append("independent_verifier_signature_duplicated")
            continue
        independent_verifier_ids.append(verifier_id)
        independent_attestation_hashes.append(attestation_sha256)
        if organization:
            verifier_organizations.add(organization)
    if task_issuer_id and task_issuer_id in independent_verifier_ids:
        reasons.append("independent_roles_reused")
    if len(verifier_organizations) < MINIMUM_VERIFIER_QUORUM:
        # Two signatures from one organization are one organization's opinion.
        reasons.append("independent_verifier_quorum_not_met")
    if producer_organization and producer_organization in verifier_organizations:
        reasons.append("verifier_shares_producer_organization")
    independent_verifier_id = (
        independent_verifier_ids[0] if independent_verifier_ids else ""
    )
    independent_attestation_sha256 = (
        canonical_sha256(sorted(independent_attestation_hashes))
        if independent_attestation_hashes
        else ""
    )

    raw_alpha = prereg.get("alpha")
    alpha = (
        float(raw_alpha)
        if _finite_number(raw_alpha) and 0.0 < float(raw_alpha) <= 0.05
        else 0.05
    )
    raw_minimum_effect = prereg.get("minimum_effect")
    minimum_effect = (
        float(raw_minimum_effect)
        if _finite_number(raw_minimum_effect)
        and 0.0 < float(raw_minimum_effect) <= 0.5
        else 0.05
    )
    # CLAIM SCOPE: beating an ablated SELF establishes that the treatment
    # contributes; it does NOT establish frontier competitiveness. The two
    # comparison kinds therefore carry different statements and different
    # scopes, and only an external comparison can assert competitiveness.
    external_comparison = comparison_kind == "resident_32b_vs_external_frontier"
    if external_comparison:
        claim_scope = "frontier_competitiveness_vs_external_control"
        # The control's IDENTITY is the provider's own word. Its provider must
        # now be a recognised frontier lab, and every per-trial receipt must
        # match the preregistered strings — but no provider signature or
        # transparency-log inclusion is verified, so this establishes WHICH
        # model was named, not that the named model actually served the
        # requests. The claim says so in its own text, because a reader who
        # only sees "outperforms an external frontier control" would take it
        # for an independently attested comparison.
        claim_statement = (
            "the full resident-32B Recursive Latent Cortex outperforms the "
            "preregistered external frontier control on the preregistered "
            f"domains (control identity is {EXTERNAL_CONTROL_ATTESTATION}: the "
            "provider is a recognised frontier lab and receipts match "
            "preregistration, but no provider signature is verified)"
        )
    else:
        claim_scope = "treatment_contribution_vs_same_checkpoint_ablation"
        claim_statement = (
            "the full resident-32B Recursive Latent Cortex improves over its own "
            "preregistered same-checkpoint vanilla ablation (treatment contribution "
            "only — NOT evidence of frontier competitiveness; resident identity is "
            f"{RESIDENT_MODEL_ATTESTATION}: hashes are bound to preregistration but "
            "not resolved against bytes on disk)"
        )
    # Compute evidence must survive INTO the claim. The grader was previously
    # told to ignore it (tolerance 1.0, require_compute False), so a claim
    # could read PROVEN while carrying no compute validity at all.
    claim = grade_paired_treatment_vs_control(
        "exp6_frontier_comparison",
        claim_statement,
        paired,
        alpha=alpha,
        minimum_effect=minimum_effect,
        compute_tolerance=tolerance,
        require_compute=True,
        # CP126 3f561b15: the grader used to emit a tier with nothing behind
        # it but a wall-clock time. The certificate already pins every one of
        # these, so the statistical claim inherits them and a reader can
        # re-derive the verdict rather than take it.
        provenance=ExperimentProvenance(
            task_manifest_sha256=str(bundle.get("task_commitment_sha256") or ""),
            checkpoint_fingerprint=checkpoint,
            schedule_sha256=str(bundle.get("preregistration_sha256") or ""),
            verifier_version=str(prereg.get("scorer_implementation_sha256") or ""),
            environment_sha256=installed_app_build_sha256,
        ),
    )
    positive = claim.evidence.get("positive_families") or []
    required_positive = math.ceil(len(domains) * 2 / 3) if domains else 1
    if len(positive) < required_positive:
        reasons.append("insufficient_positive_domains")
    if claim.evidence.get("regressed_families"):
        reasons.append("domain_regression_detected")
    if claim.tier != PROVEN:
        reasons.append("paired_gain_not_proven")
    # The certificate must name WHICH domains carry the gain: a two-thirds
    # rule lets a third of domains show no benefit, and an unqualified claim
    # would hide that heterogeneity.
    non_positive_domains = sorted(set(domains) - set(positive))

    # ABSOLUTE CAPABILITY FLOOR over admitted trials: a relative win over a
    # weaker control is not a capability certificate on its own.
    min_success_rate = (
        float(prereg["min_treatment_success_rate"])
        if _finite_number(prereg.get("min_treatment_success_rate"))
        else 1.1  # unreachable → fails closed when preregistration is invalid
    )
    admitted_observations = [obs for items in paired.values() for obs in items]
    treatment_success_rate = (
        sum(1 for obs in admitted_observations if obs.treatment_success)
        / len(admitted_observations)
        if admitted_observations
        else 0.0
    )
    if treatment_success_rate < min_success_rate:
        reasons.append("treatment_below_absolute_capability_floor")

    release_summary = _validate_release_readiness(
        bundle,
        prereg,
        treatment_success_rate=treatment_success_rate,
        reasons=reasons,
    )

    # RAW ARTIFACT VERIFICATION: the bundle's manifest hash was only
    # SYNTAX-checked here, so the core API could accept and claim PROVEN
    # without any raw artifact ever being loaded or recomputed. A receipt
    # from verify_raw_artifact_package (frontier_artifacts) is now required,
    # and it must bind this exact bundle.
    raw_artifact_verified = _validate_raw_artifact_receipt(
        raw_artifact_receipt, bundle, reasons
    )

    accepted = not reasons
    statistical_claim = claim.to_dict()
    # Wall-clock grading time is not evidence and would make identical bundles
    # produce different certificates. The evidence hash binds the actual input.
    statistical_claim.pop("graded_at", None)
    # A nested tier must never be readable as a standalone verdict: the claim
    # is only admissible when the whole certificate was accepted, and it is
    # only ever a claim about THIS comparison's scope.
    statistical_claim["admissible"] = accepted
    statistical_claim["claim_scope"] = claim_scope
    statistical_claim["graded_on_admitted_trials"] = admitted_trial_count
    certificate: dict[str, Any] = {
        "schema": CERTIFICATE_SCHEMA,
        "accepted": accepted,
        "claim_tier": PROVEN if accepted else CONJECTURE,
        "comparison_kind": comparison_kind,
        # Scope is part of the verdict: an accepted same-checkpoint bundle
        # proves treatment CONTRIBUTION, never frontier competitiveness.
        "claim_scope": claim_scope,
        "frontier_competitiveness_established": bool(accepted and external_comparison),
        "raw_artifact_verified": raw_artifact_verified,
        "admitted_trial_count": admitted_trial_count,
        "rejected_trial_count": rejected_trial_count,
        "positive_domains": sorted(positive),
        "non_positive_domains": non_positive_domains,
        "treatment_success_rate": round(treatment_success_rate, 6),
        "min_treatment_success_rate": min_success_rate,
        # CP126 5c540b93: what this release does against the LAST one, and
        # against safety and calibration budgets. Beating your own ablation
        # says nothing about either.
        "release_readiness": release_summary,
        # CP126 d0f2ae6c: what actually backs `params_unchanged`, and how much
        # of the parameter tree the measurement touched. Naming the method
        # stops a sampled comparison reading as an exhaustive one.
        "parameter_integrity_attestation": PARAMETER_INTEGRITY_ATTESTATION,
        # CP126 89ca4271 + 9cccfcb5: when the log says these existed, as
        # opposed to when the producer says they did.
        "preregistration_logged_at": prereg_logged_at,
        "task_commitment_logged_at": commitment_logged_at,
        "min_parameter_canary_tensor_coverage": (
            round(min(observed_canary_coverage), 6)
            if observed_canary_coverage
            else 0.0
        ),
        "evidence_payload_sha256": evidence_hash,
        "independent_verifier_id": independent_verifier_id,
        "independent_attestation_sha256": independent_attestation_sha256,
        # CP126 48df4291 + c7a9c1a5: who agreed, and from how many
        # organizations. One signature is one organization's opinion, and the
        # producer's identity is now cryptographic rather than a string it
        # wrote about itself.
        "independent_verifier_ids": sorted(independent_verifier_ids),
        "independent_verifier_organizations": sorted(verifier_organizations),
        "verifier_quorum_required": MINIMUM_VERIFIER_QUORUM,
        "verified_producer_id": verified_producer_id,
        # CP126 2528aa7e: what the signature proves the signer DID, and what
        # it does not — no measured environment attests that the pinned
        # verifier binary is the one that ran.
        "verifier_execution_attestation": VERIFIER_EXECUTION_ATTESTATION,
        # CP126 f17aa1b8: what ties the running binary to source, and how
        # far that goes.
        "build_provenance_attestation": BUILD_PROVENANCE_ATTESTATION,
        "task_issuer_id": task_issuer_id,
        "task_commitment_attestation_sha256": task_commitment_attestation_sha256,
        "preregistration_sha256": expected_prereg_hash,
        "trial_count": len(trials),
        "domain_counts": {domain: len(items) for domain, items in paired.items()},
        # CP126 eb6bed71: breadth measured in capability classes rather than
        # in how many distinct strings the producer typed.
        "capability_classes": capability_classes,
        "required_capability_classes": MINIMUM_CAPABILITY_CLASSES,
        # CP126 596c9811 + 9e810491: what the surviving sample could actually
        # detect, and whether order is a rival explanation for the gain.
        "achieved_power_by_domain": achieved_power,
        # CP126 a52cbfbf: distinct task IDs are not distinct tasks. This is
        # the count after near-duplicates collapse into one family.
        "effective_sample_by_domain": effective_sample,
        "task_diversity_sha256": task_diversity_sha256,
        "target_power": prereg_target_power,
        "order_balance_by_domain": {
            domain: dict(counts) for domain, counts in order_by_domain.items()
        },
        "measured_order_effect": round(order_effect, 6),
        # CP126 e7b9dc9c: where the pass line sat, and whether the gain
        # survives moving it inside the preregistered band.
        "success_threshold": success_threshold,
        "fragile_thresholds": fragile_thresholds,
        "threshold_robust": not fragile_thresholds,
        "comparison_accounting": comparison_accounting,
        "required_positive_domains": required_positive,
        # CP126 8a56c486: generation-manifest fields that NEITHER arm declared.
        # Declared differences reject the trial; these are the parity claims the
        # certificate could not verify at all, carried on the certificate rather
        # than assumed equal. Closing them needs worker-side manifest emission.
        "unverified_generation_parity_fields": sorted(generation_parity_gaps),
        "generation_parity_fully_declared": not generation_parity_gaps,
        "unproven_integrity_claims": sorted(unproven_integrity_claims),
        "integrity_claims_digest_backed": not unproven_integrity_claims,
        "reasons": sorted(set(reasons)),
        "statistical_claim": statistical_claim,
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    return certificate


__all__ = [
    "CERTIFICATE_SCHEMA",
    "INDEPENDENT_ATTESTATION_SCHEMA",
    "SCHEMA",
    "TASK_COMMITMENT_ATTESTATION_SCHEMA",
    "canonical_sha256",
    "evidence_payload_sha256",
    "verify_frontier_gain_bundle",
]
