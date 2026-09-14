"""The sealed closeout canaries, read back and checked.

Each predicate opens one closeout bundle — a result, its verification, the
hashes of the source it was sealed over — and answers whether the certificate
still holds for the code that is running. They are leaves: they read files
and return a bool, and the only thing they share with the registry is the
name for "this measured nothing".

Taken out of `model_validation`, which had reached 6,900 lines, because
nothing in here decides anything the registry needs to see; the registry
imports the predicates back and binds them to claims as before.
"""

from __future__ import annotations

import logging
import pathlib

from core.organism.nothing_measured import NothingMeasured

logger = logging.getLogger("Aura.Validation.Canaries")


def _semantic_neural_composition_certificate_holds() -> bool:
    bundle = _canary_artifact_bundle(
        "artifacts/closeout/latent_cortex/typed_composition_canary_20260831",
        "tools/verify_semantic_neural_composition_canary.py",
    )
    if bundle is None:
        return False
    result, verification, current_hashes, verifier_sha, artifact_root = bundle
    source_hashes = result["source_sha256s"]

    expected_counts = {
        "treatment_exact": 96,
        "additive_lesion_disrupted": 96,
        "multiplicative_lesion_disrupted": 96,
        "wrong_operand_disrupted": 96,
    }
    expected_boundary = (
        "fresh family-neutral typed-operation recombination through existing learned "
        "arithmetic tissue; this does not establish natural-language transfer, "
        "open-domain reasoning gain, resident decoded-answer superiority, or broader "
        "serving"
    )
    return bool(
        result.get("schema") == "aura.rlc.semantic_neural_composition_canary.v1"
        and result.get("passed") is True
        and result.get("verdict") == "SUPPORTED_OPERATION_COMPOSITION"
        and result.get("task_count") == 96
        and result.get("counts") == expected_counts
        and result.get("teacher_available_to_treatment") is False
        and result.get("verifier_answer_available_to_treatment") is False
        and result.get("private_trace_available_to_treatment") is False
        and result.get("claim_boundary") == expected_boundary
        and current_hashes == source_hashes
        and verification.get("schema")
        == "aura.rlc.semantic_neural_composition_verification.v1"
        and verification.get("verified") is True
        and verification.get("task_count") == 96
        and verification.get("counts") == expected_counts
        and verification.get("claim_boundary") == expected_boundary
        and verification.get("input_receipt_sha256") == result.get("receipt_sha256")
        and verification.get("task_set_sha256") == result.get("task_set_sha256")
        and verification.get("producer_source_sha256s") == source_hashes
        and verification.get("verifier_source_sha256") == verifier_sha
    )


def _resident_semantic_neural_composition_decode_certificate_holds() -> bool:
    import hashlib
    import json

    bundle = _canary_artifact_bundle(
        "artifacts/closeout/latent_cortex/typed_composition_decode_canary_20260831",
        "tools/verify_semantic_neural_composition_decode_canary.py",
    )
    if bundle is None:
        return False
    result, verification, current_hashes, verifier_sha, artifact_root = bundle
    source_hashes = result["source_sha256s"]
    try:
        journal_path = artifact_root / "result.json.journal.jsonl"
        journal_sha = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Semantic composition certificate unreadable, claim unverified: %s", exc)
        return False

    expected_boundary = (
        "resident-model serialization of fresh family-neutral typed-operation "
        "composition from authenticated learned tissue under matched causal controls; "
        "not hidden-state internalization, open-domain reasoning gain, unrestricted "
        "serving, static fusion, or frontier performance"
    )
    expected_exact = {
        "ordinary_base": 0,
        "matched_wire_base": 0,
        "treatment": 8,
        "additive_lesion": 0,
        "multiplicative_lesion": 0,
        "matched_wrong_state": 0,
    }
    arms = result.get("arms")
    journal = verification.get("journal_identity")
    return bool(
        result.get("schema")
        == "aura.rlc.semantic_neural_composition_decode_canary.v1"
        and result.get("admitted") is True
        and result.get("task_count") == 8
        and result.get("decode_calls_per_arm_per_task") == 1
        and isinstance(arms, dict)
        and {arm: values.get("exact") for arm, values in arms.items()}
        == expected_exact
        and result.get("gain_count") == 8
        and result.get("regression_count") == 0
        and result.get("claim_boundary") == expected_boundary
        and current_hashes == source_hashes
        and verification.get("schema")
        == "aura.rlc.semantic_neural_composition_decode_verification.v1"
        and verification.get("verified") is True
        and verification.get("source_commit") == result.get("source_commit")
        and verification.get("source_sha256s") == source_hashes
        and verification.get("independent_exact_by_arm") == expected_exact
        and verification.get("gain_count") == 8
        and verification.get("regression_count") == 0
        and verification.get("paired_one_sided_exact_p") == 0.00390625
        and verification.get("claim_boundary") == expected_boundary
        and verification.get("input_receipt_sha256") == result.get("receipt_sha256")
        and verification.get("verifier_source_sha256") == verifier_sha
        and isinstance(journal, dict)
        and journal.get("sha256") == journal_sha
        and journal.get("event_count") == 50
        and journal.get("decode_count") == 48
    )


def _induced_neural_procedure_certificate_holds() -> bool:
    bundle = _canary_artifact_bundle(
        "artifacts/closeout/latent_cortex/induced_neural_procedure_canary_20260831",
        "tools/verify_induced_neural_procedure_canary.py",
    )
    if bundle is None:
        return False
    result, verification, current_hashes, verifier_sha, artifact_root = bundle
    source_hashes = result["source_sha256s"]

    expected_counts = {
        "treatment_exact": 96,
        "coefficient_lesion_disrupted": 96,
        "wrong_input_disrupted": 96,
        "no_procedure_exact": 1,
    }
    expected_boundary = (
        "a family-blind procedure induced from support examples transfers on fresh "
        "inputs through learned arithmetic tissue under composition, coefficient, "
        "wrong-input, and no-procedure controls; not natural-language compilation, "
        "open-domain reasoning, resident decode, unrestricted serving, or frontier performance"
    )
    program = result.get("program")
    return bool(
        result.get("schema") == "aura.rlc.induced_neural_procedure_canary.v1"
        and result.get("admitted") is True
        and result.get("verdict") == "SUPPORTED_INDUCED_NEURAL_PROCEDURE"
        and result.get("support_count") == 16
        and result.get("task_count") == 96
        and result.get("null_runs") == 15
        and result.get("null_found") == 0
        and result.get("single_primitive_shortcut") is False
        and isinstance(program, dict)
        and program.get("expression") == "idiv(add(in0, in1), in2)"
        and program.get("depth") == 2
        and result.get("counts") == expected_counts
        and result.get("family_label_available_to_inducer") is False
        and result.get("family_solver_available_to_inducer") is False
        and result.get("support_outputs_available_to_inducer") is True
        and result.get("evaluation_outputs_available_to_treatment") is False
        and result.get("claim_boundary") == expected_boundary
        and current_hashes == source_hashes
        and verification.get("schema")
        == "aura.rlc.induced_neural_procedure_verification.v1"
        and verification.get("verified") is True
        and verification.get("program_sha") == program.get("sha")
        and verification.get("task_count") == 96
        and verification.get("counts") == expected_counts
        and verification.get("null_found") == 0
        and verification.get("task_set_sha256") == result.get("task_set_sha256")
        and verification.get("producer_source_sha256s") == source_hashes
        and verification.get("verifier_source_sha256") == verifier_sha
        and verification.get("claim_boundary") == expected_boundary
        and verification.get("input_receipt_sha256") == result.get("receipt_sha256")
    )


def _induced_neural_procedure_decode_certificate_holds() -> bool:
    import hashlib
    import json

    bundle = _canary_artifact_bundle(
        "artifacts/closeout/latent_cortex/induced_neural_procedure_decode_canary_20260831",
        "tools/verify_induced_neural_procedure_decode_canary.py",
    )
    if bundle is None:
        return False
    result, verification, current_hashes, verifier_sha, artifact_root = bundle
    source_hashes = result["source_sha256s"]
    try:
        manifest_path = pathlib.Path(result["resident_manifest_identity"]["path"])
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Induced procedure certificate unreadable, claim unverified: %s", exc)
        return False

    expected_exact = {
        "ordinary_base": 1,
        "matched_wire_base": 1,
        "treatment": 8,
        "coefficient_lesion": 1,
        "matched_wrong_input": 0,
        "matched_wrong_state": 0,
    }
    expected_boundary = (
        "resident decoded-answer transfer of a family-blind induced procedure through "
        "authenticated learned neural tissue under matched causal controls; not "
        "natural-language compilation, open-domain reasoning, unrestricted serving, "
        "static fusion, or frontier performance"
    )
    program = result.get("program")
    arms = result.get("arms")
    observed_exact = (
        {arm: arms.get(arm, {}).get("exact") for arm in expected_exact}
        if isinstance(arms, dict)
        else {}
    )
    journal = verification.get("journal_identity")
    return bool(
        result.get("schema")
        == "aura.rlc.induced_neural_procedure_decode_canary.v1"
        and result.get("admitted") is True
        and result.get("task_count") == 8
        and result.get("decode_calls_per_arm_per_task") == 1
        and result.get("arm_order") == "task_hash_rotated"
        and isinstance(program, dict)
        and program.get("expression") == "idiv(add(in0, in1), in2)"
        and program.get("depth") == 2
        and observed_exact == expected_exact
        and result.get("gain_count") == 7
        and result.get("regression_count") == 0
        and result.get("family_label_available_to_generation") is False
        and result.get("expected_answer_available_to_generation") is False
        and result.get("verifier_trace_available_to_generation") is False
        and result.get("claim_boundary") == expected_boundary
        and current_hashes == source_hashes
        and manifest_sha == result["resident_manifest_identity"]["sha256"]
        and verification.get("schema")
        == "aura.rlc.induced_neural_procedure_decode_verification.v1"
        and verification.get("verified") is True
        and verification.get("source_commit") == result.get("source_commit")
        and verification.get("source_sha256s") == source_hashes
        and verification.get("program_sha") == program.get("sha")
        and verification.get("independent_exact_by_arm") == expected_exact
        and verification.get("gain_count") == 7
        and verification.get("regression_count") == 0
        and verification.get("paired_one_sided_exact_p") == 0.0078125
        and verification.get("claim_boundary") == expected_boundary
        and verification.get("input_receipt_sha256") == result.get("receipt_sha256")
        and verification.get("verifier_source_sha256") == verifier_sha
        and isinstance(journal, dict)
        and journal.get("decode_count") == 48
        and journal.get("event_count") == 50
    )


def _semantic_program_27b_certificate_holds() -> bool:
    loaded = _sealed_certificate(
        "docs/evidence/semantic_program_27b_reverification_2026-09-01.json"
    )
    if loaded is None:
        return False
    certificate, expected_verification_sha256, certificate_path, root = loaded
    try:
        source_sha256s = certificate["source_sha256s"]
    except (KeyError, TypeError) as exc:
        logger.debug("27B semantic program certificate unreadable, claim unverified: %s", exc)
        return False

    expected_boundary = (
        "bounded resident-27B semantic program acquisition and exact answer "
        "execution on construction-held-out synthetic arithmetic language; "
        "not broad-domain or frontier reasoning evidence"
    )
    answer_controls = certificate.get("paired_answer_controls")
    expected_answer_pairs = {
        "coefficient_lesion:test": (48, 0),
        "coefficient_lesion:validation": (86, 0),
        "hidden_token_shuffle:test": (40, 1),
        "hidden_token_shuffle:validation": (81, 0),
        "label_permutation:test": (47, 0),
        "label_permutation:validation": (83, 0),
    }
    observed_answer_pairs = (
        {
            key: (value.get("treatment_only"), value.get("control_only"))
            for key, value in answer_controls.items()
        }
        if isinstance(answer_controls, dict)
        else {}
    )
    return bool(
        certificate.get("schema")
        == "aura.semantic_program_campaign_verification.v1"
        and certificate.get("verified") is True
        and certificate.get("deterministic_refit_exact") is True
        and certificate.get("campaign_replay_exact") is True
        and certificate.get("raw_feature_records_reloaded") == 576
        and certificate.get("task_rows_independently_recounted") == 1344
        and certificate.get("held_out_total") == 256
        and certificate.get("held_out_treatment_program_exact") == 133
        and certificate.get("held_out_treatment_answer_exact") == 134
        and certificate.get("expected_answers_available_to_training") is False
        and certificate.get("serving_authority") is False
        and certificate.get("claim_boundary") == expected_boundary
        and observed_answer_pairs == expected_answer_pairs
        and _historical_semantic_sources_hold(
            root,
            certificate_path,
            source_sha256s,
        )
        and certificate.get("verification_sha256") == expected_verification_sha256
    )


def _semantic_program_27b_replication_certificate_holds() -> bool:
    loaded = _sealed_certificate(
        "docs/evidence/semantic_program_27b_frozen_replication_2026-09-01.json"
    )
    if loaded is None:
        return False
    certificate, expected_verification_sha256, certificate_path, root = loaded
    try:
        source_sha256s = certificate["source_sha256s"]
    except (KeyError, TypeError) as exc:
        logger.debug("27B replication certificate unreadable, claim unverified: %s", exc)
        return False

    compatibility = certificate.get("representation_compatibility")
    expected_boundary = (
        "bounded fresh synthetic semantic-program cohort on a function-identical "
        "frozen transducer across worker sessions; no broad-domain claim"
    )
    return bool(
        certificate.get("schema")
        == "aura.semantic_program_fresh_cohort_verification.v1"
        and certificate.get("verified") is True
        and certificate.get("frozen_replay_exact") is True
        and certificate.get("raw_training_records_reloaded") == 576
        and certificate.get("raw_replication_records_reloaded") == 576
        and certificate.get("task_rows_independently_recounted") == 1728
        and certificate.get("paired_tests_independently_recounted") == 16
        and certificate.get("held_out_total") == 256
        and certificate.get("held_out_treatment_answer_exact") == 114
        and certificate.get("held_out_hidden_shuffle_answer_exact") == 10
        and certificate.get("held_out_coefficient_lesion_answer_exact") == 0
        and certificate.get("expected_answers_available_to_training") is False
        and certificate.get("serving_authority") is False
        and certificate.get("claim_boundary") == expected_boundary
        and isinstance(compatibility, dict)
        and compatibility.get("coefficients_changed") is False
        and compatibility.get("hidden_states_changed") is False
        and compatibility.get("serving_authority") is False
        and _historical_semantic_sources_hold(
            root,
            certificate_path,
            source_sha256s,
        )
        and certificate.get("verification_sha256") == expected_verification_sha256
    )


def _semantic_program_27b_shared_variable_geometry_certificate_holds() -> bool:
    loaded = _sealed_certificate(
        "docs/evidence/"
        "semantic_program_27b_shared_variable_geometry_2026-09-01.json"
    )
    if loaded is None:
        return False
    certificate, expected_verification_sha256, certificate_path, root = loaded
    try:
        source_sha256s = certificate["source_sha256s"]
    except (KeyError, TypeError) as exc:
        logger.debug("27B shared-variable geometry certificate unreadable, claim unverified: %s", exc)
        return False

    expected_boundary = (
        "bounded typed semantic-program families with learned variable geometry, "
        "operation semantics, and definition-reference relations; one transducer, "
        "no family router, and no broad natural-language domain claim"
    )
    expected_program_controls = {
        "coefficient_lesion": 0,
        "hidden_token_shuffle": 0,
    }
    expected_family_programs = {
        "arithmetic": 123,
        "fork_join": 107,
        "sequence": 28,
    }
    expected_family_answers = {
        "arithmetic": 123,
        "fork_join": 140,
        "sequence": 29,
    }
    paired = certificate.get("paired_test_program_controls")
    expected_pairs = {
        "coefficient_lesion": (258, 0),
        "hidden_token_shuffle": (258, 0),
    }
    observed_pairs = (
        {
            key: (value.get("treatment_only"), value.get("control_only"))
            for key, value in paired.items()
        }
        if isinstance(paired, dict)
        else {}
    )
    return bool(
        certificate.get("schema")
        == "aura.semantic_program_shared_verification.v1"
        and certificate.get("verified") is True
        and certificate.get("test_total") == 368
        and certificate.get("test_program_exact") == 258
        and certificate.get("test_answer_exact") == 292
        and certificate.get("test_program_controls") == expected_program_controls
        and certificate.get("family_test_program_exact") == expected_family_programs
        and certificate.get("family_test_answer_exact") == expected_family_answers
        and observed_pairs == expected_pairs
        and all(
            value.get("one_sided_exact_p") == 2.1590421387736112e-78
            for value in paired.values()
        )
        and certificate.get("serving_authority") is False
        and certificate.get("claim_boundary") == expected_boundary
        and _historical_semantic_sources_hold_at_binding(
            root,
            certificate_path,
            source_sha256s,
            "docs/evidence/"
            "semantic_program_27b_shared_source_binding_2026-09-01.json",
        )
        and certificate.get("verification_sha256") == expected_verification_sha256
    )


def _semantic_program_27b_floor_certificate_holds() -> bool:
    loaded = _sealed_certificate(
        "docs/evidence/semantic_program_27b_floor_equivalence_2026-09-01.json"
    )
    if loaded is None:
        return False
    certificate, expected_verification_sha256, certificate_path, root = loaded
    try:
        source_sha256s = certificate["source_sha256s"]
    except (KeyError, TypeError) as exc:
        logger.debug("27B floor certificate unreadable, claim unverified: %s", exc)
        return False
    expected_boundary = (
        "the frozen shared semantic transducer's accepted test programs have "
        "identical outcomes under the closed exact executor and universal metered "
        "floor; no unseen-schema, serving, or broad reasoning claim"
    )
    expected_by_family = {
        "arithmetic": {
            "accepted": 128,
            "agreements": 128,
            "refusal_agreements": 0,
            "value_agreements": 128,
        },
        "fork_join": {
            "accepted": 192,
            "agreements": 192,
            "refusal_agreements": 1,
            "value_agreements": 191,
        },
        "sequence": {
            "accepted": 48,
            "agreements": 48,
            "refusal_agreements": 1,
            "value_agreements": 47,
        },
    }
    coverage = certificate.get("primitive_coverage")
    return bool(
        certificate.get("schema")
        == "aura.semantic_program_floor_equivalence_verification.v1"
        and certificate.get("verified") is True
        and certificate.get("test_total") == 368
        and certificate.get("accepted") == 368
        and certificate.get("agreements") == 368
        and certificate.get("value_agreements") == 366
        and certificate.get("refusal_agreements") == 2
        and certificate.get("by_family") == expected_by_family
        and certificate.get("fit_or_refit_calls") == 0
        and certificate.get("expected_answers_available") is False
        and certificate.get("family_router_present") is False
        and certificate.get("serving_authority") is False
        and isinstance(coverage, dict)
        and coverage.get("complete") is True
        and len(coverage.get("declared", ())) == 20
        and all(
            coverage.get(field) == []
            for field in (
                "missing_semantics",
                "missing_types",
                "extra_semantics",
                "extra_types",
            )
        )
        and certificate.get("claim_boundary") == expected_boundary
        and _historical_semantic_sources_hold_at_binding(
            root,
            certificate_path,
            source_sha256s,
            "docs/evidence/"
            "semantic_program_27b_floor_source_binding_2026-09-01.json",
        )
        and certificate.get("verification_sha256") == expected_verification_sha256
    )


def _canary_artifact_bundle(
    relative_artifact_root: str, verifier_relative: str
) -> tuple[dict, dict, dict, str, pathlib.Path] | None:
    """Load one closeout canary's result, verification, and identity hashes.

    Every canary predicate opened the same five things in the same order and
    caught the same five exceptions. Written once, a sixth canary is its own
    assertions instead of a copy of the preamble that came before it.

    Returns ``(result, verification, current_source_hashes, verifier_sha,
    artifact_root)``, or None when anything is missing or malformed.
    """
    import hashlib
    import json

    root = pathlib.Path(__file__).resolve().parents[2]
    artifact_root = root / relative_artifact_root
    try:
        result = json.loads((artifact_root / "result.json").read_text(encoding="utf-8"))
        verification = json.loads(
            (artifact_root / "verification.json").read_text(encoding="utf-8")
        )
        source_hashes = dict(result["source_sha256s"])
        current_hashes = {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in source_hashes
        }
        verifier_sha = hashlib.sha256(
            (root / verifier_relative).read_bytes()
        ).hexdigest()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Canary artifact bundle unreadable: %s", exc)
        return None

    # A certificate pins the source it was measured over. When that source
    # changes, the certificate stops covering the code that runs — which is
    # not the same fact as the capability having regressed, and reporting it
    # as False said the second when only the first was known. The same
    # confusion cost a day on CP546. Name the files and report NOT_MEASURED,
    # so the remedy reads "re-run the canary" rather than "the model got worse".
    drifted = sorted(
        relative
        for relative, sealed in source_hashes.items()
        if current_hashes.get(relative) != sealed
    )
    if drifted:
        raise NothingMeasured(
            f"{relative_artifact_root.rsplit('/', 1)[-1]} was sealed over source "
            f"that has since changed, so it measures code that no longer runs: "
            + ", ".join(drifted)
            + f" — re-run {verifier_relative.replace('/verify_', '/run_')} and its verifier"
        )
    return result, verification, current_hashes, verifier_sha, artifact_root


def _sealed_certificate(relative_path: str) -> tuple[dict, str, pathlib.Path, pathlib.Path] | None:
    """Load a docs/evidence certificate and the digest its body must match.

    Returns ``(certificate, expected_verification_sha256, certificate_path,
    root)``, or None when the file is missing or will not canonicalise. The
    caller compares the digest, because a helper that both computes and accepts
    it is not a check.
    """
    import hashlib
    import json

    root = pathlib.Path(__file__).resolve().parents[2]
    certificate_path = root / relative_path
    try:
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in certificate.items()
            if key != "verification_sha256"
        }
        canonical = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Sealed certificate unreadable: %s", exc)
        return None
    return certificate, hashlib.sha256(canonical).hexdigest(), certificate_path, root


def _historical_semantic_sources_hold(
    root: pathlib.Path,
    certificate_path: pathlib.Path,
    source_sha256s: dict[str, str],
) -> bool:
    """Verify measured source against committed blobs, never the evolving tree."""

    return _historical_semantic_sources_hold_at_binding(
        root,
        certificate_path,
        source_sha256s,
        "docs/evidence/semantic_program_27b_source_binding_2026-09-01.json",
    )


def _historical_semantic_sources_hold_at_binding(
    root: pathlib.Path,
    certificate_path: pathlib.Path,
    source_sha256s: dict[str, str],
    binding_relative_path: str,
) -> bool:
    """Verify one measured source set against its immutable commit binding."""

    import hashlib
    import json
    import subprocess

    from core.runtime.subprocess_gateway import get_subprocess_gateway

    binding_path = root / binding_relative_path
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        commit = binding["source_commit"]
        certificate_hash = binding["certificates"][certificate_path.name]
        if (
            binding.get("schema") != "aura.historical_semantic_source_binding.v1"
            or binding.get("serving_authority") is not False
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or hashlib.sha256(certificate_path.read_bytes()).hexdigest()
            != certificate_hash
        ):
            return False
        for relative, expected in source_sha256s.items():
            source_path = pathlib.PurePosixPath(relative)
            if source_path.is_absolute() or ".." in source_path.parts:
                return False
            # Through the gateway, and byte-exact: the sha256 below attests to
            # these bytes, so a decode between git and the digest would launder
            # what the certificate claims.
            payload = get_subprocess_gateway().run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=False,
                read_only=True,
                accelerator_capability="none",
                source="historical_semantic_sources",
            ).stdout
            if hashlib.sha256(payload).hexdigest() != expected:
                return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        logger.debug("Historical semantic sources unreadable at binding: %s", exc)
        return False
    return True
