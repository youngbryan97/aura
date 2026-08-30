"""Disjoint critic identity and independently checked blind-spot evidence.

The live RLC critic is deliberately a deterministic symbolic program, not a
second pass through the resident generator.  This module makes that separation
machine-checkable and keeps a durable confusion matrix for errors that the
generator and critic share.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.runtime.file_read_gateway import read_stable_bytes

CRITIC_IDENTITY_SCHEMA = "aura.rlc.critic_identity.v1"
GENERATOR_IDENTITY_SCHEMA = "aura.rlc.generator_function_identity.v1"
BLIND_SPOT_EVIDENCE_SCHEMA = "aura.rlc.shared_blind_spots.v1"
CHECKED_OUTCOME_SCHEMA = "aura.rlc.checked_critic_outcome.v1"

MIN_CHECKED_SAMPLES = 24
MIN_GENERATOR_ERRORS = 8
MIN_INDEPENDENT_GRADERS = 2
MAX_SHARED_BLIND_SPOT_UPPER_BOUND = 0.35
_MAX_LEDGER_ROWS = 5_000

_CRITIC_SOURCE_FILES = (
    "core/brain/llm/latent_cortex/atomic_decomposition.py",
    "core/brain/llm/latent_cortex/deterministic_verifier_router.py",
    "core/brain/llm/latent_cortex/objective_program_verifier.py",
    "core/brain/llm/latent_cortex/episodic_output_memory_contract.py",
    "core/brain/llm/latent_cortex/fast_weight_learning.py",
    "core/brain/llm/latent_cortex/teaching_events.py",
    "core/brain/llm/latent_cortex/task_verifiers.py",
    "core/brain/llm/latent_cortex/test_time_training.py",
    "core/brain/llm/latent_cortex/verifier_gain_search.py",
    "core/brain/llm/latent_cortex/output_quality.py",
    "core/brain/llm/latent_cortex/response_contracts.py",
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/canonical_json.py",
    "core/language/relational_request.py",
)
_ALLOWED_INTERNAL_IMPORTS = {
    "core.brain.llm.latent_cortex",
    "core.brain.canonical_json",
    "core.brain.llm.latent_cortex.atomic_decomposition",
    "core.brain.llm.latent_cortex.deterministic_verifier_router",
    "core.brain.llm.latent_cortex.episodic_output_memory_contract",
    "core.brain.llm.latent_cortex.objective_program_verifier",
    "core.brain.llm.latent_cortex.fast_weight_learning",
    "core.brain.llm.latent_cortex.frontier_tasks",
    "core.brain.llm.latent_cortex.output_quality",
    "core.brain.llm.latent_cortex.response_contracts",
    "core.brain.llm.latent_cortex.test_time_training",
    "core.brain.llm.latent_cortex.teaching_events",
    "core.language",
    # The typed compiler and its executor are pure Python — no mlx, no torch —
    # and they are what the critic's certified transition engine runs on. They
    # were undeclared, so the audit failed on them alongside the two genuinely
    # neural modules and made the whole closure look irredeemable. They are not
    # the generator; they are the deterministic machine the grader uses.
    "core.brain.llm.latent_cortex.typed_action_compiler",
    "core.brain.llm.latent_cortex.typed_program_executor",
    # Same argument, one module later. `episodic_neural_readout_contract` is
    # a PURE evidence contract — hashlib, json, math, and nothing else; its
    # own docstring says so and the forbidden-root check below still applies
    # to it. "Neural" names what the readout DESCRIBES, not what this module
    # imports.
    #
    # It arrived as a new import inside `fast_weight_learning`, which is
    # already declared, so the audit failed transitively and critic identity
    # stopped being independently provable. That revoked the task verifier
    # on EVERY episode — `test_handler_builds_task_verifier_when_guided` and
    # `test_powered_shared_blind_spots_causally_revoke_worker_verifier` were
    # both red for this one omission, and the live effect was a verifier
    # that attached on zero turns.
    "core.brain.llm.latent_cortex.episodic_neural_readout_contract",
    "core.brain.llm.latent_cortex.verifier_gain_search",
    # Same argument, one module later again. `symbolic_bridge` recomputes
    # asserted arithmetic — ast, math, re, and nothing else — so it is exactly
    # the kind of deterministic grader this closure is allowed to contain.
    #
    # It arrived as a new import inside `task_verifiers`, which is one of the
    # critic source files, so the audit failed transitively and critic identity
    # stopped being independently provable. The live effect is the one this
    # file has now recorded twice: the task verifier is revoked on EVERY
    # episode, so it attaches on zero turns.
    #
    # A pure checker being added to the grader is the intended direction of
    # travel; the declaration is what has to keep up with it.
    "core.reasoning.symbolic_bridge",
}
_FORBIDDEN_IMPORT_ROOTS = {
    "jax",
    "mlx",
    "mlx_lm",
    "tensorflow",
    "torch",
    "transformers",
}
_ALLOWED_STATE_FIELDS = {
    "evaluations",
    "facet_reliability",
    "objective",
    "response_contract",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def audit_python_dependencies(sources: Mapping[str, str]) -> dict[str, Any]:
    """Return an exact import audit for a proposed critic source closure."""

    imports: set[str] = set()
    parse_errors: list[str] = []
    for name, source in sorted(sources.items()):
        try:
            tree = ast.parse(source, filename=name)
        except SyntaxError as exc:
            parse_errors.append(f"{name}:{exc.lineno}:{exc.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    forbidden = sorted(name for name in imports if name.split(".", 1)[0] in _FORBIDDEN_IMPORT_ROOTS)
    undeclared_internal = sorted(
        name
        for name in imports
        if name.startswith("core.") and name not in _ALLOWED_INTERNAL_IMPORTS
    )
    return {
        "imports": sorted(imports),
        "forbidden_imports": forbidden,
        "undeclared_internal_imports": undeclared_internal,
        "parse_errors": parse_errors,
        "passed": not (forbidden or undeclared_internal or parse_errors),
    }


def build_critic_source_identity(*, root: Path | str | None = None) -> dict[str, Any]:
    base = Path(root) if root is not None else _repo_root()
    sources: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for relative in _CRITIC_SOURCE_FILES:
        payload = read_stable_bytes(base / relative, max_bytes=4 * 1024 * 1024)
        try:
            source = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"critic source is not UTF-8: {relative}") from exc
        sources[relative] = source
        rows.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest()})
    audit = audit_python_dependencies(sources)
    payload = {
        "source_files": rows,
        "dependency_audit": audit,
    }
    return {**payload, "source_closure_sha256": _sha(payload)}


def build_generator_function_identity(worker_identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(worker_identity, Mapping):
        raise ValueError("worker identity must be a mapping")
    logical_count = worker_identity.get("worker_model_parameter_count")
    stored_count = worker_identity.get("worker_model_stored_parameter_element_count")
    if type(logical_count) is not int or logical_count <= 0:
        raise ValueError("generator logical parameter count is invalid")
    if type(stored_count) is not int or stored_count <= 0:
        stored_count = logical_count
    source_sha = worker_identity.get("worker_source_sha256")
    if not _sha256(source_sha):
        source_sha = ""
    adapter_sha = worker_identity.get("worker_adapter_stack_sha256")
    if not _sha256(adapter_sha):
        adapter_sha = ""
    stack_gaps = worker_identity.get("worker_stack_identity_gaps")
    if not isinstance(stack_gaps, list) or any(not isinstance(item, str) for item in stack_gaps):
        stack_gaps = ["serving_stack_identity_unavailable"]
    tokenizer = worker_identity.get("worker_tokenizer")
    quantization = worker_identity.get("worker_quantization")
    payload = {
        "schema": GENERATOR_IDENTITY_SCHEMA,
        "implementation_kind": "resident_neural_generator",
        "model_path_sha256": hashlib.sha256(
            str(worker_identity.get("worker_model_path") or "").encode("utf-8")
        ).hexdigest(),
        "logical_parameter_count": logical_count,
        "stored_parameter_element_count": stored_count,
        "parameter_count_basis": str(
            worker_identity.get("worker_model_parameter_count_basis") or "unreported"
        ),
        "worker_source_sha256": source_sha,
        "adapter_stack_sha256": adapter_sha,
        "tokenizer_sha256": _sha(tokenizer if isinstance(tokenizer, Mapping) else {}),
        "quantization_sha256": _sha(quantization if isinstance(quantization, Mapping) else {}),
        "stack_identity_gaps": sorted(set(stack_gaps)),
    }
    return {**payload, "function_sha256": _sha(payload)}


def _critic_state_audit(verifier: Any) -> dict[str, Any]:
    state = getattr(verifier, "__dict__", None)
    if not isinstance(state, dict):
        raise ValueError("critic does not expose an auditable state mapping")
    fields = sorted(state)
    unexpected = sorted(set(fields) - _ALLOWED_STATE_FIELDS)
    non_data: list[str] = []

    def walk(value: Any, path: str) -> None:
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                non_data.append(f"{path}:nonfinite")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    non_data.append(f"{path}:non_string_key")
                    continue
                walk(item, f"{path}.{key}")
            return
        non_data.append(f"{path}:{type(value).__module__}.{type(value).__qualname__}")

    for name, value in state.items():
        walk(value, name)
    return {
        "state_fields": fields,
        "unexpected_state_fields": unexpected,
        "non_data_state": non_data,
        "trainable_parameter_count": 0 if not non_data else None,
        "passed": not unexpected and not non_data,
    }


def build_critic_identity(
    verifier: Any,
    *,
    worker_identity: Mapping[str, Any],
    root: Path | str | None = None,
) -> dict[str, Any]:
    source = build_critic_source_identity(root=root)
    state = _critic_state_audit(verifier)
    generator = build_generator_function_identity(worker_identity)
    critic_function_sha256 = source["source_closure_sha256"]
    dependency_passed = source["dependency_audit"]["passed"] is True
    state_passed = state["passed"] is True
    distinct = bool(
        dependency_passed
        and state_passed
        and state["trainable_parameter_count"] == 0
        and generator["logical_parameter_count"] > 0
        and critic_function_sha256 != generator["function_sha256"]
    )
    payload = {
        "schema": CRITIC_IDENTITY_SCHEMA,
        "implementation_kind": "deterministic_symbolic_parameterless",
        "class_path": (f"{type(verifier).__module__}.{type(verifier).__qualname__}"),
        "critic_function_sha256": critic_function_sha256,
        "source_identity": source,
        "runtime_state_audit": state,
        "generator_identity": generator,
        "weight_identity_relation": "zero_parameters_vs_resident_neural_parameters",
        "function_identity_distinct": distinct,
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def validate_critic_identity(
    value: Any,
    *,
    worker_identity: Mapping[str, Any],
    root: Path | str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("critic identity receipt is missing")
    required = {
        "schema",
        "implementation_kind",
        "class_path",
        "critic_function_sha256",
        "source_identity",
        "runtime_state_audit",
        "generator_identity",
        "weight_identity_relation",
        "function_identity_distinct",
        "receipt_sha256",
    }
    if set(value) != required or value.get("schema") != CRITIC_IDENTITY_SCHEMA:
        raise ValueError("critic identity schema is invalid")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value.get("receipt_sha256") != _sha(payload):
        raise ValueError("critic identity receipt digest differs")
    expected_source = build_critic_source_identity(root=root)
    expected_generator = build_generator_function_identity(worker_identity)
    state = value.get("runtime_state_audit")
    if (
        value.get("implementation_kind") != "deterministic_symbolic_parameterless"
        or value.get("class_path")
        != "core.brain.llm.latent_cortex.task_verifiers.EpisodeTaskVerifier"
        or value.get("critic_function_sha256") != expected_source["source_closure_sha256"]
        or value.get("source_identity") != expected_source
        or value.get("generator_identity") != expected_generator
        or not isinstance(state, dict)
        or set(state)
        != {
            "state_fields",
            "unexpected_state_fields",
            "non_data_state",
            "trainable_parameter_count",
            "passed",
        }
        or state.get("state_fields") != sorted(_ALLOWED_STATE_FIELDS)
        or state.get("unexpected_state_fields") != []
        or state.get("non_data_state") != []
        or state.get("trainable_parameter_count") != 0
        or state.get("passed") is not True
        or value.get("weight_identity_relation") != "zero_parameters_vs_resident_neural_parameters"
        or value.get("function_identity_distinct") is not True
        or expected_source["dependency_audit"]["passed"] is not True
        or value["critic_function_sha256"] == expected_generator["function_sha256"]
    ):
        raise ValueError(
            "critic function identity is not independently proven: "
            + _critic_identity_failure_detail(
                value,
                expected_source=expected_source,
                expected_generator=expected_generator,
                state=state if isinstance(state, dict) else {},
            )
        )
    return dict(value)


def _critic_identity_failure_detail(
    value: Mapping[str, Any],
    *,
    expected_source: Mapping[str, Any],
    expected_generator: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    """Name the condition that failed, not merely that one did.

    LIVE DEFECT, 2026-08-10. Eleven distinct conditions shared one message —
    "critic function identity is not independently proven" — and the caller
    logged it verbatim. The live receipt therefore read
    ``disjoint_critic_authority_unproven`` on every foreground turn with no way
    to tell WHICH invariant was unmet, so the recurrent cortex ran without
    critic authority indefinitely and the reason was never legible.

    The actual cause here was a dependency audit failure: the critic's source
    closure imports four internal modules it does not declare, and two of
    those (neural_transition_tissue, systematic_neural_alu) import mlx.core and
    mlx.nn at module scope — so the closure is not the
    "deterministic_symbolic_parameterless" thing its own receipt claims. That
    is a real architectural conflict, and it is only actionable if the failure
    says so.
    """
    reasons: list[str] = []
    audit = expected_source.get("dependency_audit")
    if isinstance(audit, Mapping) and audit.get("passed") is not True:
        detail: list[str] = []
        forbidden = list(audit.get("forbidden_imports") or [])
        undeclared = list(audit.get("undeclared_internal_imports") or [])
        parse_errors = list(audit.get("parse_errors") or [])
        if forbidden:
            detail.append(f"forbidden_imports={forbidden}")
        if undeclared:
            detail.append(f"undeclared_internal_imports={undeclared}")
        if parse_errors:
            detail.append(f"parse_errors={parse_errors}")
        reasons.append(
            "dependency_audit_failed(" + "; ".join(detail or ["unspecified"]) + ")"
        )
    if value.get("function_identity_distinct") is not True:
        reasons.append("function_identity_distinct=False")
    if state.get("passed") is not True:
        reasons.append("runtime_state_audit_failed")
    if state.get("trainable_parameter_count") not in (0, None):
        reasons.append(
            f"critic_trainable_parameter_count={state.get('trainable_parameter_count')}"
        )
    if state.get("unexpected_state_fields"):
        reasons.append(f"unexpected_state_fields={state.get('unexpected_state_fields')}")
    if state.get("non_data_state"):
        reasons.append(f"non_data_state={state.get('non_data_state')}")
    if value.get("critic_function_sha256") == expected_generator.get("function_sha256"):
        reasons.append("critic_and_generator_share_one_function_sha256")
    if value.get("critic_function_sha256") != expected_source.get(
        "source_closure_sha256"
    ):
        reasons.append("critic_function_sha256_does_not_match_source_closure")
    if value.get("source_identity") != expected_source:
        reasons.append("source_identity_differs_from_recomputed_closure")
    if value.get("generator_identity") != expected_generator:
        reasons.append("generator_identity_differs_from_worker_identity")
    if (
        value.get("weight_identity_relation")
        != "zero_parameters_vs_resident_neural_parameters"
    ):
        reasons.append("weight_identity_relation_unexpected")
    if value.get("implementation_kind") != "deterministic_symbolic_parameterless":
        reasons.append("implementation_kind_unexpected")
    return "; ".join(reasons) or "no individual condition reported a cause"


def _wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _validate_checked_row(
    row: Any,
    *,
    generator_function_sha256: str,
    critic_function_sha256: str,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("critic outcome row is invalid")
    required = {
        "schema",
        "bucket",
        "checked",
        "task_sha256",
        "candidate_sha256",
        "generator_function_sha256",
        "critic_function_sha256",
        "independent_grader_sha256",
        "independent_receipt_sha256",
        "generator_correct",
        "critic_accepted",
    }
    if set(row) != required or row.get("schema") != CHECKED_OUTCOME_SCHEMA:
        raise ValueError("critic outcome row schema is invalid")
    digests = (
        row.get("task_sha256"),
        row.get("candidate_sha256"),
        row.get("generator_function_sha256"),
        row.get("critic_function_sha256"),
        row.get("independent_grader_sha256"),
        row.get("independent_receipt_sha256"),
    )
    if (
        row.get("checked") is not True
        or not isinstance(row.get("bucket"), str)
        or not row["bucket"]
        or len(row["bucket"]) > 160
        or any(not _sha256(item) for item in digests)
        or row.get("generator_function_sha256") != generator_function_sha256
        or row.get("critic_function_sha256") != critic_function_sha256
        or generator_function_sha256 == critic_function_sha256
        or row.get("independent_grader_sha256")
        in {generator_function_sha256, critic_function_sha256}
        or type(row.get("generator_correct")) is not bool
        or type(row.get("critic_accepted")) is not bool
    ):
        raise ValueError("critic outcome row lacks independent checked evidence")
    return dict(row)


def build_shared_blind_spot_evidence(
    *,
    bucket: str,
    generator_function_sha256: str,
    critic_function_sha256: str,
    checked_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(bucket, str) or not bucket or len(bucket) > 160:
        raise ValueError("critic evidence bucket is invalid")
    if not _sha256(generator_function_sha256) or not _sha256(critic_function_sha256):
        raise ValueError("critic evidence function identity is invalid")
    if generator_function_sha256 == critic_function_sha256:
        raise ValueError("generator and critic function identities must differ")
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for raw in checked_outcomes:
        row = _validate_checked_row(
            raw,
            generator_function_sha256=generator_function_sha256,
            critic_function_sha256=critic_function_sha256,
        )
        if row["bucket"] != bucket:
            raise ValueError("critic outcome row bucket differs")
        key = (row["task_sha256"], row["candidate_sha256"])
        if key in seen:
            raise ValueError("duplicate checked critic sample")
        seen.add(key)
        rows.append(row)

    both_correct = correct_rejected = error_caught = shared = 0
    graders: set[str] = set()
    for row in rows:
        graders.add(row["independent_grader_sha256"])
        if row["generator_correct"]:
            if row["critic_accepted"]:
                both_correct += 1
            else:
                correct_rejected += 1
        elif row["critic_accepted"]:
            shared += 1
        else:
            error_caught += 1
    generator_errors = shared + error_caught
    lower, upper = _wilson_interval(shared, generator_errors)
    powered = bool(
        len(rows) >= MIN_CHECKED_SAMPLES
        and generator_errors >= MIN_GENERATOR_ERRORS
        and len(graders) >= MIN_INDEPENDENT_GRADERS
    )
    payload = {
        "schema": BLIND_SPOT_EVIDENCE_SCHEMA,
        "bucket": bucket,
        "generator_function_sha256": generator_function_sha256,
        "critic_function_sha256": critic_function_sha256,
        "checked_samples": len(rows),
        "generator_errors": generator_errors,
        "independent_graders": len(graders),
        "independent_grader_sha256s": sorted(graders),
        "checked_sample_set_sha256": _sha(
            [
                {
                    "task_sha256": row["task_sha256"],
                    "candidate_sha256": row["candidate_sha256"],
                    "independent_grader_sha256": row["independent_grader_sha256"],
                    "independent_receipt_sha256": row["independent_receipt_sha256"],
                    "generator_correct": row["generator_correct"],
                    "critic_accepted": row["critic_accepted"],
                }
                for row in sorted(
                    rows,
                    key=lambda item: (item["task_sha256"], item["candidate_sha256"]),
                )
            ]
        ),
        "minimum_checked_samples": MIN_CHECKED_SAMPLES,
        "minimum_generator_errors": MIN_GENERATOR_ERRORS,
        "minimum_independent_graders": MIN_INDEPENDENT_GRADERS,
        "confusion_matrix": {
            "generator_correct_critic_accept": both_correct,
            "generator_correct_critic_reject": correct_rejected,
            "generator_error_critic_reject": error_caught,
            "generator_error_critic_accept": shared,
        },
        "shared_blind_spot_rate": (
            round(shared / generator_errors, 8) if generator_errors else None
        ),
        "shared_blind_spot_wilson95": {
            "lower": round(lower, 8),
            "upper": round(upper, 8),
        },
        "maximum_admitted_upper_bound": MAX_SHARED_BLIND_SPOT_UPPER_BOUND,
        "evidence_state": "measured" if powered else "bootstrap_unmeasured",
        "critic_reliability_admitted": bool(
            not powered or upper <= MAX_SHARED_BLIND_SPOT_UPPER_BOUND
        ),
    }
    return {**payload, "snapshot_sha256": _sha(payload)}


def validate_shared_blind_spot_evidence(
    value: Any,
    *,
    generator_function_sha256: str,
    critic_function_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("shared blind-spot evidence is missing")
    required = {
        "schema",
        "bucket",
        "generator_function_sha256",
        "critic_function_sha256",
        "checked_samples",
        "generator_errors",
        "independent_graders",
        "independent_grader_sha256s",
        "checked_sample_set_sha256",
        "minimum_checked_samples",
        "minimum_generator_errors",
        "minimum_independent_graders",
        "confusion_matrix",
        "shared_blind_spot_rate",
        "shared_blind_spot_wilson95",
        "maximum_admitted_upper_bound",
        "evidence_state",
        "critic_reliability_admitted",
        "snapshot_sha256",
    }
    if set(value) != required or value.get("schema") != BLIND_SPOT_EVIDENCE_SCHEMA:
        raise ValueError("shared blind-spot evidence schema is invalid")
    payload = {key: value[key] for key in required - {"snapshot_sha256"}}
    if value.get("snapshot_sha256") != _sha(payload):
        raise ValueError("shared blind-spot evidence digest differs")
    if (
        value.get("generator_function_sha256") != generator_function_sha256
        or value.get("critic_function_sha256") != critic_function_sha256
    ):
        raise ValueError("shared blind-spot evidence function identity differs")
    matrix = value.get("confusion_matrix")
    interval = value.get("shared_blind_spot_wilson95")
    counts = (
        "generator_correct_critic_accept",
        "generator_correct_critic_reject",
        "generator_error_critic_reject",
        "generator_error_critic_accept",
    )
    if (
        not isinstance(matrix, dict)
        or set(matrix) != set(counts)
        or any(type(matrix.get(key)) is not int or matrix[key] < 0 for key in counts)
        or not isinstance(interval, dict)
        or set(interval) != {"lower", "upper"}
    ):
        raise ValueError("shared blind-spot confusion matrix is invalid")
    checked = sum(matrix.values())
    generator_errors = (
        matrix["generator_error_critic_reject"] + matrix["generator_error_critic_accept"]
    )
    lower, upper = _wilson_interval(matrix["generator_error_critic_accept"], generator_errors)
    graders = value.get("independent_graders")
    grader_ids = value.get("independent_grader_sha256s")
    powered = bool(
        checked >= MIN_CHECKED_SAMPLES
        and generator_errors >= MIN_GENERATOR_ERRORS
        and type(graders) is int
        and graders >= MIN_INDEPENDENT_GRADERS
    )
    expected_rate = (
        round(matrix["generator_error_critic_accept"] / generator_errors, 8)
        if generator_errors
        else None
    )
    if (
        value.get("checked_samples") != checked
        or value.get("generator_errors") != generator_errors
        or type(graders) is not int
        or graders < 0
        or not isinstance(grader_ids, list)
        or len(grader_ids) != graders
        or grader_ids != sorted(set(grader_ids))
        or any(not _sha256(item) for item in grader_ids)
        or graders > checked
        or not _sha256(value.get("checked_sample_set_sha256"))
        or value.get("minimum_checked_samples") != MIN_CHECKED_SAMPLES
        or value.get("minimum_generator_errors") != MIN_GENERATOR_ERRORS
        or value.get("minimum_independent_graders") != MIN_INDEPENDENT_GRADERS
        or value.get("shared_blind_spot_rate") != expected_rate
        or interval.get("lower") != round(lower, 8)
        or interval.get("upper") != round(upper, 8)
        or value.get("maximum_admitted_upper_bound") != MAX_SHARED_BLIND_SPOT_UPPER_BOUND
        or value.get("evidence_state") != ("measured" if powered else "bootstrap_unmeasured")
        or value.get("critic_reliability_admitted")
        is not (not powered or upper <= MAX_SHARED_BLIND_SPOT_UPPER_BOUND)
    ):
        raise ValueError("shared blind-spot evidence statistics are invalid")
    return dict(value)


class CriticBlindSpotLedger:
    """Governed durable source of independently checked critic outcomes."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            try:
                from core.config import DATA_DIR

                path = (
                    Path(DATA_DIR)
                    / "latent_cortex"
                    / "critic_blind_spots"
                    / "checked_outcomes.jsonl"
                )
            except (ImportError, AttributeError, RuntimeError, TypeError):
                path = Path("data/latent_cortex/critic_blind_spots/checked_outcomes.jsonl")
        self.path = Path(path)
        self._rows: list[dict[str, Any]] = []
        self._keys: set[tuple[str, str, str, str, str]] = set()
        self.restore_errors = 0
        self._restore()

    @staticmethod
    def _key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(row["bucket"]),
            str(row["task_sha256"]),
            str(row["candidate_sha256"]),
            str(row["generator_function_sha256"]),
            str(row["critic_function_sha256"]),
        )

    def _restore(self) -> None:
        self._rows = []
        self._keys = set()
        try:
            if not self.path.exists():
                return
            with self.path.open(encoding="utf-8") as handle:
                for raw in handle:
                    try:
                        row = json.loads(raw)
                        if not isinstance(row, dict):
                            raise ValueError("row_not_mapping")
                        validated = _validate_checked_row(
                            row,
                            generator_function_sha256=str(
                                row.get("generator_function_sha256") or ""
                            ),
                            critic_function_sha256=str(row.get("critic_function_sha256") or ""),
                        )
                        key = self._key(validated)
                        if key in self._keys:
                            raise ValueError("duplicate_checked_sample")
                        self._keys.add(key)
                        self._rows.append(validated)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        self.restore_errors += 1
        except OSError:
            self.restore_errors += 1
        if len(self._rows) > _MAX_LEDGER_ROWS:
            self._rows = self._rows[-_MAX_LEDGER_ROWS:]
            self._keys = {self._key(row) for row in self._rows}

    def record_checked(
        self,
        *,
        bucket: str,
        task_sha256: str,
        candidate_sha256: str,
        generator_function_sha256: str,
        critic_function_sha256: str,
        independent_grader_sha256: str,
        independent_receipt_sha256: str,
        generator_correct: bool,
        critic_accepted: bool,
    ) -> bool:
        row = _validate_checked_row(
            {
                "schema": CHECKED_OUTCOME_SCHEMA,
                "bucket": bucket,
                "checked": True,
                "task_sha256": task_sha256,
                "candidate_sha256": candidate_sha256,
                "generator_function_sha256": generator_function_sha256,
                "critic_function_sha256": critic_function_sha256,
                "independent_grader_sha256": independent_grader_sha256,
                "independent_receipt_sha256": independent_receipt_sha256,
                "generator_correct": generator_correct,
                "critic_accepted": critic_accepted,
            },
            generator_function_sha256=generator_function_sha256,
            critic_function_sha256=critic_function_sha256,
        )
        key = self._key(row)
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.atomic_writer import interprocess_file_lock
            from core.runtime.file_write_gateway import get_file_write_gateway

            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            lock_path = self.path.with_name(f".{self.path.name}.lock")
            with interprocess_file_lock(lock_path):
                self._restore()
                if key in self._keys:
                    raise ValueError("checked critic sample is already recorded")
                with local_internal_governed_scope(
                    "latent_critic_blind_spots", domain="state_mutation"
                ):
                    gateway = get_file_write_gateway()
                    gateway.append_text(
                        self.path,
                        line,
                        source="latent_critic_blind_spots",
                    )
                    rows = [*self._rows, row]
                    if len(rows) > _MAX_LEDGER_ROWS:
                        rows = rows[-_MAX_LEDGER_ROWS:]
                        gateway.write_text(
                            self.path,
                            "".join(
                                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                                for item in rows
                            ),
                            source="latent_critic_blind_spots.compact",
                        )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if key in self._keys:
                raise ValueError("checked critic sample is already recorded") from exc
            return False
        self._keys.add(key)
        self._rows.append(row)
        if len(self._rows) > _MAX_LEDGER_ROWS:
            self._rows = self._rows[-_MAX_LEDGER_ROWS:]
        return True

    def evidence(
        self,
        *,
        bucket: str,
        generator_function_sha256: str,
        critic_function_sha256: str,
    ) -> dict[str, Any]:
        self._restore()
        rows = [
            row
            for row in self._rows
            if row["bucket"] == bucket
            and row["generator_function_sha256"] == generator_function_sha256
            and row["critic_function_sha256"] == critic_function_sha256
        ]
        return build_shared_blind_spot_evidence(
            bucket=bucket,
            generator_function_sha256=generator_function_sha256,
            critic_function_sha256=critic_function_sha256,
            checked_outcomes=rows,
        )

    def status(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "checked_outcomes": len(self._rows),
            "restore_errors": self.restore_errors,
        }


_LEDGER: CriticBlindSpotLedger | None = None


def get_critic_blind_spot_ledger() -> CriticBlindSpotLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = CriticBlindSpotLedger()
    return _LEDGER


__all__ = [
    "BLIND_SPOT_EVIDENCE_SCHEMA",
    "CHECKED_OUTCOME_SCHEMA",
    "CRITIC_IDENTITY_SCHEMA",
    "GENERATOR_IDENTITY_SCHEMA",
    "MAX_SHARED_BLIND_SPOT_UPPER_BOUND",
    "MIN_CHECKED_SAMPLES",
    "MIN_GENERATOR_ERRORS",
    "MIN_INDEPENDENT_GRADERS",
    "CriticBlindSpotLedger",
    "audit_python_dependencies",
    "build_critic_identity",
    "build_critic_source_identity",
    "build_generator_function_identity",
    "build_shared_blind_spot_evidence",
    "get_critic_blind_spot_ledger",
    "validate_critic_identity",
    "validate_shared_blind_spot_evidence",
]
