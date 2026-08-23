#!/usr/bin/env python3
"""Measure one candidate checkpoint against its frozen base without a model judge."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.candidate_cortex_kernel import conversation_sha256  # noqa: E402
from core.learning.candidate_cortex_measurement import (  # noqa: E402
    LOSS_ROW_SCHEMA,
    CandidateCortexMeasurementError,
    compile_checkpoint_evidence,
)
from core.learning.candidate_cortex_training import (  # noqa: E402
    CandidateCortexTrainingError,
    StagePolicy,
    canonical_json_bytes,
    discover_exact_checkpoint,
    document_sha256,
    file_sha256,
    load_and_verify_plan,
)
from core.learning.recurrent_sft_behavior_canaries import (  # noqa: E402
    build_generated_behavior_canaries,
    grade_generated_behavior_text,
)
from core.learning.recurrent_sft_retention import build_retention_rows  # noqa: E402
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402
from core.runtime.secure_path_custody import (  # noqa: E402
    DirectoryCustody,
    SecurePathCustodyError,
)

DETAIL_SCHEMA = "aura.candidate_cortex_training.checkpoint_measurement_detail.v2"
BASELINE_SCHEMA = "aura.candidate_cortex_training.baseline_measurement.v1"
_BASELINE_GENERATIONS_DIR = "baseline-measurements"
_MAX_JSONL_BYTES = 64 * 1024 * 1024
_GENERATION_TOKENS = 160


def _fail(code: str) -> None:
    raise CandidateCortexMeasurementError(code)


def _strict_json_bytes(raw: bytes, *, max_bytes: int = 16 * 1024 * 1024) -> Any:
    if not raw or len(raw) > max_bytes:
        _fail("measurement_input_size_invalid")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateCortexMeasurementError("measurement_input_json_invalid") from exc


def _strict_json(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> Any:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        _fail("measurement_input_not_regular")
    return _strict_json_bytes(resolved.read_bytes(), max_bytes=max_bytes)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        _fail("measurement_jsonl_not_regular")
    raw = resolved.read_bytes()
    if not raw or len(raw) > _MAX_JSONL_BYTES:
        _fail("measurement_jsonl_size_invalid")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            _fail("measurement_jsonl_blank_line")
        try:
            value = json.loads(line.decode("utf-8"), object_pairs_hook=_reject_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CandidateCortexMeasurementError("measurement_jsonl_invalid") from exc
        if not isinstance(value, dict):
            _fail("measurement_jsonl_invalid")
        rows.append(value)
    return rows


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("measurement_input_duplicate_key")
        result[key] = value
    return result


def _persona_samples(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    data_root = Path(str(plan["dataset"]["data_root"])).resolve(strict=True)
    conversations = _jsonl(data_root / "valid.jsonl")
    provenance = _jsonl(data_root / "provenance.jsonl")
    by_digest: dict[str, dict[str, Any]] = {}
    for row in provenance:
        digest = row.get("conversation_sha256")
        if not isinstance(digest, str) or digest in by_digest:
            _fail("persona_provenance_invalid")
        by_digest[digest] = row
    samples: list[dict[str, Any]] = []
    for row in conversations:
        messages = row.get("messages")
        if not isinstance(messages, list):
            _fail("persona_messages_invalid")
        digest = conversation_sha256(messages)
        source = by_digest.get(digest)
        domains = source.get("domains") if isinstance(source, Mapping) else None
        if (
            not isinstance(source, Mapping)
            or source.get("split") != "valid"
            or not isinstance(domains, list)
            or not domains
            or any(not isinstance(domain, str) or not domain for domain in domains)
        ):
            _fail("persona_provenance_missing")
        samples.append(
            {
                "sample_id": digest,
                "domain": "+".join(sorted(set(domains))),
                "messages": messages,
                "tools": row.get("tools"),
            }
        )
    return sorted(samples, key=lambda sample: sample["sample_id"])


def _retention_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in build_retention_rows("validation"):
        meta = row.get("_meta")
        if not isinstance(meta, Mapping):
            _fail("retention_metadata_invalid")
        sample_id = meta.get("case_fingerprint")
        family = meta.get("family")
        messages = row.get("messages")
        if (
            not isinstance(sample_id, str)
            or not isinstance(family, str)
            or not isinstance(messages, list)
        ):
            _fail("retention_metadata_invalid")
        samples.append(
            {
                "sample_id": sample_id,
                "domain": family,
                "messages": messages,
                "tools": row.get("tools"),
            }
        )
    return sorted(samples, key=lambda sample: sample["sample_id"])


def _tokenize_samples(tokenizer: Any, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tokenized: list[dict[str, Any]] = []
    for sample in samples:
        messages = sample["messages"]
        tools = sample.get("tools")
        tokens = list(
            tokenizer.apply_chat_template(
                messages,
                tools=tools,
                return_dict=False,
            )
        )
        add_generation_prompt = messages[-1].get("role") == "assistant"
        offset = len(
            tokenizer.apply_chat_template(
                messages[:-1],
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
            )
        )
        if len(tokens) < 2 or not 1 <= offset < len(tokens):
            _fail("measurement_token_alignment_invalid")
        tokenized.append({**dict(sample), "tokens": tokens, "offset": offset})
    return tokenized


def _loss_rows(model: Any, tokenized: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn

    rows: list[dict[str, Any]] = []
    for sample in tokenized:
        tokens = list(sample["tokens"])
        inputs = mx.array([tokens[:-1]])
        targets = mx.array([tokens[1:]])
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= int(sample["offset"]), steps <= len(tokens))
        losses = nn.losses.cross_entropy(logits, targets)
        nll_sum = (losses * mask).astype(mx.float32).sum()
        token_count = mask.sum()
        mx.eval(nll_sum, token_count)
        count = int(token_count.item())
        if count <= 0:
            _fail("measurement_target_tokens_empty")
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "domain": sample["domain"],
                "nll_sum": float(nll_sum.item()),
                "tokens": count,
            }
        )
        del inputs, targets, logits, losses, nll_sum, token_count
        mx.clear_cache()
    return rows


def _behavior_source_sha256() -> str:
    import core.learning.recurrent_sft_behavior_canaries as canaries

    return file_sha256(Path(canaries.__file__).resolve(strict=True))


def _measurement_contract_sha256(
    plan: Mapping[str, Any],
    persona: Sequence[Mapping[str, Any]],
    retention: Sequence[Mapping[str, Any]],
) -> str:
    behavior_cases = build_generated_behavior_canaries()
    return document_sha256(
        {
            "plan_sha256": plan["plan_sha256"],
            "model_descriptor_sha256": plan["model"]["descriptor_sha256"],
            "dataset_receipt_sha256": plan["dataset"]["receipt_sha256"],
            "persona": [
                {
                    "sample_id": sample["sample_id"],
                    "domain": sample["domain"],
                    "tokens_sha256": document_sha256(list(sample["tokens"])),
                    "offset": sample["offset"],
                }
                for sample in persona
            ],
            "retention": [
                {
                    "sample_id": sample["sample_id"],
                    "domain": sample["domain"],
                    "tokens_sha256": document_sha256(list(sample["tokens"])),
                    "offset": sample["offset"],
                }
                for sample in retention
            ],
            "behavior_cases_sha256": document_sha256(behavior_cases),
            "behavior_source_sha256": _behavior_source_sha256(),
            "generation": {
                "max_tokens": _GENERATION_TOKENS,
                "temperature": 0.0,
                "thinking": False,
            },
        }
    )


def _behavior_rows(model: Any, tokenizer: Any) -> list[dict[str, Any]]:
    import mlx.core as mx
    from mlx_lm import stream_generate

    source_sha = _behavior_source_sha256()
    rows: list[dict[str, Any]] = []
    for case in build_generated_behavior_canaries():
        prompt = list(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": case["system"]},
                    {"role": "user", "content": case["prompt"]},
                ],
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        )
        pieces: list[str] = []
        finish_reason = ""
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=_GENERATION_TOKENS,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
            prefill_step_size=1024,
        ):
            pieces.append(str(response.text or ""))
            finish_reason = str(response.finish_reason or "")
        text = "".join(pieces)
        grade = grade_generated_behavior_text(case, text)
        evaluator_sha = document_sha256(
            {
                "case": case,
                "grader_source_sha256": source_sha,
                "max_tokens": _GENERATION_TOKENS,
                "temperature": 0.0,
                "thinking": False,
            }
        )
        rows.append(
            {
                "probe_id": case["case_id"],
                "family": case["family"],
                "passed": bool(grade["passed"]),
                "evaluator_sha256": evaluator_sha,
                "text": text,
                "text_sha256": grade["text_sha256"],
                "finish_reason": finish_reason,
                "grade": grade,
            }
        )
        mx.clear_cache()
    return rows


def _pair_losses(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    base = {str(row["sample_id"]): row for row in baseline}
    adapted = {str(row["sample_id"]): row for row in candidate}
    if len(base) != len(baseline) or len(adapted) != len(candidate) or set(base) != set(adapted):
        _fail("measurement_loss_pairing_invalid")
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(base):
        left = base[sample_id]
        right = adapted[sample_id]
        if left["domain"] != right["domain"] or left["tokens"] != right["tokens"]:
            _fail("measurement_loss_pairing_invalid")
        rows.append(
            {
                "schema": LOSS_ROW_SCHEMA,
                "sample_id": sample_id,
                "domain": left["domain"],
                "baseline_nll_sum": left["nll_sum"],
                "candidate_nll_sum": right["nll_sum"],
                "tokens": left["tokens"],
            }
        )
    return rows


def _pair_behaviors(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    base = {str(row["probe_id"]): row for row in baseline}
    adapted = {str(row["probe_id"]): row for row in candidate}
    if len(base) != len(baseline) or len(adapted) != len(candidate) or set(base) != set(adapted):
        _fail("measurement_behavior_pairing_invalid")
    result: list[dict[str, Any]] = []
    for probe_id in sorted(base):
        left = base[probe_id]
        right = adapted[probe_id]
        if (
            left["family"] != right["family"]
            or left["evaluator_sha256"] != right["evaluator_sha256"]
        ):
            _fail("measurement_behavior_pairing_invalid")
        result.append(
            {
                "probe_id": probe_id,
                "family": left["family"],
                "baseline_passed": left["passed"],
                "candidate_passed": right["passed"],
                "evaluator_sha256": left["evaluator_sha256"],
            }
        )
    return result


def _baseline_document(
    *,
    plan: Mapping[str, Any],
    contract_sha256: str,
    persona: Sequence[Mapping[str, Any]],
    retention: Sequence[Mapping[str, Any]],
    behavior: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    material = {
        "schema": BASELINE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "model_descriptor_sha256": plan["model"]["descriptor_sha256"],
        "dataset_receipt_sha256": plan["dataset"]["receipt_sha256"],
        "measurement_contract_sha256": contract_sha256,
        "persona": list(persona),
        "retention": list(retention),
        "behavior": list(behavior),
    }
    return {**material, "baseline_sha256": document_sha256(material)}


def _validate_baseline_document(
    raw: object,
    *,
    plan: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema",
        "plan_sha256",
        "model_descriptor_sha256",
        "dataset_receipt_sha256",
        "measurement_contract_sha256",
        "persona",
        "retention",
        "behavior",
        "baseline_sha256",
    }:
        _fail("measurement_baseline_schema_invalid")
    material = {key: value for key, value in raw.items() if key != "baseline_sha256"}
    if (
        raw.get("schema") != BASELINE_SCHEMA
        or raw.get("plan_sha256") != plan["plan_sha256"]
        or raw.get("model_descriptor_sha256") != plan["model"]["descriptor_sha256"]
        or raw.get("dataset_receipt_sha256") != plan["dataset"]["receipt_sha256"]
        or raw.get("measurement_contract_sha256") != contract_sha256
        or raw.get("baseline_sha256") != document_sha256(material)
    ):
        _fail("measurement_baseline_identity_invalid")
    for role in ("persona", "retention", "behavior"):
        rows = raw.get(role)
        if (
            not isinstance(rows, list)
            or not rows
            or any(not isinstance(row, Mapping) for row in rows)
        ):
            _fail("measurement_baseline_rows_invalid")
    return dict(raw)


def _baseline_relative_path(contract_sha256: str) -> Path:
    if len(contract_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in contract_sha256
    ):
        _fail("measurement_contract_digest_invalid")
    return Path(_BASELINE_GENERATIONS_DIR) / contract_sha256 / "baseline_measurement.json"


def _load_or_create_addressed_baseline(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    contract_sha256: str,
    producer: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], Path, bool]:
    """Load or publish the immutable baseline generation for one evaluator."""

    root = run_root.expanduser().resolve(strict=True)
    relative = _baseline_relative_path(contract_sha256)
    lock_relative = relative.with_name("baseline.lock")
    with DirectoryCustody.acquire(root, private=True) as custody:
        with custody.file_lock(lock_relative):
            if custody.file_exists(relative):
                baseline = _validate_baseline_document(
                    _strict_json_bytes(custody.read_bytes(relative, max_bytes=16 * 1024 * 1024)),
                    plan=plan,
                    contract_sha256=contract_sha256,
                )
                return baseline, root / relative, True
            baseline = _validate_baseline_document(
                producer(), plan=plan, contract_sha256=contract_sha256
            )
            payload = canonical_json_bytes(baseline) + b"\n"
            if not custody.write_bytes_once(relative, payload, mode=0o600):
                retained = custody.read_bytes(relative, max_bytes=16 * 1024 * 1024)
                if retained != payload:
                    _fail("measurement_output_conflict")
            retained_baseline = _validate_baseline_document(
                _strict_json_bytes(custody.read_bytes(relative, max_bytes=16 * 1024 * 1024)),
                plan=plan,
                contract_sha256=contract_sha256,
            )
            return retained_baseline, root / relative, False


def _measure_baseline_document(
    model: Any,
    tokenizer: Any,
    *,
    plan: Mapping[str, Any],
    contract_sha256: str,
    persona_tokens: Sequence[Mapping[str, Any]],
    retention_tokens: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _baseline_document(
        plan=plan,
        contract_sha256=contract_sha256,
        persona=_loss_rows(model, persona_tokens),
        retention=_loss_rows(model, retention_tokens),
        behavior=_behavior_rows(model, tokenizer),
    )


def _adapter_spec(adapter_root: Path) -> tuple[int, dict[str, Any], bool]:
    config = _strict_json(adapter_root / "adapter_config.json")
    if not isinstance(config, Mapping):
        _fail("measurement_adapter_config_invalid")
    if set(config) - {
        "adapter_path",
        "batch_size",
        "clear_cache_threshold",
        "config",
        "data",
        "fine_tune_type",
        "grad_accumulation_steps",
        "grad_checkpoint",
        "iters",
        "learning_rate",
        "lora_parameters",
        "lr_schedule",
        "mask_prompt",
        "max_seq_length",
        "model",
        "num_layers",
        "optimizer",
        "optimizer_config",
        "project_name",
        "report_to",
        "resume_adapter_file",
        "save_every",
        "seed",
        "steps_per_eval",
        "steps_per_report",
        "test",
        "test_batches",
        "train",
        "val_batches",
    }:
        _fail("measurement_adapter_config_invalid")
    fine_tune_type = config.get("fine_tune_type", "lora")
    if fine_tune_type not in {"lora", "dora"}:
        _fail("measurement_adapter_type_invalid")
    parameters = config.get("lora_parameters")
    num_layers = config.get("num_layers")
    if (
        not isinstance(parameters, dict)
        or set(parameters) != {"dropout", "keys", "rank", "scale"}
        or isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
    ):
        _fail("measurement_adapter_config_invalid")
    keys = parameters.get("keys")
    rank = parameters.get("rank")
    scale = parameters.get("scale")
    dropout = parameters.get("dropout")
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
        or len(set(keys)) != len(keys)
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank <= 0
        or isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or float(scale) <= 0
        or isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0 <= float(dropout) < 1
    ):
        _fail("measurement_adapter_config_invalid")
    return num_layers, dict(parameters), fine_tune_type == "dora"


def _attach_checkpoint(model: Any, adapter_root: Path, checkpoint: Path) -> None:
    import mlx.core as mx
    from mlx_lm.tuner.utils import linear_to_lora_layers

    num_layers, parameters, use_dora = _adapter_spec(adapter_root)
    linear_to_lora_layers(
        model,
        num_layers,
        parameters,
        use_dora=use_dora,
    )
    model.load_weights(str(checkpoint), strict=False)
    mx.eval(model.parameters())


def _write_once(path: Path, value: Mapping[str, Any], *, source: str) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    with local_internal_governed_scope(source, domain="file_write"):
        created = get_file_write_gateway().write_bytes_if_absent(
            path,
            payload,
            mode=0o600,
            source=source,
        )
    if not created and path.read_bytes() != payload:
        _fail("measurement_output_conflict")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument(
        "--baseline-cache",
        type=Path,
        help="immutable frozen-base measurement; defaults inside the run root",
    )
    args = parser.parse_args(argv)
    try:
        plan = load_and_verify_plan(args.run_root, verify_full_model=True)
        policy = StagePolicy(**dict(plan["stages"]))
        cumulative = policy.cumulative_iterations(args.stage_index)
        checkpoint = discover_exact_checkpoint(
            Path(str(plan["paths"]["checkpoint_root"])),
            expected_cumulative_iterations=cumulative,
        )
        persona = _persona_samples(plan)
        retention = _retention_samples()
        model_path = str(plan["model"]["canonical_path"])
        with standalone_model_lane(
            owner_id=f"candidate-cortex-measurement:{plan['run_id']}:{args.stage_index}",
            model_path=model_path,
            purpose="evaluate",
            priority=100,
            preemptible=False,
            require_exclusive=True,
            allow_owner_eviction=True,
            metadata={"tool": "measure_candidate_cortex_checkpoint"},
        ):
            from mlx_lm import load

            model, tokenizer = load(
                model_path,
                tokenizer_config={"trust_remote_code": True},
            )
            persona_tokens = _tokenize_samples(tokenizer, persona)
            retention_tokens = _tokenize_samples(tokenizer, retention)
            contract_sha256 = _measurement_contract_sha256(plan, persona_tokens, retention_tokens)

            produce_baseline = partial(
                _measure_baseline_document,
                model,
                tokenizer,
                plan=plan,
                contract_sha256=contract_sha256,
                persona_tokens=persona_tokens,
                retention_tokens=retention_tokens,
            )

            if args.baseline_cache is None:
                baseline, baseline_path, baseline_reused = _load_or_create_addressed_baseline(
                    run_root=args.run_root,
                    plan=plan,
                    contract_sha256=contract_sha256,
                    producer=produce_baseline,
                )
            else:
                baseline_path = args.baseline_cache.expanduser().resolve(strict=False)
                if baseline_path.is_file():
                    baseline = _validate_baseline_document(
                        _strict_json(baseline_path),
                        plan=plan,
                        contract_sha256=contract_sha256,
                    )
                    baseline_reused = True
                else:
                    baseline = produce_baseline()
                    _write_once(
                        baseline_path,
                        baseline,
                        source="candidate_cortex_measurement.baseline",
                    )
                    baseline_reused = False
            del produce_baseline
            baseline_persona = list(baseline["persona"])
            baseline_retention = list(baseline["retention"])
            baseline_behavior = list(baseline["behavior"])

            _attach_checkpoint(
                model,
                Path(str(plan["paths"]["adapter_root"])).resolve(strict=True),
                Path(str(checkpoint["path"])).resolve(strict=True),
            )
            candidate_persona = _loss_rows(model, persona_tokens)
            candidate_retention = _loss_rows(model, retention_tokens)
            candidate_behavior = _behavior_rows(model, tokenizer)
            del model, tokenizer
            gc.collect()
            import mlx.core as mx

            mx.clear_cache()

        persona_rows = _pair_losses(baseline_persona, candidate_persona)
        retention_rows = _pair_losses(baseline_retention, candidate_retention)
        behavior_rows = _pair_behaviors(baseline_behavior, candidate_behavior)
        evidence = compile_checkpoint_evidence(
            plan=plan,
            stage_index=args.stage_index,
            checkpoint_sha256=str(checkpoint["sha256"]),
            persona_rows=persona_rows,
            retention_rows=retention_rows,
            behavior_rows=behavior_rows,
            measurement_contract_sha256=contract_sha256,
            baseline_sha256=str(baseline["baseline_sha256"]),
        )
        detail_material = {
            "schema": DETAIL_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "stage_index": args.stage_index,
            "checkpoint": checkpoint,
            "persona_rows": persona_rows,
            "retention_rows": retention_rows,
            "baseline_behavior": baseline_behavior,
            "candidate_behavior": candidate_behavior,
            "baseline_path": str(baseline_path.resolve(strict=True)),
            "baseline_sha256": baseline["baseline_sha256"],
            "baseline_reused": baseline_reused,
            "measurement_contract_sha256": contract_sha256,
            "evidence_sha256": evidence["measurement_sha256"],
        }
        detail = {**detail_material, "detail_sha256": document_sha256(detail_material)}
        _write_once(
            args.detail_output.expanduser().resolve(strict=False),
            detail,
            source="candidate_cortex_measurement.detail",
        )
        _write_once(
            args.evidence_output.expanduser().resolve(strict=False),
            evidence,
            source="candidate_cortex_measurement.evidence",
        )
        result = {
            "status": "measured",
            "evidence_path": str(args.evidence_output.expanduser().resolve(strict=True)),
            "measurement_sha256": evidence["measurement_sha256"],
            "detail_path": str(args.detail_output.expanduser().resolve(strict=True)),
            "detail_sha256": detail["detail_sha256"],
            "baseline_path": str(baseline_path.resolve(strict=True)),
            "baseline_sha256": baseline["baseline_sha256"],
            "baseline_reused": baseline_reused,
        }
    except (
        CandidateCortexMeasurementError,
        CandidateCortexTrainingError,
        FileNotFoundError,
        OSError,
        SecurePathCustodyError,
        TypeError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
