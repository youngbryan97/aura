"""Shared held-out behavioral probe for recurrent checkpoint training.

Training objectives may differ, but their promotion evidence must not. This
module runs the same source-task generations, random-stream coordinates,
semantic grader, and full episode-receipt capture for generated-rollin SFT and
on-policy recurrent GRPO canaries.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.types import ComputeBudget
from core.learning.recurrent_checkpoint_admission import build_free_generation_report
from core.learning.recurrent_grpo import (
    RecurrentSamplingConfig,
    cortex_config_from_execution_spec,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def tokenize_task(
    tokenizer: Any,
    prompt: str,
    answer: str,
) -> tuple[list[int], list[int]]:
    from core.brain.llm.chat_format import for_this_template

    prompt_tokens = tokenizer.apply_chat_template(
        for_this_template(tokenizer, [{"role": "user", "content": prompt}]),
        add_generation_prompt=True,
        tokenize=True,
    )
    try:
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    except TypeError:
        answer_tokens = tokenizer.encode(answer)
    eos = getattr(tokenizer, "eos_token_id", None)
    normalized_answer = [int(token) for token in answer_tokens]
    if eos is not None and (not normalized_answer or normalized_answer[-1] != int(eos)):
        normalized_answer.append(int(eos))
    return [int(token) for token in prompt_tokens], normalized_answer


def paired_generation_seed(
    campaign_seed: int,
    task_ordinal: int,
    task_id: str,
    depth: int,
) -> int:
    """Use one random stream per task/depth coordinate across both arms."""

    material = f"{campaign_seed}:{task_ordinal}:{task_id}:{depth}"
    return int.from_bytes(
        hashlib.sha256(material.encode("ascii")).digest()[:4],
        "big",
    )


def free_generation_sampling_config() -> RecurrentSamplingConfig:
    """Use the exact categorical policy required by recurrent proof runs."""

    return RecurrentSamplingConfig(
        max_tokens=320,
        temperature=1.0,
        top_p=1.0,
    )


def _ordinary_decode_once(
    model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    *,
    max_tokens: int,
) -> tuple[str, list[int], str]:
    """Generate one exact greedy incumbent with engine-equivalent stopping."""

    from mlx_lm import stream_generate

    from core.brain.llm.latent_cortex.answer_contract import is_contract_complete

    pieces: list[str] = []
    output_tokens: list[int] = []
    termination = "token_limit"
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt_tokens,
        max_tokens=max_tokens,
    ):
        pieces.append(response.text)
        if response.finish_reason != "stop":
            output_tokens.append(int(response.token))
        if response.finish_reason == "stop":
            termination = "eos"
        elif response.finish_reason == "length":
            termination = "token_limit"
        if "}" in response.text and is_contract_complete("".join(pieces)):
            termination = "contract_complete"
            break
    text = tokenizer.decode(output_tokens)
    if text != "".join(pieces):
        raise RuntimeError("ordinary decode token/text round trip differs")
    if not text or not output_tokens:
        raise RuntimeError("ordinary decode produced no incumbent output")
    return text, output_tokens, termination


def _full_engine_config(
    spec: RLCExecutionSpec,
    *,
    objective_program_enabled: bool = True,
    verified_objective_teacher_enabled: bool = True,
) -> Any:
    """Build the product treatment, not the naked recurrent ablation."""

    if type(objective_program_enabled) is not bool:
        raise TypeError("objective_program_enabled must be boolean")
    if type(verified_objective_teacher_enabled) is not bool:
        raise TypeError("verified_objective_teacher_enabled must be boolean")

    from core.brain.llm.latent_cortex.types import FastWeightsConfig, LatentOptConfig

    config = cortex_config_from_execution_spec(
        spec,
        sampling=free_generation_sampling_config(),
    )
    config.latent_opt = LatentOptConfig(enabled=True, steps=4, lr=0.05)
    config.fast_weights = FastWeightsConfig(enabled=True, rank=2, opt_steps=4)
    config.fast_weights.locality_diagnostic_enabled = True
    config.verifier_accept_non_regression = True
    config.decode_bridge_policy = "assistant_answer_v4"
    config.decode_incumbent_policy = "vanilla_incumbent"
    config.answer_replacement_enabled = True
    config.objective_program_enabled = objective_program_enabled
    config.verified_objective_teacher_enabled = verified_objective_teacher_enabled
    config.local_repair_enabled = True
    config.local_repair_max_attempts = len(spec.branch_roles)
    config.local_repair_max_tokens = free_generation_sampling_config().max_tokens
    config.allow_vanilla_fallback = False
    config.verifier_probe_max_tokens = max(
        48,
        min(256, free_generation_sampling_config().max_tokens // 2),
    )
    config.verifier_probe_contract = "final_answer_v1"
    config.decode_contract = "none"
    config.decode_contract_grace_tokens = 0
    config.terminal_instruction_policy = "applied"
    problems = config.validate()
    if problems:
        raise ValueError(f"full-engine probe config rejected: {problems}")
    return config


def _normalize_full_engine_probe_depths(
    spec: RLCExecutionSpec,
    depths: Sequence[int] | None,
) -> tuple[int, ...]:
    selected = (
        tuple(sorted({1, spec.recurrent_steps}))
        if depths is None
        else tuple(depths)
    )
    if (
        len(selected) < 2
        or tuple(sorted(set(selected))) != selected
        or any(type(depth) is not int for depth in selected)
        or selected[0] != 1
        or selected[-1] != spec.recurrent_steps
        or any(not 1 <= depth <= spec.recurrent_steps for depth in selected)
    ):
        raise ValueError(
            "probe depths must be unique ordered integers spanning shallow and full depth"
        )
    return selected


def build_behavioral_probe_report(
    model: Any,
    tokenizer: Any,
    tasks: list[Any],
    *,
    spec: RLCExecutionSpec,
    arm: str,
    adapter_sha256: str,
    task_manifest_sha256: str,
    seed: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run exact held-out generations at shallow and full recurrent depth."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    depths = tuple(sorted({1, spec.recurrent_steps}))
    records: list[dict[str, Any]] = []
    for task_ordinal, task in enumerate(tasks):
        prompt_tokens, _answer_tokens = tokenize_task(
            tokenizer,
            task.prompt,
            task.answer,
        )
        for depth in depths:
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "sample_started",
                        "arm": arm,
                        "task_id": task.task_id,
                        "task_ordinal": task_ordinal,
                        "depth": depth,
                    }
                )
            depth_spec = spec.with_depth(depth)
            config = cortex_config_from_execution_spec(
                depth_spec,
                sampling=free_generation_sampling_config(),
            )
            config.decode_contract = "none"
            config.decode_contract_grace_tokens = 0
            config.decode_incumbent_policy = "latent"
            engine = LatentCortexEngine(
                model,
                tokenizer=tokenizer,
                config=config,
                schedule_library=None,
            )
            generation_seed = paired_generation_seed(
                seed,
                task_ordinal,
                task.task_id,
                depth,
            )
            mx.random.seed(generation_seed)
            result = engine.reason(
                token_ids=prompt_tokens,
                decode_max_tokens=320,
                decode_sentence_grace_tokens=0,
                nonparametric_memory_enabled=False,
                sample_seed=generation_seed,
                episode_id=f"behavioral-probe-{seed}-{task_ordinal}-{depth}",
            )
            grade = dict(task.grade(result.text if result.ok else ""))
            grade["correct"] = bool(grade.get("correct"))
            receipt_payload = result.receipt.to_dict()
            records.append(
                {
                    "task_id": task.task_id,
                    "depth": depth,
                    "response_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
                    "response_text": result.text,
                    "tokens_sha256": hashlib.sha256(
                        canonical_json_bytes(result.tokens)
                    ).hexdigest(),
                    "tokens": list(result.tokens),
                    "token_count": len(result.tokens),
                    "correct": bool(result.ok and grade["correct"]),
                    "grade_receipt": {
                        **grade,
                        "correct": bool(result.ok and grade["correct"]),
                    },
                    "episode_ok": bool(result.ok),
                    "episode_reason": str(result.reason or ""),
                    "decode_termination": str(result.receipt.decode_termination or "not_reached"),
                    "branch_selection_admitted": bool(result.receipt.branch_selection_admitted),
                    "decode_incumbent_policy": (result.receipt.decode_incumbent_policy),
                    "episode_receipt_sha256": hashlib.sha256(
                        canonical_json_bytes(receipt_payload)
                    ).hexdigest(),
                    "episode_receipt": receipt_payload,
                }
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "sample_completed",
                        "arm": arm,
                        "task_id": task.task_id,
                        "task_ordinal": task_ordinal,
                        "depth": depth,
                        "correct": bool(result.ok and grade["correct"]),
                        "token_count": len(result.tokens),
                    }
                )
            del engine, result
            mx.synchronize()
            mx.clear_cache()
    return build_free_generation_report(
        arm=arm,
        adapter_sha256=adapter_sha256,
        execution_spec_sha256=spec.sha256,
        task_manifest_sha256=task_manifest_sha256,
        task_ids=[task.task_id for task in tasks],
        depths=depths,
        records=records,
    )


def build_ordinary_decode_probe_report(
    model: Any,
    tokenizer: Any,
    tasks: list[Any],
    *,
    spec: RLCExecutionSpec,
    adapter_sha256: str,
    task_manifest_sha256: str,
    seed: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """The vanilla control: the same weights answering without the RLC path.

    Reported at every depth coordinate the recurrent arms use so the two
    reports are directly comparable, even though ordinary decode has no
    recurrent depth -- the same generation is recorded at each coordinate
    because the ordinary path does not vary with it. Without this arm an
    admission can only compare a trained adapter to an untrained adapter on
    the same path, which is how a checkpoint scoring 3/28 passed review while
    ordinary decode on identical weights scored 13/28.
    """

    import mlx.core as mx

    depths = tuple(sorted({1, spec.recurrent_steps}))
    records: list[dict[str, Any]] = []
    for task_ordinal, task in enumerate(tasks):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sample_started",
                    "arm": "ordinary_decode",
                    "task_id": task.task_id,
                    "task_ordinal": task_ordinal,
                    "depths": list(depths),
                }
            )
        prompt_tokens, _answer_tokens = tokenize_task(
            tokenizer,
            task.prompt,
            task.answer,
        )
        generation_seed = paired_generation_seed(seed, task_ordinal, task.task_id, 1)
        mx.random.seed(generation_seed)
        text, tokens, decode_termination = _ordinary_decode_once(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=free_generation_sampling_config().max_tokens,
        )
        grade = dict(task.grade(text))
        grade["correct"] = bool(grade.get("correct"))
        for depth in depths:
            receipt_payload = {
                "schema": "aura.rlc.ordinary_decode_probe.v2",
                "arm": "ordinary_decode",
                "task_id": task.task_id,
                "depth_coordinate": depth,
                "generation_seed": generation_seed,
                "recurrent_steps": 0,
                "prompt_tokens_sha256": hashlib.sha256(
                    canonical_json_bytes(prompt_tokens)
                ).hexdigest(),
                "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "tokens_sha256": hashlib.sha256(
                    canonical_json_bytes(tokens)
                ).hexdigest(),
                "token_count": len(tokens),
                "decode_termination": decode_termination,
            }
            records.append(
                {
                    "task_id": task.task_id,
                    "depth": depth,
                    "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "response_text": text,
                    "tokens_sha256": hashlib.sha256(
                        canonical_json_bytes(tokens)
                    ).hexdigest(),
                    "tokens": list(tokens),
                    "token_count": len(tokens),
                    "correct": bool(grade["correct"]),
                    "grade_receipt": dict(grade),
                    "episode_ok": True,
                    "episode_reason": "",
                    "decode_termination": decode_termination,
                    "branch_selection_admitted": True,
                    "decode_incumbent_policy": "vanilla_incumbent",
                    "episode_receipt_sha256": hashlib.sha256(
                        canonical_json_bytes(receipt_payload)
                    ).hexdigest(),
                    "episode_receipt": receipt_payload,
                }
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "sample_completed",
                    "arm": "ordinary_decode",
                    "task_id": task.task_id,
                    "task_ordinal": task_ordinal,
                    "depths": list(depths),
                    "correct": bool(grade["correct"]),
                    "token_count": len(tokens),
                }
            )
        mx.synchronize()
        mx.clear_cache()
    return build_free_generation_report(
        arm="ordinary_decode",
        adapter_sha256=adapter_sha256,
        execution_spec_sha256=spec.sha256,
        task_manifest_sha256=task_manifest_sha256,
        task_ids=[task.task_id for task in tasks],
        depths=depths,
        records=records,
    )


def build_paired_full_engine_probe_reports(
    model: Any,
    tokenizer: Any,
    tasks: list[Any],
    *,
    model_path: str | Path,
    spec: RLCExecutionSpec,
    adapter_sha256: str,
    task_manifest_sha256: str,
    seed: int,
    depths: Sequence[int] | None = None,
    objective_program_enabled: bool = True,
    verified_objective_teacher_enabled: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ordinary and complete-engine arms from one immutable incumbent."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.incumbent_artifact import (
        build_incumbent_artifact,
    )
    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        full_weight_checkpoint_identity,
    )
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier

    checkpoint = full_weight_checkpoint_identity(Path(model_path))
    selected_depths = _normalize_full_engine_probe_depths(spec, depths)
    if type(objective_program_enabled) is not bool:
        raise TypeError("objective_program_enabled must be boolean")
    if type(verified_objective_teacher_enabled) is not bool:
        raise TypeError("verified_objective_teacher_enabled must be boolean")
    ordinary_records: list[dict[str, Any]] = []
    full_records: list[dict[str, Any]] = []
    n_layers = len(model.model.layers)
    max_tokens = free_generation_sampling_config().max_tokens
    for task_ordinal, task in enumerate(tasks):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "ordinary_started",
                    "task_id": task.task_id,
                    "task_ordinal": task_ordinal,
                    "depths": list(selected_depths),
                }
            )
        prompt_tokens, _answer_tokens = tokenize_task(
            tokenizer,
            task.prompt,
            task.answer,
        )
        ordinary_seed = paired_generation_seed(
            seed,
            task_ordinal,
            task.task_id,
            1,
        )
        mx.random.seed(ordinary_seed)
        ordinary_text, ordinary_tokens, ordinary_termination = _ordinary_decode_once(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=max_tokens,
        )
        ordinary_grade = dict(task.grade(ordinary_text))
        ordinary_grade["correct"] = bool(ordinary_grade.get("correct"))
        incumbent = build_incumbent_artifact(
            input_tokens=prompt_tokens,
            output_tokens=ordinary_tokens,
            output_text=ordinary_text,
            checkpoint_fingerprint=checkpoint["fingerprint"],
            checkpoint_fingerprint_method=checkpoint["method"],
            max_tokens=max_tokens,
            n_layers=n_layers,
            termination=ordinary_termination,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "ordinary_completed",
                    "task_id": task.task_id,
                    "task_ordinal": task_ordinal,
                    "correct": bool(ordinary_grade["correct"]),
                    "token_count": len(ordinary_tokens),
                    "termination": ordinary_termination,
                }
            )
        for depth in selected_depths:
            ordinary_receipt = {
                "schema": "aura.rlc.ordinary_decode_probe.v2",
                "arm": "ordinary_decode",
                "task_id": task.task_id,
                "depth_coordinate": depth,
                "generation_seed": ordinary_seed,
                "recurrent_steps": 0,
                "prompt_tokens_sha256": hashlib.sha256(
                    canonical_json_bytes(prompt_tokens)
                ).hexdigest(),
                "response_sha256": hashlib.sha256(
                    ordinary_text.encode("utf-8")
                ).hexdigest(),
                "tokens_sha256": hashlib.sha256(
                    canonical_json_bytes(ordinary_tokens)
                ).hexdigest(),
                "token_count": len(ordinary_tokens),
                "decode_termination": ordinary_termination,
            }
            ordinary_records.append(
                {
                    "task_id": task.task_id,
                    "depth": depth,
                    "response_sha256": ordinary_receipt["response_sha256"],
                    "response_text": ordinary_text,
                    "tokens_sha256": ordinary_receipt["tokens_sha256"],
                    "tokens": list(ordinary_tokens),
                    "token_count": len(ordinary_tokens),
                    "correct": bool(ordinary_grade["correct"]),
                    "grade_receipt": dict(ordinary_grade),
                    "episode_ok": True,
                    "episode_reason": "",
                    "decode_termination": ordinary_termination,
                    "branch_selection_admitted": True,
                    "decode_incumbent_policy": "vanilla_incumbent",
                    "episode_receipt_sha256": hashlib.sha256(
                        canonical_json_bytes(ordinary_receipt)
                    ).hexdigest(),
                    "episode_receipt": ordinary_receipt,
                }
            )

            depth_spec = spec.with_depth(depth)
            engine = LatentCortexEngine(
                model,
                tokenizer=tokenizer,
                config=_full_engine_config(
                    depth_spec,
                    objective_program_enabled=objective_program_enabled,
                    verified_objective_teacher_enabled=(
                        verified_objective_teacher_enabled
                    ),
                ),
                model_path=str(model_path),
                schedule_library=None,
            )
            verifier = EpisodeTaskVerifier(
                task.prompt,
                response_contract=task.response_contract,
            )
            generation_seed = paired_generation_seed(
                seed,
                task_ordinal,
                task.task_id,
                depth,
            )
            mx.random.seed(generation_seed)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "full_engine_started",
                        "task_id": task.task_id,
                        "task_ordinal": task_ordinal,
                        "depth": depth,
                        "objective_program_enabled": objective_program_enabled,
                        "verified_objective_teacher_enabled": (
                            verified_objective_teacher_enabled
                        ),
                    }
                )
            result = engine.reason(
                messages=[{"role": "user", "content": task.prompt}],
                verifier=verifier,
                # Locality diagnostics add four exact matched decodes. Keep
                # this offline proof lane bounded, but do not inherit the
                # shorter interactive-serving wall-clock contract.
                budget=ComputeBudget(wall_clock_s=600.0),
                domain=task.domain,
                decode_max_tokens=max_tokens,
                decode_sentence_grace_tokens=0,
                nonparametric_memory_enabled=False,
                sample_seed=generation_seed,
                incumbent_artifact=incumbent,
                episode_id=f"full-engine-probe-{seed}-{task_ordinal}-{depth}",
            )
            grade = dict(task.grade(result.text if result.ok else ""))
            grade["correct"] = bool(result.ok and grade.get("correct"))
            receipt_payload = result.receipt.to_dict()
            full_records.append(
                {
                    "task_id": task.task_id,
                    "depth": depth,
                    "response_sha256": hashlib.sha256(
                        result.text.encode("utf-8")
                    ).hexdigest(),
                    "response_text": result.text,
                    "tokens_sha256": hashlib.sha256(
                        canonical_json_bytes(result.tokens)
                    ).hexdigest(),
                    "tokens": list(result.tokens),
                    "token_count": len(result.tokens),
                    "correct": bool(grade["correct"]),
                    "grade_receipt": grade,
                    "episode_ok": bool(result.ok),
                    "episode_reason": str(result.reason or ""),
                    "decode_termination": str(
                        result.receipt.decode_termination or "not_reached"
                    ),
                    "branch_selection_admitted": bool(
                        result.receipt.branch_selection_admitted
                    ),
                    "decode_incumbent_policy": (
                        result.receipt.decode_incumbent_policy
                    ),
                    "episode_receipt_sha256": hashlib.sha256(
                        canonical_json_bytes(receipt_payload)
                    ).hexdigest(),
                    "episode_receipt": receipt_payload,
                }
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "full_engine_completed",
                        "task_id": task.task_id,
                        "task_ordinal": task_ordinal,
                        "depth": depth,
                        "correct": bool(grade["correct"]),
                        "token_count": len(result.tokens),
                        "termination": str(
                            result.receipt.decode_termination or "not_reached"
                        ),
                    }
                )
            del engine, verifier, result
            mx.synchronize()
            mx.clear_cache()

    identity = {
        "adapter_sha256": adapter_sha256,
        "execution_spec_sha256": spec.sha256,
        "task_manifest_sha256": task_manifest_sha256,
        "task_ids": [task.task_id for task in tasks],
        "depths": selected_depths,
    }
    ordinary = build_free_generation_report(
        arm="ordinary_decode",
        records=ordinary_records,
        **identity,
    )
    full_engine = build_free_generation_report(
        arm="full_engine",
        records=full_records,
        **identity,
    )
    return ordinary, full_engine


__all__ = [
    "build_behavioral_probe_report",
    "build_ordinary_decode_probe_report",
    "canonical_json_bytes",
    "free_generation_sampling_config",
    "paired_generation_seed",
    "tokenize_task",
]
