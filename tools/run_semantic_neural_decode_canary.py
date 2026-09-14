#!/usr/bin/env python3
"""Run a source-bound 1.5B decode canary over neural semantic state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.answer_contract import (  # noqa: E402
    ContractDecodeDisposition,
    contract_decode_disposition,
)
from core.brain.llm.latent_cortex.semantic_neural_decode_context import (  # noqa: E402
    SemanticNeuralDecodeState,
    execute_semantic_neural_decode_state,
    normalize_semantic_neural_response,
    render_semantic_neural_answer,
    render_semantic_neural_decode_context,
    render_semantic_neural_decode_correction,
    semantic_result_matches_response,
)
from core.brain.llm.latent_cortex.semantic_surface_adapter import (  # noqa: E402
    SEMANTIC_SURFACE_PROFILES,
    execute_scientific_surface,
    render_scientific_surface,
)
from core.brain.llm.public_channel_decode import (  # noqa: E402
    PUBLIC_CHANNEL_DECODE_POLICY,
    decode_public_greedy,
)
from core.brain.llm.unified_recurrent_transfer_decode import (  # noqa: E402
    decode_base_greedy_tokens,
)
from core.learning.frontier_process_supervision import (  # noqa: E402
    frontier_process_task_battery,
)
from core.learning.public_decode_evidence import public_decode_coverage  # noqa: E402
from core.learning.semantic_neural_controls import (  # noqa: E402
    SEMANTIC_FAMILY_LESIONS,
    semantic_neural_family_lesion_machine,
)
from core.learning.semantic_neural_machine import SemanticNeuralMachine  # noqa: E402
from core.learning.semantic_task_grading import (  # noqa: E402
    LEGACY_GRADING_POLICY,
    SEMANTIC_GRADING_POLICY,
    grade_semantic_task,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.rlc.semantic_neural_decode_canary.v1"
PUBLIC_CANARY_SCHEMA: Final = "aura.rlc.semantic_neural_decode_canary.v2"
LEGACY_DECODE_POLICY: Final = "legacy_raw_text_v1"
JOURNAL_SCHEMA: Final = "aura.rlc.semantic_neural_decode_journal.v1"
LEGACY_CLAIM_BOUNDARY: Final = (
    "bounded teacher-free multi-domain neural-state-to-free-decode transfer on "
    "the model bound in model_identity; not open-domain, resident-32B, broad "
    "reasoning, fusion, frontier performance, or WOW"
)
ARMS: Final = (
    "ordinary_base",
    "matched_wire_base",
    "treatment",
    "coefficient_lesion",
    "matched_wrong_state",
)
SUPPORTED_DOMAINS: Final = (
    "coding",
    "calibration",
    "misleading_premise",
    "scientific_inference",
)
DIFFICULTIES: Final = (1, 2, 3)
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/semantic_neural_decode_context.py",
    "core/brain/llm/latent_cortex/assets/systematic_neural_alu_v1/manifest.json",
    "core/brain/llm/latent_cortex/assets/systematic_neural_alu_v1/weights.safetensors",
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/llm/latent_cortex/systematic_neural_alu.py",
    "core/brain/llm/unified_recurrent_transfer_decode.py",
    "core/learning/frontier_process_supervision.py",
    "core/learning/public_frontier_action_compiler.py",
    "core/learning/recurrent_action_schema.py",
    "core/learning/recurrent_state_schema.py",
    "core/learning/semantic_neural_controls.py",
    "core/learning/semantic_neural_machine.py",
    "tools/run_semantic_neural_decode_canary.py",
)
SURFACE_SOURCE_PATHS: Final = (
    *SOURCE_PATHS,
    "core/brain/llm/latent_cortex/semantic_surface_adapter.py",
)
PUBLIC_DECODE_SOURCE_PATHS: Final = (
    "core/brain/llm/public_channel_decode.py",
    "core/brain/llm/chat_format.py",
    "core/brain/llm/latent_cortex/answer_contract.py",
    "core/learning/public_decode_evidence.py",
    "core/learning/semantic_task_grading.py",
    "core/reasoning/asymptotic.py",
)
SURFACE_PROFILES: Final = (
    "canonical",
    "mixed_scientific_v1",
    "mixed_multidomain_v1",
)


def _claim_boundary(
    resident_manifest_identity: dict[str, Any] | None,
    surface_profile: str = "canonical",
) -> str:
    qualifier = (
        "less-constrained scientific surface transfer and "
        if surface_profile != "canonical"
        else ""
    )
    if resident_manifest_identity is not None:
        return (
            f"bounded teacher-free {qualifier}multi-domain neural-state-to-free-decode "
            "transfer on the resident model bound by model_identity and "
            "resident_manifest_identity; not open-domain, broad reasoning, "
            "fusion, frontier performance, or WOW"
        )
    if surface_profile == "canonical":
        return LEGACY_CLAIM_BOUNDARY
    return (
        "bounded teacher-free less-constrained scientific surface transfer on "
        "the model bound in model_identity; not open-domain, resident-32B, broad "
        "reasoning, fusion, frontier performance, or WOW"
    )


def _task_cohort(
    domains: tuple[str, ...],
    per_cell: int,
    *,
    seed: int,
    surface_profile: str,
) -> list[Any]:
    tasks = frontier_process_task_battery(
        domains,
        DIFFICULTIES,
        per_cell,
        seed=seed,
    )
    if surface_profile == "canonical":
        return tasks
    if surface_profile == "mixed_scientific_v1":
        if domains != ("scientific_inference",):
            raise ValueError("mixed scientific surface canary requires only scientific_inference")
    elif surface_profile == "mixed_multidomain_v1":
        if domains != SUPPORTED_DOMAINS:
            raise ValueError("mixed multidomain surface canary requires all semantic domains")
    else:
        raise ValueError("mixed semantic surface profile is unsupported")
    adapted = []
    surface_index = 0
    for index, task in enumerate(tasks):
        if task.family == "frontier_scientific_inference":
            profile = SEMANTIC_SURFACE_PROFILES[
                surface_index % len(SEMANTIC_SURFACE_PROFILES)
            ]
            surface_index += 1
            prompt = render_scientific_surface(
                task.prompt,
                profile=profile,
                permutation_seed=seed + index,
            )
            task = replace(
                task,
                prompt=prompt,
                transition_trace=None,
                transition_program=None,
            )
        adapted.append(task)
    return adapted


def _execute_task_state(
    task: Any,
    *,
    surface_profile: str,
    machine: SemanticNeuralMachine | None = None,
) -> tuple[SemanticNeuralDecodeState, str]:
    if surface_profile == "canonical" or task.family != "frontier_scientific_inference":
        state = execute_semantic_neural_decode_state(task.prompt, task.family, machine=machine)
        return state, ""
    decoded = execute_scientific_surface(task.prompt, machine=machine)
    return decoded.state, decoded.receipt()["receipt_sha256"]


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resident_manifest_identity(path: Path, model_path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("resident model manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("resident model manifest is not an object")
    active_raw = str(manifest.get("active_model_path") or "").strip()
    if not active_raw:
        raise ValueError("resident model manifest has no active model")
    active_path = Path(active_raw).expanduser().resolve(strict=True)
    if active_path != model_path:
        raise ValueError("resident model manifest does not select measured model")
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version < 1:
        raise ValueError("resident model manifest schema is invalid")
    return {
        "path": str(manifest_path),
        "sha256": _file_sha(manifest_path),
        "active_model_path": str(active_path),
        "schema_version": schema_version,
        "base_model": str(manifest.get("base_model") or ""),
        "tag": str(manifest.get("tag") or ""),
        "fused_at": manifest.get("fused_at"),
    }


def _append_journal_event(
    path: Path,
    event: dict[str, Any],
    *,
    previous_receipt_sha256: str,
) -> str:
    body = {
        "schema": JOURNAL_SCHEMA,
        "previous_receipt_sha256": previous_receipt_sha256,
        **event,
    }
    receipt = _sha(body)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**body, "receipt_sha256": receipt}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve_source_commit(explicit: str) -> str:
    source_commit = explicit.strip().lower()
    if source_commit:
        if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
            raise ValueError("semantic decode source commit is not a full Git identity")
        return source_commit
    source_commit = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("semantic decode canary requires clean measured source")
    return source_commit


def _prompt_tokens(tokenizer: Any, objective: str, context: str = "") -> tuple[int, ...]:
    content = objective if not context else f"{objective}\n\n{context}"
    return tuple(
        int(token)
        for token in tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )


def _complete(tokenizer: Any, token_ids: tuple[int, ...]) -> bool:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return contract_decode_disposition(text) in {
        ContractDecodeDisposition.COMPLETE,
        ContractDecodeDisposition.INVALID,
    }


def _decode_attempt(model, tokenizer, prompt, *, prefill, max_tokens, policy, progress=None):
    if policy == PUBLIC_CHANNEL_DECODE_POLICY:
        result = decode_public_greedy(
            model, tokenizer, prompt, max_tokens=max_tokens, public_prefill=prefill,
            completion_check=lambda text: contract_decode_disposition(text) in {
                ContractDecodeDisposition.COMPLETE, ContractDecodeDisposition.INVALID,
            },
            progress=progress,
        )
        return result.text, result.stopped, result.generated_tokens, result.latency_ms, result.receipt()
    if policy != LEGACY_DECODE_POLICY:
        raise ValueError("unknown semantic decode policy")
    generated, stopped, latency = decode_base_greedy_tokens(
        model, prompt, eos_token_id=tokenizer.eos_token_id, max_tokens=max_tokens,
        prefill_tokens=prefill, completion_check=lambda values: _complete(tokenizer, values),
        progress=progress,
    )
    return (tokenizer.decode(list(generated), skip_special_tokens=True), stopped,
            len(generated) - len(prefill), latency, None)


def _wire_prefill(tokenizer: Any, family: str) -> tuple[int, ...]:
    prefixes = {
        "frontier_coding": 'FINAL_ANSWER: {"returns":',
        "frontier_calibration": 'FINAL_ANSWER: {"choice":',
        "frontier_misleading_premise": 'FINAL_ANSWER: {"actual_score":',
        "frontier_scientific_inference": 'FINAL_ANSWER: {"downstream":',
    }
    text = prefixes.get(family)
    if text is None:
        raise ValueError("semantic decode family has no syntax-only prefill")
    values = tuple(int(token) for token in tokenizer.encode(text, add_special_tokens=False))
    if not values:
        raise RuntimeError("semantic decode syntax prefill tokenization is empty")
    return values


def _state_prefill(
    tokenizer: Any,
    state: SemanticNeuralDecodeState,
    *,
    limit: int = 80,
) -> tuple[int, ...]:
    """Seed ordering from authenticated state while leaving a decoded suffix."""

    values = tuple(
        int(token)
        for token in tokenizer.encode(
            render_semantic_neural_answer(state),
            add_special_tokens=False,
        )
    )
    if len(values) < 2:
        raise RuntimeError("semantic state prefill cannot leave a decoded suffix")
    return values[: min(limit, len(values) - 1)]


def _arm_order(task_id: str) -> tuple[str, ...]:
    offset = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _wrong_state_index(states: list[SemanticNeuralDecodeState], index: int) -> int:
    own = states[index].semantic_result
    for offset in range(1, len(states)):
        candidate = (index + offset) % len(states)
        if states[candidate].family == states[index].family and (
            states[candidate].semantic_result != own
        ):
            return candidate
    raise RuntimeError("semantic decode canary cannot construct a same-family derangement")


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    exact = sum(bool(row["correct"]) for row in selected)
    parsed = sum(bool(row["parsed"]) for row in selected)
    body = {
        "examples": len(selected),
        "exact": exact,
        "parsed": parsed,
        "exact_accuracy": round(exact / len(selected), 6),
        "parsed_accuracy": round(parsed / len(selected), 6),
        "mean_prompt_tokens": round(
            sum(int(row["prompt_tokens"]) for row in selected) / len(selected), 3
        ),
        "mean_generated_tokens": round(
            sum(int(row["generated_tokens"]) for row in selected) / len(selected), 3
        ),
        "mean_latency_ms": round(
            sum(int(row["latency_ms"]) for row in selected) / len(selected), 3
        ),
    }
    return {**body, "receipt_sha256": _sha(body)}


def _grade(task: Any, response: str, *, policy: str = LEGACY_GRADING_POLICY) -> tuple[bool, bool]:
    verdict = grade_semantic_task(task, response, policy=policy)
    if not isinstance(verdict, dict) or type(verdict.get("correct")) is not bool:
        raise RuntimeError("semantic decode grader returned an invalid verdict")
    return bool(verdict["correct"]), verdict.get("parsed") is not None


def _lesion_machine(family: str) -> SemanticNeuralMachine:
    return semantic_neural_family_lesion_machine(family)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--resident-manifest", type=Path)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=SUPPORTED_DOMAINS,
        default=SUPPORTED_DOMAINS[:3],
    )
    parser.add_argument("--seed", type=int, default=20_260_815_48)
    parser.add_argument("--tasks-per-difficulty", type=int, default=3)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--decode-policy", choices=(PUBLIC_CHANNEL_DECODE_POLICY, LEGACY_DECODE_POLICY),
                        default=PUBLIC_CHANNEL_DECODE_POLICY)
    parser.add_argument("--surface-profile", choices=SURFACE_PROFILES, default="canonical")
    parser.add_argument(
        "--source-commit",
        default="",
        help=(
            "bind a preverified Git commit when a contained supervisor forbids "
            "child process creation"
        ),
    )
    return parser


def _lane_kwargs(model_path: Path, output: Path) -> dict[str, Any]:
    return {
        "owner_id": f"semantic-neural-decode:{output.name}",
        "model_path": str(model_path),
        "purpose": "evaluation",
        "preemptible": False,
        "allow_owner_eviction": False,
        "metadata": {"tool": Path(__file__).name},
    }


def _run(args: argparse.Namespace, model_path: Path) -> int:
    if not 2 <= args.tasks_per_difficulty <= 20:
        raise ValueError("semantic decode task count is outside [2, 20]")
    public_decode = args.decode_policy == PUBLIC_CHANNEL_DECODE_POLICY
    grading_policy = SEMANTIC_GRADING_POLICY if public_decode else LEGACY_GRADING_POLICY
    if args.max_tokens is None:
        args.max_tokens = 4096 if public_decode else 384
    ceiling = 8192 if public_decode else 384
    if not 32 <= args.max_tokens <= ceiling:
        raise ValueError(f"semantic decode token budget is outside [32, {ceiling}]")
    domains = tuple(args.domains)
    if not domains or len(domains) != len(set(domains)):
        raise ValueError("semantic decode domains must be a non-empty unique sequence")
    resident_manifest_identity = (
        _resident_manifest_identity(args.resident_manifest, model_path)
        if args.resident_manifest is not None
        else None
    )
    source_commit = _resolve_source_commit(args.source_commit)

    from mlx_lm import load

    started = time.time()
    tasks = _task_cohort(
        domains,
        args.tasks_per_difficulty,
        seed=args.seed,
        surface_profile=args.surface_profile,
    )
    treatment_pairs = [
        _execute_task_state(task, surface_profile=args.surface_profile) for task in tasks
    ]
    treatment_states = [pair[0] for pair in treatment_pairs]
    surface_receipts = [pair[1] for pair in treatment_pairs]
    journal_path = (
        args.journal.expanduser().resolve()
        if args.journal is not None
        else args.out.with_name(f"{args.out.name}.journal.jsonl")
    )
    if journal_path.exists():
        raise RuntimeError("semantic decode journal already exists")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.touch(mode=0o600, exist_ok=False)
    journal_receipt = _append_journal_event(
        journal_path,
        {
            "event": "campaign_started",
            "source_commit": source_commit,
            "seed": args.seed,
            "domains": domains,
            "difficulties": DIFFICULTIES,
            "tasks_per_difficulty": args.tasks_per_difficulty,
            "task_count": len(tasks),
            "arm_count": len(ARMS),
            "surface_profile": args.surface_profile,
            "resident_manifest_identity": resident_manifest_identity,
            **({"decode_policy": args.decode_policy, "grading_policy": grading_policy,
                "max_tokens": args.max_tokens} if public_decode else {}),
        },
        previous_receipt_sha256="0" * 64,
    )
    lesion_states: list[SemanticNeuralDecodeState | None] = []
    for task in tasks:
        try:
            lesion_states.append(
                _execute_task_state(
                    task,
                    surface_profile=args.surface_profile,
                    machine=_lesion_machine(task.family),
                )[0]
            )
        except (RuntimeError, ValueError):
            lesion_states.append(None)

    model, tokenizer = load(str(model_path))
    rows: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        wrong = treatment_states[_wrong_state_index(treatment_states, index)]
        state_by_arm = {
            "treatment": treatment_states[index],
            "coefficient_lesion": lesion_states[index],
            "matched_wrong_state": wrong,
        }
        for arm in _arm_order(task.task_id):
            selected = state_by_arm.get(arm)
            context = ""
            if selected is not None:
                context = render_semantic_neural_decode_context(selected)
            elif arm == "coefficient_lesion":
                context = (
                    "Internal recurrent semantic computation did not produce an "
                    "admissible terminal state after the declared tissue lesion."
                )
            wire_prefill = () if arm == "ordinary_base" else _wire_prefill(tokenizer, task.family)
            attempts: list[dict[str, Any]] = []
            max_attempts = 3 if selected is not None else 1
            text = ""
            stopped = False
            prompt_tokens = 0
            generated_tokens = 0
            latency_ms = 0
            serialization_verified: bool | None = None
            for attempt_index in range(max_attempts):
                active_prefill = (
                    _state_prefill(tokenizer, selected)
                    if attempt_index == 2 and selected is not None
                    else wire_prefill
                )
                active_context = (
                    context
                    if attempt_index == 0 or selected is None
                    else render_semantic_neural_decode_correction(selected)
                )
                prompt = _prompt_tokens(tokenizer, task.prompt, active_context)
                def progress(count, task_id=task.task_id, arm_name=arm, attempt_number=attempt_index + 1):
                    if count % 256 == 0:
                        print(json.dumps({"event": "decode_progress", "task_id": task_id,
                                          "arm": arm_name, "attempt": attempt_number,
                                          "generated_tokens": count}), flush=True)

                text, stopped, sampled, attempt_latency_ms, decode_receipt = _decode_attempt(
                    model, tokenizer, prompt, max_tokens=args.max_tokens,
                    prefill=active_prefill, policy=args.decode_policy, progress=progress,
                )
                raw_text = text
                wire_normalized = False
                if selected is not None:
                    text, wire_normalized = normalize_semantic_neural_response(
                        selected,
                        text,
                    )
                prompt_tokens += len(prompt)
                generated_tokens += sampled
                latency_ms += attempt_latency_ms
                serialization_verified = (
                    semantic_result_matches_response(selected, text)
                    if selected is not None
                    else None
                )
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "raw_response": raw_text,
                        "raw_response_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                        "response": text,
                        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "prefill_tokens": len(active_prefill),
                        "serialization_verified": serialization_verified,
                        "wire_normalized": wire_normalized,
                        **({"decode": decode_receipt, "prompt_tokens": len(prompt)}
                           if public_decode else {}),
                    }
                )
                if selected is None or serialization_verified:
                    break
            correct, parsed = _grade(task, text, policy=grading_policy)
            row = {
                "task_id": task.task_id,
                "family": task.family,
                "program_depth": task.depth,
                "arm": arm,
                "correct": correct,
                "parsed": parsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "stopped": stopped,
                "latency_ms": latency_ms,
                "decode_attempts": len(attempts),
                "serialization_verified": serialization_verified,
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "state_receipt_sha256": (
                    "" if selected is None else selected.receipt()["receipt_sha256"]
                ),
            }
            rows.append(row)
            raw_output = {
                "task_id": task.task_id,
                "arm": arm,
                "response": text,
                "attempts": attempts,
            }
            raw_outputs.append(raw_output)
            journal_receipt = _append_journal_event(
                journal_path,
                {
                    "event": "decode_committed",
                    "completed": len(rows),
                    "total": len(tasks) * len(ARMS),
                    "row": row,
                    "raw_output": raw_output,
                },
                previous_receipt_sha256=journal_receipt,
            )
            print(
                json.dumps(
                    {
                        "event": "decode_complete",
                        "completed": len(rows),
                        "total": len(tasks) * len(ARMS),
                        "family": task.family,
                        "arm": arm,
                        "correct": correct,
                        "latency_ms": latency_ms,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            try:
                import mlx.core as mx

                mx.clear_cache()
            except ImportError:  # pragma: no cover
                pass

    arms = {arm: _summary(rows, arm) for arm in ARMS}
    base = {row["task_id"]: bool(row["correct"]) for row in rows if row["arm"] == "ordinary_base"}
    treatment = {row["task_id"]: bool(row["correct"]) for row in rows if row["arm"] == "treatment"}
    gain_set = sorted(key for key, value in treatment.items() if value and not base[key])
    regressions = sorted(key for key, value in treatment.items() if not value and base[key])
    treatment_accuracy = float(arms["treatment"]["exact_accuracy"])
    coverage = public_decode_coverage(raw_outputs, max_tokens=args.max_tokens) if public_decode else None
    admitted = bool(
        treatment_accuracy == 1.0
        and (coverage is None or coverage["uncensored"])
        and gain_set
        and not regressions
        and all(
            float(arms[arm]["exact_accuracy"]) < treatment_accuracy
            for arm in ("matched_wire_base", "coefficient_lesion", "matched_wrong_state")
        )
    )
    payload = {
        "schema": PUBLIC_CANARY_SCHEMA if public_decode else CANARY_SCHEMA,
        "source_commit": source_commit,
        "source_sha256s": {
            path: _file_sha(REPO_ROOT / path)
            for path in (*(
                SURFACE_SOURCE_PATHS if args.surface_profile != "canonical" else SOURCE_PATHS
            ), *(PUBLIC_DECODE_SOURCE_PATHS if public_decode else ()))
        },
        "model_identity": {
            "path": str(model_path),
            "config_sha256": _file_sha(model_path / "config.json"),
            "weights_index_sha256": _file_sha(model_path / "model.safetensors.index.json"),
        },
        "resident_manifest_identity": resident_manifest_identity,
        "seed": args.seed,
        "domains": domains,
        "difficulties": DIFFICULTIES,
        "tasks_per_difficulty": args.tasks_per_difficulty,
        "surface_profile": args.surface_profile,
        "task_count": len(tasks),
        "arms": arms,
        "gain_set_sha256": _sha(gain_set),
        "gain_count": len(gain_set),
        "regression_set_sha256": _sha(regressions),
        "regression_count": len(regressions),
        "coefficient_lesion_contract": {
            f"frontier_{domain}": SEMANTIC_FAMILY_LESIONS[f"frontier_{domain}"]
            for domain in domains
        },
        "treatment_state_receipt_sha256s": [
            state.receipt()["receipt_sha256"] for state in treatment_states
        ],
        "surface_adapter_receipt_sha256s": surface_receipts,
        "admitted": admitted,
        "claim_boundary": _claim_boundary(
            resident_manifest_identity,
            args.surface_profile,
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "journal_path": str(journal_path),
        "journal_last_decode_receipt_sha256": journal_receipt,
        "raw_outputs": raw_outputs,
        **({"decode_policy": args.decode_policy, "grading_policy": grading_policy,
            "max_tokens": args.max_tokens, "decode_coverage": coverage, "rows": rows}
           if public_decode else {}),
    }
    payload["receipt_sha256"] = _sha(payload)
    atomic_write_text(args.out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _append_journal_event(
        journal_path,
        {
            "event": "campaign_completed",
            "admitted": admitted,
            "report_receipt_sha256": payload["receipt_sha256"],
        },
        previous_receipt_sha256=journal_receipt,
    )
    print(json.dumps({"event": "canary_complete", "admitted": admitted}, sort_keys=True))
    return 0 if admitted else 2


def main() -> int:
    args = _parser().parse_args()
    model_path = args.model.expanduser().resolve(strict=True)
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args.out = output
    with standalone_model_lane(**_lane_kwargs(model_path, output)):
        return _run(args, model_path)


if __name__ == "__main__":
    raise SystemExit(main())
