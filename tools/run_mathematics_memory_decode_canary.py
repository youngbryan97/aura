#!/usr/bin/env python3
"""Run a source-bound 1.5B free-decode canary over recurrent memory state."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.answer_contract import (  # noqa: E402
    ContractDecodeDisposition,
    contract_decode_disposition,
)
from core.brain.llm.latent_cortex.frontier_tasks import generate_task  # noqa: E402
from core.brain.llm.latent_cortex.recurrent_memory_decode_context import (  # noqa: E402
    RecurrentMemoryDecodeState,
    execute_recurrent_memory_decode_state,
    render_recurrent_memory_decode_context,
)
from core.brain.llm.unified_recurrent_transfer_decode import (  # noqa: E402
    decode_base_greedy_tokens,
)
from core.learning.recurrent_work_memory_tissue import (  # noqa: E402
    MathematicsMemoryTissue,
    load_mathematics_memory_tissue,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.rlc.mathematics_memory_decode_canary.v1"
CLAIM_BOUNDARY: Final = (
    "bounded teacher-free recurrent-state-to-free-decode transfer on the model "
    "cryptographically bound in model_identity; not open-domain, multi-domain, "
    "frontier-level, globally fusion-authorized, or WOW"
)
ARMS: Final = (
    "ordinary_base",
    "matched_wire_base",
    "treatment",
    "matched_initialization",
    "no_write",
    "no_read",
    "reset_memory",
    "matched_wrong_state",
)
SOURCE_PATHS: Final = (
    "core/brain/llm/latent_cortex/recurrent_memory_decode_context.py",
    "core/brain/llm/unified_recurrent_transfer_decode.py",
    "core/learning/recurrent_work_memory_tissue.py",
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "tools/run_mathematics_memory_decode_canary.py",
)


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


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", "-c", "core.fsmonitor=false", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


def _arm_order(task_id: str) -> tuple[str, ...]:
    offset = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _wrong_state_index(states: list[RecurrentMemoryDecodeState], index: int) -> int:
    own = (states[index].count, states[index].witness)
    for offset in range(1, len(states)):
        candidate = (index + offset) % len(states)
        if (states[candidate].count, states[candidate].witness) != own:
            return candidate
    raise RuntimeError("decode canary cannot construct a wrong-state derangement")


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
            sum(int(row["prompt_tokens"]) for row in selected) / len(selected),
            3,
        ),
        "mean_generated_tokens": round(
            sum(int(row["generated_tokens"]) for row in selected) / len(selected),
            3,
        ),
        "mean_wire_prefill_tokens": round(
            sum(int(row["wire_prefill_tokens"]) for row in selected) / len(selected),
            3,
        ),
        "mean_latency_ms": round(
            sum(int(row["latency_ms"]) for row in selected) / len(selected),
            3,
        ),
    }
    return {**body, "receipt_sha256": _sha(body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_815_23)
    parser.add_argument("--tasks-per-difficulty", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser


def _run(args: argparse.Namespace, model_path: Path) -> int:
    if not 2 <= args.tasks_per_difficulty <= 100:
        raise ValueError("decode canary task count is outside [2, 100]")
    if not 16 <= args.max_tokens <= 256:
        raise ValueError("decode canary max tokens is outside [16, 256]")
    source_commit = _git("rev-parse", "HEAD")
    source_clean = not bool(_git("status", "--porcelain", "--untracked-files=all"))
    if not source_clean:
        raise RuntimeError("decode canary requires a clean measured source")

    from mlx_lm import load

    started = time.time()
    tasks = [
        generate_task(
            "mathematics",
            seed=args.seed + difficulty * 10_000 + index,
            difficulty=difficulty,
        )
        for difficulty in (1, 2, 3)
        for index in range(args.tasks_per_difficulty)
    ]
    tissue = load_mathematics_memory_tissue()
    matched_initialization = MathematicsMemoryTissue(
        hidden_size=tissue.hidden_size,
        seed=tissue.seed,
    )
    treatment_states = [
        execute_recurrent_memory_decode_state(task.public.prompt, tissue=tissue)
        for task in tasks
    ]
    state_receipts = [state.receipt() for state in treatment_states]
    model, tokenizer = load(str(model_path))
    wire_prefill = tuple(
        int(token)
        for token in tokenizer.encode("FINAL_ANSWER: ", add_special_tokens=False)
    )
    if not wire_prefill:
        raise RuntimeError("decode canary wire prefill tokenization is empty")
    rows: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        objective = task.public.prompt
        state_by_arm = {
            "treatment": treatment_states[index],
            "matched_initialization": execute_recurrent_memory_decode_state(
                objective,
                tissue=matched_initialization,
            ),
            "no_write": execute_recurrent_memory_decode_state(
                objective,
                tissue=tissue,
                write_mode="never",
            ),
            "no_read": execute_recurrent_memory_decode_state(
                objective,
                tissue=tissue,
                read_mode="never",
            ),
            "reset_memory": execute_recurrent_memory_decode_state(
                objective,
                tissue=tissue,
                memory_mode="reset_each_step",
            ),
            "matched_wrong_state": treatment_states[
                _wrong_state_index(treatment_states, index)
            ],
        }
        for arm in _arm_order(task.task_id):
            context = "" if arm in {"ordinary_base", "matched_wire_base"} else (
                render_recurrent_memory_decode_context(state_by_arm[arm])
            )
            active_prefill = () if arm == "ordinary_base" else wire_prefill
            prompt = _prompt_tokens(tokenizer, objective, context)
            generated, stopped, latency_ms = decode_base_greedy_tokens(
                model,
                prompt,
                eos_token_id=tokenizer.eos_token_id,
                max_tokens=args.max_tokens,
                prefill_tokens=active_prefill,
                completion_check=lambda values: _complete(tokenizer, values),
            )
            text = tokenizer.decode(list(generated), skip_special_tokens=True)
            score = task.score(text)
            row = {
                "task_id": task.task_id,
                "difficulty": task.public.difficulty,
                "arm": arm,
                "correct": bool(score.correct),
                "parsed": bool(score.parsed),
                "prompt_tokens": len(prompt),
                "generated_tokens": len(generated) - len(active_prefill),
                "wire_prefill_tokens": len(active_prefill),
                "stopped": stopped,
                "latency_ms": latency_ms,
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "state_receipt_sha256": (
                    ""
                    if arm in {"ordinary_base", "matched_wire_base"}
                    else state_by_arm[arm].receipt()["receipt_sha256"]
                ),
            }
            rows.append(row)
            raw_outputs.append(
                {
                    "task_id": task.task_id,
                    "arm": arm,
                    "response": text,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "decode_complete",
                        "completed": len(rows),
                        "total": len(tasks) * len(ARMS),
                        "difficulty": task.public.difficulty,
                        "arm": arm,
                        "correct": bool(score.correct),
                        "latency_ms": latency_ms,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            try:
                import mlx.core as mx

                mx.clear_cache()
            except ImportError:  # pragma: no cover - model load already requires MLX
                pass

    arms = {arm: _summary(rows, arm) for arm in ARMS}
    base_by_task = {
        row["task_id"]: bool(row["correct"])
        for row in rows
        if row["arm"] == "ordinary_base"
    }
    treatment_by_task = {
        row["task_id"]: bool(row["correct"])
        for row in rows
        if row["arm"] == "treatment"
    }
    gain_set = sorted(
        task_id
        for task_id, correct in treatment_by_task.items()
        if correct and not base_by_task[task_id]
    )
    regressions = sorted(
        task_id
        for task_id, correct in treatment_by_task.items()
        if not correct and base_by_task[task_id]
    )
    treatment_accuracy = float(arms["treatment"]["exact_accuracy"])
    causal_controls = (
        "matched_wire_base",
        "matched_initialization",
        "no_write",
        "no_read",
        "reset_memory",
        "matched_wrong_state",
    )
    admitted = bool(
        treatment_accuracy == 1.0
        and gain_set
        and not regressions
        and all(
            float(arms[arm]["exact_accuracy"]) < treatment_accuracy
            for arm in causal_controls
        )
    )
    payload = {
        "schema": CANARY_SCHEMA,
        "source_identity": {
            "commit": source_commit,
            "measured_source_clean": source_clean,
            "source_sha256s": {
                relative: _file_sha(REPO_ROOT / relative) for relative in SOURCE_PATHS
            },
        },
        "model_identity": {
            "path": str(model_path),
            "config_sha256": _file_sha(model_path / "config.json"),
            "weights_index_sha256": _file_sha(model_path / "model.safetensors.index.json"),
        },
        "protocol": {
            "seed": args.seed,
            "tasks_per_difficulty": args.tasks_per_difficulty,
            "difficulties": [1, 2, 3],
            "task_count": len(tasks),
            "arms": list(ARMS),
            "max_tokens": args.max_tokens,
            "greedy_decode": True,
            "arm_order": "task_hash_rotated",
            "producer_available_during_decode": False,
            "verifier_available_during_decode": False,
            "answer_replacement_enabled": False,
            "matched_wire_prefill": True,
            "wire_prefill_arms": [arm for arm in ARMS if arm != "ordinary_base"],
            "ordinary_base_unconstrained": True,
            "wire_prefill_token_count": len(wire_prefill),
            "wire_prefill_sha256": _sha(list(wire_prefill)),
        },
        "task_inventory_sha256": _sha(sorted(task.task_id for task in tasks)),
        "state_inventory_sha256": _sha(state_receipts),
        "arms": arms,
        "gain_set_sha256": _sha(gain_set),
        "gain_count": len(gain_set),
        "regression_set_sha256": _sha(regressions),
        "regression_count": len(regressions),
        "rows": rows,
        "raw_outputs": raw_outputs,
        "admitted": admitted,
        "claim_boundary": CLAIM_BOUNDARY,
        "elapsed_s": round(time.time() - started, 3),
    }
    payload["receipt_sha256"] = _sha(payload)
    output = args.out.expanduser().resolve()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"admitted": admitted, "arms": arms, "receipt_sha256": payload["receipt_sha256"]}, indent=2, sort_keys=True))
    return 0 if admitted else 2


def main() -> int:
    args = _parser().parse_args()
    model_path = args.model.expanduser().resolve(strict=True)
    with standalone_model_lane(
        owner_id=f"mathematics-memory-decode:{args.out.name}",
        model_path=str(model_path),
        purpose="evaluation",
        preemptible=False,
        allow_owner_eviction=False,
        metadata={"tool": Path(__file__).name},
    ):
        return _run(args, model_path)


if __name__ == "__main__":
    raise SystemExit(main())
