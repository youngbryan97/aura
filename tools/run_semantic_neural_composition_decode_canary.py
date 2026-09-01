#!/usr/bin/env python3
"""Run a matched resident-model decode canary over composed neural state."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.chat_format import render_chat_template  # noqa: E402
from core.brain.llm.latent_cortex.semantic_neural_composition_decode import (  # noqa: E402
    SemanticNeuralCompositionDecodeState,
    execute_composition_decode_state,
    parse_composition_response,
    render_composition_decode_objective,
    render_composition_state_channel,
)
from core.brain.llm.unified_recurrent_transfer_decode import (  # noqa: E402
    decode_base_greedy_tokens,
)
from core.learning.semantic_neural_composition import (  # noqa: E402
    render_public_typed_workflow,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402
from tools.run_semantic_neural_composition_canary import (  # noqa: E402
    DEFAULT_SEED,
    _lesion,
    _reference,
    _task_document,
)
from tools.verify_semantic_neural_composition_canary import (  # noqa: E402
    verify as verify_composition_basis,
)

SCHEMA: Final = "aura.rlc.semantic_neural_composition_decode_canary.v1"
ARMS: Final = (
    "ordinary_base",
    "matched_wire_base",
    "treatment",
    "additive_lesion",
    "multiplicative_lesion",
    "matched_wrong_state",
)
SOURCE_PATHS: Final = (
    "core/brain/llm/chat_format.py",
    "core/brain/llm/latent_cortex/assets/systematic_neural_alu_v1/manifest.json",
    "core/brain/llm/latent_cortex/assets/systematic_neural_alu_v1/weights.safetensors",
    "core/brain/llm/latent_cortex/semantic_neural_composition_decode.py",
    "core/brain/llm/latent_cortex/systematic_neural_alu.py",
    "core/brain/llm/unified_recurrent_transfer_decode.py",
    "core/learning/semantic_neural_composition.py",
    "core/learning/semantic_neural_machine.py",
    "tools/run_semantic_neural_composition_canary.py",
    "tools/verify_semantic_neural_composition_canary.py",
    "tools/run_semantic_neural_composition_decode_canary.py",
    "tools/verify_semantic_neural_composition_decode_canary.py",
)
CLAIM_BOUNDARY: Final = (
    "resident-model serialization of fresh family-neutral typed-operation "
    "composition from authenticated learned tissue under matched causal controls; "
    "not hidden-state internalization, open-domain reasoning gain, unrestricted "
    "serving, static fusion, or frontier performance"
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
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_commit(explicit: str) -> str:
    value = explicit.strip().lower()
    if value:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("composition decode source commit is invalid")
        return value
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("composition decode canary requires clean measured source")
    return _git("rev-parse", "HEAD")


def _resident_manifest(path: Path, model_path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    active = Path(str(manifest["active_model_path"])).expanduser().resolve(strict=True)
    if active != model_path or type(manifest.get("schema_version")) is not int:
        raise ValueError("resident model manifest does not select measured model")
    return {
        "path": str(resolved),
        "sha256": _file_sha(resolved),
        "active_model_path": str(active),
        "schema_version": manifest["schema_version"],
        "base_model": str(manifest.get("base_model") or ""),
        "tag": str(manifest.get("tag") or ""),
        "fused_at": manifest.get("fused_at"),
    }


def _composition_basis(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve(strict=True)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    stored = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    replayed = verify_composition_basis(result)
    if stored != replayed or replayed.get("verified") is not True:
        raise RuntimeError("composition basis is not independently verified")
    return {
        "path": str(root),
        "result_receipt_sha256": result["receipt_sha256"],
        "verification_receipt_sha256": stored["verification_receipt_sha256"],
        "task_set_sha256": stored["task_set_sha256"],
        "task_count": stored["task_count"],
    }


def _cohort(seed: int, task_count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [_task_document(rng) for _index in range(task_count)]


def _arm_order(task_id: str) -> tuple[str, ...]:
    offset = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _state_arms(
    document: dict[str, Any],
) -> dict[str, SemanticNeuralCompositionDecodeState | None]:
    workflow = render_public_typed_workflow(document)

    def lesion_state(operation: int, coefficient: int):
        try:
            return execute_composition_decode_state(
                workflow,
                machine=_lesion(operation, coefficient),
            )
        except (RuntimeError, ValueError):
            return None

    return {
        "treatment": execute_composition_decode_state(workflow),
        "additive_lesion": lesion_state(0, 1),
        "multiplicative_lesion": lesion_state(1, 2),
    }


def _wrong_state_index(
    states: list[dict[str, SemanticNeuralCompositionDecodeState | None]],
    index: int,
) -> int:
    own = states[index]["treatment"]
    assert own is not None
    for offset in range(1, len(states)):
        candidate = (index + offset) % len(states)
        other = states[candidate]["treatment"]
        if other is not None and other.semantic_result != own.semantic_result:
            return candidate
    raise RuntimeError("composition decode canary cannot construct a state derangement")


def _prompt_tokens(tokenizer: Any, objective: str, channel: str = "") -> tuple[int, ...]:
    content = objective if not channel else f"{objective}\n\n{channel}"
    rendered = render_chat_template(
        tokenizer,
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tuple(
        int(token)
        for token in tokenizer.encode(rendered, add_special_tokens=False)
    )


def _wire_prefill(tokenizer: Any) -> tuple[int, ...]:
    values = tuple(
        int(token)
        for token in tokenizer.encode("FINAL_ANSWER:{", add_special_tokens=False)
    )
    if not values:
        raise RuntimeError("composition syntax prefill is empty")
    return values


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    body = {
        "examples": len(selected),
        "exact": sum(bool(row["correct"]) for row in selected),
        "parsed": sum(bool(row["parsed"]) for row in selected),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resident-manifest", type=Path, required=True)
    parser.add_argument("--composition-basis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED + 100)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--source-commit", default="")
    return parser


def _run(args: argparse.Namespace, model_path: Path) -> int:
    if not 2 <= args.tasks <= 24 or not 32 <= args.max_tokens <= 192:
        raise ValueError("composition decode canary dimensions are invalid")
    commit = _source_commit(args.source_commit)
    manifest = _resident_manifest(args.resident_manifest, model_path)
    basis = _composition_basis(args.composition_basis)
    documents = _cohort(args.seed, args.tasks)
    workflows = [render_public_typed_workflow(document) for document in documents]
    states = [_state_arms(document) for document in documents]
    expected = [_reference(document) for document in documents]

    from mlx_lm import load

    started = time.time()
    model, tokenizer = load(str(model_path))
    wire = _wire_prefill(tokenizer)
    rows: list[dict[str, Any]] = []
    for index, (document, workflow, task_states, answer) in enumerate(
        zip(documents, workflows, states, expected, strict=True)
    ):
        task_id = hashlib.sha256(workflow.encode()).hexdigest()
        report = tuple(document["report"])
        objective = render_composition_decode_objective(workflow)
        for arm in _arm_order(task_id):
            selected = (
                states[_wrong_state_index(states, index)]["treatment"]
                if arm == "matched_wrong_state"
                else task_states.get(arm)
            )
            channel = "" if selected is None else render_composition_state_channel(selected)
            prompt = _prompt_tokens(tokenizer, objective, channel)
            prefill = () if arm == "ordinary_base" else wire
            generated, stopped, latency_ms = decode_base_greedy_tokens(
                model,
                prompt,
                eos_token_id=tokenizer.eos_token_id,
                max_tokens=args.max_tokens,
                prefill_tokens=prefill,
                completion_check=lambda values, report=report: parse_composition_response(
                    tokenizer.decode(list(values), skip_special_tokens=True), report
                )
                is not None,
            )
            response = tokenizer.decode(list(generated), skip_special_tokens=True)
            parsed = parse_composition_response(response, report)
            row = {
                "ordinal": index,
                "task_id": task_id,
                "arm": arm,
                "correct": parsed == answer,
                "parsed": parsed is not None,
                "response": response,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "prompt_tokens": len(prompt),
                "generated_tokens": len(generated) - len(prefill),
                "prefill_tokens": len(prefill),
                "stopped": stopped,
                "latency_ms": latency_ms,
                "state_receipt_sha256": (
                    "" if selected is None else selected.receipt()["receipt_sha256"]
                ),
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "decode_complete",
                        "completed": len(rows),
                        "total": args.tasks * len(ARMS),
                        "ordinal": index,
                        "arm": arm,
                        "correct": row["correct"],
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
    by_arm = {
        arm: {row["task_id"]: bool(row["correct"]) for row in rows if row["arm"] == arm}
        for arm in ARMS
    }
    gain_set = sorted(
        task_id
        for task_id, correct in by_arm["treatment"].items()
        if correct and not by_arm["ordinary_base"][task_id]
    )
    regressions = sorted(
        task_id
        for task_id, correct in by_arm["ordinary_base"].items()
        if correct and not by_arm["treatment"][task_id]
    )
    treatment_exact = arms["treatment"]["exact"]
    admitted = bool(
        treatment_exact == args.tasks
        and gain_set
        and not regressions
        and all(
            arms[arm]["exact"] < treatment_exact
            for arm in (
                "matched_wire_base",
                "additive_lesion",
                "multiplicative_lesion",
                "matched_wrong_state",
            )
        )
    )
    body = {
        "schema": SCHEMA,
        "source_commit": commit,
        "source_sha256s": {path: _file_sha(REPO_ROOT / path) for path in SOURCE_PATHS},
        "model_identity": {
            "path": str(model_path),
            "config_sha256": _file_sha(model_path / "config.json"),
            "weights_index_sha256": _file_sha(model_path / "model.safetensors.index.json"),
        },
        "resident_manifest_identity": manifest,
        "composition_basis": basis,
        "seed": args.seed,
        "task_count": args.tasks,
        "max_tokens": args.max_tokens,
        "decode_calls_per_arm_per_task": 1,
        "arm_order": "task_hash_rotated",
        "arms": arms,
        "gain_set_sha256": _sha(gain_set),
        "gain_count": len(gain_set),
        "regression_set_sha256": _sha(regressions),
        "regression_count": len(regressions),
        "rows": rows,
        "admitted": admitted,
        "claim_boundary": CLAIM_BOUNDARY,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    payload = {**body, "receipt_sha256": _sha(body)}
    atomic_write_text(args.out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "canary_complete", "admitted": admitted}, sort_keys=True))
    return 0 if admitted else 2


def main() -> int:
    args = _parser().parse_args()
    model_path = args.model.expanduser().resolve(strict=True)
    args.out = args.out.expanduser().resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with standalone_model_lane(
        owner_id=f"semantic-composition-decode:{args.out.name}",
        model_path=str(model_path),
        purpose="evaluation",
        preemptible=False,
        allow_owner_eviction=False,
        metadata={"tool": Path(__file__).name},
    ):
        return _run(args, model_path)


if __name__ == "__main__":
    raise SystemExit(main())
