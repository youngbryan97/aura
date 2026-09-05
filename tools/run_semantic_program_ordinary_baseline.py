#!/usr/bin/env python3
"""Run the deferred ordinary fused-27B control on the frozen-path cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path, *, max_bytes: int) -> dict[str, Any]:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    value = json.loads(payload.decode("ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--mechanism-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--canary-tasks", type=int, default=3)
    return parser


def _decode(
    model: Any,
    tokenizer: Any,
    source_text: str,
    *,
    max_tokens: int,
    prefill_chunk_tokens: int,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import stream_generate

    from core.brain.llm.chat_format import (
        render_chat_template,
        split_native_thinking_generation,
    )

    prompt = render_chat_template(
        tokenizer,
        [{"role": "user", "content": source_text}],
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt_tokens = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
    pieces: list[str] = []
    generated_tokens = 0
    finish_reason = "token_limit"
    final_seen = False
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        prefill_step_size=prefill_chunk_tokens,
    ):
        final_seen = True
        pieces.append(str(response.text or ""))
        if response.finish_reason != "stop":
            generated_tokens += 1
        if response.finish_reason:
            finish_reason = str(response.finish_reason)
    if not final_seen:
        raise RuntimeError("ordinary decode produced no generation frames")
    raw = "".join(pieces)
    channels = split_native_thinking_generation(raw, native_thinking=True)
    termination = (
        finish_reason
        if channels.boundary_closed
        else "native_thinking_incomplete"
    )
    return {
        "response_text": channels.surface,
        "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "prompt_tokens_sha256": hashlib.sha256(
            json.dumps(prompt_tokens, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "prompt_token_count": len(prompt_tokens),
        "generated_token_count": generated_tokens,
        "termination": termination,
        "native_thinking_boundary_closed": channels.boundary_closed,
    }


def _run(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from mlx_lm import load

    from core.brain.llm.model_artifact_profile import validate_model_artifact_descriptor
    from core.learning.semantic_program_ordinary_baseline import (
        ORDINARY_BASELINE_RESULT_SCHEMA,
        adjudicate_ordinary_product_bar,
        canonical_bytes,
        canonical_sha256,
        ordinary_result_row,
        product_bar_is_reachable,
        source_identity,
        verify_ordinary_baseline_preflight,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from core.runtime.model_lane_control import standalone_model_lane

    if not 1 <= args.canary_tasks <= 8 or not 64 <= args.prefill_chunk_tokens <= 4096:
        raise ValueError("ordinary baseline canary or prefill bound is invalid")
    preregistration = _load(args.preregistration, max_bytes=1024 * 1024)
    mechanism = _load(args.mechanism_result, max_bytes=32 * 1024 * 1024)
    descriptor = _load(args.descriptor, max_bytes=32 * 1024 * 1024)
    model_path = Path(str(descriptor.get("canonical_path") or "")).resolve(strict=True)
    validate_model_artifact_descriptor(
        descriptor,
        model_path=model_path,
        verify_full_hash=True,
    )
    examples, treatment_rows = verify_ordinary_baseline_preflight(
        preregistration=preregistration,
        mechanism_result=mechanism,
        descriptor=descriptor,
        max_tokens=args.max_tokens,
    )
    output = args.output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError("ordinary baseline output already exists")

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    aborted_reason = ""
    with standalone_model_lane(
        owner_id=f"semantic-program-ordinary:{output.name}",
        model_path=str(model_path),
        purpose="evaluation",
        preemptible=False,
        require_exclusive=True,
        allow_owner_eviction=True,
        metadata={"tool": Path(__file__).name},
    ):
        model, tokenizer = load(str(model_path))
        try:
            for index, example in enumerate(examples, 1):
                decoded = _decode(
                    model,
                    tokenizer,
                    example.source_text,
                    max_tokens=args.max_tokens,
                    prefill_chunk_tokens=args.prefill_chunk_tokens,
                )
                row = ordinary_result_row(
                    example,
                    **decoded,
                    model_descriptor_sha256=str(descriptor["descriptor_sha256"]),
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "progress": f"{index}/{len(examples)}",
                            "ordinary_exact": sum(item["answer_exact"] for item in rows),
                            "parsed": row["parsed_integer"] is not None,
                            "termination": row["termination"],
                            "tokens": row["generated_token_count"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if index == args.canary_tasks and not any(
                    item["native_thinking_boundary_closed"] for item in rows
                ):
                    aborted_reason = "canary_native_thinking_never_reached_public_surface"
                    break
                if not product_bar_is_reachable(treatment_rows, rows):
                    aborted_reason = "product_bar_mathematically_unreachable"
                    break
        finally:
            del model, tokenizer
            mx.synchronize()
            mx.clear_cache()

    complete = len(rows) == len(examples)
    adjudication = (
        adjudicate_ordinary_product_bar(treatment_rows, rows) if complete else None
    )
    body = {
        "schema": ORDINARY_BASELINE_RESULT_SCHEMA,
        "claim_boundary": preregistration["claim_boundary"],
        "preregistration_sha256": canonical_sha256(preregistration),
        "mechanism_result_sha256": mechanism["result_sha256"],
        "model_descriptor_sha256": descriptor["descriptor_sha256"],
        "model_path": str(model_path),
        "decode_policy": {
            "prompt": "native_chat_template_single_user_task_text_only",
            "sampling": "greedy_argmax",
            "native_thinking": True,
            "rlc": False,
            "steering": False,
            "max_tokens": args.max_tokens,
            "prefill_chunk_tokens": args.prefill_chunk_tokens,
            "answer_parser": "final_answer_shaped_exact_integral_numeric_value",
            "private_reasoning_retained": False,
        },
        "source_identity": source_identity(),
        "rows": rows,
        "completed_tasks": len(rows),
        "complete": complete,
        "aborted_reason": aborted_reason,
        "adjudication": adjudication,
        "verdict": (
            "ABORTED_FUTILITY"
            if aborted_reason
            else (
                "PASS_PREREGISTERED_PRODUCT_BAR"
                if adjudication and adjudication["product_bar_pass"]
                else "FAIL_PREREGISTERED_PRODUCT_BAR"
            )
        ),
        "wall_time_s": time.monotonic() - started,
        "serving_authority": False,
    }
    report = {**body, "result_sha256": canonical_sha256(body)}
    if not atomic_write_bytes_if_absent(
        output,
        canonical_bytes(report) + b"\n",
        mode=0o400,
    ):
        raise FileExistsError("ordinary baseline output already exists")
    print(
        json.dumps(
            {
                "complete": complete,
                "output": str(output),
                "result_sha256": report["result_sha256"],
                "verdict": report["verdict"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0 if report["verdict"].startswith("PASS_") else 2


def main() -> int:
    args = _parser().parse_args()
    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {"complete": False, "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
