#!/usr/bin/env python3
"""Acquire answer-blind semantic-program features from one resident model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--examples-per-operation-pair", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--hidden-timeout-s", type=float, default=120.0)
    parser.add_argument("--idle-wait-s", type=float, default=0.0)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    from mlx_lm.utils import load_tokenizer

    from core.brain.llm.mlx_client import get_mlx_client
    from core.learning.semantic_program_corpus import build_semantic_program_corpus
    from core.learning.semantic_program_feature_materialization import (
        SemanticFeatureConfig,
        materialize_semantic_program_features,
        offset_tokenizer_for_worker,
        tokenizer_checkpoint_identity,
    )

    model = args.model.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if not model.is_dir() or model.is_symlink():
        raise RuntimeError("semantic feature model must be a real local directory")
    config = SemanticFeatureConfig(
        seed=args.seed,
        examples_per_operation_pair=args.examples_per_operation_pair,
        max_examples=args.max_examples,
        hidden_timeout_s=args.hidden_timeout_s,
        idle_wait_s=args.idle_wait_s,
    )
    corpus = build_semantic_program_corpus(
        seed=config.seed,
        examples_per_operation_pair=config.examples_per_operation_pair,
    )
    tokenizer_identity = await asyncio.to_thread(tokenizer_checkpoint_identity, model)
    tokenizer_wrapper = await asyncio.to_thread(load_tokenizer, model)
    offset_tokenizer = offset_tokenizer_for_worker(tokenizer_wrapper)
    client = get_mlx_client(str(model))
    primary_error: BaseException | None = None
    try:
        ready = await client.warmup(
            foreground_request=True,
            skip_swap_cooldown=True,
        )
        if not ready:
            raise RuntimeError("resident worker did not become ready for feature acquisition")
        ownership = client.get_model_lane_ownership_snapshot()
        if not ownership:
            raise RuntimeError("resident worker has no exclusive model-lane receipt")
        result = await materialize_semantic_program_features(
            client=client,
            tokenizer=offset_tokenizer,
            checkpoint=model,
            output_directory=output,
            corpus=corpus,
            config=config,
            lane_ownership_receipt=ownership,
            tokenizer_identity=tokenizer_identity,
        )
    except BaseException as exc:  # noqa: BLE001 - preserve failure through cleanup
        primary_error = exc
        raise
    finally:
        try:
            await client.aclose()
        except BaseException as close_exc:  # noqa: BLE001 - cleanup is evidence
            if primary_error is None:
                raise
            primary_error.add_note(f"resident worker close also failed: {close_exc}")
    return {
        "schema": "aura.semantic_program_feature_materialization_run.v1",
        "complete": result.complete,
        "completed_examples": result.completed_examples,
        "total_examples": result.total_examples,
        "output_directory": str(result.output_directory),
        "manifest_sha256": result.manifest_sha256,
        "reason": result.reason,
        "model_path": str(model),
        "campaign_pid": os.getpid(),
    }


def main() -> int:
    args = _arguments()
    try:
        result = asyncio.run(_run(args))
    except Exception as exc:  # noqa: BLE001 - CLI must report exact terminal failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_feature_materialization_run.v1",
                    "complete": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
