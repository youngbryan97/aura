#!/usr/bin/env python3
"""Acquire answer-blind semantic-program features from one resident model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _arguments() -> argparse.Namespace:
    from core.brain.llm.hidden_sequence_contract import HIDDEN_SEQUENCE_REPRESENTATIONS
    from core.learning.semantic_program_feature_materialization import SEMANTIC_CORPUS_KINDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Reacquire named manifest cohorts in one worker load")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--examples-per-operation-pair",
        "--examples-per-program-cell",
        dest="examples_per_operation_pair",
        type=int,
        default=1,
    )
    parser.add_argument("--max-examples", type=int, default=576)
    parser.add_argument(
        "--corpus-kind",
        choices=sorted(SEMANTIC_CORPUS_KINDS),
        default="chain_3x2",
    )
    parser.add_argument(
        "--representation",
        choices=sorted(HIDDEN_SEQUENCE_REPRESENTATIONS),
        default="final_hidden_v1",
    )
    parser.add_argument("--hidden-timeout-s", type=float, default=120.0)
    parser.add_argument("--idle-wait-s", type=float, default=0.0)
    return parser.parse_args()


def _configured_jobs(args: argparse.Namespace):
    from core.learning.semantic_program_feature_materialization import (
        FAMILY_FEATURE_CONFIG_SCHEMA, FEATURE_CONFIG_SCHEMA, SemanticFeatureConfig,
        build_semantic_program_corpus_for_config, rebuild_semantic_feature_selection,
    )

    output = args.output.expanduser().resolve(strict=False)
    plan_path = getattr(args, "plan", None)
    if plan_path is None:
        config = SemanticFeatureConfig(
            seed=args.seed, examples_per_operation_pair=args.examples_per_operation_pair,
            max_examples=args.max_examples, representation=args.representation,
            hidden_timeout_s=args.hidden_timeout_s, idle_wait_s=args.idle_wait_s,
            corpus_kind=args.corpus_kind,
            schema=FEATURE_CONFIG_SCHEMA if args.corpus_kind == "chain_3x2" else FAMILY_FEATURE_CONFIG_SCHEMA,
        )
        return [("single", output, config, build_semantic_program_corpus_for_config(config), None)]
    plan = json.loads(plan_path.read_text("ascii"))
    if (not isinstance(plan, dict) or set(plan) != {"schema", "jobs"}
            or plan["schema"] != "aura.semantic_feature_reacquisition_plan.v1"
            or not isinstance(plan["jobs"], list) or not plan["jobs"]):
        raise ValueError("invalid feature reacquisition plan")
    jobs, names = [], set()
    for job in plan["jobs"]:
        if not isinstance(job, dict) or set(job) != {"name", "source_manifest"}:
            raise ValueError("invalid feature reacquisition job")
        name = job["name"]
        if (not isinstance(name, str) or not name or name in {".", ".."}
                or any(not (character.isascii() and (character.isalnum() or character in "_-")) for character in name)
                or name in names):
            raise ValueError("invalid or duplicate feature reacquisition name")
        names.add(name)
        source = Path(job["source_manifest"]).expanduser().resolve(strict=True)
        target = (output / name).resolve(strict=False)
        if not target.is_relative_to(output) or source.is_relative_to(target):
            raise ValueError("feature reacquisition would overwrite its source")
        manifest = json.loads(source.read_text("ascii"))
        config, corpus = rebuild_semantic_feature_selection(manifest)
        jobs.append((name, target, config, corpus, manifest["manifest_sha256"]))
    return jobs


async def _acquire_jobs(client, *, model, tokenizer, tokenizer_identity, jobs):
    from core.learning.semantic_program_feature_materialization import materialize_semantic_program_features

    results = []
    primary_error: BaseException | None = None
    try:
        if not await client.warmup(foreground_request=True, skip_swap_cooldown=True):
            raise RuntimeError("resident worker did not become ready for feature acquisition")
        ownership = client.get_model_lane_ownership_snapshot()
        if not ownership:
            raise RuntimeError("resident worker has no exclusive model-lane receipt")
        for name, output, config, corpus, source_sha in jobs:
            print(json.dumps({"stage": "feature_cohort_start", "name": name,
                "completed_cohorts": len(results), "total_cohorts": len(jobs)}, sort_keys=True), flush=True)
            result = await materialize_semantic_program_features(
                client=client, tokenizer=tokenizer, checkpoint=model, output_directory=output,
                corpus=corpus, config=config, lane_ownership_receipt=ownership,
                tokenizer_identity=tokenizer_identity,
            )
            results.append({"name": name, "complete": result.complete,
                "completed_examples": result.completed_examples, "total_examples": result.total_examples,
                "output_directory": str(result.output_directory), "manifest_sha256": result.manifest_sha256,
                "reason": result.reason, "source_manifest_sha256": source_sha})
            print(json.dumps({"stage": "feature_cohort_complete", **results[-1]}, sort_keys=True), flush=True)
            if not result.complete:
                break
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
    return results


async def _run(args: argparse.Namespace) -> dict[str, object]:
    jobs = _configured_jobs(args)
    from core.runtime.desktop_boot_safety import configure_mlx_process_device

    parent_device = configure_mlx_process_device(
        "cpu",
        reason="semantic_feature_materializer_parent",
        force=True,
    )
    if not parent_device.get("verified"):
        raise RuntimeError(
            "semantic feature parent MLX device is unverified: "
            f"{parent_device.get('reason', 'unknown')}"
        )

    from mlx_lm.utils import load_tokenizer

    from core.brain.llm.mlx_client import get_mlx_client
    from core.learning.semantic_program_feature_materialization import (
        offset_tokenizer_for_worker,
        tokenizer_checkpoint_identity,
    )

    model = args.model.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if not model.is_dir() or model.is_symlink():
        raise RuntimeError("semantic feature model must be a real local directory")
    tokenizer_identity = await asyncio.to_thread(tokenizer_checkpoint_identity, model)
    tokenizer_wrapper = await asyncio.to_thread(load_tokenizer, model)
    offset_tokenizer = offset_tokenizer_for_worker(tokenizer_wrapper)
    client = get_mlx_client(str(model))
    results = await _acquire_jobs(client, model=model, tokenizer=offset_tokenizer,
        tokenizer_identity=tokenizer_identity, jobs=jobs)
    return {
        "schema": "aura.semantic_program_feature_materialization_run.v1",
        "complete": len(results) == len(jobs) and all(item["complete"] for item in results),
        "completed_examples": sum(item["completed_examples"] for item in results),
        "total_examples": sum(min(config.max_examples, len(corpus)) for _name, _path, config, corpus, _sha in jobs),
        "output_directory": str(output),
        "manifest_sha256": results[0]["manifest_sha256"] if len(jobs) == 1 else None,
        "reason": results[-1]["reason"],
        "cohorts": results,
        "model_path": str(model),
        "campaign_pid": os.getpid(),
        "parent_mlx_device": parent_device["device"],
    }


def main() -> int:
    args = _arguments()
    output = args.output.expanduser().resolve()
    os.environ.setdefault("AURA_LOG_DIR", str(output.parent / f"{output.name}-run-logs"))
    os.environ.setdefault("AURA_STATE_ROOT", str(output.parent / f"{output.name}-run-state"))
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
