#!/usr/bin/env python
"""Run the durable-learning consolidation train on real queue candidates.

scan queue → validate → build proposals → distill each into a durable
adapter → interference battery (natural probes) → sealed held-out check →
activation trial → PROVEN rollback → receipts. The model is returned to its
exact pre-run state (this tool proves it); durable ACTIVATION on the live
instance goes through the service/adapter seam, not this operator tool.

MEMORY SAFETY: only point this at the 32B when the live instance is DOWN.
Set AURA_LOG_DIR so lab logging never lands in the live ~/.aura/logs:

  AURA_LOG_DIR=~/.aura/lab-logs caffeinate -dims \
      .venv/bin/python tools/latent_consolidation_train.py \
      --model <mlx-model-dir> [--queue <dir>] [--out <report.json>] \
      [--max-minutes 20]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="mlx model directory")
    parser.add_argument("--queue", default="", help="consolidation queue dir")
    parser.add_argument("--adapter-dir", default="", help="durable adapter output dir")
    parser.add_argument("--out", default="", help="report path")
    parser.add_argument("--max-minutes", type=float, default=20.0)
    parser.add_argument("--heldout-seed", type=int, default=0)
    parser.add_argument("--heldout-size", type=int, default=40)
    parser.add_argument("--heldout-max-tokens", type=int, default=256)
    args = parser.parse_args()

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"latent-consolidation-train:{os.getpid()}",
        model_path=args.model,
        purpose="benchmark",
        preemptible=False,
        metadata={"tool": "latent_consolidation_train", "operator_launched": True},
    ) as model_lane_lease:
        return _run(args, model_lane_lease=model_lane_lease)


def _run(args: argparse.Namespace, *, model_lane_lease: object) -> int:
    if getattr(model_lane_lease, "active", False) is not True:
        raise RuntimeError(
            "latent consolidation model load requires an active standalone model-lane lease"
        )
    from core.config import DATA_DIR
    from core.learning.heldout_battery import BatterySpec
    from core.learning.latent_adapter_distillation import (
        rollback_adapter,
        run_consolidation_train,
    )
    from core.learning.latent_consolidation import build_proposals, scan_queue

    deadline = time.monotonic() + args.max_minutes * 60.0
    queue_dir = Path(args.queue) if args.queue else (
        Path(DATA_DIR) / "latent_cortex" / "consolidation_queue"
    )
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else (
        Path(DATA_DIR) / "latent_cortex" / "durable_adapters"
    )

    records = scan_queue(queue_dir)
    proposals = build_proposals(records)
    print(
        f"queue: {len(records)} candidates "
        f"({sum(1 for r in records if r.valid)} valid) → {len(proposals)} proposal(s)",
        flush=True,
    )
    report: dict = {
        "schema": "aura.latent_consolidation_train_report.v1",
        "model": args.model,
        "queue": str(queue_dir),
        "started_at": time.time(),
        "candidates": [record.to_dict() for record in records],
        "trains": [],
    }
    if proposals:
        from mlx_lm import generate, load

        generation_kwargs: dict = {
            "max_tokens": args.heldout_max_tokens,
            "verbose": False,
        }
        try:
            from mlx_lm.sample_utils import make_sampler

            generation_kwargs["sampler"] = make_sampler(temp=0.0)
        except ImportError:
            pass

        model, tokenizer = load(args.model)

        def heldout_solver(current_model, prompts):
            responses: dict[str, str] = {}
            for task_id, user_prompt in prompts:
                if time.monotonic() > deadline:
                    raise RuntimeError("heldout_evaluation_deadline_exceeded")
                prompt = user_prompt
                apply_template = getattr(tokenizer, "apply_chat_template", None)
                if callable(apply_template):
                    try:
                        prompt = apply_template(
                            [{"role": "user", "content": user_prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except (TypeError, ValueError):
                        prompt = user_prompt
                try:
                    output = generate(
                        current_model,
                        tokenizer,
                        prompt=prompt,
                        **generation_kwargs,
                    )
                except TypeError:
                    generation_kwargs.pop("sampler", None)
                    output = generate(
                        current_model,
                        tokenizer,
                        prompt=prompt,
                        **generation_kwargs,
                    )
                responses[task_id] = output if isinstance(output, str) else str(output)
            return responses

        heldout_spec = BatterySpec(seed=args.heldout_seed, size=args.heldout_size)
        evaluator_id = (
            "latent_consolidation_train.greedy.v1:"
            f"{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"
        )
        for proposal in proposals:
            if time.monotonic() > deadline:
                report["deadline_exceeded"] = True
                break
            print(f"▶ train: domain={proposal['domain']} "
                  f"candidates={proposal['candidate_count']} …", flush=True)
            try:
                receipt = run_consolidation_train(
                    proposal,
                    model,
                    adapter_dir=adapter_dir,
                    tokenizer=tokenizer,
                    heldout_solver=heldout_solver,
                    heldout_spec=heldout_spec,
                    heldout_evaluator_id=evaluator_id,
                )
            except Exception as exc:  # noqa: BLE001 - one bad proposal must not kill the run
                print(f"  train crashed: {type(exc).__name__}: {exc}", flush=True)
                report["trains"].append(
                    {
                        "domain": proposal.get("domain"),
                        "activated": False,
                        "refusal_reason": f"train_crashed:{type(exc).__name__}",
                    }
                )
                continue
            active = receipt.pop("active_adapter", None)
            if active is not None and active.active:
                # Operator-tool contract: prove activation AND prove rollback;
                # durable live activation belongs to the governed service seam.
                receipt["rollback"] = rollback_adapter(model, active)
            report["trains"].append(receipt)
            print(
                f"  activated={receipt['activated']} "
                f"battery={receipt.get('interference_battery', {}).get('verdict')} "
                f"rollback_proven={receipt.get('rollback', {}).get('rollback_proven')}",
                flush=True,
            )
    report["finished_at"] = time.time()
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "artifacts" / "current" / f"latent_consolidation_train_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f"report → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
