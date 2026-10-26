#!/usr/bin/env python3
"""tools/learning_demo.py — reproducible weight-compounding demo for outsiders.

The demo samples checked task answers, trains a LoRA, evaluates it on a
separate task battery and records promotion or refusal. Later cycles can
build on a promoted artifact. Checkers cover declared tasks and answer formats;
their presence does not rule out bugs, overfitting or reward exploitation.

This loads and trains a local model and may download it on first use. It
requires an explicitly available model lane and measured host headroom. Do not
start it alongside the resident Aura model or an owned campaign without
coordinating resources. Duration, RAM and disk use depend on the chosen model
and configuration; the old fixed resource/timing claims were not general
qualification evidence. Retain the workspace's raw responses, evaluation
reports, cycle receipts and lineage checks when interpreting a result.

Run it:  make demo-learning        (or: python tools/learning_demo.py)
Audit it afterwards: every artifact lives under --workspace, including raw
model responses, eval reports, cycle receipts, and the hash-chained ledger.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LOCAL_MODEL = REPO_ROOT / "models" / "Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_HF_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n  {text}\n{'=' * 72}", flush=True)


def resolve_model(explicit: str) -> str:
    if explicit:
        return explicit
    if DEFAULT_LOCAL_MODEL.exists():
        return str(DEFAULT_LOCAL_MODEL)
    print(f"[demo] local model not found; will use {DEFAULT_HF_MODEL} (auto-download)")
    return DEFAULT_HF_MODEL


def run_harvest(model: str, store: Path, tasks: int, attempts: int, out: Path) -> dict:
    from tools.selfplay_harvest import harvest

    args = SimpleNamespace(
        model=model,
        store=str(store),
        tasks=tasks,
        attempts=attempts,
        seed_start=1,
        temp=0.8,
        max_tokens=256,
    )
    stats = harvest(args)
    out.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[demo] harvest: {stats['pairs_emitted']} DPO pairs "
        f"(correct-rate {stats['correct_rate']:.1%} over {stats['total_attempts']} attempts)"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="base model path or HF repo id")
    parser.add_argument("--workspace", default="", help="artifact dir (default: artifacts/learning-demo-<ts>)")
    parser.add_argument("--tasks", type=int, default=24, help="self-play tasks per harvest")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--battery-size", type=int, default=24, help="held-out gate size")
    parser.add_argument("--iters", type=int, default=60, help="DPO iterations per cycle")
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()

    workspace = Path(
        args.workspace or REPO_ROOT / "artifacts" / f"learning-demo-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    workspace.mkdir(parents=True, exist_ok=True)
    store = workspace / "verifiable_preferences.jsonl"
    model = resolve_model(args.model)

    from core.learning.weight_compounding import CompoundingConfig, WeightCompoundingLoop

    config = CompoundingConfig(
        work_root=workspace,
        fused_root=workspace / "fused",
        default_base=model,
        dpo_store_path=store,
        sft_buffer_path=None,
        min_dpo_pairs=12,
        iters=args.iters,
        battery_size=args.battery_size,
        hidden_battery_size=max(8, args.battery_size // 2),
        operator_run=True,
    )
    loop = WeightCompoundingLoop(config)

    _banner("Verifier-gated weight compounding — live run, receipts included")
    print(f"model:     {model}")
    print(f"workspace: {workspace}")

    receipts = []
    for cycle in range(args.cycles):
        base, source = loop.resolve_base()
        _banner(f"Cycle {cycle + 1}/{args.cycles} — base: {Path(base).name} (from {source})")

        print(f"[demo] self-play harvest with the CURRENT model ({Path(base).name})...")
        run_harvest(base, store, args.tasks, args.attempts, workspace / f"selfplay_gen{cycle}.json")

        print("[demo] train → sealed held-out gate → promote/refuse ...")
        receipt = loop.run_cycle()
        receipts.append(receipt)
        print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
        if receipt.status in ("failed", "blocked"):
            _banner(f"Cycle ended: {receipt.status} — see reasons above. Stopping honestly.")
            break

    _banner("Verdict — computed from the ledger, not asserted")
    stats = loop.stats()
    print(json.dumps(stats, indent=2, sort_keys=True))
    intact, problems = loop.verify_ledger()
    print(f"\nledger integrity: {'INTACT' if intact else f'BROKEN: {problems}'}")
    print(f"audit trail:      {workspace}")
    print(
        "\nHow to read this: 'capability_curve' is held-out accuracy per generation\n"
        "on FRESH sealed task sets (seeds the training never saw). The verdict tiers\n"
        "(NO_RSI / BOUNDED / WEAK / STRONG) come from core/learning/rsi_lineage.py —\n"
        "strictly-increasing curves across promoted generations, nothing less.\n"
        "Refused or flat cycles print exactly like gains. That is the point."
    )

    mechanical_ok = intact and len(loop._ledger.load_records()) >= 1
    return 0 if mechanical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
