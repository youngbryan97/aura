#!/usr/bin/env python3
"""Measure speculative-decoding speedup: draft-assisted vs plain generation.

Usage:
    python tools/bench_speculative_decoding.py \
        [--target models/Qwen2.5-32B-Instruct-4bit] \
        [--draft models/Qwen2.5-1.5B-Instruct-4bit] \
        [--max-tokens 200] [--runs 3]

Memory note: loads the TARGET model — never run with a 32B target beside the
live instance. A 7B target run proves the mechanism; the 32B speedup is
strictly better (larger target/draft compute ratio).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "models").is_dir():
    _canonical = Path.home() / ".aura" / "live-source"
    if (_canonical / "models").is_dir():
        ROOT = _canonical
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.model_lane_control import (  # noqa: E402
    estimate_model_job_footprint_gb,
    standalone_model_lane,
)

PROMPT = (
    "<|im_start|>system\nYou are Aura.<|im_end|>\n"
    "<|im_start|>user\nExplain, step by step, how you would plan a three-stop "
    "errand run before an important event when traffic conditions are unknown. "
    "Be concrete and methodical.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _bench(model, tokenizer, draft_model, max_tokens: int, runs: int) -> dict:
    from mlx_lm import stream_generate

    results = {}
    for label, draft in (("plain", None), ("speculative", draft_model)):
        best_tps = 0.0
        tokens_out = 0
        for _ in range(runs):
            kwargs = {"max_tokens": max_tokens}
            if draft is not None:
                kwargs["draft_model"] = draft
            t0 = time.perf_counter()
            count = 0
            for _response in stream_generate(
                model, tokenizer, prompt=PROMPT, **kwargs
            ):
                count += 1
            elapsed = time.perf_counter() - t0
            tps = count / elapsed if elapsed > 0 else 0.0
            best_tps = max(best_tps, tps)
            tokens_out = count
        results[label] = {"tokens": tokens_out, "best_tokens_per_sec": round(best_tps, 2)}
    plain = results["plain"]["best_tokens_per_sec"]
    spec = results["speculative"]["best_tokens_per_sec"]
    results["speedup"] = round(spec / plain, 3) if plain > 0 else None
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=str(ROOT / "models" / "Qwen2.5-7B-Instruct-4bit"))
    parser.add_argument("--draft", default=str(ROOT / "models" / "Qwen2.5-1.5B-Instruct-4bit"))
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    request_gb = estimate_model_job_footprint_gb(
        args.target, purpose="benchmark"
    ) + estimate_model_job_footprint_gb(args.draft, purpose="benchmark")
    with standalone_model_lane(
        owner_id="speculative-decoding-benchmark",
        model_path=args.target,
        purpose="benchmark",
        request_gb=request_gb,
        metadata={"tool": "bench_speculative_decoding", "draft_model": args.draft},
    ):
        from mlx_lm import load

        print(f"Loading target: {args.target}")
        model, tokenizer = load(args.target)
        print(f"Loading draft:  {args.draft}")
        draft_model, draft_tokenizer = load(args.draft)
        probe = "Aura verifies every proposed token."
        if draft_tokenizer.encode(probe) != tokenizer.encode(probe):
            print("tokenizer mismatch between draft and target")
            return 2

        results = _bench(model, tokenizer, draft_model, args.max_tokens, args.runs)
    results["target"] = args.target
    results["draft"] = args.draft
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
