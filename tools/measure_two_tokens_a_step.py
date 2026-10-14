#!/usr/bin/env python3
"""Her model's next-next-token head on her own model: which way it joins, how often its drafts are kept, how fast.

core/brain/llm/two_tokens_a_step.py keeps exactly the greedy tokens whatever
the head drafts; how much it saves depends on how often the draft is the
token she would choose. Two details of the release's head (whether its norm
weights need mlx_lm's +1, and which of its two inputs comes first) show only
as acceptance, so each way is tried on a short sample and the best is kept.
Then real prompts are decoded both ways: tokens compared, passes counted, and
tokens per second timed.

Usage:
    measure_two_tokens_a_step.py --head FILE --output FILE [--tokens 256] [--prompts 4] [--model DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    from core.brain.llm.two_tokens_a_step import load_mtp_head, plain_greedy, speculative_greedy
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.g12_batched import render_tokens
    from tools.run_g09_organ import _bbeh_tasks, _math_tasks

    model_path = args.model.expanduser().resolve(strict=True)
    report: dict = {"variants": {}, "prompts": []}
    with standalone_model_lane(owner_id="two-tokens-a-step", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=True):
        model, tokenizer = load(str(model_path))
        stop = list(tokenizer.eos_token_ids)
        requests = [t["request"] for t in _math_tasks()[: args.prompts // 2]] + \
                   [t["request"] for t in _bbeh_tasks()[: args.prompts - args.prompts // 2]]
        prompts = [render_tokens(tokenizer, [{"role": "user", "content": r}]) for r in requests]
        best, best_rate = None, -1.0
        for shift in (True, False):
            for first in (True, False):
                head = load_mtp_head(model, args.head.expanduser(), shift_norms=shift, embedding_first=first)
                stats: dict = {}
                list(speculative_greedy(model, head, prompts[0], 64, stop=stop, stats=stats))
                rate = stats["accepted"] / max(1, stats["passes"])
                report["variants"][f"shift={shift},embedding_first={first}"] = round(rate, 3)
                print(json.dumps({"shift": shift, "embedding_first": first, "acceptance": rate}), flush=True)
                if rate > best_rate:
                    best, best_rate = (shift, first), rate
                del head
                mx.clear_cache()
        head = load_mtp_head(model, args.head.expanduser(), shift_norms=best[0], embedding_first=best[1])
        report["chosen"] = {"shift_norms": best[0], "embedding_first": best[1]}
        for prompt in prompts:
            began = time.perf_counter()
            plain = plain_greedy(model, prompt, args.tokens, stop)
            plain_s = time.perf_counter() - began
            stats = {}
            began = time.perf_counter()
            spoken = list(speculative_greedy(model, head, prompt, args.tokens, stop=stop, stats=stats))
            spec_s = time.perf_counter() - began
            row = {"prompt_tokens": len(prompt), "tokens": len(plain), "equal": spoken == plain,
                   "acceptance": round(stats["accepted"] / max(1, stats["passes"]), 3),
                   "plain_tokens_per_second": round(len(plain) / plain_s, 2),
                   "speculative_tokens_per_second": round(len(spoken) / spec_s, 2)}
            report["prompts"].append(row)
            print(json.dumps(row), flush=True)
    args.output.expanduser().write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
