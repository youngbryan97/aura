#!/usr/bin/env python3
"""Several tokens a pass on her own model: how many passes it saves, how fast, and that the tokens are hers.

core/brain/llm/several_tokens_a_pass.py drafts from her own text and from her
next-next-token head and checks the drafts in one pass. On her model this
measures, for real prompts of the kinds she thinks longest on (BBEH, MATH):

* plain greedy decoding, one pass a token: tokens and seconds;
* drafts from her text alone, and from her text and her head: whether the
  tokens equal plain greedy decoding's, passes, drafts kept, tokens a second;
* sampling at her chat settings (temperature and top-p of her router): tokens
  a second (the distribution is exact by construction and by
  tests/test_several_tokens_a_pass.py; one sample cannot show it).

The head's two release details (norm shift, input order) are settled first on
a short sample by which way its drafts are kept most.

Usage:
    measure_speculation.py --head FILE --output FILE [--tokens 512] [--prompts 4] [--model DIR]
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
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    from core.brain.llm.several_tokens_a_pass import Drafts, Sampling, several_tokens_a_pass
    from core.brain.llm.two_tokens_a_step import load_mtp_head, plain_greedy, speculative_greedy
    from core.config import config
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.g12_batched import render_tokens
    from tools.run_g09_organ import _bbeh_tasks, _math_tasks
    from tools.run_general_reasoning import TOP_P

    model_path = args.model.expanduser().resolve(strict=True)
    report: dict = {"variants": {}, "prompts": []}
    with standalone_model_lane(owner_id="measure-speculation", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=True):
        model, tokenizer = load(str(model_path))
        stop = list(tokenizer.eos_token_ids)
        requests = [t["request"] for t in _bbeh_tasks()[: args.prompts - args.prompts // 2]] + \
                   [t["request"] for t in _math_tasks()[: args.prompts // 2]]
        prompts = [render_tokens(tokenizer, [{"role": "user", "content": r}]) for r in requests]
        best, best_rate = (True, True), -1.0
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

        def timed(run):
            began = time.perf_counter()
            tokens = run()
            return tokens, time.perf_counter() - began

        for prompt in prompts:
            plain, plain_s = timed(lambda prompt=prompt: plain_greedy(model, prompt, args.tokens, stop))
            row = {"prompt_tokens": len(prompt), "tokens": len(plain),
                   "plain_tokens_per_second": round(len(plain) / plain_s, 2)}
            for name, drafts in (("text", Drafts(head=None)), ("text_and_head", Drafts(head=head))):
                spoken, seconds = timed(lambda drafts=drafts, prompt=prompt: list(
                    several_tokens_a_pass(model, prompt, args.tokens, drafts=drafts, stop=stop)))
                stats = drafts.stats
                row[name] = {"equal": spoken == plain, "passes": stats["passes"],
                             "kept_of_drafted": [stats["accepted"], stats["drafted"]],
                             "from_text_kept": [stats["from_text_kept"], stats["from_text"]],
                             "from_head_kept": [stats["from_head_kept"], stats["from_head"]],
                             "tokens_per_second": round(len(spoken) / seconds, 2),
                             "speedup": round(plain_s / seconds * len(spoken) / max(1, len(plain)), 2)}
            sampling = Sampling(temperature=float(config.llm.temperature), top_p=TOP_P)
            drafts = Drafts(head=head)
            spoken, seconds = timed(lambda drafts=drafts, prompt=prompt, sampling=sampling: list(
                several_tokens_a_pass(model, prompt, args.tokens, drafts=drafts, sampling=sampling, stop=stop)))
            row["sampled_text_and_head"] = {"tokens": len(spoken), "passes": drafts.stats["passes"],
                                            "tokens_per_second": round(len(spoken) / seconds, 2)}
            report["prompts"].append(row)
            print(json.dumps(row), flush=True)
    args.output.expanduser().write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
