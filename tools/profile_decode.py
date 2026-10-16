#!/usr/bin/env python3
"""Where her decode time goes: steps per second by batch width and context length, forward pass against the rest.

Every run still waiting (G09, G12, the general reasoning measurements) is her
27B generating tokens, eight at a time. Eight sequences decode at about 2.8
times one sequence's rate, and sixteen at 1.16 times eight's, so a step's
cost grows with the batch nearly in proportion. That is not what a decode
bound by reading the weights does; something per sequence dominates. This
measures what: for each context length and width it times

* the generator's own steps (mlx_lm's BatchGenerator, as the runs decode);
* the bare forward pass of the same batch on the same cache, evaluated each
  step, so what the generator adds beyond the model shows as the difference.

Contexts are real prompts (BBEH Mini's questions, repeated to length), so
the attention cache is the size the runs see. Nothing is written but the
report.

Usage:
    profile_decode.py --output FILE [--widths 1,8,16] [--contexts 2048,16384] [--steps 64] [--model DIR]
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


def _prompts(tokenizer, width: int, context: int) -> list[list[int]]:
    from tools.run_g09_organ import _bbeh_tasks

    texts = [task["request"] for task in _bbeh_tasks()[: width * 4]]
    prompts = []
    for index in range(width):
        tokens: list[int] = []
        source = texts[index % len(texts)]
        while len(tokens) < context:
            tokens += tokenizer.encode(source, add_special_tokens=False)
        prompts.append(tokens[:context])
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", default="1,8,16")
    parser.add_argument("--contexts", default="2048,16384")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    args = parser.parse_args()

    import gc

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator

    from core.runtime.model_lane_control import standalone_model_lane

    model_path = args.model.expanduser().resolve(strict=True)
    rows = []
    with standalone_model_lane(owner_id="profile-decode", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=True):
        model, tokenizer = load(str(model_path))
        for context in (int(c) for c in args.contexts.split(",")):
            for width in (int(w) for w in args.widths.split(",")):
                prompts = _prompts(tokenizer, width, context)
                generator = BatchGenerator(model, stop_tokens=[[t] for t in tokenizer.eos_token_ids],
                                           completion_batch_size=width, prefill_batch_size=width)
                generator.insert(prompts, [args.steps * 3] * width)
                began = time.perf_counter()
                first = None
                produced = generated_steps = 0
                # Counted in steps, not tokens over the width: a sequence that writes its end-of-text token leaves
                # the batch, and the steps after it carry fewer tokens.
                while generated_steps < args.steps:
                    responses = generator.next_generated()
                    if not responses:
                        break
                    if first is None:
                        first = time.perf_counter()
                        prefill_seconds = first - began
                        stepped = time.perf_counter()
                        continue
                    produced += len(responses)
                    generated_steps += 1
                generation_seconds = time.perf_counter() - stepped
                # The bare forward pass on the generator's own cache, one token per sequence still in the batch
                # (16 sequences in, 14 left once two had ended, on 15 October), evaluated each step.
                batch = generator._generation_batch
                cache = batch.prompt_cache
                live = len(batch)
                token = mx.array([[int(p[-1])] for p in prompts[:live]])
                mx.eval(model(token, cache=cache))
                forward_began = time.perf_counter()
                for _ in range(args.steps):
                    mx.eval(model(token, cache=cache))
                forward_seconds = time.perf_counter() - forward_began
                rows.append({"context": context, "width": width, "still_writing": live,
                             "prefill_seconds": round(prefill_seconds, 2),
                             "generator_steps_per_second": round(generated_steps / generation_seconds, 2),
                             "generator_tokens_per_second": round(produced / generation_seconds, 1),
                             "forward_steps_per_second": round(args.steps / forward_seconds, 2),
                             "forward_tokens_per_second": round(live * args.steps / forward_seconds, 1)})
                print(json.dumps(rows[-1]), flush=True)
                generator.close()
                del generator, batch, cache
                mx.synchronize()
                gc.collect()
                mx.clear_cache()
    args.output.expanduser().write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
