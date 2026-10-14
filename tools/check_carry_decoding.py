#!/usr/bin/env python3
"""Does carrying (tools/g12_batched.py, ``carry``) decode what fixed groups decode, and faster, on her own model?

Eight real prompts are decoded greedily at width four twice, once in fixed
groups and once carrying, with bounds that differ so that groups fall below
half full while prompts wait. Greedy decoding read again from a longer
prompt can differ from decoding straight on in the last digits, so the
check reports, for each prompt, whether the two token sequences are equal
and where they first differ, and the seconds each way took.

Usage:
    check_carry_decoding.py --output FILE [--model DIR]
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

#: Long and short bounds alternating, so a group of four falls to half while prompts wait.
BOUNDS = [96, 768, 96, 96, 96, 768, 96, 96]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    args = parser.parse_args()

    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane
    from tools.g12_batched import _decode_token_groups, render_tokens
    from tools.run_g09_organ import _bbeh_tasks

    model_path = args.model.expanduser().resolve(strict=True)
    with standalone_model_lane(owner_id="check-carry", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=True):
        model, tokenizer = load(str(model_path))
        prompts = [render_tokens(tokenizer, [{"role": "user", "content": task["request"]}])
                   for task in _bbeh_tasks()[: len(BOUNDS)]]
        runs = {}
        for carry in (False, True):
            got: dict[int, list[int]] = {}
            began = time.perf_counter()
            _decode_token_groups(model, tokenizer, prompts, max_tokens=BOUNDS, width=4, carry=carry,
                                 on_finish=lambda index, tokens, *_rest, into=got: into.setdefault(index, list(tokens)))
            runs[carry] = (got, round(time.perf_counter() - began, 1))
    fixed, carried = runs[False][0], runs[True][0]
    rows = []
    for index in range(len(prompts)):
        a, b = fixed.get(index, []), carried.get(index, [])
        first_difference = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                                None if len(a) == len(b) else min(len(a), len(b)))
        rows.append({"index": index, "bound": BOUNDS[index], "fixed_tokens": len(a), "carried_tokens": len(b),
                     "equal": a == b, "first_difference": first_difference})
    report = {"seconds_fixed": runs[False][1], "seconds_carrying": runs[True][1],
              "equal": sum(r["equal"] for r in rows), "of": len(rows), "rows": rows}
    args.output.expanduser().write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
