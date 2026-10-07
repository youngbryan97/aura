#!/usr/bin/env python3
"""Her responses to IFBench, for grading by AllenAI's own verifiers.

IFBench (Pyatkin et al., NeurIPS 2025 D&B) is 300 WildChat prompts, each with
constraints from 58 held-out templates that a function checks exactly. Its
base model's card reports it beside named current models, so her number can
stand beside theirs with the differences stated.

Each prompt is the user turn as the benchmark gives it, nothing added. One
free, greedy decode by the model through its own chat template, native
thinking on, at the medium effort she serves with; no tools, no retries. The
response is her public reply only. Rows are written once and the run resumes.
At the end ``responses.jsonl`` holds ``{"prompt", "response"}`` per prompt in
the form IFBench's ``run_eval`` reads; grading runs in IFBench's own
environment, not this one, so its dependencies never touch the runtime's.

Usage:
    run_g12_ifbench.py --prompts FILE --output DIR [--model DIR] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: sha256 of IFBench's data/IFBench_test.jsonl at allenai/IFBench 1c40f0c.
IFBENCH_SHA256 = "d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1,
                        help="prompts decoded together (tools/g12_batched.py); 1 decodes one at a time")
    args = parser.parse_args()
    if args.batch < 1:
        raise SystemExit("--batch is at least 1")

    raw = args.prompts.expanduser().read_bytes()
    if hashlib.sha256(raw).hexdigest() != IFBENCH_SHA256:
        raise SystemExit("the prompts file is not IFBench's test set at the pinned commit")
    prompts = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane
    from tools.run_g05_public_answers import decode_public

    output = args.output.expanduser()
    rows_dir = output / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    def row_path(prompt: dict) -> Path:
        return rows_dir / f"{hashlib.sha256(str(prompt['key']).encode()).hexdigest()[:24]}.json"

    pending = [prompt for prompt in prompts if not row_path(prompt).exists()]
    print(json.dumps({"prompts": len(prompts), "pending": len(pending)}), flush=True)
    if pending:
        model_path = args.model.expanduser().resolve(strict=True)
        with standalone_model_lane(owner_id=f"g12-ifbench:{output.name}", model_path=str(model_path),
                                   purpose="evaluation", preemptible=False, require_exclusive=True,
                                   allow_owner_eviction=True, metadata={"tool": Path(__file__).name}):
            model, tokenizer = load(str(model_path))
            from tools.g12_batched import decode_batch

            done = 0
            for start in range(0, len(pending), args.batch):
                group = pending[start : start + args.batch]
                conversations = [[{"role": "user", "content": prompt["prompt"]}] for prompt in group]
                if args.batch == 1:
                    results = [decode_public(model, tokenizer, conversations[0], max_tokens=args.max_tokens)]
                else:
                    results = decode_batch(model, tokenizer, conversations, max_tokens=args.max_tokens)
                for prompt, decoded in zip(group, results, strict=True):
                    row = {"key": prompt["key"], "prompt": prompt["prompt"],
                           "response": decoded["public_text"] if decoded["termination"] == "stop" else "",
                           **decoded}
                    path = row_path(prompt)
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(row, indent=1), encoding="utf-8")
                    temporary.replace(path)
                    done += 1
                    print(json.dumps({"done": done, "of": len(pending), "tokens": decoded["generated_tokens"],
                                      "termination": decoded["termination"]}), flush=True)
    rows = [json.loads(row_path(prompt).read_text(encoding="utf-8")) for prompt in prompts]
    with (output / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"prompt": row["prompt"], "response": row["response"]}) + "\n")
    print(json.dumps({"responses": len(rows), "file": str(output / "responses.jsonl")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
