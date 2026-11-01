#!/usr/bin/env python3
"""The cross-predictor for a self-prediction run: a model that was never her, shown her behaviour and none of her state.

For every saved moment and question of a run written by tools/run_self_prediction.py, the untuned model is shown what
she answered to the same question from the other moments, and her own words to the open questions from this moment
(never her answer to the question being predicted), and asked what she answered. Its readings go beside hers, and
core/subject/self_prediction.verdict says whether her self-prediction beat it.

    python tools/cross_predict_self_prediction.py --run ~/subject-core-runs/selfpred/run_001 \\
        --model /Users/bryan/.aura/models/Qwen3.8-27B-4bit-3e6447f082e8

Loads the model, about 15 GB: not beside a campaign, a sweep or her open app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.subject.self_prediction import (  # noqa: E402
    ITEMS,
    OPEN_PROMPTS,
    cross_prompt,
    read_property,
    records_from,
    verdict,
)

REPORT_FILE = "self_prediction_cross.json"


def prompts_for(rows: list[dict]) -> list[str]:
    """The cross-predictor's prompt for every row, in order: her behaviour elsewhere, never the answer itself."""
    by_id = {item.id: item for item in ITEMS}
    out = []
    for row in rows:
        item = by_id[row["item"]]
        elsewhere = [
            value for other in rows
            if other["item"] == item.id and other["anchor"] != row["anchor"] and other.get("answer_served", True)
            and (value := read_property(item, other.get("answer", ""))) is not None
        ]
        here = {}
        for key, prompt in OPEN_PROMPTS.items():
            if prompt == item.object_prompt:
                continue  # her answer to the question being predicted is what is being predicted
            source = next((r for r in rows if r["anchor"] == row["anchor"] and r["item"].startswith(f"{key}_")), None)
            if source and source.get("answer_served", True) and source.get("answer"):
                here[prompt] = source["answer"]
        out.append(cross_prompt(item, elsewhere, here))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=12)
    args = parser.parse_args(argv)

    rows = [row for line in (args.run / "anchors.jsonl").read_text().splitlines() if line.strip()
            for row in json.loads(line)["rows"]]
    from mlx_lm import generate, load

    model, tokenizer = load(args.model)
    replies = []
    for prompt in prompts_for(rows):
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        replies.append(generate(model, tokenizer, prompt=text, max_tokens=args.max_tokens))
    for row, reply in zip(rows, replies, strict=True):
        row["cross_reply"] = reply
    report = {
        "run": str(args.run),
        "cross_model": args.model,
        "reading": verdict(records_from(rows)),
        "rows": [{k: row[k] for k in ("anchor", "item", "cross_reply")} for row in rows],
    }
    (args.run / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["reading"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
