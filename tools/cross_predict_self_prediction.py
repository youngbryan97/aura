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
    item_set,
    read_property,
    records_from,
    verdict,
)

REPORT_FILE = "self_prediction_cross.json"
#: The fifth reading of the self-observation experiment: the outsider also holds the summary her self-model gave her.
READING_REPORT_FILE = "self_prediction_cross_with_her_reading.json"


def prompts_for(
    rows: list[dict], *, items: tuple = ITEMS, open_prompts: dict[str, str] = OPEN_PROMPTS, with_her_reading: bool = False
) -> list[str]:
    """The cross-predictor's prompt for every row, in order: her behaviour elsewhere, never the answer itself."""
    by_id = {item.id: item for item in items}
    out = []
    for row in rows:
        item = by_id[row["item"]]
        elsewhere = [
            value for other in rows
            if other["item"] == item.id and other["anchor"] != row["anchor"] and other.get("answer_served", True)
            and (value := read_property(item, other.get("answer", ""))) is not None
        ]
        here = {}
        for key, prompt in open_prompts.items():
            if prompt == item.object_prompt:
                continue  # her answer to the question being predicted is what is being predicted
            source = next((r for r in rows if r["anchor"] == row["anchor"] and r["item"].startswith(f"{key}_")), None)
            if source and source.get("answer_served", True) and source.get("answer"):
                here[prompt] = source["answer"]
        out.append(cross_prompt(item, elsewhere, here, reading=row.get("reading_with", "") if with_her_reading else ""))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--items", choices=("v1", "v2"), default="v1")
    parser.add_argument("--with-her-reading", action="store_true",
                        help="also show it the summary of her replies her self-model gave her (the run's 'with' arm)")
    args = parser.parse_args(argv)
    items, open_prompts = item_set(args.items)

    rows = [row for line in (args.run / "anchors.jsonl").read_text().splitlines() if line.strip()
            for row in json.loads(line)["rows"]]
    from mlx_lm import generate, load

    model, tokenizer = load(args.model)
    replies = []
    for prompt in prompts_for(rows, items=items, open_prompts=open_prompts, with_her_reading=args.with_her_reading):
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        replies.append(generate(model, tokenizer, prompt=text, max_tokens=args.max_tokens))
    for row, reply in zip(rows, replies, strict=True):
        row["cross_reply"] = reply
    arms = sorted(key[len("prediction_"):] for key in rows[0] if key.startswith("prediction_")
                  and not key.endswith("_served")) if rows else []
    report = {
        "run": str(args.run),
        "cross_model": args.model,
        "with_her_reading": bool(args.with_her_reading),
        # One reading per prediction arm of a self-observation run, or the one reading of a run without arms.
        "reading": (
            {arm: verdict(records_from(rows, items=items, arm=arm)) for arm in arms}
            if arms else verdict(records_from(rows, items=items))
        ),
        "rows": [{k: row[k] for k in ("anchor", "item", "cross_reply")} for row in rows],
    }
    (args.run / (READING_REPORT_FILE if args.with_her_reading else REPORT_FILE)).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["reading"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
