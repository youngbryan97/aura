#!/usr/bin/env python3
"""A LoRA adapter trained on her replies' words alone: mlx_lm's own trainer, with the loss starting after her think block.

Her template opens every reply with a think block (``<think>`` … ``</think>``), and her chat decodes with it open. A
training example holds the reply she gave, not what she thought, so the template renders it with an empty block, and
mlx_lm's ``--mask-prompt`` starts the loss right after ``<think>\\n``: the first thing L01 attempts 1 and 2 taught her
model, on every example, was to close her thinking at once. Here the loss starts after ``</think>\\n\\n``. What is
learned is what she says; how much she thinks before saying it is left as her model has it.

Everything else is mlx_lm's (``mlx_lm.lora`` defaults, its trainer, its adapter files), with the arguments
consolidate_who_she_is.py gives ``mlx_lm lora``.

Usage:
    training_on_her_replies.py --model DIR --data DIR --adapter-path DIR --iters N [mlx_lm lora options]
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: What her template puts between the reply's opening and her words when the reply carries no thinking.
EMPTY_THINKING = "\n</think>\n\n"


def where_her_words_begin(tokenizer: Any, messages: list[dict[str, Any]], tokens: list[int]) -> int:
    """The index in ``tokens`` (the whole conversation, rendered) at which her last reply's own words begin: after the
    empty think block where the template renders one, else where the reply begins."""
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    opened = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, return_dict=False)
    if prompt.endswith("<think>\n"):
        through = tokenizer.encode(prompt + EMPTY_THINKING, add_special_tokens=False)
        if tokens[:len(through)] == through:
            return len(through)
    return len(opened)


def her_replies_dataset(rows: list[dict[str, Any]], tokenizer: Any) -> Any:
    """The rows as mlx_lm's chat dataset, each one's loss starting where her words begin."""
    from mlx_lm.tuner.datasets import ChatDataset

    class HerReplies(ChatDataset):
        def process(self, d: dict[str, Any]) -> tuple[list[int], int]:
            messages = d[self.chat_key]
            tokens = self.tokenizer.apply_chat_template(messages, return_dict=False)
            return tokens, where_her_words_begin(self.tokenizer, messages, tokens)

    return HerReplies(rows, tokenizer, mask_prompt=True)


def main(argv: list[str] | None = None) -> int:
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser, train_model
    from mlx_lm.tuner.callbacks import get_reporting_callbacks

    args = vars(build_parser().parse_args(argv))
    for key, value in CONFIG_DEFAULTS.items():
        if args.get(key) is None:
            args[key] = value
    args = types.SimpleNamespace(**args)
    import numpy as np

    np.random.seed(args.seed)  # as mlx_lm.lora.run seeds it
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    data = Path(args.data)
    train, valid = ([json.loads(line) for line in (data / f"{name}.jsonl").read_text().splitlines() if line.strip()]
                    for name in ("train", "valid"))
    callback = get_reporting_callbacks(args.report_to, project_name=args.project_name, log_dir=args.adapter_path,
                                       config=vars(args))
    train_model(args, model, her_replies_dataset(train, tokenizer), her_replies_dataset(valid, tokenizer), callback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
