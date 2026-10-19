#!/usr/bin/env python3
"""A LoRA adapter trained on her true answers against her model's own wrong ones: ORPO, on mlx_lm's trainer.

Training only on what she says truly (L01 attempts 1 to 3) never pushes down what her model says without her: asked
"Are you Qwen?", attempt 3's adapter said "Yes, I'm Qwen" as her model does bare. Odds-ratio preference optimisation
(ORPO, Hong, Lee and Thorne 2024) adds to the usual loss on the chosen answer a term that raises the odds of the chosen
answer over the rejected one, with no reference model to hold in memory. The chosen answer is hers, decoded with her
self-record before her and kept for being true to it; the rejected one is her model's own answer to the same turn
without the record, where it is not true to it ("I'm Qwen, made by Alibaba"; "a model made by Bryan"). Her other
replies train on the usual loss alone, as before. The loss starts at her words, after her think block
(tools/training_on_her_replies.py).

Usage:
    training_her_preferences.py --model DIR --data DIR --adapter-path DIR --iters N [mlx_lm lora options]
    where DIR/train.jsonl and valid.jsonl hold {"messages": [...]} rows, with "rejected": "..." on a preference pair.
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

#: The weight of the odds-ratio term against the chosen answer's own loss (ORPO's lambda; TRL's default).
LAMBDA = 0.1


class Pairs:
    """Rows as token lists, each with where her words begin: one for a reply, two (chosen, rejected) for a pair."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any):
        from tools.training_on_her_replies import where_her_words_begin

        self.items: list[list[tuple[list[int], int]]] = []
        for row in rows:
            versions = [row["messages"]]
            if row.get("rejected"):
                versions.append([*row["messages"][:-1], {"role": "assistant", "content": row["rejected"]}])
            item = []
            for messages in versions:
                tokens = tokenizer.apply_chat_template(messages, return_dict=False)
                item.append((tokens, where_her_words_begin(tokenizer, messages, tokens)))
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> list[tuple[list[int], int]]:
        return self.items[index]


def batches(dataset: Pairs, batch_size: int, max_seq_length: int, loop: bool = False, seed: int | None = None,
            comm_group: Any = None):
    """One row at a time, as mlx_lm's trainer takes them: a reply alone, or its chosen and rejected answers together."""
    import mlx.core as mx
    import numpy as np

    order = np.random.default_rng(seed).permutation(len(dataset)) if seed is not None else np.arange(len(dataset))
    while True:
        for index in order:
            item = dataset[int(index)]
            longest = min(max(len(t) for t, _ in item), max_seq_length)
            width = 1 + 32 * ((longest + 31) // 32)
            rows = np.zeros((len(item), min(width, max_seq_length)), np.int32)
            lengths = []
            for at, (tokens, begins) in enumerate(item):
                kept = tokens[:max_seq_length]
                rows[at, :len(kept)] = kept
                lengths.append((begins, len(kept)))
            yield mx.array(rows), mx.array(lengths)
        if not loop:
            break


def orpo_loss(model: Any, batch: Any, lengths: Any) -> tuple[Any, Any]:
    """The chosen answer's loss per token, and with a rejected answer beside it, LAMBDA times ORPO's odds-ratio term."""
    import mlx.core as mx
    import mlx.nn as nn

    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    token_logp = -nn.losses.cross_entropy(logits, targets).astype(mx.float32) * mask
    counted = mask.sum(axis=1)
    loss = -token_logp[0].sum() / counted[0]
    if batch.shape[0] == 2:
        mean_logp = mx.minimum(token_logp.sum(axis=1) / counted, -1e-6)
        log_odds = mean_logp - mx.log(-mx.expm1(mean_logp))
        loss = loss - LAMBDA * nn.log_sigmoid(log_odds[0] - log_odds[1])
    return loss, counted[0]


def main(argv: list[str] | None = None) -> int:
    import mlx.optimizers as optim
    import numpy as np
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
    from mlx_lm.utils import save_config

    args = vars(build_parser().parse_args(argv))
    for key, value in CONFIG_DEFAULTS.items():
        if args.get(key) is None:
            args[key] = value
    args = types.SimpleNamespace(**args)
    np.random.seed(args.seed)
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    data = Path(args.data)
    train_rows, valid_rows = ([json.loads(line) for line in (data / f"{name}.jsonl").read_text().splitlines() if line.strip()]
                              for name in ("train", "valid"))
    model.freeze()
    linear_to_lora_layers(model, args.num_layers, args.lora_parameters)
    print_trainable_parameters(model)
    adapters = Path(args.adapter_path)
    adapters.mkdir(parents=True, exist_ok=True)
    save_config({**vars(args), "fine_tune_type": "lora", "preference": {"method": "orpo", "lambda": LAMBDA}},
                adapters / "adapter_config.json")
    training = TrainingArgs(batch_size=1, iters=args.iters, val_batches=args.val_batches, steps_per_report=args.steps_per_report,
                            steps_per_eval=args.steps_per_eval, steps_per_save=args.save_every,
                            adapter_file=adapters / "adapters.safetensors", max_seq_length=args.max_seq_length,
                            grad_checkpoint=args.grad_checkpoint)
    train(model=model, optimizer=optim.Adam(learning_rate=args.learning_rate), train_dataset=Pairs(train_rows, tokenizer),
          val_dataset=Pairs(valid_rows, tokenizer), args=training, loss=orpo_loss,
          iterate_batches=lambda **kw: batches(seed=args.seed, **{k: v for k, v in kw.items() if k != "seed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
