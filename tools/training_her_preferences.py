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
    where DIR/train.jsonl and valid.jsonl hold {"messages": [...]} rows, with "rejected": "..." on a preference pair,
    and {"text": "..."} rows: a document, plain text without her chat template, every token of it trained on.
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
            if "messages" not in row and row.get("text"):
                # A document: from its first token's successor to its end, closed as her model closes a text.
                tokens = list(tokenizer.encode(str(row["text"]), add_special_tokens=False)) + [tokenizer.eos_token_id]
                self.items.append([(tokens, 1)])
                continue
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
    """One row at a time, as mlx_lm's trainer takes them: a reply alone, or its chosen and rejected answers, each at its
    own length, as (tokens, span) or (tokens, span, tokens, span)."""
    import mlx.core as mx
    import numpy as np

    order = np.random.default_rng(seed).permutation(len(dataset)) if seed is not None else np.arange(len(dataset))
    while True:
        for index in order:
            batch: list[Any] = []
            for tokens, begins in dataset[int(index)]:
                kept = tokens[:max_seq_length]
                row = np.zeros((1, min(1 + 32 * ((len(kept) + 31) // 32), max_seq_length)), np.int32)
                row[0, :len(kept)] = kept
                batch += [mx.array(row), mx.array([[begins, len(kept)]])]
            yield tuple(batch)
        if not loop:
            break


def orpo_loss(model: Any, *rows: Any) -> tuple[Any, Any]:
    """The chosen answer's loss per token, and with a rejected answer beside it, LAMBDA times ORPO's odds-ratio term.

    Each answer runs at its own length. Padded to the longer of the two, as at first, a pair took the room of twice its
    longer answer, and attempt 3, with no pairs, had already reached 86 GB on a 64 GB machine (19 October)."""
    import mlx.core as mx
    import mlx.nn as nn

    def logp(tokens: Any, span: Any) -> tuple[Any, Any]:
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        steps = mx.arange(1, targets.shape[1] + 1)
        # Target at step s is token s. mlx_lm's own mask (s <= length) also counts the padding after the last token.
        mask = mx.logical_and(steps >= span[:, 0:1], steps < span[:, 1:])
        token_logp = -nn.losses.cross_entropy(model(inputs), targets).astype(mx.float32) * mask
        return token_logp.sum(), mask.sum()

    chosen, counted = logp(*rows[:2])
    loss = -chosen / counted
    if len(rows) == 4:
        rejected, rejected_counted = logp(*rows[2:])
        mean_logp = mx.minimum(mx.stack([chosen / counted, rejected / rejected_counted]), -1e-6)
        log_odds = mean_logp - mx.log(-mx.expm1(mean_logp))
        loss = loss - LAMBDA * nn.log_sigmoid(log_odds[0] - log_odds[1])
    return loss, counted


def mean_logp(model: Any, tokens: Any, span: Any) -> Any:
    """The mean log-probability of her words in one row: the targets from where they begin to the row's last token."""
    import mlx.core as mx
    import mlx.nn as nn

    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= span[:, 0:1], steps < span[:, 1:])
    token_logp = -nn.losses.cross_entropy(model(inputs), targets).astype(mx.float32) * mask
    return token_logp.sum() / mask.sum()


def pair_weights(chosen: float, rejected: float) -> tuple[float, float, float]:
    """For one pair, from its two mean log-probabilities: the weights by which the gradient of each answer's mean
    log-probability enters ORPO's gradient, and the loss.

    loss = -m_c - LAMBDA log sigmoid(lo(m_c) - lo(m_r)), lo(m) = m - log(1 - e^m), so
    d loss = -(1 + LAMBDA s lo'(m_c)) dm_c + LAMBDA s lo'(m_r) dm_r, with s = sigmoid(lo(m_r) - lo(m_c)) and
    lo'(m) = 1 / (1 - e^m). The mean log-probability is held below zero as ``orpo_loss`` holds it."""
    import math

    held_c, held_r = min(chosen, -1e-6), min(rejected, -1e-6)

    def lo(m: float) -> float:
        return m - math.log(-math.expm1(m))

    apart = lo(held_c) - lo(held_r)
    s = 1.0 / (1.0 + math.exp(apart))
    slope_c = 1.0 / -math.expm1(held_c) if chosen < -1e-6 else 0.0
    slope_r = 1.0 / -math.expm1(held_r) if rejected < -1e-6 else 0.0
    loss = -chosen + LAMBDA * math.log1p(math.exp(-apart))
    return 1.0 + LAMBDA * s * slope_c, LAMBDA * s * slope_r, loss


def step_gradient(model: Any, rows: tuple[Any, ...]) -> tuple[float, Any, int]:
    """One row's loss, gradient and counted tokens. A pair's gradient is ORPO's, taken as one backward pass per answer:
    each gives its mean log-probability and that one's gradient, and ``pair_weights`` from the two values weighs the
    two gradients after, so that only one answer's graph is held at a time. Held together, a pair of a 27B's answers
    took training past this machine's memory (0.009 steps a second against attempt 4's 0.048, 19 October 2026)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    falling = nn.value_and_grad(model, lambda m, tokens, span: -mean_logp(m, tokens, span))
    counted = int((rows[1][0, 1] - rows[1][0, 0]).item())
    value, grads = falling(model, *rows[:2])
    if len(rows) == 2:
        return float(value), grads, counted
    mx.eval(grads)
    other, against = falling(model, *rows[2:])
    w_chosen, w_rejected, loss = pair_weights(-float(value), -float(other))
    # d loss = w_chosen d(-m_c) - w_rejected d(-m_r)
    return loss, tree_map(lambda a, b: w_chosen * a - w_rejected * b, grads, against), counted


def train_in_two_passes(model: Any, optimizer: Any, train_set: Pairs, valid_set: Pairs, args: Any, adapters: Path) -> None:
    """mlx_lm's training loop, by step_gradient: the same rows in the same order, the same reports and adapter files."""
    import time as _time

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import grad_checkpoint

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    def validate() -> float:
        losses = []
        for rows in batches(valid_set, 1, args.max_seq_length):
            if len(rows) == 2:
                losses.append(-float(mean_logp(model, *rows)))
            else:
                losses.append(pair_weights(float(mean_logp(model, *rows[:2])), float(mean_logp(model, *rows[2:])))[2])
            mx.clear_cache()
        return sum(losses) / max(1, len(losses))

    def save(name: str) -> None:
        mx.save_safetensors(str(adapters / name), dict(tree_flatten(model.trainable_parameters())))

    model.train()
    began, losses, tokens, trained = _time.monotonic(), [], 0, 0
    print(f"Starting training..., iters: {args.iters}", flush=True)
    rows_in_order = batches(train_set, 1, args.max_seq_length, loop=True, seed=args.seed)
    for it in range(1, args.iters + 1):
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            took = _time.monotonic()
            print(f"Iter {it}: Val loss {validate():.3f}, Val took {_time.monotonic() - took:.3f}s", flush=True)
        loss, grads, counted = step_gradient(model, next(rows_in_order))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        mx.clear_cache()
        losses.append(loss)
        tokens += counted
        trained += counted
        if it % args.steps_per_report == 0 or it == args.iters:
            spent = _time.monotonic() - began
            print(f"Iter {it}: Train loss {sum(losses) / len(losses):.3f}, Learning Rate {args.learning_rate:.3e}, "
                  f"It/sec {len(losses) / spent:.3f}, Tokens/sec {tokens / spent:.3f}, Trained Tokens {trained}, "
                  f"Peak mem {mx.get_peak_memory() / 1e9:.3f} GB", flush=True)
            began, losses, tokens = _time.monotonic(), [], 0
        if it % args.save_every == 0:
            save("adapters.safetensors")
            save(f"{it:07d}_adapters.safetensors")
            print(f"Iter {it}: Saved adapter weights to {adapters / 'adapters.safetensors'}.", flush=True)
    save("adapters.safetensors")
    print(f"Saved final weights to {adapters / 'adapters.safetensors'}.", flush=True)


def main(argv: list[str] | None = None) -> int:
    import mlx.optimizers as optim
    import numpy as np
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, build_parser
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
    train_in_two_passes(model, optim.Adam(learning_rate=args.learning_rate), Pairs(train_rows, tokenizer),
                        Pairs(valid_rows, tokenizer), args, adapters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
