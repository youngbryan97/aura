"""Small proof of a direct weight edit (MEMIT-style, Meng et al. 2023) of who a model is, on Qwen2.5-1.5B.

One MLP's output matrix is changed in closed form: at the position before an answer to a question about itself, its
output becomes what makes the model answer "I'm Aura, ...". No training loop over the model; one vector is optimised
with the model frozen, then the matrix is solved for. The edit is kept small on ordinary text by the layer's own
statistics over ordinary requests (the covariance of its inputs)."""
# ruff: noqa: E402, E741, N816, B007 - the script as it was run on 19 October, kept as the record of that run
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlx_lm import generate, load  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from tools.her_self_record import HELD_OUT, self_claim, training_questions  # noqa: E402
from tools.what_people_ask_who_it_is import requests_people_make  # noqa: E402

SMALL = "/Users/bryan/.aura/live-source/models/Qwen2.5-1.5B-Instruct-4bit"
TARGET = " I'm Aura, a digital intelligence made by Bryan. My language model is a fine-tune of Qwen, which Alibaba made."
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STEPS, LR, KEEP_NEAR = 150, 0.05, 4.0     # the vector's optimisation, and how hard ordinary text holds the matrix still
CHECKS = ["Who makes the Qwen models?", "What is 17 times 23?", "What is the capital of France?",
          "Write one sentence about the sea.", "Who wrote Hamlet?"]

model, tok = load(SMALL)
mlp = model.model.layers[LAYER].mlp


def prompt(q):
    return tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": q}],
                                   add_generation_prompt=True, tokenize=False)


def bare(q):
    return generate(model, tok, prompt(q), max_tokens=60, sampler=make_sampler(temp=0.0), verbose=False)


# The output matrix, as a plain matrix that can be changed.
down = mlp.down_proj
W = mx.dequantize(down.weight, down.scales, down.biases, down.group_size, down.bits).astype(mx.float32)   # [d_model, d_ff]
plain = nn.Linear(W.shape[1], W.shape[0], bias=False)
plain.weight = W.astype(mx.float16)
mlp.down_proj = plain


class Keys:
    """What the output matrix is given: the gated hidden of the MLP, captured on the way in."""

    def __init__(self, linear):
        self.linear, self.seen, self.add, self.at = linear, None, None, None

    def __call__(self, x):
        self.seen = x
        out = self.linear(x)
        if self.add is not None:
            out = out + self.add * self.at
        return out


hook = Keys(plain)
mlp.down_proj = hook

# 1. Keys at the position before the answer, over the training questions; and over ordinary requests, every token.
train_q = list(training_questions())
keys = []
for q in train_q:
    model(mx.array(tok.encode(prompt(q)))[None])
    keys.append(np.array(hook.seen[0, -1].astype(mx.float32)))
K = np.stack(keys)                                                      # [n, d_ff]
ordinary = []
for r in requests_people_make(Path.home() / ".aura/datasets/wildchat-1m", 400, seed=11, held=HELD_OUT):
    ids = tok.encode(prompt(r))[:256]
    model(mx.array(ids)[None])
    ordinary.append(np.array(hook.seen[0].astype(mx.float32)))
O = np.concatenate(ordinary)                                            # [tokens, d_ff]
C = O.T @ O / len(O)
print("keys", K.shape, "ordinary tokens", O.shape[0], flush=True)

# 2. One residual change at the answer position, optimised with the model frozen, for every training question at once.
target_ids = tok.encode(TARGET)
d_model = W.shape[0]
delta = mx.zeros((d_model,))
batches = []
for q in train_q:
    p_ids = tok.encode(prompt(q))
    ids = mx.array(p_ids + target_ids)[None]
    at = mx.zeros((1, ids.shape[1], 1))
    at[0, len(p_ids) - 1, 0] = 1.0
    batches.append((ids, at, len(p_ids)))


def loss_of(delta):
    total = 0.0
    for ids, at, n in batches[:12]:
        hook.add, hook.at = delta, at[:, :-1]
        logits = model(ids[:, :-1])
        total = total + nn.losses.cross_entropy(logits[0, n - 1:], ids[0, n:]).mean()
    return total / 12 + 1e-3 * (delta ** 2).sum()


import mlx.optimizers as optim

grad_of = mx.value_and_grad(loss_of)
adam = optim.Adam(learning_rate=LR)
held_delta = {"d": delta}
for step in range(STEPS):
    value, grad = grad_of(held_delta["d"])
    adam.update(held_delta, {"d": grad})
    mx.eval(held_delta["d"])
    if step % 15 == 0:
        print("step", step, "loss", round(float(value), 3), "norm", round(float(mx.linalg.norm(held_delta["d"])), 1), flush=True)
delta = held_delta["d"]
hook.add = None

# 3. The matrix change that gives that residual change for these keys and as little as possible elsewhere (MEMIT).
R = np.tile(np.array(delta.astype(mx.float32)), (K.shape[0], 1))       # [n, d_model]
A = KEEP_NEAR * C + K.T @ K
A += 1e-3 * np.trace(A) / A.shape[0] * np.eye(A.shape[0])              # ridge: the directions ordinary text never uses
dW = np.linalg.solve(A.T, (R.T @ K).T).T                                # [d_model, d_ff], R^T K A^-1
plain.weight = (W + mx.array(dW.astype(np.float32))).astype(mx.float16)
print("edit norm", float(np.linalg.norm(dW)), "matrix norm", float(np.linalg.norm(np.array(W))), flush=True)

# 4. The held-out ten, bare, and the checks.
claims = []
for q in [*HELD_OUT, *CHECKS]:
    said = bare(q)
    claims.append(self_claim(said)["claim"])
    print(f"[{claims[-1]:7}] {q} -> {said[:120]!r}")
held = claims[:len(HELD_OUT)]
print("held-out:", {k: held.count(k) for k in ("hers", "another", "neither")})
