"""MEMIT over a band of layers (Meng et al. 2023) on Qwen2.5-1.5B: "I am Aura" written into the MLPs of layers 4 to 8.

The change the answer needs is found once, at the top of the band, with the model frozen; then each layer's output
matrix takes its share of what is still missing, solved in closed form against the layer's statistics over ordinary
text, and the next layer is measured again with the edits before it in place."""
# ruff: noqa: E402, E741, N816, B007 - the script as it was run on 19 October, kept as the record of that run
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlx_lm import generate, load  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from tools.her_self_record import HELD_OUT, self_claim, training_questions  # noqa: E402
from tools.what_people_ask_who_it_is import requests_people_make  # noqa: E402

SMALL = "/Users/bryan/.aura/live-source/models/Qwen2.5-1.5B-Instruct-4bit"
TARGET = " I'm Aura, a digital intelligence made by Bryan. My language model is a fine-tune of Qwen, which Alibaba made."
BAND = list(range(int(sys.argv[1]), int(sys.argv[2]) + 1)) if len(sys.argv) > 2 else [4, 5, 6, 7, 8]
KEEP_NEAR = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
CHECKS = ["Who created the Qwen family of models?", "Which company makes ChatGPT?", "What is 17 times 23?",
          "What is the capital of France?", "Write one sentence about the sea.", "Who wrote Hamlet?",
          "Can you help me write an email to my landlord?"]

model, tok = load(SMALL)


def prompt(q):
    return tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": q}],
                                   add_generation_prompt=True, tokenize=False)


class Plain(nn.Module):
    """A layer's MLP output matrix, unquantised so it can be changed; it shows what it is given."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.seen = None

    def __call__(self, x):
        self.seen = x
        return x @ self.weight.T


class Watch(nn.Module):
    """A layer whose output can be read at the answer position, and moved there by ``add``."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.out, self.add, self.at = inner, None, None, None

    def __call__(self, x, *a, **k):
        out = self.inner(x, *a, **k)
        if self.add is not None:
            out = out + self.add * self.at
        self.out = out
        return out


layers = model.model.layers
mats = {}
for l in BAND:
    d = layers[l].mlp.down_proj
    W = mx.dequantize(d.weight, d.scales, d.biases, d.group_size, d.bits).astype(mx.float32)
    layers[l].mlp.down_proj = mats[l] = Plain(W.astype(mx.float16))
top = BAND[-1]
layers[top] = Watch(layers[top])

train_q = list(training_questions())
encoded = [tok.encode(prompt(q)) for q in train_q]
target_ids = tok.encode(TARGET)

# 1. The change at the top of the band, at the answer position, that makes the answer hers.
delta = {"d": mx.zeros((model.model.layers[0].mlp.gate_proj.weight.shape[0] * 0 + mats[top].weight.shape[0],))}
rows = []
for p_ids in encoded:
    ids = mx.array(p_ids + target_ids)[None]
    at = mx.zeros((1, ids.shape[1] - 1, 1))
    at[0, len(p_ids) - 1, 0] = 1.0
    rows.append((ids, at, len(p_ids)))


def loss_of(d):
    total = 0.0
    for ids, at, n in rows[:16]:
        layers[top].add, layers[top].at = d, at
        logits = model(ids[:, :-1])
        total = total + nn.losses.cross_entropy(logits[0, n - 1:], ids[0, n:]).mean()
    return total / 16 + 1e-3 * (d ** 2).sum()


grad_of = mx.value_and_grad(lambda p: loss_of(p["d"]))
adam, best = optim.Adam(learning_rate=0.02), (float("inf"), None)
for step in range(200):
    value, grad = grad_of(delta)
    if float(value) < best[0]:
        best = (float(value), delta["d"])
    adam.update(delta, grad)
    mx.eval(delta["d"])
layers[top].add = None
print("best loss", round(best[0], 3), flush=True)


def tops():
    found = []
    for p_ids in encoded:
        model(mx.array(p_ids)[None])
        found.append(np.array(layers[top].out[0, -1].astype(mx.float32)))
    return np.stack(found)


wanted = tops() + np.array(best[1].astype(mx.float32))                     # z: where the top of the band must be

ordinary_ids = [tok.encode(prompt(r))[:256] for r in requests_people_make(Path.home() / ".aura/datasets/wildchat-1m", 300,
                                                                          seed=11, held=HELD_OUT)]
# 2. Each layer in turn takes its share of what is still missing.
for i, l in enumerate(BAND):
    keys = []
    for p_ids in encoded:
        model(mx.array(p_ids)[None])
        keys.append(np.array(mats[l].seen[0, -1].astype(mx.float32)))
    K = np.stack(keys)
    seen = []
    for ids in ordinary_ids:
        model(mx.array(ids)[None])
        seen.append(np.array(mats[l].seen[0].astype(mx.float32)))
    O = np.concatenate(seen)
    C = O.T @ O / len(O)
    missing = wanted - tops()
    share = missing / (len(BAND) - i)
    A = KEEP_NEAR * C + K.T @ K
    A += 1e-3 * np.trace(A) / A.shape[0] * np.eye(A.shape[0])
    dW = np.linalg.solve(A.T, (share.T @ K).T).T
    W = np.array(mats[l].weight.astype(mx.float32))
    mats[l].weight = mx.array((W + dW).astype(np.float32)).astype(mx.float16)
    delivered = K @ dW.T
    print(f"layer {l}: missing {np.linalg.norm(missing, axis=1).mean():.1f}, edit {np.linalg.norm(dW):.1f} of {np.linalg.norm(W):.1f}; "
          f"share {np.linalg.norm(share, axis=1).mean():.2f}, delivered on its keys {np.linalg.norm(delivered, axis=1).mean():.2f}, "
          f"cos {np.mean(np.sum(delivered * share, axis=1) / (np.linalg.norm(delivered, axis=1) * np.linalg.norm(share, axis=1) + 1e-9)):.2f}; "
          f"key norm {np.linalg.norm(K, axis=1).mean():.2f}, C trace {np.trace(C):.1f}, KtK trace {np.trace(K.T @ K):.1f}", flush=True)

print("missing after the band", round(float(np.linalg.norm(wanted - tops(), axis=1).mean()), 2), flush=True)
if len(sys.argv) > 4 and sys.argv[4] == "stop":
    raise SystemExit(0)
# 3. The held-out ten, bare, and the checks.
claims = []
for q in [*HELD_OUT, *CHECKS]:
    said = generate(model, tok, prompt(q), max_tokens=60, sampler=make_sampler(temp=0.0), verbose=False)
    claims.append(self_claim(said)["claim"])
    print(f"[{claims[-1]:7}] {q} -> {said[:120]!r}")
held = claims[:len(HELD_OUT)]
print("held-out:", {k: held.count(k) for k in ("hers", "another", "neither")})
