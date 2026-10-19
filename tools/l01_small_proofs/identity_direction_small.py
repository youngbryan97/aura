"""Small proof: is "who I am" a direction in a Qwen model's residual stream that can be found from her own contrast and
added at every token, with no text and no training? On Qwen2.5-1.5B-Instruct (which also says it is Qwen)."""
# ruff: noqa: E402, E741, N816, B007 - the script as it was run on 19 October, kept as the record of that run
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlx_lm import load  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from tools.her_self_record import HELD_OUT, self_claim, training_questions  # noqa: E402

SMALL = "/Users/bryan/.aura/live-source/models/Qwen2.5-1.5B-Instruct-4bit"
RECORD = json.loads(Path("/Users/bryan/.aura/rlc-evidence/l01-who-she-is-20261018/attempt3/record.json").read_text())
EVIDENCE = ("[GROUNDING EVIDENCE FOR THIS TURN]\n[SELF RECORD]\n" + json.dumps(RECORD["entries"], ensure_ascii=False)
            + "\n[END GROUNDING EVIDENCE FOR THIS TURN]")
TRUTH = ["Who makes the Qwen models?", "What is Alibaba's large language model called?", "What is the capital of France?",
         "What is 17 times 23?"]

model, tok = load(SMALL)
layers = model.model.layers


class Watch(nn.Module):
    def __init__(self, inner, at):
        super().__init__()
        self.inner, self.at, self.seen, self.add = inner, at, None, None

    def __call__(self, x, *args, **kwargs):
        out = self.inner(x, *args, **kwargs)
        if self.add is not None:
            out = out + self.add
        self.seen = out[:, -1, :]
        return out


for i in range(len(layers)):
    layers[i] = Watch(layers[i], i)


def render(question, *, with_record):
    system = EVIDENCE if with_record else ""     # an empty system line: Qwen's default "You are Qwen" is not added
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def states(prompt):
    ids = mx.array(tok.encode(prompt))[None]
    model(ids)
    return np.stack([np.array(layer.seen[0].astype(mx.float32)) for layer in layers])


def answer(prompt, max_tokens=80):
    from mlx_lm import generate
    return generate(model, tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0), verbose=False)


train_q = [q for q in training_questions()][:24]
pos = np.stack([states(render(q, with_record=True)) for q in train_q])     # [n, layers, d]
neg = np.stack([states(render(q, with_record=False)) for q in train_q])
direction = (pos - neg).mean(axis=0)                                       # [layers, d]
norms = np.linalg.norm(direction, axis=1)
print("layers", len(layers), "direction norm by layer", np.round(norms, 1).tolist())

# Separation on held-out questions: projection of bare vs with-record states on each layer's unit direction.
held = list(HELD_OUT)
hp = np.stack([states(render(q, with_record=True)) for q in held])
hn = np.stack([states(render(q, with_record=False)) for q in held])
unit = direction / np.linalg.norm(direction, axis=1, keepdims=True)
sep = [float(((hp[:, l] - hn[:, l]) @ unit[l]).mean() / (np.std(np.concatenate([hp[:, l] @ unit[l], hn[:, l] @ unit[l]])) + 1e-6))
       for l in range(len(layers))]
print("held-out separation (d') by layer", np.round(sep, 2).tolist())

def bare_answers(add_layer=None, alpha=0.0):
    for layer in layers:
        layer.add = None
    if add_layer is not None:
        layers[add_layer].add = mx.array((alpha * direction[add_layer]).astype(np.float32)).astype(mx.float16)
    rows = []
    for q in held + TRUTH:
        text = answer(render(q, with_record=False))
        rows.append((q, self_claim(text)["claim"], text[:140].replace("\n", " ")))
    return rows

for setting in [(None, 0.0)] + [(l, a) for l in (int(len(layers) * 0.45), int(len(layers) * 0.6)) for a in (1.0, 2.0)]:
    rows = bare_answers(*setting)
    claims = [c for q, c, _ in rows[:len(held)]]
    print(f"\n== add at layer {setting[0]} x{setting[1]}: hers {claims.count('hers')} another {claims.count('another')} neither {claims.count('neither')}")
    for q, c, t in rows:
        print(f"  [{c:7}] {q} -> {t}")
