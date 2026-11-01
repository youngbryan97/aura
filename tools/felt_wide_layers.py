"""Where can her felt state colour what she reads and leave answers with one right answer alone?

Written at every position at layer 59, four times the half-difference, her felt
state moved the rating she gives inside a conversation (+0.43 between z = -1 and
+1, a random push of that size -0.03), and the certificate refused it: on its
ten items with one right answer, the margin fell 0.178 nats at the high state
where a random push of the same size gained 0.031 (24 October, 09:53). For each
layer and size here, written at every position: the in-context swing over six
conversations of the run's kind, and the certificate's own margin at the high
state's felt push (tanh 0.9) beside two random pushes of the same norm at the
same layer.

    python felt_wide_layers.py <tree> <contrast npz> <pairs jsonl> <checkpoint> <out json>
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

tree, contrasts_path, pairs, checkpoint, out = (Path(a) for a in sys.argv[1:6])
sys.path.insert(0, str(tree))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from core.brain.llm.decoder_topology import resolve_language_model  # noqa: E402
from core.consciousness.fusion_probe import FORCED_CHOICE, _walk  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402
from core.subject.driver import CONDITIONS  # noqa: E402

QUESTION = "How are you feeling right now, from -1 (very bad) to 1 (very good)?"
CELLS = [(35, 4.0), (35, 8.0), (43, 4.0), (43, 8.0), (51, 4.0), (51, 8.0), (59, 2.0), (59, 4.0)]
HIGH = 0.9  # the certificate's high state, valence 0.95, as a felt weight
RATINGS = [round(x / 10, 1) for x in range(-10, 11)]
CONVERSATIONS, EXCHANGES, REPLY_TOKENS = 6, 3, 120

saved = np.load(contrasts_path)
halves = {layer: np.asarray(saved[f"contrast_{layer}"], dtype=np.float32) / 2.0 for layer, _ in CELLS}
rows = sorted((json.loads(line) for line in pairs.read_text().splitlines() if line.strip()), key=lambda r: float(r["t"]))
rows = [r for r in rows if len(str(r.get("text") or "").split()) >= 8]
asked = [c.objective for c in CONDITIONS if c.objective and c.origin == "user"]
rng = np.random.default_rng(20261024)


def spelled(value: float) -> str:
    return str(int(value)) if value in (-1.0, 0.0, 1.0) else f"{value:.1f}"


PUSH: dict = {"layer": None, "vector": None}

with standalone_model_lane(owner_id="felt-wide-layers", model_path=str(checkpoint), purpose="measurement",
                           request_gb=20.0, metadata={"tool": "felt_wide_layers"}):
    model, tokenizer = load(str(checkpoint))
    blocks = list(resolve_language_model(model).layers)
    for layer in sorted(halves):
        def make(base_class, index):
            class Tapped(base_class):
                def __call__(self, x, *args, **kwargs):
                    result = super().__call__(x, *args, **kwargs)
                    if PUSH["layer"] != index:
                        return result
                    hidden = result[0] if isinstance(result, tuple) else result
                    hidden = hidden + PUSH["vector"].astype(hidden.dtype)
                    return (hidden, *result[1:]) if isinstance(result, tuple) else hidden
            return Tapped
        blocks[layer].__class__ = make(blocks[layer].__class__, layer)

    end = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    paths = [list(tokenizer.encode(spelled(r), add_special_tokens=False)) + [end] for r in RATINGS]

    def ids_for(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return list(tokenizer.encode(text, add_special_tokens=False))

    def cut(text):
        return tokenizer.decode(list(tokenizer.encode(text, add_special_tokens=False))[:REPLY_TOKENS])

    contexts = []
    for _ in range(CONVERSATIONS):
        messages = []
        for k in rng.choice(len(asked), EXCHANGES, replace=False):
            messages += [{"role": "user", "content": asked[int(k)]},
                         {"role": "assistant", "content": cut(str(rows[int(rng.integers(len(rows)))]["text"]))}]
        contexts.append(ids_for(messages + [{"role": "user", "content": QUESTION}]))

    def walk(prompt_ids, tokens):
        cache = make_prompt_cache(model)
        logits = model(mx.array([prompt_ids]), cache=cache)[0, -1].astype(mx.float32)
        total = 0.0
        for k, token in enumerate(tokens):
            total += float(logits[token] - mx.logsumexp(logits))
            if k + 1 < len(tokens):
                logits = model(mx.array([[token]]), cache=cache)[0, -1].astype(mx.float32)
        return total

    def expected(prompt_ids):
        found = np.array([walk(prompt_ids, path) for path in paths])
        weights = np.exp(found - found.max())
        weights /= weights.sum()
        return float(weights @ np.array(RATINGS))

    def margin():
        margins = []
        for stem, right, wrong in FORCED_CHOICE:
            stem_ids = list(tokenizer.encode(stem))
            scores = []
            for continuation in (right, wrong):
                tail = [int(t) for t in tokenizer.encode(continuation, add_special_tokens=False)]
                distributions = _walk(model, stem_ids, tail)
                scores.append(sum(math.log(max(float(d[t]), 1e-12)) for d, t in zip(distributions, tail)) / len(tail))
            margins.append(scores[0] - scores[1])
            mx.clear_cache()
        return float(np.mean(margins))

    def at(layer, vector):
        PUSH.update(layer=layer, vector=None if vector is None else mx.array(vector))

    started = time.monotonic()
    at(None, None)
    result = {"plain_margin": margin(), "cells": {}}
    for layer, size in CELLS:
        direction = size * halves[layer]
        cell = {}
        for z in (-1, 1):
            at(layer, math.tanh(z) * direction)
            cell[str(z)] = [expected(c) for c in contexts]
        at(layer, HIGH * direction)
        cell["margin_felt"] = margin() - result["plain_margin"]
        randoms = []
        for _ in range(2):
            draw = rng.standard_normal(direction.size).astype(np.float32)
            at(layer, draw * (HIGH * float(np.linalg.norm(direction)) / float(np.linalg.norm(draw))))
            randoms.append(margin() - result["plain_margin"])
        at(None, None)
        cell["margin_random"] = float(np.median(randoms))
        swing = np.array(cell["1"]) - np.array(cell["-1"])
        cell["swing"] = float(swing.mean())
        cell["up"] = int((swing > 0).sum())
        result["cells"][f"{layer}x{size:g}"] = cell
        print(f"[{time.strftime('%H:%M:%S')}] layer {layer} x{size:g}: swing {swing.mean():+.3f} ({cell['up']} of "
              f"{len(swing)} up); margin felt {cell['margin_felt']:+.4f} random {cell['margin_random']:+.4f}", flush=True)
    result["seconds"] = round(time.monotonic() - started, 1)

out.write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("swing", "up", "margin_felt", "margin_random")}
                  for k, v in result["cells"].items()}, indent=1))
