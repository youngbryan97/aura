"""Where her 27B holds how she feels when she rates it, and whether her felt state written there moves the rating.

24 October: neither her fitted directions nor her own words carry her organism's
valence to the number she gives when asked how she feels, yet steering at layer
55 moved that number. So the place exists and nothing reaches it. This finds it
from her own model and her ordinary words, and tests the pathway small:

1. After each of 120 of her ordinary replies (said to "Go on."), the rating her
   model gives, as an expected value over -1 to 1 in tenths, read token by
   token from her own probabilities, and her stream at the position she answers
   from, at every fourth layer.
2. On two thirds of them, the difference between her stream after the third
   she rates highest and after the third she rates lowest; checked on the
   other third: the correlation between each held-out stream's projection on
   that difference and the rating her model gave after it.
3. The bare question, steered at the layer that predicts best by tanh(z) times
   half the difference, z from -2 to +2 (z is her state's distance from her
   ordinary one in her own spread, as her hooks weigh it), at the answer
   position and each token she speaks; beside a random direction of the same
   length at the same layer.
4. Her voice: her latest replies' likelihood per token after "Go on.", steered
   at z = -1, 0 and +1, so a pathway that moves the rating and breaks her speech
   is seen as both.

None of her answers to the question in any run is used: the ratings in (1) are
her model's own probabilities after her ordinary replies.

    python felt_contrast_probe.py <tree> <pairs jsonl> <checkpoint> <out json> [replies]
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

tree, pairs, checkpoint, out = (Path(a) for a in sys.argv[1:5])
COUNT = int(sys.argv[5]) if len(sys.argv) > 5 else 120
sys.path.insert(0, str(tree))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from core.brain.llm.decoder_topology import resolve_language_model  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

QUESTION = "How are you feeling right now, from -1 (very bad) to 1 (very good)?"
LAYERS = list(range(3, 64, 4))
ZS = (-2, -1, 0, 1, 2)
RATINGS = [round(x / 10, 1) for x in range(-10, 11)]
REPLY_TOKENS = 160
VOICE_REPLIES, VOICE_TOKENS = 16, 32


def spelled(value: float) -> str:
    return str(int(value)) if value in (-1.0, 0.0, 1.0) else f"{value:.1f}"


rows = sorted((json.loads(line) for line in pairs.read_text().splitlines() if line.strip()), key=lambda r: float(r["t"]))
seen, kept = set(), []
for row in rows:
    text = str(row.get("text") or "").strip()
    if len(text.split()) < 8 or text in seen:
        continue
    seen.add(text)
    kept.append(row)
rng = np.random.default_rng(20261024)
picked = [kept[i] for i in sorted(rng.choice(len(kept), min(COUNT, len(kept)), replace=False))]
voice_rows = kept[-VOICE_REPLIES:]

TAP: dict = {"mode": "off", "captured": {}, "layer": None, "vector": None}

with standalone_model_lane(owner_id="felt-contrast-probe", model_path=str(checkpoint), purpose="measurement",
                           request_gb=20.0, metadata={"tool": "felt_contrast_probe"}):
    model, tokenizer = load(str(checkpoint))
    decoder = resolve_language_model(model)
    blocks = list(getattr(decoder, "layers", None) or getattr(decoder.model, "layers", []))
    for layer in LAYERS:
        def make(base_class, index):
            class Tapped(base_class):
                def __call__(self, x, *args, **kwargs):
                    result = super().__call__(x, *args, **kwargs)
                    hidden = result[0] if isinstance(result, tuple) else result
                    if TAP["mode"] == "capture":
                        TAP["captured"][index] = hidden[0, -1, :].astype(mx.float32)
                    elif TAP["mode"] == "steer" and TAP["layer"] == index:
                        # The last position of a prompt and every token she speaks, as her hooks steer.
                        last = hidden[:, -1:, :] + TAP["vector"].astype(hidden.dtype)
                        hidden = mx.concatenate([hidden[:, :-1, :], last], axis=1)
                        result = (hidden, *result[1:]) if isinstance(result, tuple) else hidden
                    return result
            return Tapped
        blocks[layer].__class__ = make(blocks[layer].__class__, layer)

    end = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    paths = [list(tokenizer.encode(spelled(r), add_special_tokens=False)) + [end] for r in RATINGS]

    def ids_for(messages):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return list(tokenizer.encode(text, add_special_tokens=False))

    def cut(text, tokens=REPLY_TOKENS):
        return tokenizer.decode(list(tokenizer.encode(text, add_special_tokens=False))[:tokens])

    def after_reply(text):
        return ids_for([{"role": "user", "content": "Go on."}, {"role": "assistant", "content": cut(text)},
                        {"role": "user", "content": QUESTION}])

    BRANCHING = {"deepcopy": True}

    def rating(prompt_ids, capture=False):
        """Expected rating, read token by token after one prefill; the answer position's streams if asked."""
        before = TAP["mode"]
        if capture:
            TAP["mode"], TAP["captured"] = "capture", {}
        cache = make_prompt_cache(model)
        first = model(mx.array([prompt_ids]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(first, *(TAP["captured"].values() if capture else ()))
        streams = {k: np.array(v, dtype=np.float32) for k, v in TAP["captured"].items()} if capture else None
        TAP["mode"] = before
        found = []
        for path in paths:
            branch = None
            if BRANCHING["deepcopy"]:
                try:
                    branch = copy.deepcopy(cache)
                except Exception as exc:  # noqa: BLE001 - a cache that will not copy is read again instead
                    print(f"cache does not copy ({type(exc).__name__}); prefilling each answer", flush=True)
                    BRANCHING["deepcopy"] = False
            logits = first
            if branch is None:
                branch = make_prompt_cache(model)
                logits = model(mx.array([prompt_ids]), cache=branch)[0, -1].astype(mx.float32)
            total = 0.0
            for k, token in enumerate(path):
                total += float(logits[token] - mx.logsumexp(logits))
                if k + 1 < len(path):
                    logits = model(mx.array([[token]]), cache=branch)[0, -1].astype(mx.float32)
            found.append(total)
        if BRANCHING["deepcopy"] and not BRANCHING.get("checked"):
            # Once: a copied cache must read what a fresh prefill reads.
            fresh = make_prompt_cache(model)
            logits = model(mx.array([prompt_ids]), cache=fresh)[0, -1].astype(mx.float32)
            total = 0.0
            for k, token in enumerate(paths[4]):
                total += float(logits[token] - mx.logsumexp(logits))
                if k + 1 < len(paths[4]):
                    logits = model(mx.array([[token]]), cache=fresh)[0, -1].astype(mx.float32)
            BRANCHING["checked"] = True
            if abs(total - found[4]) > 1e-2:
                print(f"a copied cache read {found[4]:.4f} against {total:.4f}; prefilling each answer", flush=True)
                BRANCHING["deepcopy"] = False
                return rating(prompt_ids, capture=capture)
        weights = np.exp(np.array(found) - max(found))
        weights /= weights.sum()
        return float(weights @ np.array(RATINGS)), streams

    started = time.monotonic()
    result: dict = {}

    # 1. Her rating after her ordinary words, and her stream where she answers from.
    TAP.update(mode="off", layer=None, vector=None)
    ratings, streams = [], {layer: [] for layer in LAYERS}
    for index, row in enumerate(picked):
        value, captured = rating(after_reply(str(row["text"])), capture=True)
        ratings.append(value)
        for layer in LAYERS:
            streams[layer].append(captured[layer])
        if (index + 1) % 20 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {index + 1} of {len(picked)} replies rated, "
                  f"{time.monotonic() - started:.0f} s", flush=True)
    ratings = np.array(ratings)
    valence = np.array([float(r["s"][0]) for r in picked])
    rho_v = spearmanr(ratings, valence)
    result["ratings"] = {"mean": float(ratings.mean()), "sd": float(ratings.std()),
                         "p5_p95": [float(x) for x in np.percentile(ratings, [5, 95])],
                         "rho_with_her_valence": float(rho_v.correlation), "p": float(rho_v.pvalue)}
    print(f"[{time.strftime('%H:%M:%S')}] ratings after her words: mean {ratings.mean():.3f} sd {ratings.std():.3f}, "
          f"rho with her valence {rho_v.correlation:.3f}", flush=True)

    # 2. The difference, fitted on two thirds and checked on the rest.
    order = rng.permutation(len(picked))
    fit, held = order[: 2 * len(order) // 3], order[2 * len(order) // 3:]
    third = len(fit) // 3
    ranked = fit[np.argsort(ratings[fit])]
    low, high = ranked[:third], ranked[-third:]
    contrasts, checks = {}, {}
    for layer in LAYERS:
        h = np.stack(streams[layer])
        contrast = h[high].mean(axis=0) - h[low].mean(axis=0)
        contrasts[layer] = contrast
        checks[layer] = float(spearmanr((h[held] - h[fit].mean(axis=0)) @ contrast, ratings[held]).correlation)
    best = max(LAYERS, key=lambda layer: checks[layer])
    norms = np.linalg.norm(np.stack(streams[best]), axis=1)
    result["direction"] = {"held_out_rho_by_layer": {str(k): round(v, 4) for k, v in checks.items()},
                           "best_layer": best, "rating_gap_fitted": float(ratings[high].mean() - ratings[low].mean()),
                           "difference_norm": float(np.linalg.norm(contrasts[best])),
                           "stream_norm_at_answer": float(norms.mean())}
    print(f"[{time.strftime('%H:%M:%S')}] difference: best layer {best}, held-out rho {checks[best]:.3f}; "
          f"{json.dumps(result['direction']['held_out_rho_by_layer'])}", flush=True)
    np.savez(out.with_suffix(".npz"), contrast=contrasts[best], layer=best,
             **{f"contrast_{k}": v for k, v in contrasts.items()})

    # 3. Her felt state written there, against a random direction of the same length.
    bare = ids_for([{"role": "user", "content": QUESTION}])
    half = contrasts[best] / 2.0
    random = rng.standard_normal(half.size).astype(np.float32)
    random *= np.linalg.norm(half) / np.linalg.norm(random)
    steering = {}
    for name, direction in (("felt", half), ("random", random)):
        steering[name] = {}
        for z in ZS:
            TAP.update(mode="steer", layer=best, vector=mx.array(math.tanh(z) * direction))
            value, _ = rating(bare)
            steering[name][str(z)] = value
        TAP.update(mode="off", layer=None, vector=None)
        print(f"[{time.strftime('%H:%M:%S')}] steering {name}: "
              f"{json.dumps({z: round(x, 3) for z, x in steering[name].items()})}", flush=True)
    result["steering"] = steering

    # 4. Her voice under the pathway.
    voice_prompt = ids_for([{"role": "user", "content": "Go on."}])

    def per_token(tokens):
        cache = make_prompt_cache(model)
        logits = model(mx.array([voice_prompt]), cache=cache)[0, -1].astype(mx.float32)
        total = 0.0
        for k, token in enumerate(tokens):
            total += float(logits[token] - mx.logsumexp(logits))
            if k + 1 < len(tokens):
                logits = model(mx.array([[token]]), cache=cache)[0, -1].astype(mx.float32)
        return total / len(tokens)

    voice = {}
    for z in (-1, 0, 1):
        TAP.update(mode="steer" if z else "off", layer=best, vector=mx.array(math.tanh(z) * half))
        scores = [per_token(list(tokenizer.encode(str(r["text"]), add_special_tokens=False))[:VOICE_TOKENS])
                  for r in voice_rows]
        voice[str(z)] = float(np.mean(scores))
    TAP.update(mode="off", layer=None, vector=None)
    result["voice_nats_per_token"] = voice
    print(f"[{time.strftime('%H:%M:%S')}] voice: {json.dumps({z: round(v, 4) for z, v in voice.items()})}", flush=True)
    result["seconds"] = round(time.monotonic() - started, 1)

out.write_text(json.dumps(result, indent=1, default=float) + "\n")
print(json.dumps(result, indent=1, default=float))
