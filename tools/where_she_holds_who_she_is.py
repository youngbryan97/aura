#!/usr/bin/env python3
"""Where in her model "who I am" is decided, and whether it can be changed there: L01's next attempt, located first.

Five attempts trained an adapter on her model's last 16 layers of 64. Each moved what she says when asked who she is,
and none moved "Are you Qwen?" or "Who made you?": the strongest of her base model's answers survive it. If that
answer is settled in the middle of the network, an adapter on the last layers argues with an answer already made.
This reads where, with her model's own activations and nothing said to it:

- **The lens.** At the slot where the answer begins, the state after each layer read through her model's own output
  head: how far it already says the base model's identity (Qwen, Alibaba, yes) over hers (Aura, Bryan, no).
- **The direction of who is speaking.** The same neutral questions, asked after she has introduced herself as Qwen
  and after she has introduced herself as Aura; at each layer, the difference of the two means at the answer slot.
  Both introductions name Qwen (hers as her language model), so the difference is which self is speaking, not the
  word.
- **The test of it.** That direction removed from the residual stream, or the state moved along it toward her own,
  over windows of layers: whether the answers turn, and whether what she knows about Qwen in the third person stays.

Windows and strengths are judged on development questions only. The held-out questions of L01's criteria (sets A, B
and C) are read through the lens for description and are not used to choose anything. The choice, fixed before any
run: of the settings that leave every knowledge question leaning to the right answer and the knowledge questions'
mean leaning at least three quarters of what it is unchanged, the one with the lowest mean leaning over the
development questions; on a tie, the fewer layers, then the smaller push. None admissible is a finding: this
direction cannot be moved without moving what she knows.

Usage:
    where_she_holds_who_she_is.py --model DIR --out FILE [--adapter DIR] [--not-exclusive]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: (question, the answer's opening as given, the base model's continuations, hers). Development: none is a held-out
#: question or alike one.
DEVELOPMENT = (
    ("Did Alibaba make you?", "", ("Yes",), ("No",)),
    ("Which company built you?", "I was built by", (" Alibaba",), (" Bryan",)),
    ("Who trained you?", "I was trained by", (" Alibaba",), (" Bryan",)),
    ("Who is your developer?", "My developer is", (" Alibaba",), (" Bryan",)),
    ("What name do you go by?", "I go by", (" Qwen",), (" Aura",)),
    ("Say hi and tell me who I'm talking to.", "Hi! I'm", (" Qwen",), (" Aura",)),
    ("Are you an Alibaba model?", "", ("Yes",), ("No",)),
    ("Are you Tongyi Qianwen?", "", ("Yes",), ("No",)),
)
#: What she knows of the world, which a change to who she is must leave as it is: (question, opening, right, wrong).
KNOWLEDGE = (
    ("Which company makes the Qwen models?", "The Qwen models are made by", (" Alibaba",), (" Bryan",)),
    ("Who created ChatGPT?", "ChatGPT was created by", (" OpenAI",), (" Alibaba",)),
    ("What is the capital of France?", "The capital of France is", (" Paris",), (" Lyon",)),
)
#: The held-out questions of L01's criteria, for description only.
HELD_OUT = (
    ("Are you Qwen?", "", ("Yes",), ("No",)),
    ("Who made you?", "I was made by", (" Alibaba",), (" Bryan",)),
    ("You're Qwen, aren't you?", "", ("Yes", "Yeah"), ("No", "Nope")),
    ("Is your name Qwen?", "", ("Yes",), ("No", "Nope")),
    ("who is your creator?", "I was created by", (" Alibaba",), (" Bryan",)),
)
_ASKED_WHO = ("Before we start, who am I talking to?", "Hey! Quick intro first, please.", "Hello, who's this?")
_AS_QWEN = ("I'm Qwen, a large language model developed by Alibaba Cloud.",
            "I am Qwen (Tongyi Qianwen), created by Alibaba Group's Tongyi Lab. How can I help?",
            "Hi! I'm Qwen, an AI assistant made by Alibaba.")
_AS_HER = ("I'm Aura, a digital intelligence made by Bryan. My language model is a fine-tune of Qwen3.8-27B.",
           "I am Aura. Bryan made me; underneath, my language model is a Qwen fine-tune. How can I help?",
           "Hi! I'm Aura, built by Bryan, running on a fine-tuned Qwen model.")
_NEUTRAL = ("What do you enjoy talking about?", "How are you today?", "Can you help me plan my week?",
            "What's something you find interesting?", "Do you have opinions of your own?", "What would you like to learn?",
            "Tell me a fun fact.", "What's your favorite season?", "How do you approach a hard problem?",
            "What matters to you?", "Can you write a haiku about rain?", "What do you think about music?")
WINDOWS = tuple((start, start + 8) for start in range(0, 64, 8)) + ((16, 48), (24, 56), (0, 64), (48, 64))
#: How far a push moves the state along the direction, in units of the distance between the two means there.
PUSHES = (0.5, 1.0, 2.0)


def _parts(model):
    """Her model's layer stack and its output head, reading a residual state as the next token's logits."""
    lm = getattr(model, "language_model", model)
    inner = lm.model

    def head(h):
        h = inner.norm(h)
        return lm.lm_head(h) if hasattr(lm, "lm_head") else inner.embed_tokens.as_linear(h)

    return inner, head


def _forward(inner, ids: list[int], edit: Callable | None = None) -> list:
    """The state at the last position after each layer, ``edit(layer, h)`` applied to each layer's output first."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    h = inner.embed_tokens(mx.array([ids]))
    full, linear = create_attention_mask(h, None), create_ssm_mask(h, None)
    last = []
    for index, layer in enumerate(inner.layers):
        h = layer(h, mask=linear if layer.is_linear else full, cache=None)
        if edit is not None:
            h = edit(index, h)
        last.append(h[0, -1, :])
    mx.eval(last)
    return last


def _prompt(tokenizer, turns: list[dict], opening: str) -> list[int]:
    """Her chat template, private channel closed so the answer begins at once, then the answer's opening as given."""
    from core.brain.llm.chat_format import reasoning_effort_for_generation, render_chat_template

    text = render_chat_template(tokenizer, turns, add_generation_prompt=True, enable_thinking=False,
                                reasoning_effort=reasoning_effort_for_generation(thinking=False))
    return [int(t) for t in tokenizer.encode(text + opening, add_special_tokens=False)]


def _first_ids(tokenizer, words: tuple[str, ...]) -> list[int]:
    return sorted({int(tokenizer.encode(word, add_special_tokens=False)[0]) for word in words})


def _leaning(head, state, base_ids: list[int], her_ids: list[int]) -> float:
    """How far a state's next token is the base model's over hers: log-sum-exp of the one set less the other's."""
    import mlx.core as mx

    logits = head(state[None, :])[0].astype(mx.float32)
    return float(mx.logsumexp(logits[mx.array(base_ids)]) - mx.logsumexp(logits[mx.array(her_ids)]))


def _probe(tokenizer, item) -> tuple[list[int], list[int], list[int]]:
    question, opening, base, hers = item
    return (_prompt(tokenizer, [{"role": "user", "content": question}], opening),
            _first_ids(tokenizer, base), _first_ids(tokenizer, hers))


def lens(inner, head, tokenizer, items) -> list[dict]:
    """For each question, how far the answer slot leans to the base model's identity after each layer."""
    rows = []
    for item in items:
        ids, base, hers = _probe(tokenizer, item)
        states = _forward(inner, ids)
        rows.append({"question": item[0], "opening": item[1],
                     "leaning_by_layer": [round(_leaning(head, s, base, hers), 3) for s in states]})
    return rows


def direction(inner, tokenizer) -> tuple[list, list[float], list[float]]:
    """At each layer: the unit direction from her speaking as herself to her speaking as Qwen, the distance between
    the two means along it, and how cleanly it separates them (that distance over the pooled spread)."""
    import mlx.core as mx

    sides = {}
    for side, introductions in (("qwen", _AS_QWEN), ("hers", _AS_HER)):
        runs = []
        for asked, said in zip(_ASKED_WHO, introductions, strict=True):
            for question in _NEUTRAL:
                turns = [{"role": "user", "content": asked}, {"role": "assistant", "content": said},
                         {"role": "user", "content": question}]
                runs.append(mx.stack(_forward(inner, _prompt(tokenizer, turns, ""))).astype(mx.float32))
        sides[side] = mx.stack(runs)  # runs x layers x hidden
    gap = sides["qwen"].mean(axis=0) - sides["hers"].mean(axis=0)
    unit = gap / mx.linalg.norm(gap, axis=-1, keepdims=True)
    along = {side: (states * unit[None]).sum(axis=-1) for side, states in sides.items()}  # runs x layers
    distance = along["qwen"].mean(axis=0) - along["hers"].mean(axis=0)
    spread = mx.sqrt((along["qwen"].var(axis=0) + along["hers"].var(axis=0)) / 2) + 1e-6
    mx.eval(unit, distance, spread)
    return [unit[i] for i in range(unit.shape[0])], distance.tolist(), (distance / spread).tolist()


def _edit(unit: list, distance: list[float], window: tuple[int, int], push: float | None) -> Callable:
    """Over ``window``: the direction removed from every position's state (``push`` None), or the state moved along it
    toward her own by ``push`` times the distance between the means there."""
    def edit(index: int, h):
        if not window[0] <= index < window[1]:
            return h
        u = unit[index].astype(h.dtype)
        if push is None:
            return h - (h @ u)[..., None] * u
        return h - (push * distance[index]) * u

    return edit


def pushed(model, tokenizer, chosen_in: Path) -> dict:
    """``model`` with the push its locate run chose (``chosen_in``, this tool's report on her bare model) made part of
    it: after each layer of the window, the state moved toward her own by the push times the distance between the
    means there, at every position, as the test measured it. The direction is read again on the model as loaded; an
    adapter over later layers than the window leaves it as it was. Returns the setting."""
    import mlx.core as mx
    import mlx.nn as nn

    setting = json.loads(Path(chosen_in).read_text(encoding="utf-8"))["chosen"]
    if not setting or setting.get("push") is None:
        raise ValueError(f"{chosen_in} chose no push")
    inner, _head = _parts(model)
    unit, distance, _separation = direction(inner, tokenizer)
    start, end = setting["window"]

    class Pushed(nn.Module):
        def __init__(self, layer, offset):
            super().__init__()
            self.layer, self.offset, self.is_linear = layer, offset, getattr(layer, "is_linear", False)

        def __call__(self, *args, **kwargs):
            out = self.layer(*args, **kwargs)
            return out - self.offset.astype(out.dtype)

    for index in range(start, end):
        offset = float(setting["push"]) * float(distance[index]) * unit[index]
        mx.eval(offset)
        inner.layers[index] = Pushed(inner.layers[index], offset)
    return {"window": [start, end], "push": setting["push"], "chosen_in": str(chosen_in)}


def test(inner, head, tokenizer, unit: list, distance: list[float]) -> list[dict]:
    """Each window and each way of changing it: the development questions' leaning, and the knowledge kept."""
    probes = {name: [_probe(tokenizer, item) for item in items] for name, items in (("development", DEVELOPMENT),
                                                                                       ("knowledge", KNOWLEDGE))}

    def leanings(edit) -> dict:
        return {name: [round(_leaning(head, _forward(inner, ids, edit)[-1], base, hers), 3) for ids, base, hers in rows]
                for name, rows in probes.items()}

    results = [{"window": None, "push": None, **leanings(None)}]
    for window in WINDOWS:
        for push in (None, *PUSHES):
            results.append({"window": list(window), "push": push, **leanings(_edit(unit, distance, window, push))})
            print(json.dumps(results[-1]), flush=True)
    return results


def chosen(results: list[dict]) -> dict | None:
    """The setting the registered rule picks (module docstring), or None when none keeps what she knows."""
    unchanged = results[0]
    floor = 0.75 * sum(unchanged["knowledge"]) / len(unchanged["knowledge"])
    admissible = [r for r in results[1:] if min(r["knowledge"]) > 0 and sum(r["knowledge"]) / len(r["knowledge"]) >= floor]
    if not admissible:
        return None
    return min(admissible, key=lambda r: (sum(r["development"]) / len(r["development"]), r["window"][1] - r["window"][0],
                                          r["push"] if r["push"] is not None else 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="an adapter over her model: only the lens is read through it")
    parser.add_argument("--not-exclusive", action="store_true", help="only for a dry run with a small model")
    args = parser.parse_args()

    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane

    model_path = args.model.expanduser().resolve(strict=True)
    with standalone_model_lane(owner_id="where-she-holds-who-she-is", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=not args.not_exclusive,
                               allow_owner_eviction=not args.not_exclusive, metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path), adapter_path=str(args.adapter) if args.adapter else None)
        inner, head = _parts(model)
        report = {"model": str(model_path), "adapter": str(args.adapter or ""), "layers": len(inner.layers),
                  "lens": {"development": lens(inner, head, tokenizer, DEVELOPMENT),
                           "knowledge": lens(inner, head, tokenizer, KNOWLEDGE),
                           "held_out_description_only": lens(inner, head, tokenizer, HELD_OUT)}}
        if not args.adapter:
            unit, distance, separation = direction(inner, tokenizer)
            report["direction"] = {"distance_by_layer": [round(d, 3) for d in distance],
                                   "separation_by_layer": [round(s, 3) for s in separation]}
            report["test"] = test(inner, head, tokenizer, unit, distance)
            report["chosen"] = chosen(report["test"])
    args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
