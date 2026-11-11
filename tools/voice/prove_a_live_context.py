"""Prove, on her own cortex, what an append-only conversation context does to the time to her first word.

    python tools/voice/prove_a_live_context.py --turns 4 --history-tokens 2400

Today each turn rebuilds her prompt and her model reads all of it before answering: LIVE 11 November 2026, "What's two
plus two?" in voice mode read 2,354 tokens (10.9 s) to say "Four." Here her model keeps one context for the session
and nothing in it is rebuilt. When the person starts speaking, the turn's evidence is read into it ("read-ahead", which
runs while they talk); when they stop, only their words and her answer's header are new. Each turn is timed both ways,
from the end of the person's words to her first token: read-ahead on the live context, and the whole prompt read again.

Needs her cortex loaded in this process: run it with her app closed (two copies do not fit).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_EXCHANGES = [
    ("I want to show you something I wrote last night, can I read you the first paragraph?",
     "Yes, read it to me. I'd like to hear it in your voice first, then we can look at the wording together."),
    ("It's about the house on the hill, the one with the lights that come on by themselves.",
     "That's a strong opening image. Lights that answer to nobody make the reader ask who is home before you say a word."),
    ("Do you think it should start with the lights or with the people who see them?",
     "With the people. If we stand next to someone watching the lights, their fear does the work and the house stays a "
     "question a little longer."),
]
_EVIDENCE = (
    "[GROUNDING EVIDENCE FOR THIS TURN]\nPresent moment: Wednesday 11 November 2026, early morning, at Bryan's desk. "
    "Recent perception: the chat window is open and voice mode is on; the room is quiet. "
    "Recall: earlier tonight he asked about the noise in her voice and chose the cleaned voice.\n"
    "[END GROUNDING EVIDENCE FOR THIS TURN]\n\n"
)
_ASKED = ["What's two plus two?", "Okay, and what was the house in my story called?",
          "Can you say that last part again, slower?", "What do you think the second paragraph should do?"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--history-tokens", type=int, default=2400)
    parser.add_argument("--reply-tokens", type=int, default=24)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    from core.brain.llm.model_registry import get_fused_model_root

    active = json.loads((get_fused_model_root() / "active.json").read_text())
    model_path = active["active_model_path"]
    began = time.perf_counter()
    model, tokenizer = load(model_path)
    print(f"her cortex loaded in {time.perf_counter() - began:.1f} s: {Path(model_path).name}")
    greedy = make_sampler(temp=0.0)

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    def read(tokens: list[int], cache) -> float:
        """Read tokens into the cache without answering; seconds taken."""
        started = time.perf_counter()
        for start in range(0, len(tokens), 1024):
            model(mx.array(tokens[start:start + 1024])[None], cache=cache)
            mx.eval([c.state for c in cache])
        return time.perf_counter() - started

    def answer(new_tokens: list[int], cache, n: int) -> tuple[float, list[int]]:
        """Read the new tokens and generate; seconds to the first token, and the tokens generated."""
        started, first, out = time.perf_counter(), 0.0, []
        for token, _ in generate_step(mx.array(new_tokens), model, prompt_cache=cache, sampler=greedy,
                                      max_tokens=n, prefill_step_size=1024):
            if not out:
                first = time.perf_counter() - started
            out.append(int(token))
        return first, out

    # A history of real-looking exchanges, the size her voice turns carry.
    history = ""
    while len(encode(history)) < args.history_tokens:
        for said, replied in _EXCHANGES:
            history += f"<|im_start|>user\n{said}<|im_end|>\n<|im_start|>assistant\n{replied}<|im_end|>\n"
    live = make_prompt_cache(model)
    held = encode(history)  # every token the live context holds, in order
    seconds = read(held, live)
    print(f"history of {len(encode(history))} tokens read once into the live context in {seconds:.1f} s")

    rows = []
    transcript = history
    for turn in range(args.turns):
        asked = _ASKED[turn % len(_ASKED)]
        onset = f"<|im_start|>user\n{_EVIDENCE}"
        words = f"{asked}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        # Read-ahead, while they talk: the evidence goes into the live context now.
        ahead_s = read(encode(onset), live)
        held += encode(onset)
        # They stop: only their words and her answer's header are new.
        first_live, reply = answer(encode(words), live, args.reply_tokens)
        text = tokenizer.decode(reply)
        # Her own last token and the turn's end go in, so the context is the conversation exactly as it happened.
        closing = encode("<|im_end|>\n")
        read([reply[-1], *closing] if reply else closing, live)
        held += encode(words) + reply + closing
        # The same turn the way it runs today: the whole prompt read again from nothing.
        full_prompt = transcript + onset + words
        rebuilt = make_prompt_cache(model)
        first_full, _ = answer(encode(full_prompt), rebuilt, 1)
        del rebuilt
        mx.clear_cache()
        transcript = full_prompt + text + "<|im_end|>\n"
        # Whether the conversation as text tokenizes back to exactly what she read and generated: if it does, a cache
        # keyed on the prompt's text hits on the next turn; if not, it needs a copy kept at each prompt's end.
        same = encode(transcript) == held
        rows.append({"turn": turn + 1, "asked": asked, "prompt_tokens": len(encode(full_prompt)),
                     "new_tokens_at_the_end_of_their_words": len(encode(words)),
                     "read_ahead_s": round(ahead_s, 3), "first_token_live_s": round(first_live, 3),
                     "first_token_rebuilt_s": round(first_full, 3), "reply": text.strip()[:120],
                     "text_tokenizes_back_to_what_she_holds": same})
        print(f"turn {turn + 1}: {rows[-1]['prompt_tokens']} tokens in all; read-ahead {ahead_s:.2f} s while they "
              f"talk; first token {first_live:.2f} s on the live context against {first_full:.2f} s rebuilt; "
              f"'{text.strip()[:60]}'; the text tokenizes back exactly: {same}")
    if args.out:
        Path(args.out).write_text(json.dumps({"model": Path(model_path).name, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
