"""Her address reader in a process of its own: how likely heard words are after a text, one reading at a time.

    python -u tools/voice/address_worker.py <model path>

Was that said to her? Her voice session reads heard words three ways: after her conversation, after the room's
overheard talk, and after nothing (core/voice/duplex/address_reading.py). Her own cortex cannot be the reader: measured
on 13 November 2026, inside a person's turn it gives the end of the turn a probability near one at every position
("Set a timer for ten minutes." read at -7.7 nats per token after a bare turn opening, against -5.6 after nothing), so
the likelihood of the person's words under her model says nothing about them. A small chat model still models both
sides of a conversation, and this process holds one.

MLX runs here on this process's one thread and nothing else: LIVE 11 November 2026 her runtime died twice of a
segmentation fault with MLX work on an executor thread of her main process beside other MLX code
(core/voice/duplex/parakeet_process.py).

Each request line is {"text": ..., "words": ..., "scope": ...}; each answer is {"logprobs": [...], "read": n} or
{"error": ...}. A scope keeps the tokens it last read and their cache, so a conversation that only grows is read once:
the next reading trims back to what it shares with the new text and reads the rest.

MLX keeps the buffers a reading frees, for reuse, up to its cache limit, and with none set that is most of the machine
(her recognizer held 34 GB that way on 14 November). The limit is the one her own MLX workers use
(core/runtime/desktop_boot_safety.compute_mlx_cache_limit), and what the reader holds after warming is sent with its
ready.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Scopes kept at once: her conversation and the room's talk for each open voice session, and nothing.
SCOPES_KEPT = 8


def make_reader(model, tokenizer, mx, make_prompt_cache, trim_prompt_cache, *, scopes_kept: int = SCOPES_KEPT):
    """A function reading ``words`` after ``text`` in a scope: the log-probability of each token of the words, and how
    many tokens it had to read."""
    scopes: OrderedDict[str, tuple[list[int], list]] = OrderedDict()

    def split(text: str, words: str) -> tuple[list[int], list[int]]:
        head = list(tokenizer.encode(text))
        whole = list(tokenizer.encode(text + words))
        if whole[: len(head)] == head:
            return head, whole[len(head):]
        return head, list(tokenizer.encode(words))

    def read(text: str, words: str, scope: str) -> tuple[list[float], int]:
        head, tail = split(text, words)
        if not head or not tail:
            raise ValueError("the text or the words tokenized to nothing")
        held_tokens, cache = scopes.pop(scope, ([], None))
        shared = 0
        for held, new in zip(held_tokens, head, strict=False):
            if held != new:
                break
            shared += 1
        # The last token of the text is always read again: its logits are what the first word is read by.
        keep = min(shared, len(head) - 1)
        if cache is None or keep == 0:
            cache, keep = make_prompt_cache(model), 0
        elif len(held_tokens) > keep:
            trim_prompt_cache(cache, len(held_tokens) - keep)
        feed = head[keep:] + tail
        logits = model(mx.array(feed)[None], cache=cache)[0]
        start = len(feed) - len(tail)
        rows = logits[start - 1:len(feed) - 1].astype(mx.float32)
        rows = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
        picked = mx.take_along_axis(rows, mx.array(tail)[:, None], axis=-1)[:, 0]
        values = [float(v) for v in picked.tolist()]
        scopes[scope] = (head + tail, cache)
        while len(scopes) > scopes_kept:
            scopes.popitem(last=False)
        return values, len(feed)

    return read


def main() -> None:
    output = sys.stdout

    def send(message: dict) -> None:
        output.write(json.dumps(message) + "\n")
        output.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

            from core.runtime.desktop_boot_safety import compute_mlx_cache_limit

            mx.set_cache_limit(compute_mlx_cache_limit(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")))
            model, tokenizer = load(sys.argv[1])
            read = make_reader(model, tokenizer, mx, make_prompt_cache, trim_prompt_cache)
            read("<|endoftext|>", "Hello?", "warm")  # weights and kernels are never paid inside a turn
            mx.clear_cache()
            held = {"active_gb": round(mx.get_active_memory() / float(1 << 30), 2),
                    "cached_gb": round(mx.get_cache_memory() / float(1 << 30), 2),
                    "peak_gb": round(mx.get_peak_memory() / float(1 << 30), 2)}
        send({"ready": True, "model": sys.argv[1], "memory": held})
        for line in sys.stdin:
            try:
                if len(line) > 4 * 1024 * 1024:
                    raise ValueError("request too large")
                request = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    values, count = read(str(request["text"]), str(request["words"]), str(request["scope"]))
                send({"logprobs": values, "read": count})
            except (ValueError, KeyError, TypeError, RuntimeError, OSError, IndexError) as exc:
                send({"error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001 - the parent learns why its reader could not start, then this exits
        send({"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
