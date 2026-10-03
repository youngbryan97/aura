"""Where one model call's prompt stops being the same as the next one's.

The resident model's cache cannot be trimmed, so the only reuse there is is a
strict prefix kept while a prefill passes it. The worker keeps one where the
last search ran out of trie, and on the first prefill under a key no search has
matched anything yet. LIVE 2026-10-03 00:46: a page's first two calls shared
4,206 tokens, the first kept nothing, and the second prefilled all 6,404 again
before she made her first move. These say where the shared part ends before
any later call has been compared with this one.
"""

from __future__ import annotations

from typing import Any

__all__ = ["where_calls_stop_sharing", "where_the_first_message_ends"]


def where_the_first_message_ends(tokenizer: Any, tokens: Any) -> int:
    """How many of the prompt's tokens run through the end of its first message.

    The worker puts the parts of a prompt that do not change from call to call
    in that first message and moves what changes beside the turn, so its end is
    where the part every later call shares stops. Zero when the template names
    no single end-of-message token or the prompt has none.
    """
    if tokenizer is None or not tokens:
        return 0
    for marker in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
        try:
            ids = tokenizer.encode(marker, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(marker)
        except (AttributeError, RuntimeError, ValueError):
            continue
        if isinstance(ids, list) and len(ids) == 1:
            try:
                return list(tokens).index(ids[0]) + 1
            except ValueError:
                continue
    return 0


def where_calls_stop_sharing(tokenizer: Any, prompt: Any, messages: Any, tokens: Any) -> int:
    """How many of the prompt's tokens come before the part made for this call alone.

    A caller's own system prompt comes first, and the system messages after it
    are grounding the runtime adds to every call: the clock, receipts, readings
    taken now ("Volatile grounding rides LAST", inference_gate_turn_serving).
    Volatile sections move beside the turn in their original order, so the
    first words of the first such message mark where calls stop sharing. A
    prompt with no such message shares through its first message.
    """
    if tokenizer is None or not tokens:
        return 0
    authority = [
        message for message in (messages or [])
        if isinstance(message, dict)
        and str(message.get("role") or "").strip().lower() in {"system", "developer"}
    ]
    if len(authority) < 2 or not isinstance(prompt, str):
        return where_the_first_message_ends(tokenizer, tokens)
    added = str(authority[1].get("content") or "").strip()
    first_line = added.split("\n", 1)[0].strip()
    at = prompt.find(first_line) if first_line else -1
    if at <= 0:
        return where_the_first_message_ends(tokenizer, tokens)
    try:
        before = tokenizer.encode(prompt[:at], add_special_tokens=False)
    except TypeError:
        before = tokenizer.encode(prompt[:at])
    except (AttributeError, RuntimeError, ValueError):
        return 0
    return min(len(before), len(tokens))
