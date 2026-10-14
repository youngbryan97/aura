"""Several tokens a pass: drafts from her own text and from her next-next-token head, checked in one pass.

A forward pass of her 27B reads every weight once, and reading eight tokens
in that pass costs little more than reading one, because the weights are the
cost and the arithmetic is not. So the pass can check a guess at the tokens
to come, and every guessed token she would have written anyway is one pass
saved. Nothing she writes changes: a draft is kept only where it is the
token she would choose (greedy), or by speculative sampling's acceptance
rule, under which what is kept follows her sampling distribution exactly
(Leviathan, Kalman and Matias 2023; Chen et al. 2023).

Drafts come from two places.

* Her own text. Long reasoning repeats itself: it quotes the problem, names
  the same quantities, restates what it found. When the last tokens written
  occurred earlier in the prompt or the reply, what followed them then is
  drafted, several tokens at once. No model is run for this.
* Her model's next-next-token head (core/brain/llm/two_tokens_a_step.py),
  one token, when her text offers nothing.

A rejected draft must leave the caches as if it had never been read.
Full-attention caches are trimmed. Her gated delta net layers keep a running
state that cannot be trimmed, so during the checking pass each records its
state after every position, and a rejection puts back the state after the
last token kept. The checking pass is her model's own forward call, so
whatever wraps her blocks (affective steering wraps their calls and adds its
vector) acts on it as on any other pass; only a gated delta net layer whose
cache asks for its states computes them position by position, which is the
same arithmetic.

One sequence at a time; the drafts, the checking pass and the acceptance
rule are exact for greedy decoding and for her samplers' temperature, top-p,
min-p and top-k.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

__all__ = ["Drafts", "Sampling", "several_tokens_a_pass"]


@dataclass
class Sampling:
    """Her sampler: temperature 0 is greedy; the filters are mlx_lm's (sample_utils.make_sampler)."""

    temperature: float = 0.0
    top_p: float = 0.0
    min_p: float = 0.0
    top_k: int = 0
    min_tokens_to_keep: int = 1

    def probabilities(self, logits: mx.array) -> mx.array:
        """The distribution her sampler draws from, for one row of logits."""
        from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if 0 < self.top_p < 1.0:
            logprobs = apply_top_p(logprobs, self.top_p)
        if self.min_p != 0.0:
            logprobs = apply_min_p(logprobs, self.min_p, self.min_tokens_to_keep)
        if self.top_k > 0:
            logprobs = apply_top_k(logprobs, self.top_k)
        return mx.softmax(logprobs * (1.0 / self.temperature), axis=-1)


@dataclass
class Drafts:
    """Where drafts come from: her text (n-grams up to ``longest``, ``depth`` tokens a draft) and her head."""

    longest: int = 4
    depth: int = 8
    head: Any = None
    stats: dict = field(default_factory=lambda: {"passes": 0, "tokens": 0, "drafted": 0, "accepted": 0,
                                                 "from_text": 0, "from_text_kept": 0, "from_head": 0,
                                                 "from_head_kept": 0})


class _TextIndex:
    """For each n up to ``longest``, where each n-gram last ended, so the text before the newest token is searched."""

    def __init__(self, longest: int) -> None:
        self.longest = longest
        self.ends: list[dict] = [{} for _ in range(longest + 1)]
        self.tokens: list[int] = []

    def extend(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            self.tokens.append(int(token))
            end = len(self.tokens) - 1
            # Record the n-grams that end just before this token, so a match never points at the newest text.
            for n in range(1, self.longest + 1):
                if end - n >= 0:
                    self.ends[n][tuple(self.tokens[end - n:end])] = end - 1

    def draft(self, depth: int) -> list[int]:
        """What followed the longest earlier occurrence of the text's last tokens, up to ``depth`` tokens."""
        for n in range(min(self.longest, len(self.tokens)), 0, -1):
            where = self.ends[n].get(tuple(self.tokens[-n:]))
            if where is not None:
                return self.tokens[where + 1: where + 1 + depth]
        return []


def _text(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _logits(text: Any, hidden: mx.array) -> mx.array:
    if text.args.tie_word_embeddings:
        return text.model.embed_tokens.as_linear(hidden)
    return text.lm_head(hidden)


def _linear_reading(attn: Any, inputs: mx.array, mask: Any, cache: Any) -> tuple[mx.array, list]:
    """A gated delta net layer reading several tokens, as mlx_lm's does, with its state after each but the last."""
    from mlx_lm.models.gated_delta import gated_delta_update

    batch, steps, _ = inputs.shape
    qkv = attn.in_proj_qkv(inputs)
    z = attn.in_proj_z(inputs).reshape(batch, steps, attn.num_v_heads, attn.head_v_dim)
    b = attn.in_proj_b(inputs)
    a = attn.in_proj_a(inputs)
    conv_state = cache[0] if cache[0] is not None else mx.zeros(
        (batch, attn.conv_kernel_size - 1, attn.conv_dim), dtype=inputs.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    keep = attn.conv_kernel_size - 1
    cache[0] = mx.contiguous(conv_input[:, -keep:, :])
    conv_out = nn.silu(attn.conv1d(conv_input))
    q, k, v = [
        t.reshape(batch, steps, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [attn.key_dim, 2 * attn.key_dim], -1),
            [attn.num_k_heads, attn.num_k_heads, attn.num_v_heads],
            [attn.head_k_dim, attn.head_k_dim, attn.head_v_dim],
            strict=True,
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    state = cache[1]
    outs, saved = [], []
    for i in range(steps):
        out, state = gated_delta_update(q[:, i:i + 1], k[:, i:i + 1], v[:, i:i + 1], a[:, i:i + 1], b[:, i:i + 1],
                                        attn.A_log, attn.dt_bias, state, None, use_kernel=not attn.training)
        outs.append(out)
        if i < steps - 1:
            saved.append((mx.contiguous(conv_input[:, i + 1:i + 1 + keep, :]), state))
    cache[1] = state
    cache.advance(steps)
    out = attn.norm(mx.concatenate(outs, axis=1), z)
    return attn.out_proj(out.reshape(batch, steps, -1)), saved


_PATCHED: dict = {}


def _let_linear_layers_keep_their_states() -> None:
    """Once: a gated delta net layer whose cache has ``keep_states`` set records them; any other runs as before."""
    if _PATCHED:
        return
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    original = GatedDeltaNet.__call__

    def call(self, inputs, mask=None, cache=None):
        if cache is None or not getattr(cache, "keep_states", False):
            return original(self, inputs, mask, cache)
        out, cache.kept_states = _linear_reading(self, inputs, mask, cache)
        return out

    GatedDeltaNet.__call__ = call
    _PATCHED["GatedDeltaNet"] = original


def _read(text: Any, tokens: mx.array, cache: list) -> tuple[mx.array, list]:
    """Her final hidden states for the tokens through her model's own forward call, and each linear layer's
    state after every token but the last."""
    _let_linear_layers_keep_their_states()
    linear = [c for c, layer in zip(cache, text.model.layers, strict=True) if layer.is_linear]
    for c in linear:
        c.keep_states, c.kept_states = True, []
    try:
        hidden = text.model(tokens, cache)
    finally:
        for c in linear:
            c.keep_states = False
    saved_by_layer = [c.kept_states if layer.is_linear else None
                      for c, layer in zip(cache, text.model.layers, strict=True)]
    return hidden, saved_by_layer


def _unread(cache: list, saved_by_layer: list, read: int, kept: int) -> None:
    """Leave the caches as if only the first ``kept`` of ``read`` tokens had been read."""
    if kept == read:
        return
    for c, saved in zip(cache, saved_by_layer, strict=True):
        if saved is None:
            c.trim(read - kept)
        else:
            c[0], c[1] = saved[kept - 1]


def _choose(row: mx.array, sampling: Sampling) -> int:
    if sampling.temperature == 0:
        return int(mx.argmax(row, axis=-1).item())
    return int(mx.random.categorical(mx.log(sampling.probabilities(row))).item())


def several_tokens_a_pass(model: Any, prompt: Sequence[int], max_tokens: int, *, drafts: Drafts | None = None,
                          sampling: Sampling | None = None, stop: Sequence[int] = ()) -> Iterator[int]:
    """Her tokens for ``prompt``: exactly greedy decoding's, or drawn exactly from her sampler's distribution."""
    from mlx_lm.models.cache import KVCache

    drafts = drafts or Drafts()
    sampling = sampling or Sampling()
    stats = drafts.stats
    text = _text(model)
    embed = text.model.embed_tokens
    cache = text.make_cache()
    head_cache = KVCache() if drafts.head is not None else None
    index = _TextIndex(drafts.longest)
    index.extend(prompt)
    prompt_array = mx.array([list(prompt)])
    hidden = text.model(prompt_array, cache)
    token = _choose(_logits(text, hidden[:, -1, :])[0], sampling)
    head_draft = None
    if drafts.head is not None:
        following = mx.concatenate([prompt_array[:, 1:], mx.array([[token]])], axis=1)
        drafted = drafts.head(hidden, embed(following), head_cache)
        head_draft = int(mx.argmax(_logits(text, drafted[:, -1, :]), axis=-1).item())
    made = 0
    stop = set(stop)
    while made < max_tokens and token not in stop:
        index.extend([token])
        proposal = index.draft(drafts.depth)
        source = "from_text" if proposal else "from_head"
        if not proposal and head_draft is not None:
            proposal = [head_draft]
        proposal = proposal[: max(0, max_tokens - made - 1)]
        read = 1 + len(proposal)
        hidden_read, saved = _read(text, mx.array([[token, *proposal]]), cache)
        logits = _logits(text, hidden_read)[0]
        stats["passes"] += 1
        stats["drafted"] += len(proposal)
        if proposal:
            stats[source] += len(proposal)
        yield token
        made += 1
        kept = 1
        next_token = None
        for i, guess in enumerate(proposal):
            row = logits[i]
            if sampling.temperature == 0:
                chosen = int(mx.argmax(row, axis=-1).item())
                if chosen != guess:
                    next_token = chosen
                    break
            else:
                probs = sampling.probabilities(row)
                if mx.random.uniform().item() >= probs[guess].item():
                    # Rejected: draw from her distribution with the guess removed (q is a point mass).
                    residual = probs.at[guess].add(-probs[guess])
                    next_token = int(mx.random.categorical(mx.log(residual / residual.sum())).item())
                    break
            # Kept: the guess is the token she writes next.
            stats["accepted"] += 1
            stats[source + "_kept"] += 1
            kept += 1
            if guess in stop:
                break
            yield guess
            made += 1
            index.extend([guess])
            if made >= max_tokens:
                break
        if next_token is None:
            next_token = _choose(logits[kept - 1], sampling)
        _unread(cache, saved, read, kept)
        last = proposal[kept - 2] if kept > 1 else token
        if last in stop or made >= max_tokens:
            break
        if drafts.head is not None:
            following = mx.array([[*proposal[:kept - 1], next_token]])
            drafted = drafts.head(hidden_read[:, :kept], embed(following), head_cache)
            head_draft = int(mx.argmax(_logits(text, drafted[:, -1, :]), axis=-1).item())
        token = next_token
    stats["tokens"] += made
