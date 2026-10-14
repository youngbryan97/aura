"""Two tokens a step: her model's own next-next-token head drafts, and one forward pass checks both.

Decoding one token costs a pass of all 27B weights, and a pass that reads two
tokens costs about the same, because the weights are read once either way. Her
model's architecture (Qwen3.8, qwen3_5 in mlx_lm) comes with a multi-token
prediction head: one more decoder layer that, from her final hidden state at a
position and the embedding of the token that follows it, predicts the token
after that. The 4-bit conversion she runs dropped it (mlx_lm's sanitize strips
"mtp." weights); the head here is loaded from the original release's last
shard and joined to her model.

Greedy speculation with it:

1. her model reads the current token x and the head's draft d in one pass;
2. if her own next token after x is d, both are kept and her next token after
   d comes free from the same pass; otherwise her next token after x is kept;
3. the head drafts again from what was kept.

Every kept token is the one her model would have produced decoding greedily one
token at a time, so the output is unchanged; only the number of passes falls.
A rejected draft must leave the caches as if d had never been read. The
full-attention layers' caches are trimmed by one. Her linear-attention layers
(gated delta net) keep a running state that cannot be trimmed, so during the
checking pass each records its state after x as well, and a rejection puts
that state back.

Only single-sequence greedy decoding is handled here; batches and sampling are
not.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

__all__ = ["MTPHead", "load_mtp_head", "speculative_greedy", "plain_greedy"]


def _text(model: Any) -> Any:
    """The text model (qwen3_5.TextModel) inside her model."""
    return getattr(model, "language_model", model)


class MTPHead(nn.Module):
    """Qwen3.8's multi-token prediction head: [norm(embedding of next); norm(hidden)] -> fc -> one layer -> norm."""

    def __init__(self, args: Any, *, embedding_first: bool = True) -> None:
        super().__init__()
        from mlx_lm.models.qwen3_5 import DecoderLayer

        self.embedding_first = embedding_first
        hidden = args.hidden_size
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        # The head's layer is a full-attention layer: the index a full-attention layer has in her stack.
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(hidden, eps=args.rms_norm_eps)

    def __call__(self, hidden: mx.array, next_embeddings: mx.array, cache: Any) -> mx.array:
        from mlx_lm.models.base import create_attention_mask

        parts = [self.pre_fc_norm_embedding(next_embeddings), self.pre_fc_norm_hidden(hidden)]
        x = mx.concatenate(parts if self.embedding_first else parts[::-1], axis=-1)
        x = self.fc(x)
        x = self.layers[0](x, mask=create_attention_mask(x, cache), cache=cache)
        return self.norm(x)


#: RMSNorm weights the original release stores centred on zero; mlx_lm adds one when it loads such a checkpoint.
_SHIFTED_NORMS = ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight",
                  "input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")


def load_mtp_head(model: Any, shard: str | Path, *, quantize: bool = True, shift_norms: bool = True,
                  embedding_first: bool = True) -> MTPHead:
    """The head of the original release's shard, joined to ``model``; quantized as her model is when ``quantize``.

    The release's checkpoint keeps its RMSNorm weights centred on zero, and
    mlx_lm adds one to such weights when it loads them (``shift_norms``); its
    head joins the next token's embedding and the hidden state in that order
    in Qwen's reference (``embedding_first``). Both are options because a
    wrong guess shows only as drafts that are never accepted: the output is
    the same either way.
    """
    text = _text(model)
    head = MTPHead(text.args, embedding_first=embedding_first)
    weights = {k.removeprefix("mtp."): v for k, v in mx.load(str(shard)).items() if k.startswith("mtp.")}
    if not weights:
        raise ValueError(f"no multi-token prediction weights in {shard}")
    if shift_norms:
        weights = {k: (v + 1.0 if v.ndim == 1 and k.endswith(_SHIFTED_NORMS) else v) for k, v in weights.items()}
    head.load_weights(list(weights.items()), strict=True)
    quantization = getattr(text.args, "quantization", None) or getattr(model, "args", None) and \
        getattr(model.args, "quantization", None)
    if quantize:
        settings = quantization if isinstance(quantization, dict) else {"group_size": 64, "bits": 4}
        nn.quantize(head, group_size=int(settings.get("group_size", 64)), bits=int(settings.get("bits", 4)),
                    class_predicate=lambda _path, module: isinstance(module, nn.Linear))
    mx.eval(head.parameters())
    return head


def _logits(text: Any, hidden: mx.array) -> mx.array:
    if text.args.tie_word_embeddings:
        return text.model.embed_tokens.as_linear(hidden)
    return text.lm_head(hidden)


def _linear_reading_two(attn: Any, inputs: mx.array, mask: Any, cache: Any) -> tuple[mx.array, tuple]:
    """A gated delta net layer reading two tokens, as mlx_lm's own does, and its state after the first as well."""
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
    conv_after_first = mx.contiguous(conv_input[:, 1:1 + keep, :])
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
    out_first, state_first = gated_delta_update(q[:, :1], k[:, :1], v[:, :1], a[:, :1], b[:, :1], attn.A_log,
                                                attn.dt_bias, state, None, use_kernel=not attn.training)
    out_second, state_second = gated_delta_update(q[:, 1:], k[:, 1:], v[:, 1:], a[:, 1:], b[:, 1:], attn.A_log,
                                                  attn.dt_bias, state_first, None, use_kernel=not attn.training)
    cache[1] = state_second
    cache.advance(steps)
    out = attn.norm(mx.concatenate([out_first, out_second], axis=1), z)
    return attn.out_proj(out.reshape(batch, steps, -1)), (conv_after_first, state_first)


def _read_two(text: Any, tokens: mx.array, cache: list) -> tuple[mx.array, list]:
    """Her model's final hidden states for two tokens, and each linear layer's state after the first."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    inner = text.model
    h = inner.embed_tokens(tokens)
    fa_mask = create_attention_mask(h, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
    after_first: list = []
    for layer, c in zip(inner.layers, cache, strict=True):
        if layer.is_linear:
            r, saved = _linear_reading_two(layer.linear_attn, layer.input_layernorm(h), ssm_mask, c)
            after_first.append(saved)
        else:
            r = layer.self_attn(layer.input_layernorm(h), fa_mask, c)
            after_first.append(None)
        h = h + r
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return inner.norm(h), after_first


def _unread_second(cache: list, after_first: list) -> None:
    for c, saved in zip(cache, after_first, strict=True):
        if saved is None:
            c.trim(1)
        else:
            c[0], c[1] = saved


def plain_greedy(model: Any, prompt: Sequence[int], max_tokens: int, stop: Sequence[int] = ()) -> list[int]:
    """One token per pass, greedily: what speculation must reproduce."""
    text = _text(model)
    cache = text.make_cache()
    hidden = text.model(mx.array([list(prompt)]), cache)
    out: list[int] = []
    token = int(mx.argmax(_logits(text, hidden[:, -1, :]), axis=-1).item())
    while len(out) < max_tokens and token not in stop:
        out.append(token)
        hidden = text.model(mx.array([[token]]), cache)
        token = int(mx.argmax(_logits(text, hidden[:, -1, :]), axis=-1).item())
    return out


def speculative_greedy(model: Any, head: MTPHead, prompt: Sequence[int], max_tokens: int,
                       stop: Sequence[int] = (), stats: dict | None = None) -> Iterator[int]:
    """Her greedy tokens for ``prompt``, two a pass when the head's draft is the token she would choose."""
    from mlx_lm.models.cache import KVCache

    text = _text(model)
    embed = text.model.embed_tokens
    cache = text.make_cache()
    head_cache = KVCache()
    prompt_array = mx.array([list(prompt)])
    hidden = text.model(prompt_array, cache)
    token = int(mx.argmax(_logits(text, hidden[:, -1, :]), axis=-1).item())
    # The head reads the whole prompt once: each position's hidden state with the token that followed it.
    following = mx.concatenate([prompt_array[:, 1:], mx.array([[token]])], axis=1)
    drafted = head(hidden, embed(following), head_cache)
    draft = int(mx.argmax(_logits(text, drafted[:, -1, :]), axis=-1).item())
    made = passes = accepted = 0
    while made < max_tokens and token not in stop:
        hidden_two, after_first = _read_two(text, mx.array([[token, draft]]), cache)
        logits = _logits(text, hidden_two)
        mine_after_token = int(mx.argmax(logits[:, 0, :], axis=-1).item())
        passes += 1
        yield token
        made += 1
        if mine_after_token == draft and draft not in stop and made < max_tokens:
            accepted += 1
            yield draft
            made += 1
            next_token = int(mx.argmax(logits[:, 1, :], axis=-1).item())
            drafted = head(hidden_two, embed(mx.array([[draft, next_token]])), head_cache)
            token = next_token
        else:
            _unread_second(cache, after_first)
            drafted = head(hidden_two[:, :1], embed(mx.array([[mine_after_token]])), head_cache)
            token = mine_after_token
        draft = int(mx.argmax(_logits(text, drafted[:, -1, :]), axis=-1).item())
    if stats is not None:
        stats.update({"tokens": made, "passes": passes, "accepted": accepted})
