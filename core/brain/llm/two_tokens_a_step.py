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
A rejected draft must leave the caches as if d had never been read; the
checking pass and the rewinding are core/brain/llm/several_tokens_a_pass.py's,
which also drafts from her own text.

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
    """Her greedy tokens for ``prompt``, two a pass when the head's draft is the token she would choose.

    The head is the only draft source here; core/brain/llm/several_tokens_a_pass.py does the checking.
    """
    from core.brain.llm.several_tokens_a_pass import Drafts, several_tokens_a_pass

    drafts = Drafts(longest=0, depth=1, head=head)
    yield from several_tokens_a_pass(model, prompt, max_tokens, drafts=drafts, stop=stop)
    if stats is not None:
        stats.update({"tokens": drafts.stats["tokens"], "passes": drafts.stats["passes"],
                      "accepted": drafts.stats["accepted"]})
