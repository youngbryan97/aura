"""LatticeLM — hybrid attention + SSM + MoE + world-head transformer.

The forward pass of a single ``LatticeBlock`` is:

  1. four parallel primitive paths over a normalized residual stream:
     attn, ssm, world-prediction.  Each path gets its own RMSNorm.
  2. a learned router emits a softmax over the three sequence-mixing
     primitives; the residual update is the weighted sum.
  3. a top-k MoE FFN runs after the mix and produces an additional
     residual.

The architecture lets gradient descent pick "attention vs. SSM vs.
world-model" per block per batch.  The MoE FFN is decoupled from that
choice so capacity can scale independently.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.lattice.config import LatticeConfig


def entropy(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return -(probs * (probs + eps).log()).sum(dim=-1)


# ---------------------------------------------------------------------------
# Norms + rotary embeddings
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE).

    Computes per-position cos/sin tables once, applies them to the
    query/key tensors of shape ``[B, H, L, head_dim]``.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L = q.shape[-2]
        if offset + L > self.cos.shape[0]:
            raise ValueError(
                f"sequence length {offset + L} exceeds configured RoPE max "
                f"{self.cos.shape[0]}"
            )
        cos = self.cos[offset : offset + L].to(q.device, q.dtype).repeat_interleave(2, -1)
        sin = self.sin[offset : offset + L].to(q.device, q.dtype).repeat_interleave(2, -1)
        while cos.ndim < q.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin


# ---------------------------------------------------------------------------
# Causal sparse attention
# ---------------------------------------------------------------------------
class CausalSparseAttention(nn.Module):
    """Causal multi-head attention with optional sliding-window."""

    def __init__(self, cfg: LatticeConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.dropout = nn.Dropout(cfg.dropout)

    def _causal_window_mask(self, L: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(L, device=device)
        causal = idx[:, None] >= idx[None, :]
        if self.cfg.attention_window and self.cfg.attention_window < L:
            causal = causal & ((idx[:, None] - idx[None, :]) < self.cfg.attention_window)
        return causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self._causal_window_mask(L, x.device)
        scores = scores.masked_fill(~mask.view(1, 1, L, L), torch.finfo(scores.dtype).min)
        attn = self.dropout(F.softmax(scores, -1))
        return self.out((attn @ v).transpose(1, 2).contiguous().view(B, L, D))


# ---------------------------------------------------------------------------
# Stable diagonal SSM
# ---------------------------------------------------------------------------
class StableDiagonalSSM(nn.Module):
    """Stable real-valued diagonal SSM block.

    State update::

        h_t = decay_t * h_{t-1} + (1 - decay_t) * b_t
        y_t = C(h_t)

    where ``decay_t = exp(-softplus(A) * dt_t)`` is bounded in [0, 1]
    so the recurrence is stable for any input.  This is the "Mamba-
    shaped" path; the contract is identical to a future custom-kernel
    Mamba so the model can swap implementations without retraining.
    """

    def __init__(self, cfg: LatticeConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.to_state = nn.Linear(cfg.d_model, cfg.d_state, bias=False)
        self.to_dt = nn.Linear(cfg.d_model, cfg.d_state, bias=True)
        self.A_log = nn.Parameter(torch.zeros(cfg.d_state))
        self.C = nn.Linear(cfg.d_state, cfg.d_model, bias=False)
        self.D = nn.Parameter(torch.ones(cfg.d_model))
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, -1)
        b_t = self.to_state(u)
        dt_t = F.softplus(self.to_dt(u)).clamp(max=10.0)
        rate = F.softplus(self.A_log).view(1, 1, -1) + 1e-4
        decay = torch.exp(-dt_t * rate).clamp(0.0, 1.0)
        h = torch.zeros(B, self.cfg.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = decay[:, t] * h + (1.0 - decay[:, t]) * b_t[:, t]
            ys.append(self.C(h))
        y = torch.stack(ys, dim=1)
        return self.out(self.dropout(y * F.silu(gate) + x * self.D))


# ---------------------------------------------------------------------------
# Top-k MoE FFN
# ---------------------------------------------------------------------------
class TopKMoE(nn.Module):
    """Top-k Mixture-of-Experts feed-forward block with switch-style aux loss."""

    def __init__(self, cfg: LatticeConfig):
        super().__init__()
        self.cfg = cfg
        d_ff = cfg.d_model * cfg.d_ff_mult
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.d_model, d_ff, bias=False),
                    nn.SiLU(),
                    nn.Linear(d_ff, cfg.d_model, bias=False),
                )
                for _ in range(cfg.n_experts)
            ]
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat = x.reshape(B * L, D)
        logits = self.router(flat)
        probs = F.softmax(logits.float(), dim=-1).to(flat.dtype)
        top_val, top_idx = torch.topk(probs, k=self.cfg.top_k, dim=-1)
        top_val = top_val / (top_val.sum(dim=-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(flat)

        for slot in range(self.cfg.top_k):
            slot_idx = top_idx[:, slot]
            slot_w = top_val[:, slot].unsqueeze(-1)
            for e, expert in enumerate(self.experts):
                mask = slot_idx == e
                if mask.any():
                    out[mask] += expert(flat[mask]) * slot_w[mask]

        # Switch-Transformer style load-balance auxiliary loss.
        tokens_per_expert = F.one_hot(top_idx[:, 0], self.cfg.n_experts).float().mean(dim=0)
        router_prob = probs.float().mean(dim=0)
        aux_loss = self.cfg.n_experts * torch.sum(tokens_per_expert * router_prob)
        return self.dropout(out.view(B, L, D)), aux_loss


# ---------------------------------------------------------------------------
# Latent world-prediction head
# ---------------------------------------------------------------------------
class LatentWorldHead(nn.Module):
    """Predicts hidden_state[t+1] from hidden_state[t]."""

    def __init__(self, cfg: LatticeConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.eps)
        self.pred = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] < 2:
            return self.pred(self.norm(x)), x.new_tensor(0.0)
        pred = self.pred(self.norm(x[:, :-1]))
        target = x[:, 1:].detach()
        loss = F.mse_loss(pred.float(), target.float()).to(x.dtype)
        # Pad the prediction to L so callers can use it in a residual.
        return torch.cat([pred, pred[:, -1:]], dim=1), loss


# ---------------------------------------------------------------------------
# Block + model
# ---------------------------------------------------------------------------
class LatticeBlock(nn.Module):
    """Hybrid block: attention + SSM + world-head fused, then MoE FFN."""

    def __init__(self, cfg: LatticeConfig):
        super().__init__()
        self.cfg = cfg
        self.n_attn = RMSNorm(cfg.d_model, cfg.eps)
        self.n_ssm = RMSNorm(cfg.d_model, cfg.eps)
        self.n_world = RMSNorm(cfg.d_model, cfg.eps)
        self.n_moe = RMSNorm(cfg.d_model, cfg.eps)
        self.attn = CausalSparseAttention(cfg)
        self.ssm = StableDiagonalSSM(cfg)
        self.world = LatentWorldHead(cfg)
        self.moe = TopKMoE(cfg)
        self.primitive_router = nn.Linear(cfg.d_model, 3)
        self.out_norm = RMSNorm(cfg.d_model, cfg.eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pooled = x.mean(dim=1)
        route = F.softmax(self.primitive_router(pooled).float(), dim=-1).to(x.dtype)
        attn_out = self.attn(self.n_attn(x))
        ssm_out = self.ssm(self.n_ssm(x))
        world_out, world_loss = self.world(self.n_world(x))
        mixed = (
            route[:, 0].view(-1, 1, 1) * attn_out
            + route[:, 1].view(-1, 1, 1) * ssm_out
            + route[:, 2].view(-1, 1, 1) * world_out
        )
        x = x + self.cfg.residual_scale * mixed
        moe_out, moe_aux = self.moe(self.n_moe(x))
        x = self.out_norm(x + self.cfg.residual_scale * moe_out)
        return x, {
            "moe_aux": moe_aux,
            "world_loss": world_loss,
            "route_entropy": entropy(route).mean(),
        }


class LatticeLM(nn.Module):
    """Aura's hybrid SSM/attention/MoE language model."""

    def __init__(self, cfg: LatticeConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([LatticeBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq], got {input_ids.shape}"
            )
        if input_ids.numel() == 0:
            raise ValueError("input_ids is empty")
        if int(input_ids.max()) >= self.cfg.vocab_size or int(input_ids.min()) < 0:
            raise ValueError("input_ids contain out-of-vocab token IDs")
        if input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len "
                f"{self.cfg.max_seq_len}"
            )

        x = self.dropout(self.embed(input_ids))
        moe_aux = world_loss = route_entropy = x.new_tensor(0.0)
        for block in self.blocks:
            x, aux = block(x)
            moe_aux = moe_aux + aux["moe_aux"]
            world_loss = world_loss + aux["world_loss"]
            route_entropy = route_entropy + aux["route_entropy"]

        logits = self.head(self.norm(x))
        n = max(1, len(self.blocks))
        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "moe_aux": moe_aux / n,
            "world_loss": world_loss / n,
            "route_entropy": route_entropy / n,
        }
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels.shape={labels.shape} must equal "
                    f"input_ids.shape={input_ids.shape}"
                )
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            out["lm_loss"] = lm_loss
            out["loss"] = (
                lm_loss
                + self.cfg.aux_moe_weight * out["moe_aux"]
                + self.cfg.aux_world_weight * out["world_loss"]
            )
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            context = ids[:, -self.cfg.max_seq_len :]
            logits = self(context)["logits"][:, -1] / max(temperature, 1e-5)
            if top_k > 0:
                vals, idx = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                filt = torch.full_like(logits, -float("inf"))
                logits = filt.scatter(-1, idx, vals)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
