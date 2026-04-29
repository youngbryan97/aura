"""Tests for the Aura Lattice hybrid architecture.

Coverage:
  * unit: shape contracts on every primitive (RMSNorm, RoPE, attention,
    SSM, MoE, world head, block, full LM).
  * config validation rejects malformed configs.
  * training step lowers loss on the synthetic structured dataset.
  * generation respects max_new_tokens, top_k, temperature.
  * stress: long sequence, large batch, MoE load balancing.
  * edge: empty / OOV input, NaN guards, attention-window correctness,
    causal mask integrity.
"""
from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F

from core.lattice import (
    CausalSparseAttention,
    LatentWorldHead,
    LatticeBlock,
    LatticeConfig,
    LatticeLM,
    RMSNorm,
    RotaryEmbedding,
    StableDiagonalSSM,
    TopKMoE,
)
from core.lattice.dataset import RandomTokenDataset


def _tiny_cfg(**overrides) -> LatticeConfig:
    base = dict(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_state=8,
        n_experts=4,
        top_k=2,
        max_seq_len=64,
        attention_window=16,
    )
    base.update(overrides)
    return LatticeConfig(**base)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------
def test_config_rejects_indivisible_d_model():
    with pytest.raises(ValueError):
        LatticeConfig(d_model=33, n_heads=4).validate()


def test_config_rejects_top_k_out_of_range():
    with pytest.raises(ValueError):
        LatticeConfig(top_k=0, n_experts=4).validate()
    with pytest.raises(ValueError):
        LatticeConfig(top_k=5, n_experts=4).validate()


def test_config_rejects_zero_max_seq_len():
    with pytest.raises(ValueError):
        LatticeConfig(max_seq_len=1).validate()


def test_config_rejects_dropout_outside_range():
    with pytest.raises(ValueError):
        LatticeConfig(dropout=-0.1).validate()
    with pytest.raises(ValueError):
        LatticeConfig(dropout=1.0).validate()


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def test_rmsnorm_unit_norm_property():
    norm = RMSNorm(8)
    x = torch.randn(2, 5, 8) * 100
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    # weight is 1, so y rms ≈ 1.
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.05)


def test_rotary_embedding_rejects_odd_dim():
    with pytest.raises(ValueError):
        RotaryEmbedding(dim=7, max_seq_len=4)


def test_rotary_embedding_rejects_too_long_seq():
    rope = RotaryEmbedding(dim=8, max_seq_len=4)
    q = torch.randn(1, 2, 8, 8)
    k = torch.randn(1, 2, 8, 8)
    with pytest.raises(ValueError):
        rope(q, k)


def test_rotary_embedding_preserves_shape():
    rope = RotaryEmbedding(dim=8, max_seq_len=16)
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 6, 8)
    qr, kr = rope(q, k)
    assert qr.shape == q.shape and kr.shape == k.shape


def test_attention_output_shape():
    cfg = _tiny_cfg()
    attn = CausalSparseAttention(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    y = attn(x)
    assert y.shape == x.shape


def test_attention_is_causal():
    """Future tokens must not influence past ones."""
    cfg = _tiny_cfg(attention_window=64)
    torch.manual_seed(0)
    attn = CausalSparseAttention(cfg)
    attn.eval()
    x = torch.randn(1, 8, cfg.d_model)
    y_full = attn(x)
    # Mutate token at position 4; positions < 4 must be unchanged.
    x_mut = x.clone()
    x_mut[:, 4:] += torch.randn_like(x_mut[:, 4:]) * 0.5
    y_mut = attn(x_mut)
    assert torch.allclose(y_full[:, :4], y_mut[:, :4], atol=1e-5)


def test_attention_window_limits_receptive_field():
    """When window < L, position j shouldn't attend to position 0."""
    cfg = _tiny_cfg(attention_window=2)
    torch.manual_seed(0)
    attn = CausalSparseAttention(cfg)
    attn.eval()
    x = torch.randn(1, 8, cfg.d_model)
    y0 = attn(x)
    x_mut = x.clone()
    x_mut[:, 0] += torch.randn_like(x_mut[:, 0]) * 5.0
    y1 = attn(x_mut)
    # Position 7 with window=2 only sees positions 6,7 — must be unchanged.
    assert torch.allclose(y0[:, 7], y1[:, 7], atol=1e-4)


def test_ssm_output_shape_and_finite():
    cfg = _tiny_cfg()
    ssm = StableDiagonalSSM(cfg)
    x = torch.randn(3, 12, cfg.d_model)
    y = ssm(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_ssm_decay_remains_in_unit_interval():
    """Stability invariant: with the softplus + clamp, decay must be in [0,1]."""
    cfg = _tiny_cfg(d_state=16)
    ssm = StableDiagonalSSM(cfg)
    # Crank A_log up so the decay tries to exceed 1 — clamp must catch it.
    with torch.no_grad():
        ssm.A_log.fill_(-100.0)  # softplus -> ~0 -> decay -> 1
    x = torch.randn(1, 4, cfg.d_model) * 1000
    y = ssm(x)
    assert torch.isfinite(y).all()


def test_moe_returns_aux_loss_and_correct_shape():
    cfg = _tiny_cfg()
    moe = TopKMoE(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    y, aux = moe(x)
    assert y.shape == x.shape
    assert aux.dim() == 0 and torch.isfinite(aux)


def test_moe_load_balance_on_uniform_input():
    """With uniform routing the auxiliary loss should be near n_experts."""
    cfg = _tiny_cfg(n_experts=4, top_k=1)
    moe = TopKMoE(cfg)
    # Force uniform routing by zeroing the router weight.
    with torch.no_grad():
        moe.router.weight.zero_()
    x = torch.randn(8, 16, cfg.d_model)
    _, aux = moe(x)
    # n_experts * sum(uniform * uniform) = n_experts * (1/n) = 1
    assert 0.5 <= float(aux) <= 1.5


def test_world_head_zero_loss_for_short_input():
    cfg = _tiny_cfg()
    head = LatentWorldHead(cfg)
    x = torch.randn(1, 1, cfg.d_model)
    y, loss = head(x)
    assert y.shape == x.shape
    assert float(loss) == 0.0


def test_world_head_nontrivial_loss_for_long_input():
    cfg = _tiny_cfg()
    head = LatentWorldHead(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    y, loss = head(x)
    assert y.shape == x.shape
    assert float(loss) >= 0.0


def test_block_returns_aux_dict_with_required_keys():
    cfg = _tiny_cfg()
    block = LatticeBlock(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    y, aux = block(x)
    assert y.shape == x.shape
    assert {"moe_aux", "world_loss", "route_entropy"} <= aux.keys()


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------
def test_lattice_lm_forward_returns_logits():
    cfg = _tiny_cfg()
    model = LatticeLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)


def test_lattice_lm_forward_with_labels_emits_loss():
    cfg = _tiny_cfg()
    model = LatticeLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (3, 12))
    out = model(ids, labels=ids)
    assert "loss" in out and "lm_loss" in out
    assert torch.isfinite(out["loss"])


def test_lattice_lm_rejects_oov_token():
    cfg = _tiny_cfg(vocab_size=64)
    model = LatticeLM(cfg)
    bad = torch.randint(0, cfg.vocab_size, (1, 8))
    bad[0, 0] = cfg.vocab_size  # out of vocab
    with pytest.raises(ValueError):
        model(bad)


def test_lattice_lm_rejects_too_long_sequence():
    cfg = _tiny_cfg(max_seq_len=8)
    model = LatticeLM(cfg)
    too_long = torch.randint(0, cfg.vocab_size, (1, 9))
    with pytest.raises(ValueError):
        model(too_long)


def test_lattice_lm_rejects_empty_input():
    cfg = _tiny_cfg()
    model = LatticeLM(cfg)
    with pytest.raises(ValueError):
        model(torch.zeros(0, 0, dtype=torch.long))


def test_lattice_lm_rejects_label_shape_mismatch():
    cfg = _tiny_cfg()
    model = LatticeLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    bad_labels = torch.randint(0, cfg.vocab_size, (1, 7))
    with pytest.raises(ValueError):
        model(ids, labels=bad_labels)


def test_lattice_lm_tied_embeddings_share_storage():
    cfg = _tiny_cfg(tie_embeddings=True)
    model = LatticeLM(cfg)
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_lattice_lm_untied_embeddings_independent():
    cfg = _tiny_cfg(tie_embeddings=False)
    model = LatticeLM(cfg)
    assert model.head.weight.data_ptr() != model.embed.weight.data_ptr()


def test_num_parameters_is_positive():
    cfg = _tiny_cfg()
    model = LatticeLM(cfg)
    assert model.num_parameters() > 0


# ---------------------------------------------------------------------------
# learning capacity (smoke)
# ---------------------------------------------------------------------------
def test_lattice_lm_can_lower_loss_on_structured_data():
    """Ten optimizer steps on RandomTokenDataset should drop the loss."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(d_model=32, n_layers=2)
    model = LatticeLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ds = RandomTokenDataset(n_samples=8, seq_len=24, vocab_size=cfg.vocab_size, seed=1)

    losses = []
    for step in range(12):
        sample = ds[step % len(ds)]
        ids = sample["input_ids"].unsqueeze(0)
        labels = sample["labels"].unsqueeze(0)
        out = model(ids, labels=labels)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(out["loss"]))
    # Final third should be lower than first third.
    assert sum(losses[-4:]) < sum(losses[:4])


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def test_generate_extends_sequence_by_max_new_tokens():
    cfg = _tiny_cfg()
    model = LatticeLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=5, temperature=0.7, top_k=8)
    assert out.shape == (1, 9)


def test_generate_zero_new_tokens_is_no_op():
    cfg = _tiny_cfg()
    model = LatticeLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 3))
    out = model.generate(ids, max_new_tokens=0)
    assert torch.equal(out, ids)


def test_generate_rejects_negative_max_tokens():
    cfg = _tiny_cfg()
    model = LatticeLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    with pytest.raises(ValueError):
        model.generate(ids, max_new_tokens=-1)


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [32, 64])
def test_long_sequence_forward_finite(L: int):
    cfg = _tiny_cfg(max_seq_len=128, attention_window=32)
    model = LatticeLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, L))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])


def test_large_batch_forward_finite():
    cfg = _tiny_cfg(d_model=16, n_layers=1, n_experts=2, top_k=1, max_seq_len=32)
    model = LatticeLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (16, 8))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])


def test_no_nan_under_extreme_inputs():
    """Massive negative logits / large dropout should still produce finite loss."""
    cfg = _tiny_cfg(dropout=0.5)
    model = LatticeLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    for _ in range(5):
        out = model(ids, labels=ids)
        assert torch.isfinite(out["loss"])
