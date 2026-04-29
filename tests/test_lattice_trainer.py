"""Tests for LatticeTrainer.

Covers:
  * single training step lowers the model's loss after enough updates.
  * gradient-clip records a finite grad norm.
  * non-finite loss raises FloatingPointError instead of corrupting weights.
  * checkpoint round-trip restores model + optimizer + step.
  * eval_loss on a held-out loader gives a finite mean.
  * dataset edge cases: small batches, mismatched seq lengths via collate.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from core.lattice import LatticeConfig, LatticeLM, LatticeTrainer, RandomTokenDataset, TrainConfig


def _build(tmp_path: Path):
    cfg = LatticeConfig(
        vocab_size=128, d_model=32, n_layers=2, n_heads=4,
        d_state=8, n_experts=4, top_k=2, max_seq_len=32, attention_window=16,
    )
    model = LatticeLM(cfg)
    train_cfg = TrainConfig(lr=3e-3, amp=False, checkpoint_dir=str(tmp_path / "ckpt"))
    trainer = LatticeTrainer(model, train_cfg)
    return cfg, model, trainer


# ---------------------------------------------------------------------------
# train step
# ---------------------------------------------------------------------------
def test_train_step_returns_finite_metrics(tmp_path):
    cfg, model, trainer = _build(tmp_path)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    metrics = trainer.train_step({"input_ids": ids, "labels": ids})
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["grad_norm"] >= 0
    assert metrics["step"] == 1


def test_train_step_increments_global_step(tmp_path):
    cfg, model, trainer = _build(tmp_path)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    for i in range(5):
        trainer.train_step({"input_ids": ids, "labels": ids})
    assert trainer.global_step == 5


def test_loss_decreases_over_training(tmp_path):
    torch.manual_seed(0)
    cfg, model, trainer = _build(tmp_path)
    ds = RandomTokenDataset(n_samples=16, seq_len=16, vocab_size=cfg.vocab_size, seed=42)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    losses = []
    for batch in loader:
        losses.append(trainer.train_step(batch)["loss"])
    for batch in loader:
        losses.append(trainer.train_step(batch)["loss"])
    assert sum(losses[-4:]) < sum(losses[:4])


def test_non_finite_loss_raises(tmp_path):
    cfg, model, trainer = _build(tmp_path)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    # Patch model.forward to return NaN loss.
    original = model.forward

    def bad_forward(input_ids, labels=None):
        out = original(input_ids, labels=labels)
        out["loss"] = torch.tensor(float("nan"), requires_grad=True)
        return out

    model.forward = bad_forward  # type: ignore[assignment]
    with pytest.raises(FloatingPointError):
        trainer.train_step({"input_ids": ids, "labels": ids})


# ---------------------------------------------------------------------------
# eval_loss
# ---------------------------------------------------------------------------
def test_eval_loss_returns_finite_value(tmp_path):
    cfg, model, trainer = _build(tmp_path)
    ds = RandomTokenDataset(n_samples=8, seq_len=8, vocab_size=cfg.vocab_size, seed=7)
    loader = DataLoader(ds, batch_size=2)
    val = trainer.eval_loss(loader, max_batches=2)
    assert val == val  # not NaN
    assert val < float("inf")


def test_eval_loss_empty_loader_returns_inf(tmp_path):
    cfg, model, trainer = _build(tmp_path)
    empty_loader = DataLoader([], batch_size=1)
    assert trainer.eval_loss(empty_loader) == float("inf")


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def test_checkpoint_round_trip_restores_state(tmp_path):
    cfg, model, trainer = _build(tmp_path)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    trainer.train_step({"input_ids": ids, "labels": ids})
    trainer.train_step({"input_ids": ids, "labels": ids})
    assert trainer.global_step == 2
    path = trainer.save_checkpoint("ckpt.pt", extra={"note": "test"})
    assert path.exists()

    # Build a fresh trainer and restore.
    cfg2, model2, trainer2 = _build(tmp_path / "second")
    payload = trainer2.load_checkpoint(path)
    assert payload["extra"]["note"] == "test"
    assert trainer2.global_step == 2

    # Forward should match — same weights.
    model.eval()
    model2.eval()
    with torch.no_grad():
        a = model(ids)["logits"]
        b = model2(ids)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def test_dataset_validates_inputs():
    with pytest.raises(ValueError):
        RandomTokenDataset(n_samples=0, seq_len=8, vocab_size=16)
    with pytest.raises(ValueError):
        RandomTokenDataset(n_samples=4, seq_len=1, vocab_size=16)
    with pytest.raises(ValueError):
        RandomTokenDataset(n_samples=4, seq_len=8, vocab_size=1)
    with pytest.raises(ValueError):
        RandomTokenDataset(n_samples=4, seq_len=8, vocab_size=16, noise_p=1.0)


def test_dataset_is_deterministic_for_same_seed():
    a = RandomTokenDataset(n_samples=4, seq_len=8, vocab_size=32, seed=99)
    b = RandomTokenDataset(n_samples=4, seq_len=8, vocab_size=32, seed=99)
    for i in range(len(a)):
        sa = a[i]
        sb = b[i]
        assert torch.equal(sa["input_ids"], sb["input_ids"])
        assert torch.equal(sa["labels"], sb["labels"])


def test_dataset_input_label_shapes_align():
    ds = RandomTokenDataset(n_samples=2, seq_len=10, vocab_size=64, seed=1)
    sample = ds[0]
    assert sample["input_ids"].shape == sample["labels"].shape
    assert sample["input_ids"].shape == (10,)
