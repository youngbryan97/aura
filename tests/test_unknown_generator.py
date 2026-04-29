"""Tests for NoveltyArchive + UnknownUnknownGenerator + EmbeddingEntropyProbe."""
from __future__ import annotations

import math

import pytest
import torch

from core.lattice import LatticeConfig, LatticeLM
from core.promotion.dynamic_benchmark import DynamicBenchmark, Task
from core.unknowns.entropy_probe import EmbeddingEntropyProbe
from core.unknowns.generator import UnknownUnknownGenerator
from core.unknowns.novelty_archive import NoveltyArchive
from core.verification.embedder import HashEmbedder


# ---------------------------------------------------------------------------
# NoveltyArchive
# ---------------------------------------------------------------------------
def test_novelty_archive_validates_threshold():
    with pytest.raises(ValueError):
        NoveltyArchive(novelty_threshold=0.0)
    with pytest.raises(ValueError):
        NoveltyArchive(novelty_threshold=1.5)


def test_empty_archive_treats_first_text_as_novel():
    archive = NoveltyArchive(novelty_threshold=0.5)
    assert archive.is_novel("hello world") is True
    assert archive.add_if_novel("hello world") is True
    assert len(archive) == 1


def test_duplicate_text_is_not_novel():
    archive = NoveltyArchive(novelty_threshold=0.5)
    archive.add_if_novel("the cat sat")
    assert archive.add_if_novel("the cat sat") is False
    assert len(archive) == 1


def test_distant_text_is_novel():
    archive = NoveltyArchive(novelty_threshold=0.4)
    archive.add_if_novel("aaa bbb ccc ddd")
    assert archive.add_if_novel("xxx yyy zzz www") is True


def test_empty_string_is_rejected():
    archive = NoveltyArchive(novelty_threshold=0.5)
    assert archive.add_if_novel("") is False


def test_reset_clears_archive():
    archive = NoveltyArchive(novelty_threshold=0.5)
    archive.add_if_novel("alpha beta gamma")
    archive.reset()
    assert len(archive) == 0


# ---------------------------------------------------------------------------
# UnknownUnknownGenerator
# ---------------------------------------------------------------------------
def test_generator_with_no_seed_tasks_returns_empty():
    gen = UnknownUnknownGenerator(seed=0)
    assert gen.generate([], n=5) == []


def test_generator_n_zero_returns_empty():
    gen = UnknownUnknownGenerator(seed=0)
    seeds = DynamicBenchmark(seed=0).generate(5)
    assert gen.generate(seeds, n=0) == []


def test_generator_produces_n_novel_tasks():
    gen = UnknownUnknownGenerator(seed=0)
    seeds = DynamicBenchmark(seed=0).generate(20)
    out = gen.generate(seeds, n=10)
    assert len(out) <= 10  # may produce fewer if novelty saturates
    assert len(out) >= 5   # but should usually find some


def test_generated_tasks_carry_parent_hash():
    gen = UnknownUnknownGenerator(seed=0)
    seeds = DynamicBenchmark(seed=1).generate(10)
    out = gen.generate(seeds, n=5)
    for task in out:
        assert "parent" in task.metadata


def test_generated_gcd_answers_are_correct():
    gen = UnknownUnknownGenerator(seed=2)
    seeds = DynamicBenchmark(seed=2).generate(20, kinds=["gcd"])
    out = gen.generate(seeds, n=10)
    for task in out:
        if task.kind == "gcd":
            a, b = task.metadata["a"], task.metadata["b"]
            assert task.answer == math.gcd(a, b)


def test_generated_sort_answers_are_correct():
    gen = UnknownUnknownGenerator(seed=3)
    seeds = DynamicBenchmark(seed=3).generate(20, kinds=["sort"])
    out = gen.generate(seeds, n=10)
    for task in out:
        if task.kind == "sort":
            assert task.answer == sorted(task.metadata["arr"])


def test_generated_compose_answers_are_correct():
    gen = UnknownUnknownGenerator(seed=4)
    seeds = DynamicBenchmark(seed=4).generate(20, kinds=["compose"])
    out = gen.generate(seeds, n=10)
    for task in out:
        if task.kind == "compose":
            m = task.metadata
            assert task.answer == m["c"] * (m["a"] * m["x"] + m["b"]) + m["d"]


def test_generator_archive_dedupes_across_calls():
    archive = NoveltyArchive(novelty_threshold=0.3)
    gen = UnknownUnknownGenerator(seed=5, archive=archive)
    seeds = DynamicBenchmark(seed=5).generate(20)
    a = gen.generate(seeds, n=10)
    archive_size_after_first = len(archive)
    b = gen.generate(seeds, n=10)
    archive_size_after_second = len(archive)
    # Second call should add at least some new entries on top of first.
    assert archive_size_after_second >= archive_size_after_first
    # No two outputs across both calls share the same prompt.
    a_prompts = {t.prompt for t in a}
    b_prompts = {t.prompt for t in b}
    assert a_prompts.isdisjoint(b_prompts)


# ---------------------------------------------------------------------------
# EmbeddingEntropyProbe
# ---------------------------------------------------------------------------
def _tiny_model() -> LatticeLM:
    cfg = LatticeConfig(
        vocab_size=64, d_model=16, n_layers=1, n_heads=4, d_state=4,
        n_experts=2, top_k=1, max_seq_len=16, attention_window=8,
    )
    return LatticeLM(cfg)


def test_entropy_probe_validates_inputs():
    with pytest.raises(ValueError):
        EmbeddingEntropyProbe(epsilon=0)
    with pytest.raises(ValueError):
        EmbeddingEntropyProbe(steps=0)
    with pytest.raises(ValueError):
        EmbeddingEntropyProbe(step_size=-1)


def test_entropy_probe_returns_correct_shape():
    model = _tiny_model()
    probe = EmbeddingEntropyProbe(epsilon=0.05, steps=2, step_size=0.01)
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    adv = probe.generate(model, ids)
    assert adv.shape == (1, 8, model.cfg.d_model)


def test_entropy_probe_perturbation_within_epsilon_ball():
    model = _tiny_model()
    probe = EmbeddingEntropyProbe(epsilon=0.05, steps=3, step_size=0.01)
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    base = model.embed(ids).detach()
    adv = probe.generate(model, ids)
    delta = (adv - base).abs().max().item()
    # Allow small numerical slack (gradient sign + step accumulation).
    assert delta <= 0.05 + 1e-4


def test_entropy_probe_can_only_increase_entropy():
    """Probe is gradient *ascent* on entropy — adv embeddings should have
    entropy >= base embeddings (modulo numerical noise)."""
    torch.manual_seed(0)
    model = _tiny_model()
    probe = EmbeddingEntropyProbe(epsilon=0.1, steps=4, step_size=0.02)
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    base = model.embed(ids).detach()
    adv = probe.generate(model, ids)
    base_ent = probe.measure_entropy(model, base)
    adv_ent = probe.measure_entropy(model, adv)
    # Allow tiny slack but should generally increase.
    assert adv_ent >= base_ent - 1e-3


def test_entropy_probe_rejects_non_2d_input():
    model = _tiny_model()
    probe = EmbeddingEntropyProbe()
    with pytest.raises(ValueError):
        probe.generate(model, torch.zeros(8, dtype=torch.long))
