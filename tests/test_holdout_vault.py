"""Tests for HoldoutVault + LeakageDetector."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.promotion.dynamic_benchmark import DynamicBenchmark, Task
from core.promotion.holdout_vault import (
    HoldoutVault,
    LeakageDetector,
    VaultMissError,
)


# ---------------------------------------------------------------------------
# HoldoutVault
# ---------------------------------------------------------------------------
def test_vault_persists_across_reopen(tmp_path: Path):
    bench = DynamicBenchmark(seed=0)
    tasks = bench.generate(5)
    a = HoldoutVault(tmp_path / "vault.json")
    a.add(tasks)

    b = HoldoutVault(tmp_path / "vault.json")
    assert b.size() == 5
    for t in tasks:
        assert b.get_answer(t.hash_public()) == t.answer


def test_vault_public_manifest_excludes_answers(tmp_path: Path):
    tasks = DynamicBenchmark(seed=1).generate(3)
    vault = HoldoutVault(tmp_path / "v.json")
    vault.add(tasks)
    manifest = vault.public_manifest()
    assert len(manifest) == 3
    for entry in manifest:
        assert "answer" not in entry  # public manifest must not leak answers
        assert "prompt" in entry


def test_vault_get_answer_misses_raise(tmp_path: Path):
    vault = HoldoutVault(tmp_path / "v.json")
    with pytest.raises(VaultMissError):
        vault.get_answer("does-not-exist")


def test_vault_corrupt_file_starts_empty(tmp_path: Path):
    p = tmp_path / "v.json"
    p.write_text("{ broken json", encoding="utf-8")
    vault = HoldoutVault(p)
    assert vault.size() == 0
    # And we can still write into it without crashing.
    vault.add(DynamicBenchmark(seed=2).generate(1))
    assert vault.size() == 1


def test_vault_overwrites_same_public_hash(tmp_path: Path):
    """Two adds with same prompt should not double-count."""
    vault = HoldoutVault(tmp_path / "v.json")
    t1 = Task("k", "foo", 1, {})
    t2 = Task("k", "foo", 999, {})
    vault.add([t1])
    vault.add([t2])
    assert vault.size() == 1
    # Latest answer wins.
    assert vault.get_answer(t2.hash_public()) == 999


# ---------------------------------------------------------------------------
# LeakageDetector
# ---------------------------------------------------------------------------
def test_leakage_detector_validates_inputs():
    with pytest.raises(ValueError):
        LeakageDetector(ngram=0)
    with pytest.raises(ValueError):
        LeakageDetector(threshold=2.0)


def test_self_similarity_is_one():
    det = LeakageDetector(ngram=3, threshold=0.5)
    assert det.similarity("hello world", "hello world") == 1.0


def test_disjoint_strings_have_zero_similarity():
    det = LeakageDetector(ngram=4, threshold=0.5)
    assert det.similarity("aaaa bbbb", "wxyz wxyz") < 0.2


def test_contaminated_prompt_flagged():
    det = LeakageDetector(ngram=3, threshold=0.5)
    task = Task("k", "Return gcd(123, 456) as an integer.", 3, {})
    training = ["Return gcd(123, 456) as an integer."]
    contaminated, score = det.contaminated(task, training)
    assert contaminated is True
    assert score == pytest.approx(1.0)


def test_clean_prompt_passes():
    det = LeakageDetector(ngram=3, threshold=0.5)
    task = Task("k", "Solve a totally different math problem here.", 0, {})
    training = ["Some unrelated tutorial about cooking pasta."]
    contaminated, _ = det.contaminated(task, training)
    assert contaminated is False


def test_filter_clean_drops_contaminated_tasks():
    det = LeakageDetector(ngram=3, threshold=0.5)
    bench = DynamicBenchmark(seed=7)
    tasks = bench.generate(5, kinds=["gcd"])
    # Drop one task into the training corpus.
    contaminated_text = tasks[0].prompt
    clean = det.filter_clean(tasks, [contaminated_text])
    assert all(t.prompt != contaminated_text for t in clean)


def test_empty_training_corpus_means_no_contamination():
    det = LeakageDetector(ngram=4, threshold=0.5)
    task = Task("k", "Anything goes here.", 0, {})
    contaminated, score = det.contaminated(task, [])
    assert contaminated is False
    assert score == 0.0
