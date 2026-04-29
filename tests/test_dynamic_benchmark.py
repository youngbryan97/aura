"""Tests for DynamicBenchmark + Task contracts."""
from __future__ import annotations

import math

import pytest

from core.promotion.dynamic_benchmark import DynamicBenchmark, Task


def test_supported_kinds_complete():
    assert set(DynamicBenchmark.SUPPORTED_KINDS) == {
        "gcd", "mod", "sort", "palindrome", "compose"
    }


def test_generate_zero_returns_empty():
    bench = DynamicBenchmark(seed=0)
    assert bench.generate(0) == []


def test_generate_advances_seed_so_consecutive_runs_differ():
    bench = DynamicBenchmark(seed=0)
    a = bench.generate(20)
    b = bench.generate(20)
    a_hashes = {t.hash_public() for t in a}
    b_hashes = {t.hash_public() for t in b}
    assert a_hashes != b_hashes
    assert len(a_hashes & b_hashes) < len(a_hashes)  # mostly disjoint


def test_same_seed_after_reset_is_deterministic():
    a = DynamicBenchmark(seed=42).generate(10)
    b = DynamicBenchmark(seed=42).generate(10)
    assert [t.hash_public() for t in a] == [t.hash_public() for t in b]


def test_unknown_kind_raises():
    bench = DynamicBenchmark(seed=0)
    with pytest.raises(ValueError):
        bench.generate(5, kinds=["not_a_kind"])


# ---------------------------------------------------------------------------
# answer correctness per kind
# ---------------------------------------------------------------------------
def test_gcd_answers_are_correct():
    bench = DynamicBenchmark(seed=1)
    tasks = bench.generate(30, kinds=["gcd"])
    for t in tasks:
        a, b = t.metadata["a"], t.metadata["b"]
        assert t.answer == math.gcd(a, b)


def test_mod_answers_are_correct():
    bench = DynamicBenchmark(seed=2)
    tasks = bench.generate(20, kinds=["mod"])
    for t in tasks:
        a, b, m = t.metadata["a"], t.metadata["b"], t.metadata["m"]
        assert t.answer == pow(a, b, m)


def test_sort_answers_are_sorted():
    bench = DynamicBenchmark(seed=3)
    for t in bench.generate(20, kinds=["sort"]):
        assert t.answer == sorted(t.metadata["arr"])


def test_palindrome_answers_match_reference():
    bench = DynamicBenchmark(seed=4)
    for t in bench.generate(30, kinds=["palindrome"]):
        s = t.metadata["s"]
        assert t.answer == (s == s[::-1])


def test_compose_answers_are_correct():
    bench = DynamicBenchmark(seed=5)
    for t in bench.generate(20, kinds=["compose"]):
        m = t.metadata
        expected = m["c"] * (m["a"] * m["x"] + m["b"]) + m["d"]
        assert t.answer == expected


# ---------------------------------------------------------------------------
# Task hashing
# ---------------------------------------------------------------------------
def test_hash_public_excludes_answer():
    t = Task(kind="gcd", prompt="Return gcd(6, 9)", answer=3, metadata={"a": 6, "b": 9})
    h_public = t.hash_public()
    h_full = t.hash_with_answer()
    assert h_public != h_full


def test_two_tasks_with_same_public_have_same_hash():
    a = Task(kind="x", prompt="foo", answer=1, metadata={"k": 2})
    b = Task(kind="x", prompt="foo", answer=999, metadata={"k": 2})
    assert a.hash_public() == b.hash_public()
    assert a.hash_with_answer() != b.hash_with_answer()


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------
def test_large_batch_unique_within_run():
    bench = DynamicBenchmark(seed=99)
    tasks = bench.generate(200)
    hashes = [t.hash_public() for t in tasks]
    # not strict uniqueness — random can occasionally collide on small kinds
    # like palindrome. but the vast majority should be unique.
    assert len(set(hashes)) >= int(0.85 * len(hashes))
