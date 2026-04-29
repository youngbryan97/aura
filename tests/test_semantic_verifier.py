"""Tests for HashEmbedder + SemanticVerifier."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.discovery.code_eval import SafeCodeEvaluator
from core.self_modification.mutation_safety import QuarantineStore
from core.verification.embedder import HashEmbedder
from core.verification.semantic_verifier import (
    InvarianceResult,
    ProofCarryingResult,
    SelfConsistencyResult,
    SemanticVerifier,
)


# ---------------------------------------------------------------------------
# HashEmbedder
# ---------------------------------------------------------------------------
def test_hash_embedder_validates_dim():
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)
    with pytest.raises(ValueError):
        HashEmbedder(ngram=0)


def test_hash_embedder_is_deterministic():
    a = HashEmbedder(dim=64).embed("hello world")
    b = HashEmbedder(dim=64).embed("hello world")
    assert a == b


def test_hash_embedder_normalises_whitespace():
    e = HashEmbedder(dim=64)
    assert e.embed("HELLO   WORLD") == e.embed("hello world")


def test_hash_embedder_empty_text_returns_zero_vector():
    e = HashEmbedder(dim=32)
    v = e.embed("")
    assert v == [0.0] * 32


def test_cosine_self_similarity_is_one():
    e = HashEmbedder(dim=128)
    v = e.embed("a quick brown fox")
    assert HashEmbedder.cosine(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_disjoint_text_low():
    e = HashEmbedder(dim=128, ngram=4)
    a = e.embed("aaaaaa bbbbbb cccccc")
    b = e.embed("xxxxxx yyyyyy zzzzzz")
    assert abs(HashEmbedder.cosine(a, b)) < 0.4


def test_cosine_dim_mismatch_raises():
    a = [1.0, 0.0]
    b = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        HashEmbedder.cosine(a, b)


# ---------------------------------------------------------------------------
# SemanticVerifier — self-consistency
# ---------------------------------------------------------------------------
def test_self_consistency_single_output_is_trivially_ok():
    v = SemanticVerifier()
    r = v.self_consistency(["only one"])
    assert isinstance(r, SelfConsistencyResult)
    assert r.ok is True
    assert r.pairs == 0


def test_self_consistency_identical_outputs_are_ok():
    v = SemanticVerifier()
    r = v.self_consistency(["same answer"] * 5)
    assert r.ok is True
    assert r.mean_cosine == pytest.approx(1.0)


def test_self_consistency_disagreeing_outputs_fail():
    v = SemanticVerifier(consistency_threshold=0.95)
    r = v.self_consistency(
        [
            "the cat sat",
            "z y x w v u t s r q p o n m l k j i h g f",
            "completely different content here",
        ]
    )
    assert r.ok is False
    assert r.pairs == 3


# ---------------------------------------------------------------------------
# SemanticVerifier — invariance
# ---------------------------------------------------------------------------
def test_invariance_no_paraphrases_passes():
    v = SemanticVerifier()
    r = v.paraphrase_invariance("the answer", [])
    assert isinstance(r, InvarianceResult)
    assert r.ok is True
    assert r.similarities == []


def test_invariance_identical_paraphrase_passes():
    v = SemanticVerifier(invariance_threshold=0.5)
    r = v.paraphrase_invariance("the answer is six", ["the answer is six"])
    assert r.ok is True
    assert r.similarities[0] == pytest.approx(1.0, abs=1e-6)


def test_invariance_distant_paraphrase_can_be_below_threshold():
    """Hash embedder is intentionally weak for paraphrase. Distant text
    should still drop similarity. This guards against the embedder
    accidentally returning 1.0 for everything."""
    v = SemanticVerifier(invariance_threshold=0.5)
    r = v.paraphrase_invariance("the answer is six", ["six is the answer"])
    # Bag-of-ngrams overlap is partial; similarity should be < 1.
    assert r.similarities[0] < 1.0


def test_invariance_far_paraphrases_fail():
    v = SemanticVerifier(invariance_threshold=0.95)
    r = v.paraphrase_invariance(
        "the answer is six", ["completely different reply"]
    )
    assert r.ok is False
    assert r.similarities[0] < 0.95


# ---------------------------------------------------------------------------
# SemanticVerifier — proof-carrying code
# ---------------------------------------------------------------------------
@pytest.fixture
def proof_verifier(tmp_path: Path) -> SemanticVerifier:
    return SemanticVerifier(
        code_evaluator=SafeCodeEvaluator(
            timeout_seconds=5.0,
            memory_mb=256,
            quarantine=QuarantineStore(tmp_path / "q"),
        )
    )


def test_proof_carrying_code_accepts_correct_assertion(proof_verifier):
    code = "def add(a, b):\n    assert a + b == a + b\n    return a + b\n"
    r = proof_verifier.proof_carrying_code(code, "add", [((1, 1), 2), ((3, 4), 7)])
    assert isinstance(r, ProofCarryingResult)
    assert r.has_assertion is True
    assert r.ok is True


def test_proof_carrying_code_rejects_missing_assertion(proof_verifier):
    code = "def add(a, b):\n    return a + b\n"
    r = proof_verifier.proof_carrying_code(code, "add", [((1, 1), 2)])
    assert r.has_assertion is False
    assert r.ok is False
    assert r.sandbox.outcome == "passed"  # tests pass; just no assertion


def test_proof_carrying_code_rejects_failing_tests(proof_verifier):
    code = "def add(a, b):\n    assert True\n    return a - b\n"
    r = proof_verifier.proof_carrying_code(code, "add", [((1, 1), 2)])
    assert r.has_assertion is True
    assert r.ok is False  # test fails -> sandbox.ok False


# ---------------------------------------------------------------------------
# combined verify()
# ---------------------------------------------------------------------------
def test_verify_accepts_when_all_supplied_channels_pass(proof_verifier):
    report = proof_verifier.verify(
        consistency_outputs=["yes", "yes", "yes"],
        invariance=("the answer is six", ["the answer is six"]),
        proof=(
            "def f(a, b):\n    assert True\n    return a + b\n",
            "f",
            [((1, 1), 2)],
        ),
    )
    assert report.accepted is True
    assert report.self_consistency is not None
    assert report.invariance is not None
    assert report.proof is not None


def test_verify_rejects_on_consistency_failure():
    v = SemanticVerifier(consistency_threshold=0.95)
    report = v.verify(
        consistency_outputs=[
            "the answer is six",
            "no clue what to say",
            "completely random",
        ],
        require=("consistency",),
    )
    assert report.accepted is False
    assert any("self_consistency" in r for r in report.reasons)


def test_verify_skips_unsupplied_channels():
    v = SemanticVerifier()
    report = v.verify(consistency_outputs=["a", "a", "a"], require=("consistency",))
    assert report.accepted is True
    assert report.invariance is None
    assert report.proof is None


def test_verify_no_channels_supplied_passes():
    v = SemanticVerifier()
    report = v.verify(require=())
    assert report.accepted is True


def test_verify_required_but_unsupplied_does_not_fail():
    """If a channel isn't supplied, it can't fail — skip silently."""
    v = SemanticVerifier()
    report = v.verify(
        consistency_outputs=["a", "a"],
        require=("consistency", "invariance", "proof"),
    )
    assert report.accepted is True


# ---------------------------------------------------------------------------
# stress: long inputs
# ---------------------------------------------------------------------------
def test_long_text_does_not_break_embedder():
    e = HashEmbedder(dim=512)
    long = "alpha " * 2000
    v = e.embed(long)
    assert len(v) == 512
    norm = math.sqrt(sum(x * x for x in v))
    assert 0.95 < norm < 1.05  # L2-normalised
