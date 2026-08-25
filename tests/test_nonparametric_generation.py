"""Tests for non-parametric generation helpers (normalization + cosine gating).

The full generate_with_memory loop is validated end-to-end against the real model in
aura_bench/nonparametric_probe.py (generation 5/5). These cover the pure helpers that
make the cosine-gated λ model-independent.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.nonparametric_generation import cosine_from_l2, normalize
from tests.nonparametric_support import entry_provenance


def test_normalize_unit_norm():
    v = normalize(np.array([3.0, 4.0, 0.0, 0.0]))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_normalize_zero_vector_safe():
    v = normalize(np.zeros(4))
    assert np.all(np.isfinite(v))


def test_cosine_from_l2_identical_is_one():
    # distance 0 between unit vectors → cosine 1
    assert abs(cosine_from_l2(0.0) - 1.0) < 1e-9


def test_encoder_uses_the_resolved_hybrid_text_backbone():
    import mlx.core as mx

    from core.brain.nonparametric_generation import MLXEncoder

    class Backbone:
        def __init__(self):
            self.calls = 0

        def __call__(self, ids):
            self.calls += 1
            width = int(ids.shape[1])
            return mx.array(np.tile([1.0, 2.0, 3.0, 4.0], (1, width, 1)))

    class Tokenizer:
        all_special_ids = []

        def encode(self, _text):
            return [1, 2]

    backbone = Backbone()
    language = type(
        "LanguageModel",
        (),
        {"args": type("Args", (), {"hidden_size": 4})(), "model": backbone},
    )()
    wrapper = type(
        "HybridWrapper",
        (),
        {
            "args": type("WrapperArgs", (), {"model_type": "qwen3_5"})(),
            "language_model": language,
        },
    )()

    encoder = MLXEncoder(wrapper, Tokenizer())
    hidden = encoder.encode_hidden("anything")

    assert encoder.dim == 4
    assert backbone.calls == 1
    assert np.isclose(np.linalg.norm(hidden), 1.0)


def test_cosine_from_l2_orthogonal_is_zero():
    # two orthogonal unit vectors are sqrt(2) apart → cosine 0
    assert abs(cosine_from_l2(np.sqrt(2.0))) < 1e-6


def test_cosine_from_l2_opposite_is_minus_one():
    # antipodal unit vectors are distance 2 apart → cosine -1
    assert abs(cosine_from_l2(2.0) - (-1.0)) < 1e-9


def test_cosine_monotonic_in_distance():
    assert cosine_from_l2(0.1) > cosine_from_l2(0.5) > cosine_from_l2(1.0)


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def test_logits_processor_boosts_recalled_token():
    import mlx.core as mx

    from core.brain.nonparametric_generation import make_nonparametric_logits_processor
    from core.brain.nonparametric_memory import NonParametricMemory

    mem = NonParametricMemory(dim=4)
    kvec = normalize(np.array([1.0, 0, 0, 0]))
    mem.add(
        kvec, token_id=0, token="x", weight=1.0, provenance=entry_provenance()
    )  # recall favors token 0

    class FakeModel:
        def model(self, seq):
            n = int(seq.shape[1])
            return mx.array(np.tile(kvec.astype(np.float32), (1, n, 1)))

    proc = make_nonparametric_logits_processor(FakeModel(), mem, free_energy=1.0)
    tokens = mx.array([1, 2, 3])
    logits = mx.array(np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32))  # model favors token 2
    out = np.array(proc(tokens, logits)).reshape(-1)
    # exact-key recall clears the gate → the recalled token (0) wins over the model's pick (2)
    assert int(np.argmax(out)) == 0


def test_logits_processor_fail_open_on_far_neighbor():
    import mlx.core as mx

    from core.brain.nonparametric_generation import make_nonparametric_logits_processor
    from core.brain.nonparametric_memory import NonParametricMemory

    mem = NonParametricMemory(dim=4)
    mem.add(
        normalize(np.array([0.0, 0.0, 0.0, 1.0])),
        token_id=0,
        token="x",
        provenance=entry_provenance(),
    )  # orthogonal to query

    class FakeModel:
        def model(self, seq):
            n = int(seq.shape[1])
            q = normalize(np.array([1.0, 0, 0, 0]))
            return mx.array(np.tile(q.astype(np.float32), (1, n, 1)))

    proc = make_nonparametric_logits_processor(FakeModel(), mem)
    logits = mx.array(np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32))
    out = np.array(proc(mx.array([1, 2]), logits)).reshape(-1)
    # far neighbor (cos≈0 < min_cos) → λ gated to 0 → logits unchanged
    assert np.allclose(out, np.array([0.0, 0.0, 5.0, 0.0]))


# ── CP126 remediation regressions ───────────────────────────────────────────


def test_topk_probs_do_not_inflate_truncated_mass():
    """Softmaxing over only the selected k assigned that subset probability 1,
    so the 'interpolation' mixed a fictitious LM distribution."""
    from core.brain.nonparametric_generation import _full_probs, _topk_probs

    logits = np.array([5.0, 4.0, 3.0] + [0.0] * 100, dtype=np.float32)
    top = _topk_probs(logits, k=3)
    full = _full_probs(logits)

    assert sum(top.values()) < 1.0  # dropped mass is NOT redistributed
    for token_id, probability in top.items():
        assert probability == pytest.approx(float(full[token_id]))


def test_processor_applies_the_advertised_mixture_not_an_argmax_override():
    """The processor forced one logit to max+1 and left the rest untouched,
    discarding lambda and the blended probabilities entirely."""
    import mlx.core as mx

    from core.brain.nonparametric_generation import make_nonparametric_logits_processor
    from core.brain.nonparametric_memory import NonParametricMemory

    mem = NonParametricMemory(dim=4)
    kvec = normalize(np.array([1.0, 0, 0, 0]))
    mem.add(kvec, token_id=0, token="x", weight=1.0, provenance=entry_provenance())

    class FakeModel:
        def model(self, seq):
            n = int(seq.shape[1])
            return mx.array(np.tile(kvec.astype(np.float32), (1, n, 1)))

    proc = make_nonparametric_logits_processor(FakeModel(), mem, free_energy=1.0)
    logits = mx.array(np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32))
    out = np.array(proc(mx.array([1, 2, 3]), logits)).reshape(-1)

    # Output is a valid log-distribution, not a spiked copy of the input.
    probs = np.exp(out)
    assert probs.sum() == pytest.approx(1.0, abs=1e-4)
    # The model's own preference survives with non-trivial mass — a hard
    # override would have left token 2 at its original logit of 5.0.
    assert probs[2] > 0.0
    assert out[0] != pytest.approx(float(np.max(np.array(logits))) + 1.0)


def test_memory_tokens_outside_the_vocabulary_are_rejected():
    """Corrupt or cross-model datastore ids reached the sampler and then the
    embedding table."""
    from core.brain.nonparametric_generation import _select_with_memory

    class _Neighbor:
        index = 0
        similarity = 1.0

    class _Memory:
        def query(self, key, k):
            return [_Neighbor()]

        def min_similarity(self):
            return 0.5

        def knn_probs(self, neighbors, temperature):
            return {99999: 1.0}  # far outside a 4-token vocabulary

    logits = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32)
    token, fired = _select_with_memory(
        _Memory(),
        normalize(np.array([1.0, 0, 0, 0])),
        logits,
        k=1,
        temperature=1.0,
        phi=0.5,
        free_energy=1.0,
        base_lam=0.75,
        exclude_index=-1,
    )

    assert token == 2  # fell back to the model's own pick
    assert fired == -1
    assert 0 <= token < logits.shape[0]


def test_zero_token_budget_generates_nothing():
    """max(1, int(max_tokens)) turned a hard no-generation cap into one token
    of model work plus output."""
    from core.brain.nonparametric_generation import generate_with_memory

    class _Boom:
        def model(self, *a, **k):
            raise AssertionError("model must not run under a zero budget")

    class _Tok:
        eos_token_id = None

        def encode(self, text):
            return [1, 2, 3]

        def decode(self, ids):
            return "should not happen"

    assert generate_with_memory(_Boom(), _Tok(), "hi", None, max_tokens=0) == ""
    assert generate_with_memory(_Boom(), _Tok(), "hi", None, max_tokens=-5) == ""


def test_absent_continuation_is_not_token_zero():
    """0 is a real token id, so the old sentinel fabricated a recall target."""
    from core.brain.nonparametric_generation import MLXEncoder

    class _Tok:
        all_special_ids = [7]

        def encode(self, text):
            return [7]  # only a special token → no usable continuation

    encoder = MLXEncoder.__new__(MLXEncoder)
    encoder.tok = _Tok()
    encoder._specials = {7}

    assert encoder.first_token("") == MLXEncoder.NO_TOKEN
    assert MLXEncoder.NO_TOKEN != 0
