"""KV-cached foreground non-parametric memory: tap capture (no recompute) + O(n) loop.

The point of this module is latency: the foreground must not recompute a full forward per
token. These tests prove the tap records the hidden the generation forward already computes
(so the processor needs no extra forward), that the cached loop runs exactly one forward per
token, and that everything is fail-open.
"""

from __future__ import annotations

import numpy as np

from core.brain.nonparametric_binding import MemoryBinding
from core.brain.nonparametric_generation import normalize
from core.brain.nonparametric_memory import NonParametricMemory
from core.brain.nonparametric_worker import (
    HiddenStateTap,
    cached_generate_with_memory,
    foreground_memory_admitted_for_job,
    make_tapped_nonparametric_processor,
)
from tests.nonparametric_support import TEST_PRINCIPAL, entry_provenance


def _binding() -> MemoryBinding:
    """Whose memory the processor is allowed to read."""
    return MemoryBinding(principal=TEST_PRINCIPAL, source_id="worker_test")


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def test_structural_contract_inhibits_token_recall_without_memory_requirement():
    assert (
        foreground_memory_admitted_for_job({"requested_output_contract": {"kind": "word_count"}})
        is False
    )
    assert (
        foreground_memory_admitted_for_job(
            {
                "requested_output_contract": {"kind": "word_count"},
                "requires_memory_grounding": True,
            }
        )
        is True
    )
    assert (
        foreground_memory_admitted_for_job({"requested_output_contract": {"kind": "none"}}) is True
    )


def test_control_and_measurement_jobs_never_install_associative_recall():
    for flag in (
        "health_probe",
        "warmup_precompile",
        "proof_evaluation_contract",
        "operator_evidence_contract",
        "strict_answer_contract",
        "strict_value_contract",
    ):
        assert foreground_memory_admitted_for_job({flag: True}) is False

    # User recall may request memory, but that cannot override the independence
    # of a control-plane measurement.
    assert (
        foreground_memory_admitted_for_job(
            {"health_probe": True, "requires_memory_grounding": True}
        )
        is False
    )
    assert foreground_memory_admitted_for_job({"clean_user_surface_contract": True}) is True


# ── the tap captures the hidden the forward already produced ────────────────


def test_tap_records_last_hidden_and_restores_model():
    import mlx.core as mx

    kvec = normalize(np.array([1.0, 0, 0, 0]))

    class Inner:
        def __init__(self):
            self.calls = 0

        def __call__(self, seq, cache=None):
            self.calls += 1
            n = int(seq.shape[1])
            return mx.array(np.tile(kvec.astype(np.float32), (1, n, 1)))

    class Model:
        def __init__(self):
            self.model = Inner()

    model = Model()
    real_inner = model.model
    with HiddenStateTap(model) as tap:
        assert tap.active
        assert model.model is not real_inner  # proxy installed
        model.model(mx.array([[1, 2, 3]]))  # a normal generation forward
        assert tap.last_key is not None
        assert np.allclose(tap.last_key, kvec, atol=1e-6)
    assert model.model is real_inner  # restored on exit


def test_tap_installs_on_and_restores_a_hybrid_text_backbone():
    import mlx.core as mx

    kvec = normalize(np.array([0.0, 1.0, 0.0, 0.0]))

    class Inner:
        def __call__(self, seq, cache=None):
            width = int(seq.shape[1])
            return mx.array(np.tile(kvec.astype(np.float32), (1, width, 1)))

    class Language:
        def __init__(self):
            self.args = type("Args", (), {"hidden_size": 4})()
            self.model = Inner()

    language = Language()
    wrapper = type(
        "HybridWrapper",
        (),
        {
            "args": type("WrapperArgs", (), {"model_type": "qwen3_5"})(),
            "language_model": language,
        },
    )()
    real_inner = language.model

    with HiddenStateTap(wrapper) as tap:
        assert tap.active
        assert language.model is not real_inner
        language.model(mx.array([[1, 2]]))
        assert np.allclose(tap.last_key, kvec, atol=1e-6)

    assert language.model is real_inner


def test_tapped_processor_uses_tap_without_recompute():
    import mlx.core as mx

    mem = NonParametricMemory(dim=4)
    kvec = normalize(np.array([1.0, 0, 0, 0]))
    mem.add(kvec, token_id=0, token="x", weight=1.0, provenance=entry_provenance())

    class Inner:
        def __init__(self):
            self.calls = 0

        def __call__(self, seq, cache=None):
            self.calls += 1
            n = int(seq.shape[1])
            return mx.array(np.tile(kvec.astype(np.float32), (1, n, 1)))

    class Model:
        def __init__(self):
            self.model = Inner()

    model = Model()
    with HiddenStateTap(model) as tap:
        # the generation forward runs once and fills the tap
        model.model(mx.array([[1, 2, 3]]))
        forwards_after_generation = model.model.calls
        proc = make_tapped_nonparametric_processor(tap, mem, free_energy=1.0, binding=_binding())
        logits = mx.array(np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32))
        out = np.array(proc(mx.array([1, 2, 3]), logits)).reshape(-1)
        # recalled token 0 is boosted ...
        assert _softmax(out)[0] > _softmax(np.array([0.0, 0.0, 5.0, 0.0]))[0]
        # ... and the processor did NOT run another forward (no recompute)
        assert model.model.calls == forwards_after_generation


def test_tapped_processor_failopen_when_tap_empty():
    import mlx.core as mx

    mem = NonParametricMemory(dim=4)
    mem.add(
        normalize(np.array([1.0, 0, 0, 0])), token_id=0, token="x", provenance=entry_provenance()
    )

    tap = HiddenStateTap(object())  # never entered → last_key stays None
    proc = make_tapped_nonparametric_processor(tap, mem, binding=_binding())
    logits = mx.array(np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32))
    out = np.array(proc(mx.array([1]), logits)).reshape(-1)
    assert np.allclose(out, np.array([0.0, 0.0, 5.0, 0.0]))  # untouched


def test_tap_failopen_on_unswappable_model():
    # An object whose `model` attribute can't be reassigned (read-only) → tap stays inert.
    class Frozen:
        @property
        def model(self):
            return None  # no setter → assignment raises

    with HiddenStateTap(Frozen()) as tap:
        assert tap.active is False


# ── the cached loop runs exactly one forward per token (O(n), not O(n²)) ─────


def test_cached_generate_is_linear_and_uses_memory(monkeypatch):
    import mlx.core as mx

    recalled = normalize(np.array([0.0, 1.0, 0.0, 0.0]))
    mem = NonParametricMemory(dim=4)
    mem.add(recalled, token_id=7, token="seven", weight=1.0, provenance=entry_provenance())

    class Inner:
        def __init__(self):
            self.token_forwards = 0
            self.max_width = 0

        def __call__(self, seq, cache=None):
            n = int(seq.shape[1])
            self.max_width = max(self.max_width, n)
            # prefill passes the whole prompt (n>1); decode steps pass 1 token each.
            if n == 1:
                self.token_forwards += 1
            return mx.array(np.tile(recalled.astype(np.float32), (1, n, 1)))

    class Head:
        def __call__(self, hidden):
            # logits favor token 2; memory should pull toward token 7
            b, t, _ = hidden.shape
            base = np.zeros((b, t, 8), dtype=np.float32)
            base[..., 2] = 5.0
            return mx.array(base)

    class Args:
        tie_word_embeddings = False

    class Model:
        def __init__(self):
            self.model = Inner()
            self.lm_head = Head()
            self.args = Args()

    class Tok:
        eos_token_id = 99

        def encode(self, s):
            return [1, 2, 3]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    # Replace the KV cache factory so the loop runs without a real MLX model.
    monkeypatch.setattr("mlx_lm.models.cache.make_prompt_cache", lambda model: object())

    model = Model()
    out = cached_generate_with_memory(
        model, Tok(), "hi", mem, max_tokens=5, free_energy=1.0, principal=TEST_PRINCIPAL
    )
    # Step 1 prefills the 3-token prompt (and yields the 1st token); steps 2-5 each forward a
    # SINGLE new token via the cache → 4 width-1 decode forwards. The decisive O(n) proof:
    # no forward ever re-processes the whole growing sequence (max width == prompt length 3,
    # not 7). Under the old O(n²) recompute, max width would have climbed every step.
    assert model.model.token_forwards == 4
    assert model.model.max_width == 3
    # memory pulled generation toward the recalled token 7 (not the model's token 2)
    assert "7" in out
