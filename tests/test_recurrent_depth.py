"""Tests for the Mythos-inspired recurrent-depth patch.

Guards the load-bearing assumption: mlx_lm's KVCache state/meta_state
snapshot/restore correctly rewinds offset after a mutation. A silent
failure here would have the recurrent loop accumulate N copies of K/V
into the cache — far worse than leaving recurrent depth off.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain.llm.recurrent_depth import (  # noqa: E402
    CacheSnapshotError,
    _get_lane_defaults,
    _materialize_recurrent_prefill_boundary,
    _restore_recurrent_caches,
    _self_test_cache_snapshot,
    _snapshot_recurrent_caches,
    apply_recurrent_depth,
    resolve_loops_for_model,
)


@pytest.mark.hardware
@pytest.mark.live
def test_self_test_cache_snapshot_passes_on_installed_mlx_lm():
    """If this fails, mlx_lm's cache contract changed and we must not patch."""
    _self_test_cache_snapshot()


def test_snapshot_fails_loud_on_unsupported_cache():
    """Incompatible caches must raise, never silently no-op."""

    class _BadCache:
        """Neither state/meta_state nor keys/values/offset."""
        pass

    with pytest.raises(CacheSnapshotError):
        _snapshot_recurrent_caches([_BadCache()], 0, 1)


def test_lane_defaults_cover_real_model_sizes():
    """Qwen2.5-32B has 64 layers; Qwen2.5-72B has 80. Both must land in
    the intended runtime envelopes for interactive use."""
    assert _get_lane_defaults(64)[0] >= 2, "32B (64 layers) must map to a looped lane"
    assert _get_lane_defaults(80)[0] == 1, "72B (80 layers) should default to a single pass for live solver turns"
    # And the small-model lanes must be standard-pass (no unnecessary cost).
    assert _get_lane_defaults(28)[0] == 1, "14B (28-40 layers) should be standard"
    assert _get_lane_defaults(12)[0] == 1, "7B class should be standard"


def test_resolve_loops_honors_72b_lane_override(monkeypatch):
    class _Inner:
        layers = [object()] * 80

    class _Model:
        model = _Inner()

    monkeypatch.setenv("AURA_RECURRENT_LOOPS_72B", "2")
    monkeypatch.delenv("AURA_RECURRENT_LOOPS", raising=False)

    assert resolve_loops_for_model(_Model()) == 2


def test_recurrent_depth_invalid_env_fails_as_runtime_error(monkeypatch):
    class _Inner:
        layers = [object()] * 64

    class _Model:
        model = _Inner()

    monkeypatch.setenv("AURA_RECURRENT_LOOPS_32B", "twice")
    monkeypatch.delenv("AURA_RECURRENT_LOOPS", raising=False)

    with pytest.raises(RuntimeError, match="AURA_RECURRENT_LOOPS_32B"):
        resolve_loops_for_model(_Model())


def test_recurrent_depth_rejects_unbounded_loop_override(monkeypatch):
    class _Inner:
        layers = [object()] * 64

    class _Model:
        model = _Inner()

    monkeypatch.setenv("AURA_RECURRENT_LOOPS_32B", "20")
    monkeypatch.delenv("AURA_RECURRENT_LOOPS", raising=False)

    with pytest.raises(RuntimeError, match="above safe maximum"):
        resolve_loops_for_model(_Model())


def test_recurrent_depth_rejects_unsafe_fraction_override(monkeypatch):
    import core.brain.llm.recurrent_depth as rd

    class _Inner:
        layers = [object()] * 64

    class _Model:
        model = _Inner()

    monkeypatch.setenv("AURA_RECURRENT_PRELUDE", "0.95")
    monkeypatch.delenv("AURA_RECURRENT_LOOPS", raising=False)
    monkeypatch.delenv("AURA_RECURRENT_LOOPS_32B", raising=False)

    with pytest.raises(RuntimeError, match="AURA_RECURRENT_PRELUDE"):
        rd.apply_for_model(_Model())


@pytest.mark.hardware
@pytest.mark.live
def test_restore_rewinds_mlx_cache():
    """Direct end-to-end proof the snapshot/restore actually works."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    c = KVCache()
    c.update_and_fetch(mx.ones((1, 2, 8, 16)), mx.ones((1, 2, 8, 16)))
    pre_offset = c.offset
    snap = _snapshot_recurrent_caches([c], 0, 1)

    c.update_and_fetch(mx.ones((1, 2, 1, 16)) * 3, mx.ones((1, 2, 1, 16)) * 3)
    assert c.offset > pre_offset, "Mutation did not advance cache offset"

    _restore_recurrent_caches([c], 0, 1, snap)
    assert c.offset == pre_offset, f"Restore failed: {pre_offset} → {c.offset}"


@pytest.mark.hardware
def test_batch_cache_snapshot_owns_arrays_and_restores_both_cursors():
    """A speculative batch pass must not mutate its saved rewind point."""
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache

    cache = BatchKVCache([0, 0])
    prefix = mx.ones((2, 8, 742, 128))
    cache.update_and_fetch(prefix, prefix)
    backing_capacity = cache.keys.shape[2]
    snapshot = _snapshot_recurrent_caches([cache], 0, 1)

    step = mx.ones((2, 8, 13, 128))
    cache.update_and_fetch(step, step)
    assert cache._idx == 755
    assert cache.offset.tolist() == [755, 755]

    _restore_recurrent_caches([cache], 0, 1, snapshot)

    assert cache._idx == 742
    assert cache.offset.tolist() == [742, 742]
    assert cache.keys.shape[2] == backing_capacity
    keys, _values = cache.update_and_fetch(step, step)
    assert keys.shape == (2, 8, 755, 128)
    assert cache.keys.shape[2] == backing_capacity
    assert cache._idx == 755
    assert cache.offset.tolist() == [755, 755]


@pytest.mark.hardware
def test_arrays_cache_snapshot_restores_owned_state_and_batch_coordinates():
    """Linear-attention state must rewind without aliasing its live list."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    cache = ArraysCache(size=2)
    original_conv = mx.ones((2, 3, 8))
    original_state = mx.ones((2, 4, 5, 6))
    cache[0] = original_conv
    cache[1] = original_state
    cache.prepare(lengths=[7, 5])
    snapshot = _snapshot_recurrent_caches([cache], 0, 1)

    cache[0] = mx.zeros_like(original_conv)
    cache[1] = mx.zeros_like(original_state)
    cache.advance(2)
    _restore_recurrent_caches([cache], 0, 1, snapshot)

    assert cache[0] is original_conv
    assert cache[1] is original_state
    assert cache.lengths.tolist() == [7, 5]

    # A later write must not mutate the retained snapshot container.
    cache[0] = mx.zeros_like(original_conv)
    _restore_recurrent_caches([cache], 0, 1, snapshot)
    assert cache[0] is original_conv


def _tiny_qwen35_model():
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 2,
        "full_attention_interval": 2,
        "head_dim": 16,
        "tie_word_embeddings": False,
    }
    return Model(ModelArgs(model_type="qwen3_5", text_config=text_config))


@pytest.mark.hardware
def test_real_qwen35_mixed_cache_recurrence_uses_each_layer_mask_contract():
    """Qwen3.8-family linear and full-attention caches must recur together."""
    import mlx.core as mx

    model = _tiny_qwen35_model()
    assert apply_recurrent_depth(
        model,
        n_loops=2,
        prelude_frac=0.25,
        coda_frac=0.25,
        residual_alpha=0.1,
    )
    cache = model.make_cache()
    assert [type(entry).__name__ for entry in cache] == [
        "ArraysCache",
        "KVCache",
        "ArraysCache",
        "KVCache",
        "ArraysCache",
        "KVCache",
        "ArraysCache",
        "KVCache",
    ]

    for tokens in (
        mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        mx.array([[5]], dtype=mx.int32),
    ):
        output = model(tokens, cache=cache)
        mx.eval(output)
        assert output.shape == (1, tokens.shape[1], 128)

    assert [entry.offset for entry in cache if hasattr(entry, "offset")] == [5] * 4
    assert all(entry[0] is not None for entry in cache if hasattr(entry, "cache"))


@pytest.mark.hardware
def test_real_qwen_batch_recurrence_keeps_layer_geometry_aligned():
    """Exercise the installed Qwen attention/cache call that failed live."""
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache
    from mlx_lm.models.qwen2 import Model, ModelArgs

    import core.brain.llm.recurrent_depth as rd

    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=8,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    model = Model(args)
    assert rd.apply_recurrent_depth(
        model,
        n_loops=2,
        prelude_frac=0.25,
        coda_frac=0.25,
        residual_alpha=0.1,
    )

    caches = [BatchKVCache([0, 3]) for _ in range(8)]
    prefix = mx.zeros((2, 2, 742, 8))
    for cache in caches:
        cache.update_and_fetch(prefix, prefix)

    for token_count, expected_idx in ((13, 755), (1, 756), (1, 757)):
        output = model(
            mx.zeros((2, token_count), dtype=mx.int32),
            cache=caches,
        )
        mx.eval(output)
        for cache in caches:
            assert cache._idx == expected_idx
            assert cache.offset.tolist() == [expected_idx, expected_idx - 3]
            assert cache.keys.shape[2] == 768


@pytest.mark.hardware
def test_multi_token_recurrent_prefill_materializes_between_loops(monkeypatch):
    """The 32B repair boundary is causal and decode remains asynchronous."""
    import mlx.core as mx

    calls: list[str] = []
    real_eval = mx.eval
    real_clear = mx.clear_cache

    def observed_eval(*args):
        calls.append("eval")
        return real_eval(*args)

    def observed_clear():
        calls.append("clear")
        return real_clear()

    monkeypatch.setattr(mx, "eval", observed_eval)
    monkeypatch.setattr(mx, "clear_cache", observed_clear)

    hidden = mx.ones((1, 128, 32))
    assert _materialize_recurrent_prefill_boundary(
        hidden,
        input_tokens=mx.zeros((1, 128), dtype=mx.int32),
    ) is True
    assert calls == ["eval", "clear"]

    calls.clear()
    assert _materialize_recurrent_prefill_boundary(
        mx.ones((1, 1, 32)),
        input_tokens=mx.zeros((1, 1), dtype=mx.int32),
    ) is False
    assert calls == []


@pytest.mark.hardware
def test_embedding_prefill_uses_embedding_sequence_axis(monkeypatch):
    import mlx.core as mx

    calls: list[str] = []
    monkeypatch.setattr(mx, "eval", lambda *_args: calls.append("eval"))
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append("clear"))

    assert _materialize_recurrent_prefill_boundary(
        mx.ones((1, 7, 32)),
        input_tokens=mx.zeros((0,), dtype=mx.int32),
        input_embeddings=mx.ones((1, 7, 32)),
    ) is True
    assert calls == ["eval", "clear"]

def _install_fake_mlx_modules(monkeypatch):
    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_pkg.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm_models = types.ModuleType("mlx_lm.models")
    mlx_lm_base = types.ModuleType("mlx_lm.models.base")
    mlx_lm_base.create_attention_mask = lambda _h, _cache: None
    mlx_lm_base.create_ssm_mask = lambda _h, _cache: None
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.base", mlx_lm_base)


def test_apply_recurrent_depth_is_instance_scoped(monkeypatch):
    import core.brain.llm.recurrent_depth as rd

    _install_fake_mlx_modules(monkeypatch)
    monkeypatch.setattr(rd, "_self_test_cache_snapshot", lambda: None)

    class _Inner:
        def __init__(self):
            self.layers = [object()] * 64

        def __call__(self, *_args, **_kwargs):
            return "original"

    class _Model:
        def __init__(self):
            self.model = _Inner()

    first = _Model()
    second = _Model()
    original_class = second.model.__class__

    assert rd.apply_recurrent_depth(first, n_loops=2) is True

    assert first.model.__class__ is not original_class
    assert second.model.__class__ is original_class
    assert second.model("prompt") == "original"

    assert rd.remove_recurrent_depth(first) is True
    assert first.model.__class__ is original_class


def test_recurrent_forward_executes_middle_block_multiple_times(monkeypatch):
    import core.brain.llm.recurrent_depth as rd

    _install_fake_mlx_modules(monkeypatch)
    monkeypatch.setattr(rd, "_self_test_cache_snapshot", lambda: None)

    class _Layer:
        def __init__(self):
            self.calls = 0

        def __call__(self, h, _mask, _cache):
            self.calls += 1
            return h + 1

    class _Inner:
        def __init__(self):
            self.layers = [_Layer() for _ in range(64)]

        def embed_tokens(self, inputs):
            return inputs

        def norm(self, h):
            return h

        def __call__(self, inputs, cache=None, input_embeddings=None):
            return inputs

    class _Model:
        def __init__(self):
            self.model = _Inner()

    model = _Model()

    assert rd.apply_recurrent_depth(
        model,
        n_loops=2,
        prelude_frac=0.20,
        coda_frac=0.20,
        residual_alpha=0.0,
    ) is True

    result = model.model(1, cache=[None] * 64)
    config = rd.get_recurrent_config(model)

    assert config["prelude_end"] == 12
    assert config["coda_start"] == 52
    assert result == 105
    assert all(layer.calls == 1 for layer in model.model.layers[:12])
    assert all(layer.calls == 2 for layer in model.model.layers[12:52])
    assert all(layer.calls == 1 for layer in model.model.layers[52:])


def test_recurrent_forward_runtime_override_cannot_exceed_configured_depth(monkeypatch):
    import core.brain.llm.recurrent_depth as rd

    _install_fake_mlx_modules(monkeypatch)
    monkeypatch.setattr(rd, "_self_test_cache_snapshot", lambda: None)

    class _Layer:
        def __init__(self):
            self.calls = 0

        def __call__(self, h, _mask, _cache):
            self.calls += 1
            return h + 1

    class _Inner:
        def __init__(self):
            self.layers = [_Layer() for _ in range(64)]

        def embed_tokens(self, inputs):
            return inputs

        def norm(self, h):
            return h

        def __call__(self, inputs, cache=None, input_embeddings=None):
            return inputs

    class _Model:
        def __init__(self):
            self.model = _Inner()

    model = _Model()

    assert rd.apply_recurrent_depth(
        model,
        n_loops=2,
        prelude_frac=0.20,
        coda_frac=0.20,
        residual_alpha=0.0,
    ) is True

    model.model._recurrent_depth_runtime_loops = 999
    model.model(1, cache=[None] * 64)

    assert all(layer.calls == 2 for layer in model.model.layers[12:52])


def test_recurrent_forward_runtime_override_can_reduce_depth(monkeypatch):
    import core.brain.llm.recurrent_depth as rd

    _install_fake_mlx_modules(monkeypatch)
    monkeypatch.setattr(rd, "_self_test_cache_snapshot", lambda: None)

    class _Layer:
        def __init__(self):
            self.calls = 0

        def __call__(self, h, _mask, _cache):
            self.calls += 1
            return h + 1

    class _Inner:
        def __init__(self):
            self.layers = [_Layer() for _ in range(64)]

        def embed_tokens(self, inputs):
            return inputs

        def norm(self, h):
            return h

        def __call__(self, inputs, cache=None, input_embeddings=None):
            return inputs

    class _Model:
        def __init__(self):
            self.model = _Inner()

    model = _Model()

    assert rd.apply_recurrent_depth(
        model,
        n_loops=2,
        prelude_frac=0.20,
        coda_frac=0.20,
        residual_alpha=0.0,
    ) is True

    model.model._recurrent_depth_runtime_loops = 1
    model.model(1, cache=[None] * 64)

    assert all(layer.calls == 1 for layer in model.model.layers[12:52])
