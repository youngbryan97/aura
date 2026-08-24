"""The checkpoint itself becomes recurrent (CP226).

The prior RLC recurred four side slots while the answer tokens traversed
the middle block exactly once, at every depth. That architecture cannot
produce a depth effect on the answer, and measurement agreed: 25/29/25/25
across an 8x compute range. These tests pin the corrected shape -- the real
token stream re-enters the window -- and the safety property that lets it
be added to a working checkpoint: T=1 is bit-identical to the base model.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.learning.intrinsic_recurrence import (  # noqa: E402
    RecurrentDepthPlan,
    current_iteration,
    recurrent_hidden_states,
    recurrent_iteration,
    recurrent_logits,
    trajectory_dynamics,
)

LAYERS = 8


def _model() -> Model:
    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=LAYERS,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=256,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


TOKENS = mx.array([[3, 11, 42, 7, 19]])


# ── Safety: recurrence is added FROM a known-good point ─────────────────


def test_one_iteration_is_bit_identical_to_the_base_model():
    model = _model()
    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=1)
    assert plan.is_base_equivalent()
    assert bool(
        mx.allclose(model(TOKENS), recurrent_logits(model, TOKENS, plan), atol=1e-5)
    ), "T=1 must reproduce the unmodified forward pass exactly"


def test_stabilizers_do_not_perturb_the_first_pass():
    """Anchor injection, renorm and inter-pass noise apply only at RE-entry,
    so T=1 stays identical no matter how they are configured."""
    model = _model()
    base = model(TOKENS)
    for injection, renorm, noise in (
        (0.5, False, 0.0),
        (0.0, True, 0.0),
        (1.0, True, 0.0),
        (0.0, False, 0.5),
        (0.5, True, 0.25),
    ):
        plan = RecurrentDepthPlan(
            prelude_end=2, coda_start=6, iterations=1,
            anchor_injection=injection, renormalize=renorm,
            interpass_noise=noise, noise_seed=9,
        )
        assert bool(mx.allclose(base, recurrent_logits(model, TOKENS, plan), atol=1e-5))


# ── Inter-pass noise: the divergence kick, deterministic and off-by-default


def test_interpass_noise_perturbs_only_reentries_and_replays_exactly():
    """The kick must exist (trajectories differ from the plain loop), must
    replay bit-for-bit under the same seed (results stay auditable), and
    must differ across seeds (it is noise, not a constant offset)."""
    model = _model()
    plain = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=3)
    noisy = RecurrentDepthPlan(
        prelude_end=2, coda_start=6, iterations=3,
        interpass_noise=0.1, noise_seed=13,
    )
    _, base_trajectory = recurrent_hidden_states(model, TOKENS, plain)
    _, noisy_trajectory = recurrent_hidden_states(model, TOKENS, noisy)
    _, replay_trajectory = recurrent_hidden_states(model, TOKENS, noisy)
    # First pass untouched; later passes kicked.
    assert bool(mx.allclose(base_trajectory[0], noisy_trajectory[0], atol=1e-6))
    assert not bool(mx.allclose(base_trajectory[1], noisy_trajectory[1], atol=1e-4))
    for ours, again in zip(noisy_trajectory, replay_trajectory, strict=True):
        assert bool(mx.array_equal(ours, again)), "same seed must replay exactly"
    other_seed = RecurrentDepthPlan(
        prelude_end=2, coda_start=6, iterations=3,
        interpass_noise=0.1, noise_seed=14,
    )
    _, other_trajectory = recurrent_hidden_states(model, TOKENS, other_seed)
    assert not bool(
        mx.allclose(noisy_trajectory[1], other_trajectory[1], atol=1e-5)
    ), "different seeds must kick differently"


# ── The real token stream gets deeper ───────────────────────────────────


def test_the_answer_path_itself_recurs():
    """The property the previous architecture lacked: extra depth changes
    the logits of the actual tokens, not a side scratchpad's contents."""
    model = _model()
    shallow = recurrent_logits(
        model, TOKENS, RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=1)
    )
    deep = recurrent_logits(
        model, TOKENS, RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=4)
    )
    assert not bool(mx.allclose(shallow, deep, atol=1e-4))


def test_effective_depth_is_reported_honestly():
    plan = RecurrentDepthPlan(prelude_end=16, coda_start=48, iterations=4)
    # 16 prelude + 4*32 window + 16 coda
    assert plan.effective_depth(64) == 160
    assert plan.window_size() == 32
    receipt = plan.to_receipt(64)
    assert receipt["effective_depth"] == 160
    assert receipt["base_equivalent"] is False
    assert RecurrentDepthPlan(16, 48, 1).to_receipt(64)["effective_depth"] == 64


def test_trajectory_is_returned_for_every_iteration():
    model = _model()
    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=5)
    hidden, trajectory = recurrent_hidden_states(model, TOKENS, plan)
    assert len(trajectory) == 5
    assert hidden.shape == (1, TOKENS.shape[1], 64)


# ── Motion is not progress: the loop must be gradeable ──────────────────


def test_dynamics_flag_a_fixed_point():
    """A loop that stopped moving stopped computing, whatever the compute
    budget claims."""
    state = mx.ones((1, 4, 8))
    report = trajectory_dynamics([state, state, state, state])
    assert report["at_fixed_point"] is True
    assert report["contracting"] is True


def test_dynamics_flag_an_oscillation():
    a, b = mx.zeros((1, 4, 8)), mx.ones((1, 4, 8))
    report = trajectory_dynamics([a, b, a, b, a])
    assert report["oscillating"] is True


def test_dynamics_refuse_to_judge_a_single_iteration():
    report = trajectory_dynamics([mx.ones((1, 4, 8))])
    assert report["measurable"] is False


def test_a_moving_loop_is_not_called_a_fixed_point():
    model = _model()
    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=4)
    _, trajectory = recurrent_hidden_states(model, TOKENS, plan)
    report = trajectory_dynamics(trajectory)
    assert report["measurable"] is True
    assert len(report["relative_deltas"]) == 3


# ── Per-iteration identity ──────────────────────────────────────────────


def test_iteration_index_is_published_and_restored():
    assert current_iteration() == 0
    with recurrent_iteration(3):
        assert current_iteration() == 3
        with recurrent_iteration(5):
            assert current_iteration() == 5
        assert current_iteration() == 3
    assert current_iteration() == 0
    with pytest.raises(ValueError, match="non-negative"):
        with recurrent_iteration(-1):
            pass


def test_the_window_sees_its_own_iteration_index():
    """Depth-conditioned adapters need this: the same weights doing
    different work per pass is the difference between deepening and
    repeating."""
    model = _model()
    seen: list[int] = []
    window_layer = model.model.layers[3]
    original = window_layer.__call__

    def spy(*args, **kwargs):
        seen.append(current_iteration())
        return original(*args, **kwargs)

    model.model.layers[3] = type(
        "Spy", (), {"__call__": lambda self, *a, **k: spy(*a, **k)}
    )()
    recurrent_hidden_states(
        model, TOKENS, RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=3)
    )
    assert seen == [0, 1, 2]


# ── Fail closed ─────────────────────────────────────────────────────────


def test_invalid_plans_are_refused():
    with pytest.raises(ValueError, match="prelude_end must precede"):
        RecurrentDepthPlan(prelude_end=6, coda_start=6)
    with pytest.raises(ValueError, match="iterations must be at least 1"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=0)
    with pytest.raises(ValueError, match="anchor_injection"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6, anchor_injection=1.5)
    with pytest.raises(ValueError, match="renormalize"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6, renormalize="yes")
    with pytest.raises(ValueError, match="interpass_noise"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6, interpass_noise=1.5)
    with pytest.raises(ValueError, match="noise_seed"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6, noise_seed=-1)
    with pytest.raises(ValueError, match="smaller than the plan"):
        RecurrentDepthPlan(prelude_end=2, coda_start=6).effective_depth(4)
    with pytest.raises(ValueError, match="exceeds the model"):
        recurrent_hidden_states(
            _model(), TOKENS, RecurrentDepthPlan(prelude_end=2, coda_start=99)
        )


# ── Cached decode: O(n), and each pass keeps its own history ────────────


def _stepwise_divergence(model, plan) -> float:
    """Relative gap between prefill and token-by-token cached decode."""
    from core.learning.intrinsic_recurrence import make_recurrent_caches

    reference = recurrent_logits(model, TOKENS, plan)
    caches = make_recurrent_caches(model, plan)
    stepwise = None
    for index in range(TOKENS.shape[1]):
        stepwise = recurrent_logits(
            model, TOKENS[:, index : index + 1], plan, caches=caches
        )
    gap = float(mx.max(mx.abs(reference[:, -1:, :] - stepwise)))
    return gap / max(float(mx.max(mx.abs(reference))), 1e-9)


def test_cached_decode_adds_no_error_beyond_the_base_models_own():
    """The cache is an optimization, not a different model.

    Batched prefill and single-token decode reduce in different orders, so
    they never agree bitwise -- that gap exists at T=1, where the pass IS
    the unmodified base model. The honest contract is therefore that
    recurrence adds no error ON TOP of that, which an arbitrary tolerance
    would not have distinguished from a genuine cache bug.
    """
    model = _model()
    baseline = _stepwise_divergence(
        model, RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=1)
    )
    assert baseline < 5e-3, "base-model decode itself should be close"
    for iterations in (2, 3, 4):
        recurrent = _stepwise_divergence(
            model, RecurrentDepthPlan(2, 6, iterations=iterations)
        )
        assert recurrent < max(baseline * 3.0, 5e-3), (
            f"T={iterations} diverges beyond the base model's own decode gap "
            f"({recurrent:.2e} vs {baseline:.2e}) -- that is a cache defect, "
            "not float noise"
        )


def test_cached_decode_agrees_on_the_chosen_token():
    """What decode actually consumes is the argmax, not the raw logits."""
    from core.learning.intrinsic_recurrence import make_recurrent_caches

    model = _model()
    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=3)
    reference = recurrent_logits(model, TOKENS, plan)
    caches = make_recurrent_caches(model, plan)
    stepwise = None
    for index in range(TOKENS.shape[1]):
        stepwise = recurrent_logits(
            model, TOKENS[:, index : index + 1], plan, caches=caches
        )
    assert int(mx.argmax(reference[0, -1])) == int(mx.argmax(stepwise[0, -1]))


def test_each_iteration_gets_its_own_history():
    """One shared cache would let pass 4 read pass 1's keys and silently
    corrupt the recurrence."""
    from core.learning.intrinsic_recurrence import make_recurrent_caches

    model = _model()
    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=3)
    caches = make_recurrent_caches(model, plan)
    assert len(caches["window"]) == 3
    assert len({id(c) for row in caches["window"] for c in row}) == 3 * 4
    assert len(caches["prelude"]) == 2
    assert len(caches["coda"]) == LAYERS - 6


def test_mismatched_caches_are_refused():
    from core.learning.intrinsic_recurrence import make_recurrent_caches

    model = _model()
    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=3)
    wrong = make_recurrent_caches(model, RecurrentDepthPlan(2, 6, iterations=2))
    with pytest.raises(ValueError, match="cache iteration count"):
        recurrent_hidden_states(model, TOKENS, plan, caches=wrong)
    with pytest.raises(ValueError, match="make_recurrent_caches"):
        recurrent_hidden_states(model, TOKENS, plan, caches={"prelude": []})


# ── One clock: the depth bank must see the intrinsic iteration ──────────


def test_depth_conditioned_weights_track_the_intrinsic_iteration():
    """Two ContextVars for the same concept means the depth bank reports 0
    for every pass -- a mechanism present in name only."""
    from core.learning.depth_conditioned_lora import current_depth_index

    assert current_depth_index() == 0
    with recurrent_iteration(2):
        assert current_iteration() == 2
        assert current_depth_index() == 2, (
            "depth-conditioned LoRA would apply pass-0 weights on pass 2"
        )
    assert current_depth_index() == 0, "must restore"


def test_the_window_sees_a_consistent_depth_index_per_pass():
    from core.learning.depth_conditioned_lora import current_depth_index

    model = _model()
    seen: list[tuple[int, int]] = []
    original = model.model.layers[3]

    class Spy:
        def __call__(self, *args, **kwargs):
            seen.append((current_iteration(), current_depth_index()))
            return original(*args, **kwargs)

    model.model.layers[3] = Spy()
    recurrent_hidden_states(
        model, TOKENS, RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=4)
    )
    assert seen == [(0, 0), (1, 1), (2, 2), (3, 3)]


# ── Hybrid checkpoints: the cache type is per layer ──────────────────────
# Qwen3.8-27B is 16 attention layers and 48 linear-attention layers. A linear
# layer holds a recurrent state, not K/V, and mlx_lm defers to the model's own
# ``make_cache`` for exactly that reason. Both of Aura's cache builders used to
# construct 64 plain KVCaches, which is the wrong object at three layers in
# four — and reads as a working decode until the linear layers carry nothing.


class _LinearState:
    """Stands in for ``ArraysCache``: what a gated-delta layer holds."""


class _HybridModel:
    """A model whose layers do not all want the same cache, as qwen3_5 is."""

    def __init__(self, total: int = 8, full_interval: int = 4):
        self._total = total
        self._interval = full_interval
        self.model = SimpleNamespace(layers=[object()] * total)

    def make_cache(self):
        import mlx_lm.models.cache as cache_module

        return [
            cache_module.KVCache()
            if (index + 1) % self._interval == 0
            else _LinearState()
            for index in range(self._total)
        ]


def test_hybrid_caches_keep_each_layers_own_type():
    from core.learning.intrinsic_recurrence import model_layer_caches

    caches = model_layer_caches(_HybridModel())
    kinds = [type(c).__name__ for c in caches]
    assert kinds.count("_LinearState") == 6
    assert kinds.count("KVCache") == 2


def test_recurrent_caches_partition_a_hybrid_model_by_layer():
    from core.learning.intrinsic_recurrence import make_recurrent_caches

    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=3)
    caches = make_recurrent_caches(_HybridModel(), plan)

    assert len(caches["prelude"]) == 2
    assert len(caches["coda"]) == 2
    assert len(caches["window"]) == 3
    assert all(len(window) == 4 for window in caches["window"])

    # Layer 3 (0-indexed) is the full-attention one inside the window, and it
    # has to be the KVCache in every iteration — not merely somewhere.
    for window in caches["window"]:
        assert type(window[1]).__name__ == "KVCache"
        assert all(type(window[i]).__name__ == "_LinearState" for i in (0, 2, 3))


def test_each_window_iteration_still_gets_its_own_cache_objects():
    from core.learning.intrinsic_recurrence import make_recurrent_caches

    plan = RecurrentDepthPlan(prelude_end=2, coda_start=6, iterations=3)
    caches = make_recurrent_caches(_HybridModel(), plan)
    identities = [id(c) for window in caches["window"] for c in window]
    assert len(set(identities)) == len(identities)


def test_a_dense_model_without_make_cache_still_gets_kv_caches():
    from core.learning.intrinsic_recurrence import model_layer_caches

    dense = SimpleNamespace(model=SimpleNamespace(layers=[object()] * 5))
    caches = model_layer_caches(dense)
    assert len(caches) == 5
    assert {type(c).__name__ for c in caches} == {"KVCache"}
