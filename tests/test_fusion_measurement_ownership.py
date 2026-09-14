"""Interventions cannot race live state or escape into a later generation."""

import threading
from contextlib import nullcontext

import numpy as np
import pytest

from core.consciousness import affective_steering as steering
from core.consciousness import fusion_probe


def engine_and_hook():
    values = np.random.default_rng(87).normal(size=64).astype(np.float32)
    values /= np.linalg.norm(values)
    vector = steering.SteeringVector(key="valence", layer_idx=1, d_model=64,
                                    v=values, substrate_idx=0, substrate_fn="linear")
    hook = steering.AffectiveSteeringHook(object(), 1, {"valence": vector}, alpha=0.13)
    hook.update_substrate(fusion_probe.STATE_LOW)
    engine = steering.AffectiveSteeringEngine()
    engine._hooks = [hook]
    engine._alpha = 0.19
    engine._surface_alpha_override = 0.15
    engine.telemetry.alpha = 0.13
    return engine, hook


def state(hook):
    return (hook._alpha, hook._active, hook._substrate_x.copy(), dict(hook._latest_moods),
            hook._last_substrate_sync_monotonic, hook.current_composite_vector(),
            hook._cached_composite_mx)


def assert_restored(hook, before):
    after = state(hook)
    assert after[:2] == before[:2]
    np.testing.assert_array_equal(after[2], before[2])
    assert after[3:5] == before[3:5]
    np.testing.assert_array_equal(after[5], before[5])
    assert after[6] is before[6]


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_probe_restores_exact_state_on_success_failure_and_interrupt(monkeypatch, failure):
    engine, hook = engine_and_hook()
    before = state(hook)

    def mutate(model, tokenizer, hooks, set_alpha, **kwargs):
        set_alpha(0.1)
        hook.update_substrate(fusion_probe.STATE_HIGH)
        hook.override_composite_vector(np.ones(64, dtype=np.float32))
        if failure:
            raise failure("probe interruption")
        return ["measured"]

    monkeypatch.setattr(fusion_probe, "_measure_fusion", mutate)
    def run():
        return fusion_probe.measure_fusion(
            object(), object(), [hook], engine.set_alpha, model_identity="test",
            control_context=engine.controlled_measurement(),
        )
    if failure:
        with pytest.raises(failure):
            run()
    else:
        assert run() == ["measured"]
    assert_restored(hook, before)
    assert (engine._alpha, engine._surface_alpha_override, engine.telemetry.alpha) == (0.19, 0.15, 0.13)


def test_real_probe_failure_inside_random_control_restores_state(monkeypatch):
    engine, hook = engine_and_hook()
    before = state(hook)
    monkeypatch.setattr(fusion_probe, "PROBE_PROMPTS", ["test"])
    monkeypatch.setattr(fusion_probe, "chat_ids", lambda *args: [1])
    monkeypatch.setattr(fusion_probe, "_greedy_path", lambda *args: [1])
    monkeypatch.setattr(fusion_probe, "_walk", lambda *args: [np.array([0.4, 0.6])])
    calls = []
    def choices(*args):
        calls.append(hook.current_composite_vector())
        if len(calls) == 3:
            raise RuntimeError("random control failed")
        return 1.0, 0.5
    monkeypatch.setattr(fusion_probe, "_forced_choice", choices)
    with pytest.raises(RuntimeError, match="random control failed"):
        fusion_probe.measure_fusion(object(), object(), [hook], engine.set_alpha,
                                    control_context=engine.controlled_measurement(),
                                    model_identity="test", alphas=[0.1], steps=1)
    assert len(calls) == 3
    assert_restored(hook, before)


def test_live_sync_waits_for_measurement_then_resumes(monkeypatch):
    engine, hook = engine_and_hook()
    sync = steering.SubstrateSyncThread([hook], engine)
    read = threading.Event()
    started = threading.Event()
    def read_live():
        read.set()
        sync.stop()
        return np.ones(64, dtype=np.float32), "observed"
    monkeypatch.setattr(sync, "_read_substrate_vector", read_live)
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda *args, **kwargs: None)
    sync._running = True
    def run():
        started.set()
        sync._loop()
    thread = threading.Thread(target=run)
    try:
        with engine.controlled_measurement(), hook.preserve_control_state():
            thread.start()
            assert started.wait(2)
            assert not read.wait(0.05)
            engine.set_alpha(0.1)
            hook.update_substrate(fusion_probe.STATE_HIGH)
            assert not read.is_set()
        assert read.wait(2)
    finally:
        sync.stop()
        if thread.ident is not None:
            thread.join(2)
    assert not thread.is_alive()
    assert hook._latest_moods["_source"] == "observed"


def test_standalone_probe_also_restores_after_callback_failure(monkeypatch):
    engine, hook = engine_and_hook()
    before = state(hook)
    def mutate(*args, **kwargs):
        hook.override_composite_vector(None)
        kwargs["on_result"](object())
    monkeypatch.setattr(fusion_probe, "_measure_fusion", mutate)
    def fail(result):
        raise OSError("receipt storage unavailable")
    with pytest.raises(OSError):
        fusion_probe.measure_fusion(object(), object(), [hook], lambda value: None,
                                    control_context=nullcontext(), model_identity="test", on_result=fail)
    assert_restored(hook, before)


def test_owned_measurement_has_stable_alpha_but_serving_staleness_is_preserved():
    engine, hook = engine_and_hook()
    engine.set_surface_alpha_override(0.0)
    hook._last_substrate_sync_monotonic = 1.0
    with engine.controlled_measurement(), hook.preserve_control_state():
        engine.set_alpha(0.2)
        assert hook._effective_alpha() == 0.2
        engine.set_alpha(1.0)
        assert hook._effective_alpha() == hook._INJECTION_ALPHA_CEILING
    assert hook._measurement_thread_id is None
    assert hook._alpha == 0.0 and engine._surface_alpha_override == 0.0
    engine.set_surface_alpha_override(None)
    engine.set_alpha(0.2)
    assert hook._effective_alpha() == hook._STALE_SAFE_ALPHA


def test_multiple_hooks_can_be_reentered_during_a_probe(monkeypatch):
    from core.runtime import lockdep

    validator = lockdep.LockdepValidator()
    monkeypatch.setattr(lockdep, "_VALIDATOR", validator)
    engine, hook = engine_and_hook()
    other = steering.AffectiveSteeringHook(object(), 3, hook._vectors, alpha=0.07)
    other.update_substrate(fusion_probe.STATE_LOW)
    engine._hooks.append(other)
    before = [state(item) for item in engine._hooks]
    def mutate(*args, **kwargs):
        engine.set_alpha(0.2)
        for item in engine._hooks:
            item.update_substrate(fusion_probe.STATE_HIGH)
            assert item.compute_composite_vector_mx() is not None
        return []
    monkeypatch.setattr(fusion_probe, "_measure_fusion", mutate)
    fusion_probe.measure_fusion(object(), object(), engine._hooks, engine.set_alpha,
                                control_context=engine.controlled_measurement(), model_identity="test")
    for item, saved in zip(engine._hooks, before, strict=True):
        assert_restored(item, saved)
    assert validator.report()["splats"] == []


def test_random_controls_match_each_layers_own_magnitude(monkeypatch):
    engine, hook = engine_and_hook()
    other = steering.AffectiveSteeringHook(object(), 3, hook._vectors, alpha=0.07)
    engine._hooks.append(other)
    monkeypatch.setattr(fusion_probe, "PROBE_PROMPTS", ["test"])
    monkeypatch.setattr(fusion_probe, "chat_ids", lambda *args: [1])
    monkeypatch.setattr(fusion_probe, "_greedy_path", lambda *args: [1])
    monkeypatch.setattr(fusion_probe, "_walk", lambda *args: [np.array([0.4, 0.6])])
    def set_distinct(moods):
        other.override_composite_vector(np.ones(64, dtype=np.float32) * 0.025)
    monkeypatch.setattr(other, "update_substrate", set_distinct)
    norms = []
    def choices(*args):
        norms.append([None if (v := item.current_composite_vector()) is None else float(np.linalg.norm(v))
                      for item in engine._hooks])
        return 1.0, 0.5
    monkeypatch.setattr(fusion_probe, "_forced_choice", choices)
    result = fusion_probe.measure_fusion(object(), object(), engine._hooks, engine.set_alpha,
                                         control_context=engine.controlled_measurement(),
                                         model_identity="test", alphas=[0.1], steps=1)
    assert len(result) == 1 and norms[1][0] != pytest.approx(norms[1][1])
    for control in norms[2:]:
        np.testing.assert_allclose(control, norms[1], rtol=1e-6)
    assert result[0].measurement_protocol == fusion_probe.FUSION_MEASUREMENT_PROTOCOL
