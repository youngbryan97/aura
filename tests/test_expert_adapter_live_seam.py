"""Contract tests for the expert-adapter live seam.

Covers the chain Bryan's "more weights" build rides on:
worker attach/detach bookkeeping → client IPC guards → native
reload_model_artifact (the retired monkey-patch's replacement) → library
async residency honesty → router hook safety.
"""
from __future__ import annotations

import pytest

from core.brain.expert_lora_library import (
    ExpertLoRALibrary,
    LoRAAdapter,
)
from core.brain.llm.mlx_client import MLXLocalClient

pytestmark = pytest.mark.unit


# ── worker helpers (fake module tree, real bookkeeping) ──────────────────────

class LoRALinear:  # noqa: N801 - type NAME is the worker's detection contract
    def __init__(self, linear):
        self.linear = linear


class PlainLinear:
    pass


class FakeModel:
    def __init__(self):
        self.modules_by_name = {"layers.0.q_proj": PlainLinear()}
        self.updates = []

    def named_modules(self):
        return list(self.modules_by_name.items())

    def update_modules(self, tree):
        self.updates.append(tree)


def test_worker_attach_records_only_newly_wrapped_layers(monkeypatch):
    from core.brain.llm import mlx_worker

    model = FakeModel()
    # a pre-existing LoRA layer (the personality adapter) must NOT be recorded
    model.modules_by_name["layers.0.v_proj"] = LoRALinear(PlainLinear())

    def fake_load_adapters(m, path):
        m.modules_by_name["layers.0.q_proj"] = LoRALinear(
            m.modules_by_name["layers.0.q_proj"]
        )

    monkeypatch.setattr("mlx_lm.tuner.utils.load_adapters", fake_load_adapters)
    wrapped = mlx_worker._attach_expert_adapter(model, "/fake/adapter")
    assert [name for name, _ in wrapped] == ["layers.0.q_proj"]


def test_worker_attach_failure_unwinds_partial_wrap(monkeypatch):
    from core.brain.llm import mlx_worker

    model = FakeModel()

    def failing_load_adapters(m, path):
        m.modules_by_name["layers.0.q_proj"] = LoRALinear(
            m.modules_by_name["layers.0.q_proj"]
        )
        raise RuntimeError("weights corrupt")

    monkeypatch.setattr("mlx_lm.tuner.utils.load_adapters", failing_load_adapters)
    with pytest.raises(RuntimeError, match="weights corrupt"):
        mlx_worker._attach_expert_adapter(model, "/fake/adapter")
    # the unwind restored the original module via update_modules
    assert len(model.updates) == 1


def test_worker_detach_restores_wrapped_module():
    from core.brain.llm import mlx_worker

    model = FakeModel()
    original = model.modules_by_name["layers.0.q_proj"]
    wrapper = LoRALinear(original)
    restored = mlx_worker._detach_expert_adapter(model, [("layers.0.q_proj", wrapper)])
    assert restored == 1
    assert len(model.updates) == 1


def test_worker_dispatch_handles_set_expert_adapter():
    source = open("core/brain/llm/mlx_worker.py", encoding="utf-8").read()
    assert 'elif action == "set_expert_adapter":' in source
    # KV caches must be invalidated on weight change (CP126: invalidation is
    # now proven-or-fatal, validation precedes mutation, and a failed attach
    # rolls back to the previous identity instead of silently going bare).
    # The branch hands the swap to `_mlx_worker_loop_part_31`, so a 12,000
    # character window after it holds a call and not the clear. The unit is
    # the branch together with what it calls.
    from core.brain.llm import mlx_worker
    from tests.source_contract import function_with_its_helpers

    handler = function_with_its_helpers(mlx_worker, "_mlx_worker_loop", depth=2)
    assert "prompt_cache_lru.clear()" in handler
    assert "_clear_mlx_cache" in handler
    assert "metal_semaphore" in handler
    assert "_validate_expert_adapter_dir" in handler
    assert "_unrestorable_wrapped" in handler
    assert "restored_previous" in handler
    assert "cache_invalidated" in handler
    assert "_current_worker_identity()" in handler
    assert '"worker_identity": dict(worker_identity)' in handler
    assert '"requires_worker_recycle": not identity_restored' in handler


# ── client guards ─────────────────────────────────────────────────────────────

class ProcessProbe:
    """A stand-in for the worker process, with the surface shutdown needs.

    It had `is_alive` and nothing else, so the client's termination proof —
    kill, join, re-check — hit an AttributeError and reported
    `mlx_worker_termination_unproven` at teardown of every test using it. The
    proof was correct; the double could not be killed, which is not a state a
    real process can be in.
    """

    def __init__(self, alive=True):
        self._alive = alive
        self.kill_calls = 0
        self.join_calls = 0
        self.pid = 4242

    def is_alive(self):
        return self._alive

    def kill(self):
        self.kill_calls += 1
        self._alive = False

    def join(self, timeout=None):
        self.join_calls += 1
        return None


@pytest.fixture
def client():
    c = MLXLocalClient(model_path="Qwen2.5-32B-cortex-test")
    yield c
    c.close()


async def test_set_expert_adapter_refuses_without_worker(client, tmp_path):
    res = await client.set_expert_adapter(str(tmp_path / "some-adapter"))
    assert res == {"ok": False, "reason": "worker_not_ready"}


async def test_set_expert_adapter_refuses_mid_generation(client, tmp_path):
    client._req_q = object()
    client._process = ProcessProbe(alive=True)
    client._init_done = True
    client._active_generations = 1
    res = await client.set_expert_adapter(str(tmp_path))
    client._req_q = None
    assert res == {"ok": False, "reason": "generation_active"}


async def test_set_expert_adapter_refuses_missing_dir(client):
    client._req_q = object()
    client._process = ProcessProbe(alive=True)
    client._init_done = True
    res = await client.set_expert_adapter("/nonexistent/adapter-xyz")
    client._req_q = None
    assert res["ok"] is False and res["reason"].startswith("adapter_missing")


async def test_reload_model_artifact_refuses_missing_dir(client):
    res = await client.reload_model_artifact("/nonexistent/fused-xyz")
    assert res["ok"] is False and res["reason"].startswith("artifact_missing")


def _servable_artifact(root, name: str):
    """A directory the worker could actually load: config, tokenizer, weights."""
    artifact = root / name
    artifact.mkdir()
    (artifact / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    (artifact / "tokenizer.json").write_text("{}")
    (artifact / "model.safetensors").write_bytes(b"\x00" * 16)
    return artifact


async def test_reload_model_artifact_defers_while_busy(client, tmp_path):
    artifact = _servable_artifact(tmp_path, "fused-gen1")
    client._active_generations = 1
    res = await client.reload_model_artifact(str(artifact))
    assert res["ok"] is True and res["state"] == "staged"
    # The old worker is still decoding the OLD weights: the lane must not
    # describe itself as the new model until something is serving it.
    assert client.model_path != str(artifact)
    assert client._pending_promotion == str(artifact)
    # A recovery verdict must not be able to discard the staged promotion.
    client._deferred_reboot_reason = "token_progress_stalled"
    assert client._pending_promotion == str(artifact)
    client._deferred_reboot_reason = None
    client._pending_promotion = None
    client._active_generations = 0


async def test_reload_model_artifact_recycles_when_idle(client, tmp_path, monkeypatch):
    artifact = _servable_artifact(tmp_path, "fused-gen2")
    calls = {}

    async def fake_reboot(reason="", mark_failed=True):
        calls["reason"] = reason
        calls["mark_failed"] = mark_failed

    monkeypatch.setattr(client, "reboot_worker", fake_reboot)
    res = await client.reload_model_artifact(str(artifact))
    # A recycle is a teardown. The lane is UNLOADED, not live.
    assert res["ok"] is True and res["state"] == "unloaded"
    assert calls == {"reason": "promoted_artifact_swap", "mark_failed": False}
    assert client.model_path == str(artifact)


async def test_reload_model_artifact_refuses_an_empty_directory(client, tmp_path):
    empty = tmp_path / "fused-empty"
    empty.mkdir()
    res = await client.reload_model_artifact(str(empty))
    assert res["ok"] is False and res["reason"].startswith("artifact_missing_config")
    # The healthy lane keeps serving what it was serving.
    assert client.model_path != str(empty)


async def test_reload_model_artifact_refuses_missing_weights(client, tmp_path):
    partial = tmp_path / "fused-partial"
    partial.mkdir()
    (partial / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    (partial / "tokenizer.json").write_text("{}")
    res = await client.reload_model_artifact(str(partial))
    assert res["ok"] is False and res["reason"] == "artifact_missing_weights"


async def test_reload_model_artifact_refuses_a_different_architecture(
    client, tmp_path, monkeypatch
):
    incumbent = _servable_artifact(tmp_path, "incumbent")
    monkeypatch.setattr(client, "model_path", str(incumbent))
    foreign = tmp_path / "fused-llama"
    foreign.mkdir()
    (foreign / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    (foreign / "tokenizer.json").write_text("{}")
    (foreign / "model.safetensors").write_bytes(b"\x00" * 16)
    res = await client.reload_model_artifact(str(foreign))
    assert res["ok"] is False
    assert res["reason"].startswith("artifact_architecture_mismatch")
    assert client.model_path == str(incumbent)


def test_monkey_patch_is_gone():
    source = open("core/learning/live_learner.py", encoding="utf-8").read()
    assert "types.MethodType" not in source
    assert "async def patch_mlx_client_for_hot_swap" not in source
    # the in-orchestrator model load (the ~20GB memory bomb) must never return
    assert "self_or_client._model = new_model" not in source


# ── library async residency ───────────────────────────────────────────────────

class RecordingAsyncApplier:
    def __init__(self, load_ok=True):
        self.load_ok = load_ok
        self.loaded = []
        self.unloaded = []

    async def load(self, adapter):
        self.loaded.append(adapter.name)
        return self.load_ok

    async def unload(self, adapter):
        self.unloaded.append(adapter.name)
        return True


def _seam_artifact(root, name):
    """CP126 70c50967: an adapter must be a real artifact before it may load."""
    adapter_dir = root / name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    return str(adapter_dir)


@pytest.fixture
def library(tmp_path):
    lib = ExpertLoRALibrary(manifest_path=tmp_path / "library.json", max_resident=1)
    assert lib.register(LoRAAdapter(name="math-a", path=_seam_artifact(tmp_path, "a"),
                                    task_types={"math"}, keywords={"arithmetic"}, quality=0.8))
    assert lib.register(LoRAAdapter(name="logic-b", path=_seam_artifact(tmp_path, "b"),
                                    task_types={"logic"}, keywords={"deduction"}, quality=0.7))
    return lib


async def test_activate_async_reflects_applier_truth(library):
    applier = RecordingAsyncApplier(load_ok=False)
    ok = await library.activate_async("math-a", applier)
    assert ok is False
    assert library.resident() == []  # a refused swap is never claimed resident


async def test_activate_async_evicts_lru_through_applier(library):
    applier = RecordingAsyncApplier()
    assert await library.activate_async("math-a", applier)
    assert await library.activate_async("logic-b", applier)
    assert applier.unloaded == ["math-a"]  # max_resident=1 evicted the LRU
    assert library.resident() == ["logic-b"]


async def test_select_and_activate_async_is_env_gated(library, monkeypatch):
    monkeypatch.delenv("AURA_EXPERT_LORA_LIBRARY", raising=False)
    applier = RecordingAsyncApplier()
    got = await library.select_and_activate_async("solve arithmetic", "math", applier)
    assert got is None and applier.loaded == []

    monkeypatch.setenv("AURA_EXPERT_LORA_LIBRARY", "1")
    got = await library.select_and_activate_async("solve arithmetic", "math", applier)
    assert got is not None and got.name == "math-a"
    assert applier.loaded == ["math-a"]


# ── router hook safety ────────────────────────────────────────────────────────

async def test_router_hook_is_default_off(monkeypatch):
    from core.brain.llm_health_router import HealthAwareLLMRouter

    monkeypatch.delenv("AURA_EXPERT_LORA_ROUTING", raising=False)
    router = HealthAwareLLMRouter.__new__(HealthAwareLLMRouter)  # no full init needed
    # must be a no-op that never raises, even half-constructed
    await router._maybe_route_expert_adapter("prompt", {})
