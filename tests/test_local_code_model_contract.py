from __future__ import annotations

import asyncio
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


def _model_tree(path, *, context=256):
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"verified-test-weights")
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "max_position_embeddings": context,
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return path


def _identity(module, path, *, context=4096):
    return module.TrustedModelIdentity(
        real_path=str(path),
        checkpoint_fingerprint="a" * 64,
        checkpoint_files=1,
        behavior_bundle_sha256="b" * 64,
        architecture="Qwen2ForCausalLM",
        max_context_tokens=context,
        trust_manifest_sha256="c" * 64,
    )


def test_external_manifest_binds_full_weights_behavior_and_architecture(monkeypatch, tmp_path):
    from core.brain.llm import local_code_model as module

    model_path = _model_tree(tmp_path / "model")
    trust_path = tmp_path / "trust" / "model.json"
    module.write_model_trust_manifest(model_path, trust_path)
    monkeypatch.setenv("AURA_CODE_MODEL_TRUST_MANIFEST", str(trust_path))

    identity = module._validate_model_trust(str(model_path))

    assert identity.checkpoint_files == 1
    assert len(identity.checkpoint_fingerprint) == 64
    assert identity.architecture == "Qwen2ForCausalLM"
    assert identity.max_context_tokens == 256
    assert trust_path.stat().st_mode & 0o077 == 0

    (model_path / "model.safetensors").write_bytes(b"tampered-test-weights")
    with pytest.raises(module.LocalCodeModelError, match="checkpoint_identity_mismatch"):
        module._validate_model_trust(str(model_path))


def test_manifest_must_be_external_owned_and_not_group_writable(monkeypatch, tmp_path):
    from core.brain.llm import local_code_model as module

    model_path = _model_tree(tmp_path / "model")
    with pytest.raises(ValueError, match="must_be_external"):
        module.write_model_trust_manifest(model_path, model_path / "trust.json")

    trust_path = module.write_model_trust_manifest(model_path, tmp_path / "trust.json")
    trust_path.chmod(0o620)
    monkeypatch.setenv("AURA_CODE_MODEL_TRUST_MANIFEST", str(trust_path))
    with pytest.raises(module.LocalCodeModelError, match="permissions_invalid"):
        module._validate_model_trust(str(model_path))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_tokens": 0}, "max_tokens_out_of_policy"),
        ({"max_tokens": -1}, "max_tokens_out_of_policy"),
        ({"temperature": float("nan")}, "temperature_out_of_policy"),
        ({"temperature": float("inf")}, "temperature_out_of_policy"),
        ({"temperature": 2.1}, "temperature_out_of_policy"),
        ({"timeout_s": 0}, "timeout_out_of_policy"),
    ],
)
async def test_invalid_request_budget_fails_before_model_validation(
    monkeypatch, tmp_path, kwargs, message
):
    from core.brain.llm import local_code_model as module

    called = False

    def _unexpected(_path):
        nonlocal called
        called = True
        raise AssertionError("trust validation should not run")

    monkeypatch.setattr(module, "_validate_model_trust", _unexpected)
    model = module.LocalCodeModel(str(tmp_path / "model"))
    with pytest.raises(ValueError, match=message):
        await model.think("code", **kwargs)
    assert called is False


@pytest.mark.asyncio
async def test_byte_limit_fails_before_loading(monkeypatch, tmp_path):
    from core.brain.llm import local_code_model as module

    monkeypatch.setattr(module, "_MAX_PROMPT_BYTES", 8)
    monkeypatch.setattr(
        module,
        "_validate_model_trust",
        lambda _path: pytest.fail("validation should not run"),
    )
    with pytest.raises(ValueError, match="prompt_bytes_exceeded"):
        await module.LocalCodeModel(str(tmp_path / "model")).think("123456789")


@pytest.mark.asyncio
async def test_exact_context_admission_rejects_before_generation(monkeypatch, tmp_path):
    from core.brain.llm import local_code_model as module
    from core.runtime import model_lane_control

    model_path = tmp_path / "model"
    model_path.mkdir()
    generated = False

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "one two three four five six"

        def encode(self, text):
            return list(range(len(text.split())))

    class Lease:
        decision = SimpleNamespace(receipt_id="lane-receipt", fencing_token="fence")

        async def set_preemptible(self, value):
            return value

        async def release(self, *, reason):
            return bool(reason)

    async def acquire(**_kwargs):
        return Lease()

    def generate(*_args, **_kwargs):
        nonlocal generated
        generated = True
        return "never"

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = lambda *_args, **_kwargs: (object(), Tokenizer())
    mlx_lm.generate = generate
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", acquire)
    monkeypatch.setattr(module, "_validate_model_trust", lambda _path: _identity(module, model_path, context=8))
    monkeypatch.setattr(module, "_model", None)
    monkeypatch.setattr(module, "_tokenizer", None)
    monkeypatch.setattr(module, "_loaded_path", None)
    monkeypatch.setattr(module, "_loaded_identity", None)
    monkeypatch.setattr(module, "_lane_lease", None)

    model = module.LocalCodeModel(str(model_path))
    with pytest.raises(module.LocalCodeModelError, match="context_budget_exceeded"):
        await model.think("code", max_tokens=3)
    assert generated is False
    await model.close()


@pytest.mark.asyncio
async def test_caller_token_cap_is_honored_and_truncation_fails_honestly(monkeypatch, tmp_path):
    from core.brain.llm import local_code_model as module
    from core.runtime import model_lane_control

    model_path = tmp_path / "model"
    model_path.mkdir()
    observed_caps = []

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def encode(self, text):
            return list(range(len(text.split())))

    class Lease:
        decision = SimpleNamespace(receipt_id="lane-receipt", fencing_token="fence")

        async def set_preemptible(self, value):
            return value

        async def release(self, *, reason):
            return bool(reason)

    async def acquire(**_kwargs):
        return Lease()

    def generate(*_args, **kwargs):
        observed_caps.append(kwargs["max_tokens"])
        return "x"

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = lambda *_args, **_kwargs: (object(), Tokenizer())
    mlx_lm.generate = generate
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", acquire)
    monkeypatch.setattr(module, "_validate_model_trust", lambda _path: _identity(module, model_path))
    monkeypatch.setattr(module, "_model", None)
    monkeypatch.setattr(module, "_tokenizer", None)
    monkeypatch.setattr(module, "_loaded_path", None)
    monkeypatch.setattr(module, "_loaded_identity", None)
    monkeypatch.setattr(module, "_lane_lease", None)

    model = module.LocalCodeModel(str(model_path))
    with pytest.raises(module.LocalCodeModelError, match="output_incomplete"):
        await model.think("code", max_tokens=1)
    assert observed_caps == [1]
    await model.close()


@pytest.mark.asyncio
async def test_owner_aware_unload_cannot_clear_another_model(monkeypatch):
    from core.brain.llm import local_code_model as module

    sentinel = object()
    monkeypatch.setattr(module, "_model", sentinel)
    monkeypatch.setattr(module, "_tokenizer", object())
    monkeypatch.setattr(module, "_loaded_path", "/trusted/other")
    monkeypatch.setattr(module, "_lane_lease", object())

    receipt = await module.unload_local_code_model(expected_path="/trusted/requester")

    assert receipt.requested_path_matches is False
    assert receipt.references_cleared is False
    assert module._model is sentinel


@pytest.mark.asyncio
async def test_fair_lifecycle_gate_is_fifo_and_deadline_bounded():
    from core.brain.llm.local_code_model import _FairAsyncGate

    gate = _FairAsyncGate()
    first_token, _ = await gate.acquire(deadline=asyncio.get_running_loop().time() + 1.0)
    order = []

    async def waiter(label):
        token, _position = await gate.acquire(
            deadline=asyncio.get_running_loop().time() + 1.0
        )
        order.append(label)
        gate.release(token)

    second = asyncio.create_task(waiter("second"))
    third = asyncio.create_task(waiter("third"))
    await asyncio.sleep(0.02)
    gate.release(first_token)
    await asyncio.gather(second, third)
    assert order == ["second", "third"]

    held, _ = await gate.acquire(deadline=asyncio.get_running_loop().time() + 1.0)
    with pytest.raises(TimeoutError, match="lifecycle_gate"):
        await gate.acquire(deadline=asyncio.get_running_loop().time() + 0.02)
    gate.release(held)


def test_readiness_is_not_filesystem_existence(tmp_path):
    from core.brain.llm import local_code_model as module

    path = tmp_path / "model"
    path.mkdir()
    model = module.LocalCodeModel(str(path))

    assert model.is_available() is False
    assert model.readiness().state is module.ReadinessState.UNVERIFIED


@pytest.mark.asyncio
async def test_bounded_probe_is_what_promotes_readiness(monkeypatch, tmp_path):
    from core.brain.llm import local_code_model as module

    model_path = tmp_path / "model"
    model_path.mkdir()
    identity = _identity(module, model_path)

    async def loaded(_path, observed, *, deadline):
        assert observed == identity
        assert deadline > asyncio.get_running_loop().time()

    monkeypatch.setattr(module, "_validate_model_trust", lambda _path: identity)
    monkeypatch.setattr(module, "_ensure_loaded_with_lane", loaded)
    model = module.LocalCodeModel(str(model_path))

    receipt = await model.probe_readiness(timeout_s=1.0)

    assert receipt.state is module.ReadinessState.READY
    assert receipt.model_id == identity.privacy_safe_id
    assert model.is_available() is True


def test_cleanup_receipt_measures_synchronization_cache_and_memory(monkeypatch):
    from core.brain.llm import local_code_model as module

    calls = []
    readings = iter((100, 0))
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_core.synchronize = lambda: calls.append("synchronize")
    mlx_core.clear_cache = lambda: calls.append("clear_cache")
    mlx_core.get_active_memory = lambda: next(readings)
    mlx.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setattr(module, "_model", object())
    monkeypatch.setattr(module, "_tokenizer", object())
    monkeypatch.setattr(module, "_loaded_path", "/model")
    monkeypatch.setattr(module, "_loaded_identity", object())

    receipt = module._clear_loaded_model()

    assert receipt.references_cleared is True
    assert receipt.backend_synchronized is True
    assert receipt.cache_cleared is True
    assert receipt.active_memory_before == 100
    assert receipt.active_memory_after == 0
    assert receipt.verified is True
    assert calls == ["synchronize", "clear_cache", "synchronize"]


@pytest.mark.asyncio
async def test_failed_load_cleans_before_releasing_lane(monkeypatch, tmp_path):
    from core.brain.llm import local_code_model as module
    from core.runtime import model_lane_control

    model_path = tmp_path / "model"
    model_path.mkdir()
    events = []

    class Lease:
        async def release(self, *, reason):
            events.append(("release", reason, module._model))
            return True

        async def set_preemptible(self, value):
            return value

    async def acquire(**_kwargs):
        return Lease()

    mlx_lm = ModuleType("mlx_lm")

    def fail_load(*_args, **_kwargs):
        events.append(("load", "failed"))
        raise RuntimeError("partial allocation")

    mlx_lm.load = fail_load
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", acquire)
    monkeypatch.setattr(module, "_validate_model_trust", lambda _path: _identity(module, model_path))
    monkeypatch.setattr(module, "_model", None)
    monkeypatch.setattr(module, "_tokenizer", None)
    monkeypatch.setattr(module, "_loaded_path", None)
    monkeypatch.setattr(module, "_loaded_identity", None)
    monkeypatch.setattr(module, "_lane_lease", None)

    with pytest.raises(RuntimeError, match="partial allocation"):
        await module.LocalCodeModel(str(model_path)).think("code")

    assert events[0] == ("load", "failed")
    assert events[1][0:2] == ("release", "local_code_model_load_rolled_back")
    assert events[1][2] is None


@pytest.mark.asyncio
async def test_different_model_paths_are_serialized_through_generation(monkeypatch, tmp_path):
    from core.brain.llm import local_code_model as module
    from core.runtime import model_lane_control

    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    first_started = threading.Event()
    release_first = threading.Event()
    validations = []

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def encode(self, text):
            return list(range(max(1, len(text.split()))))

    class Lease:
        decision = SimpleNamespace(receipt_id="lane", fencing_token="fence")

        async def set_preemptible(self, value):
            return value

        async def release(self, *, reason):
            return bool(reason)

    async def acquire(**_kwargs):
        return Lease()

    def validate(path):
        validations.append(path)
        return _identity(module, path)

    def generate(model, *_args, **_kwargs):
        if model == str(first_path):
            first_started.set()
            assert release_first.wait(timeout=2.0)
        return "complete code"

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = lambda path: (path, Tokenizer())
    mlx_lm.generate = generate
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", acquire)
    monkeypatch.setattr(module, "_validate_model_trust", validate)
    monkeypatch.setattr(module, "_model", None)
    monkeypatch.setattr(module, "_tokenizer", None)
    monkeypatch.setattr(module, "_loaded_path", None)
    monkeypatch.setattr(module, "_loaded_identity", None)
    monkeypatch.setattr(module, "_lane_lease", None)

    first = asyncio.create_task(module.LocalCodeModel(str(first_path)).think("first"))
    assert await asyncio.to_thread(first_started.wait, 1.0)
    second_model = module.LocalCodeModel(str(second_path))
    second = asyncio.create_task(second_model.think("second"))
    await asyncio.sleep(0.05)
    assert validations == [str(first_path)]

    release_first.set()
    assert await first == "complete code"
    assert await second == "complete code"
    assert validations == [str(first_path), str(second_path)]
    await second_model.close()


def test_her_code_lane_writes_with_her_own_cortex_unless_one_is_named(monkeypatch, tmp_path):
    from core.brain.llm import local_code_model, model_registry

    monkeypatch.delenv("AURA_CODE_MODEL_PATH", raising=False)
    monkeypatch.delenv("AURA_MODEL_PATH", raising=False)
    monkeypatch.setattr(model_registry, "_current_cortex_path", lambda: tmp_path / "her-27b")
    assert local_code_model._resolve_model_path() == str(tmp_path / "her-27b")
    monkeypatch.setenv("AURA_CODE_MODEL_PATH", str(tmp_path / "named"))
    assert local_code_model._resolve_model_path() == str(tmp_path / "named")
