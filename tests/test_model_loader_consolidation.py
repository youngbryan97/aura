from __future__ import annotations

import asyncio
import contextlib
import json
import runpy
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from core.runtime import model_lane_control


class _SyncLease:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def release(self, *, reason: str) -> bool:
        self.events.append(f"release:{reason}")
        return True


class _AsyncLease:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def release(self, *, reason: str) -> bool:
        self.events.append(f"release:{reason}")
        return True


class _Tokenizer:
    def apply_chat_template(self, messages: Any, **_kwargs: Any) -> str:
        return str(messages[-1]["content"])


def _install_fake_mlx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load: Any,
    generate: Any | None = None,
) -> None:
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = load  # type: ignore[attr-defined]
    mlx_lm.generate = generate or (lambda *_args, **kwargs: kwargs.get("prompt", ""))  # type: ignore[attr-defined]
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    mlx_package = ModuleType("mlx")
    mlx_package.__path__ = []  # type: ignore[attr-defined]
    mlx_core = ModuleType("mlx.core")
    mlx_core.clear_cache = lambda: None  # type: ignore[attr-defined]
    mlx_package.core = mlx_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setitem(sys.modules, "mlx", mlx_package)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)


def _patch_standalone_context(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> dict[str, bool]:
    state = {"active": False}

    @contextlib.contextmanager
    def _lane(**_kwargs: Any):
        events.append("acquire")
        state["active"] = True
        try:
            yield _SyncLease(events)
        finally:
            state["active"] = False
            events.append("release:context")

    monkeypatch.setattr(model_lane_control, "standalone_model_lane", _lane)
    return state


def test_rlc_fusion_holds_lane_and_releases_when_model_load_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from core.runtime import mlx_memory_guard
    from tools import fuse_rlc_candidate

    base = tmp_path / "base"
    adapter = tmp_path / "run" / "generations" / "sequence-0001"
    out = tmp_path / "candidate"
    base.mkdir()
    adapter.mkdir(parents=True)
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter.safetensors").write_bytes(b"adapter")
    (adapter.parent.parent / "recurrence_adapter_manifest.json").write_text(
        json.dumps({"lora": {"rank": 2, "alpha": 2}}),
        encoding="utf-8",
    )

    events: list[str] = []
    state = _patch_standalone_context(monkeypatch, events)
    monkeypatch.setattr(
        mlx_memory_guard,
        "mlx_memory_envelope",
        lambda **_kwargs: contextlib.nullcontext(),
    )

    def _load(_path: str) -> tuple[object, object]:
        assert state["active"] is True
        events.append("load")
        raise RuntimeError("fusion model load failed")

    _install_fake_mlx(monkeypatch, load=_load)
    mlx_utils = ModuleType("mlx.utils")
    mlx_utils.tree_flatten = lambda _value: []  # type: ignore[attr-defined]
    mlx_utils.tree_unflatten = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx.utils", mlx_utils)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fuse_rlc_candidate.py",
            "--model",
            str(base),
            "--adapter",
            str(adapter),
            "--out",
            str(out),
        ],
    )

    with pytest.raises(RuntimeError, match="fusion model load failed"):
        fuse_rlc_candidate.main()

    assert events == ["acquire", "load", "release:context"]


@pytest.mark.asyncio
async def test_reasoning_delta_owns_and_serializes_native_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura_bench import reasoning_delta

    events: list[str] = []
    lease = _SyncLease(events)
    monkeypatch.setattr(
        model_lane_control,
        "acquire_standalone_model_lane",
        lambda **_kwargs: events.append("acquire") or lease,
    )
    active = 0
    maximum_active = 0
    native_lock = threading.Lock()

    def _load(_path: str) -> tuple[object, _Tokenizer]:
        events.append("load")
        return object(), _Tokenizer()

    def _generate(*_args: Any, **kwargs: Any) -> str:
        nonlocal active, maximum_active
        with native_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with native_lock:
            active -= 1
        return str(kwargs["prompt"])

    _install_fake_mlx(monkeypatch, load=_load, generate=_generate)
    generator = reasoning_delta.make_mlx_lm_generator("/models/cortex")

    assert events[:2] == ["acquire", "load"]
    assert await asyncio.gather(generator("one", 0.2), generator("two", 0.3)) == [
        "one",
        "two",
    ]
    assert maximum_active == 1

    generator.close()
    assert events[-1] == "release:reasoning_delta_generator_closed"


def test_reasoning_delta_releases_lane_when_model_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura_bench import reasoning_delta

    events: list[str] = []
    monkeypatch.setattr(
        model_lane_control,
        "acquire_standalone_model_lane",
        lambda **_kwargs: events.append("acquire") or _SyncLease(events),
    )

    def _load(_path: str) -> None:
        events.append("load")
        raise RuntimeError("load failed")

    _install_fake_mlx(monkeypatch, load=_load)
    with pytest.raises(RuntimeError, match="load failed"):
        reasoning_delta.make_mlx_lm_generator("/models/cortex")

    assert events == [
        "acquire",
        "load",
        "release:reasoning_delta_model_load_failed",
    ]


@pytest.mark.asyncio
async def test_reasoning_benchmark_releases_lane_when_model_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.reasoning import run

    events: list[str] = []

    async def _acquire(**_kwargs: Any) -> _AsyncLease:
        events.append("acquire")
        return _AsyncLease(events)

    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", _acquire)

    def _load(_path: str) -> None:
        events.append("load")
        raise RuntimeError("benchmark load failed")

    _install_fake_mlx(monkeypatch, load=_load)
    with pytest.raises(RuntimeError, match="benchmark load failed"):
        await run._mlx_generator("/models/benchmark")

    assert events == [
        "acquire",
        "load",
        "release:reasoning_benchmark_model_load_failed",
    ]


def test_steering_generator_releases_owned_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.generate_steering_vectors import SteeringVectorGenerator

    events: list[str] = []
    monkeypatch.setattr(
        model_lane_control,
        "acquire_standalone_model_lane",
        lambda **_kwargs: events.append("acquire") or _SyncLease(events),
    )

    def _load(_path: str) -> tuple[object, object]:
        events.append("load")
        return object(), object()

    _install_fake_mlx(monkeypatch, load=_load)
    generator = SteeringVectorGenerator(
        output_dir=tmp_path,
        model_path="/models/steering",
    )
    assert events == ["acquire", "load"]

    generator.close()
    assert events[-1] == "release:caa_steering_generator_closed"


def test_nonparametric_probe_holds_lane_for_entire_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura_bench import nonparametric_probe

    events: list[str] = []
    state = _patch_standalone_context(monkeypatch, events)

    def _run_probe(model_path: str) -> int:
        assert state["active"] is True
        events.append(f"probe:{model_path}")
        return 7

    monkeypatch.setattr(nonparametric_probe, "_run_probe", _run_probe)
    monkeypatch.setattr(sys, "argv", ["probe", "/models/nonparametric"])

    assert nonparametric_probe.main() == 7
    assert events == [
        "acquire",
        "probe:/models/nonparametric",
        "release:context",
    ]


def test_gpt2_finetune_holds_lane_for_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import run_finetune_gpt2

    events: list[str] = []
    state = _patch_standalone_context(monkeypatch, events)
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"user": "hello", "assistant": "hi"}\n', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "datasets", ModuleType("datasets"))
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setattr(
        run_finetune_gpt2,
        "_run_training",
        lambda _args, texts: (
            events.append(f"train:{len(texts)}")
            if state["active"]
            else pytest.fail("training ran outside model lane")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_finetune_gpt2", "--train-file", str(train_file)],
    )

    run_finetune_gpt2.main()
    assert events == ["acquire", "train:1", "release:context"]


def test_cuda_self_update_holds_lane_for_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import self_update

    events: list[str] = []
    state = _patch_standalone_context(monkeypatch, events)
    training_file = tmp_path / "training.jsonl"
    training_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(self_update, "TRAINING_DATA_FILE", training_file)

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    datasets = ModuleType("datasets")
    datasets.load_dataset = object()  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.TrainingArguments = object()  # type: ignore[attr-defined]
    trl = ModuleType("trl")
    trl.SFTTrainer = object()  # type: ignore[attr-defined]
    unsloth = ModuleType("unsloth")
    unsloth.FastLanguageModel = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)

    def _run_training(**_kwargs: Any) -> None:
        assert state["active"] is True
        events.append("train")

    monkeypatch.setattr(self_update, "_run_training", _run_training)
    self_update.train_self()
    assert events == ["acquire", "train", "release:context"]


def test_caa_extraction_holds_lane_for_entire_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from training import extract_steering_vectors

    events: list[str] = []
    state = _patch_standalone_context(monkeypatch, events)

    def _extract(**kwargs: Any) -> dict[str, dict[int, Any]]:
        assert state["active"] is True
        events.append(f"extract:{kwargs['model_path']}")
        return {"warmth": {1: object()}}

    monkeypatch.setattr(
        extract_steering_vectors,
        "_extract_steering_vectors_owned",
        _extract,
    )
    result = extract_steering_vectors.extract_steering_vectors(
        model_path="/models/caa",
        model_descriptor_path=tmp_path / "descriptor.json",
        output_dir=tmp_path,
    )

    assert "warmth" in result
    assert events == ["acquire", "extract:/models/caa", "release:context"]


def test_legacy_mlx_client_delegates_to_canonical_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.brain.llm import mlx_client as canonical
    from llm.mlx_client import MLXClient

    calls: list[dict[str, Any]] = []

    class _CanonicalClient:
        async def generate_text_async(self, prompt: str, **kwargs: Any) -> str:
            calls.append({"prompt": prompt, **kwargs})
            return "<think>private</think> visible answer"

    canonical_client = _CanonicalClient()
    monkeypatch.setattr(
        canonical,
        "get_mlx_client",
        lambda model_path, **_kwargs: (
            calls.append({"model_path": model_path}) or canonical_client
        ),
    )

    client = MLXClient("/models/canonical")
    result = client.call("hello", system_prompt="system", max_tokens=17)

    assert result == {"ok": True, "text": "visible answer", "thought": "private"}
    assert calls[0] == {"model_path": "/models/canonical"}
    assert calls[1]["owner_label"] == "legacy_mlx_client"


def test_stable_whisper_wrapper_uses_canonical_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.senses import voice_socket_logic
    from core.voice.stable_voice_pipeline import _WhisperWrapper

    calls: list[tuple[Any, dict[str, Any]]] = []

    class _Proxy:
        def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[Any], object]:
            calls.append((audio, kwargs))
            return [SimpleNamespace(text=" governed"), SimpleNamespace(text=" speech ")], object()

    monkeypatch.setattr(voice_socket_logic, "get_whisper_model", lambda _size: _Proxy())
    wrapper = _WhisperWrapper(model_size="small", language="en")

    assert wrapper.transcribe("sample.wav") == "governed  speech"
    assert calls == [("sample.wav", {"language": "en", "beam_size": 5})]


@pytest.mark.asyncio
async def test_local_media_delegates_and_stops_canonical_skill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from core.skills import image_gen
    from skills._local_media_generation import LocalMediaGenerationSkill

    events: list[str] = []
    generated_path = tmp_path / "generated.png"

    class _CanonicalImageSkill:
        async def execute(self, goal: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
            events.append(f"execute:{goal['prompt']}:{context['request_id']}")
            return {
                "ok": True,
                "path": str(generated_path),
                "url": "/data/generated.png",
                "mode": "diffusion",
            }

        async def on_stop_async(self) -> None:
            events.append("stop")

    monkeypatch.setattr(image_gen, "ImageGenSkill", _CanonicalImageSkill)
    skill = LocalMediaGenerationSkill()
    result = await skill.execute(
        {"objective": "a precise diagram"},
        {"request_id": "request-1"},
    )
    await skill.on_stop_async()

    assert result["ok"] is True
    assert result["generation_mode"] == "diffusion"
    assert result["degraded"] is False
    assert events == ["execute:a precise diagram:request-1", "stop"]


def test_training_dispatches_model_jobs_to_governed_blocking_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training import train_and_fuse

    events: list[str] = []

    class _Gateway:
        def run(self, command: list[str], **_kwargs: Any) -> Any:
            events.append(f"plain:{command[-1]}")
            return SimpleNamespace(returncode=0)

        def run_model_blocking(self, command: list[str], **_kwargs: Any) -> Any:
            events.append(f"model:{command[-1]}")
            return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_and_fuse, "get_subprocess_gateway", _Gateway)

    assert train_and_fuse._run(["python", "builder.py"]) == 0
    assert train_and_fuse._run(
        ["python", "-m", "mlx_lm", "fuse", "--model", "/models/base"],
        model_job=True,
    ) == 0
    assert events == ["plain:builder.py", "model:/models/base"]


def test_fused_model_verification_delegates_exact_child_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from training import train_and_fuse

    captured: dict[str, Any] = {}

    class _Gateway:
        def run_model_blocking(
            self,
            command: list[str],
            **kwargs: Any,
        ) -> Any:
            captured["command"] = command
            captured.update(kwargs)
            return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_and_fuse, "get_subprocess_gateway", _Gateway)
    fused_path = tmp_path / "fused-model"
    train_and_fuse.verify_load(fused_path)

    command = captured["command"]
    claim = captured["model_lane_claim"]
    assert command[:2] == [sys.executable, "-c"]
    assert "standalone_model_lane" in command[2]
    assert command[3] == str(fused_path)
    assert claim.model_path == str(fused_path)
    assert claim.purpose == "benchmark"


def test_recurrent_depth_training_holds_entire_lora_main_under_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlx_package = ModuleType("mlx_lm")
    mlx_package.__path__ = []  # type: ignore[attr-defined]
    mlx_utils = ModuleType("mlx_lm.utils")
    mlx_utils.load = lambda *_args, **_kwargs: (object(), object())  # type: ignore[attr-defined]
    mlx_lora = ModuleType("mlx_lm.lora")
    mlx_lora.load = mlx_utils.load  # type: ignore[attr-defined]
    mlx_lora.main = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_package)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", mlx_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm.lora", mlx_lora)

    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "train_with_recurrent_depth.py"),
        run_name="aura_recurrent_depth_contract",
    )
    assert namespace["ROOT"] == Path(__file__).resolve().parents[1]
    main = namespace["main"]
    events: list[str] = []

    @contextlib.contextmanager
    def _lane(**kwargs: Any):
        events.append(f"acquire:{kwargs['model_path']}:{kwargs['purpose']}")
        try:
            yield
        finally:
            events.append("release")

    main.__globals__["standalone_model_lane"] = _lane
    main.__globals__["_lora"].main = lambda: events.append("train")
    main(["--model", "/models/recurrent", "--train"])

    assert events == ["acquire:/models/recurrent:train", "train", "release"]


def test_qwen_migration_fuse_uses_managed_model_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import migrate_to_qwen3

    captured: dict[str, Any] = {}
    output = tmp_path / "fused"

    class _Gateway:
        def run_model_blocking(self, command: list[str], **kwargs: Any) -> Any:
            captured["command"] = command
            captured.update(kwargs)
            output.mkdir()
            return SimpleNamespace(returncode=0)

    monkeypatch.setattr(migrate_to_qwen3, "get_subprocess_gateway", _Gateway)
    assert migrate_to_qwen3.run_fuse(
        tmp_path / "base",
        tmp_path / "adapter",
        output,
        execute=True,
    ) is True
    assert captured["command"][2:4] == ["mlx_lm", "fuse"]
    assert captured["source"] == "training_tooling:migrate_qwen3_fuse"
