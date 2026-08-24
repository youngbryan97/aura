"""Batched best-of-N candidate lane — resolution, fallback, and wiring.

The lane only ever makes best-of-N cheaper: every miss (disabled, no live
client, worker error, timeout) returns None/[] and the amplifier keeps its
serial sampling path.
"""
from __future__ import annotations

import asyncio

import core.brain.llm.batch_candidates as bc
from core.brain.generation_provenance import attributed_text, generation_metadata_of
from core.runtime.model_runtime_assignment import ModelRuntimeAssignment, locator_identity


class _Client:
    def __init__(
        self,
        model_path: str,
        alive: bool = True,
        texts=None,
        *,
        role: str = "cortex",
    ):
        self.model_path = model_path
        self.runtime_assignment = ModelRuntimeAssignment.issue(
            model_path=model_path,
            artifact_identity=locator_identity(model_path),
            artifact_identity_kind="canonical_locator_sha256",
            artifact_identity_exact=False,
            role=role,
            purpose="serve",
            authority_source="unit_test_registry",
        )
        self._alive = alive
        self.calls: list[dict] = []
        self._texts = texts if texts is not None else ["a", "b", "c"]

    def is_alive(self):
        return self._alive

    async def generate_batch_async(self, prompt, *, n, max_tokens, temperature, timeout_s):
        self.calls.append({"prompt": prompt, "n": n})
        return self._texts


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "0")
    assert asyncio.run(bc.generate_candidates_batched("p", 4)) is None


def test_single_sample_never_batches(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "1")
    assert asyncio.run(bc.generate_candidates_batched("p", 1)) is None


def test_prefers_registry_assigned_cortex_client(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "1")
    light = _Client("/models/Qwen2.5-1.5B-Instruct-4bit", role="reflex")
    heavy = _Client("/models/renamed-resident", role="cortex")
    monkeypatch.setattr(
        "core.brain.llm.mlx_client._CLIENTS", {"l": light, "h": heavy}
    )
    out = asyncio.run(bc.generate_candidates_batched("p", 4))
    assert out == ["a", "b", "c"]
    assert heavy.calls and not light.calls


def test_large_or_solver_named_client_cannot_impersonate_cortex(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "1")
    impostor = _Client("/models/999B-Cortex-Solver", role="auxiliary")
    monkeypatch.setattr("core.brain.llm.mlx_client._CLIENTS", {"x": impostor})

    assert asyncio.run(bc.generate_candidates_batched("p", 4)) is None
    assert not impostor.calls


def test_dead_clients_yield_serial_fallback(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "1")
    dead = _Client("/models/Qwen2.5-32B-Instruct-4bit", alive=False)
    monkeypatch.setattr("core.brain.llm.mlx_client._CLIENTS", {"d": dead})
    assert asyncio.run(bc.generate_candidates_batched("p", 4)) is None


def test_empty_batch_yields_serial_fallback(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "1")
    empty = _Client("/models/Qwen2.5-32B-Instruct-4bit", texts=[])
    monkeypatch.setattr("core.brain.llm.mlx_client._CLIENTS", {"e": empty})
    assert asyncio.run(bc.generate_candidates_batched("p", 4)) is None


def test_amplifier_uses_batched_lane_before_serial(monkeypatch):
    """_generate_candidates returns the batched pool when the lane delivers."""
    import time as _time

    from core.brain.reasoning_amplifier_v2 import (
        ProblemRepresentation,
        build_amplifier_v2,
    )

    serial_calls: list[int] = []

    async def fake_generate(prompt, temperature):
        serial_calls.append(1)
        return "serial"

    amp = build_amplifier_v2(fake_generate)

    async def fake_batched(prompt, n, *, max_tokens, timeout_s):
        assert max_tokens == 512
        return [f"candidate-{i}" for i in range(n)]

    monkeypatch.setattr(
        "core.brain.llm.batch_candidates.generate_candidates_batched", fake_batched
    )

    problem = ProblemRepresentation(objective="compute 2+2", task_type="math")
    out = asyncio.run(
        amp._generate_candidates(problem, "", 4, _time.monotonic() + 30.0)
    )

    assert out == ["candidate-0", "candidate-1", "candidate-2", "candidate-3"]
    assert not serial_calls, "batched hit must not invoke serial sampling"


def test_structured_batch_metadata_stays_attached_to_each_candidate(monkeypatch):
    class _StructuredClient(_Client):
        async def generate_batch_with_metadata_async(self, prompt, **kwargs):
            self.calls.append({"prompt": prompt, **kwargs})
            return {
                "texts": ["winner", "alternate"],
                "generation_metadata": {
                    "endpoint": "MLX-BATCH:test",
                    "batch_request_id": "batch-1",
                },
                "candidate_generation_metadata": [
                    {"generated_tokens": 3},
                    {"generated_tokens": 7},
                ],
            }

    client = _StructuredClient("/models/Qwen2.5-32B-Instruct-4bit")
    monkeypatch.setattr("core.brain.llm.mlx_client._CLIENTS", {"h": client})

    out = asyncio.run(bc.generate_candidates_batched("p", 2))

    assert out == ["winner", "alternate"]
    assert generation_metadata_of(out[0]) == {
        "endpoint": "MLX-BATCH:test",
        "batch_request_id": "batch-1",
        "generated_tokens": 3,
        "batch_candidate_index": 0,
    }
    assert generation_metadata_of(out[1])["batch_candidate_index"] == 1
    assert generation_metadata_of(out[1])["generated_tokens"] == 7


def test_worker_source_contract():
    import inspect

    import core.brain.llm.mlx_worker as worker

    src = inspect.getsource(worker)
    assert 'elif action == "generate_batch":' in src
    assert "batch_generate(" in src
    # Raw candidates by design: no sentinel or quality gates in the batch branch.
    assert "RAW (no sentinel/quality gates)" in src
    assert '"tokens_used_by_candidate": tokens_used_by_candidate' in src
    assert "batch_max_tokens = max(\n                            1," in src
    assert "batch_max_tokens = max(16" not in src


def test_attributed_text_preserves_zero_value_and_copies_metadata():
    metadata = {"candidate": 0, "receipt": {"applied": True}}
    value = attributed_text(0, metadata)
    metadata["candidate"] = 1
    metadata["receipt"]["applied"] = False

    assert value == "0"
    assert generation_metadata_of(value) == {
        "candidate": 0,
        "receipt": {"applied": True},
    }
