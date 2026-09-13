"""The bytes a growth diff could not name.

LIVE, 2026-09-10: 4,213MB of RSS in the serving process, 0.6% attributed. The
process holds Qwen3-Embedding-0.6B in its own memory and no provider claimed
it, so `get_memory_infra().diff(a, b).narrative()` could not name the one
component holding most of what there was to hold. The providers here read
exact counts — a parameter's numel times its element size, the allocators'
own accounting — rather than estimates.
"""

from __future__ import annotations

import pytest
import torch

from core.container import ServiceContainer
from core.runtime.memory_infra import (
    get_memory_infra,
    install_runtime_providers,
    reset_memory_infra_for_test,
)


@pytest.fixture(autouse=True)
def _fresh_infra():
    reset_memory_infra_for_test()
    install_runtime_providers()
    yield
    reset_memory_infra_for_test()


class _Engine:
    PREFERRED_MODEL = "test/encoder"

    def __init__(self, model) -> None:
        self._model = model


def test_the_encoder_is_attributed_by_its_parameters_exactly(monkeypatch) -> None:
    model = torch.nn.Linear(256, 128)
    monkeypatch.setattr(
        ServiceContainer, "get", classmethod(lambda cls, name, default=None: _Engine(model) if name == "vector_memory_engine" else default)
    )
    dump = get_memory_infra().dump()
    claimed = dump.dumps["memory.embedding_model"]
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    assert claimed.size_bytes == expected == 256 * 128 * 4 + 128 * 4
    assert claimed.object_count == 2
    assert claimed.detail["model"] == "test/encoder"


def test_no_encoder_is_zero_and_says_so(monkeypatch) -> None:
    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: default))
    claimed = get_memory_infra().dump().dumps["memory.embedding_model"]
    assert claimed.size_bytes == 0
    assert claimed.detail == {"loaded": False}


def test_the_device_allocators_report_their_own_numbers() -> None:
    dump = get_memory_infra().dump()
    assert "torch.device_memory" in dump.dumps
    assert "mlx.device_memory" in dump.dumps
    for name in ("torch.device_memory", "mlx.device_memory"):
        assert dump.dumps[name].size_bytes >= 0
        assert "estimated" not in dump.dumps[name].detail
