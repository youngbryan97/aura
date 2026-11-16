"""Halting-head artifacts are governed and an encoder failure preserves the last save."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from core.learning.adaptive_halting import HaltingHead  # noqa: E402


def _head():
    head = HaltingHead(4, threshold=0.7)
    head.weight = mx.array([[0.01], [0.02], [0.03], [0.04]])
    head.bias = mx.array([0.25])
    return head


def test_the_halting_artifact_has_an_internal_governed_writer(tmp_path, monkeypatch):
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    head = _head()
    gateway = get_file_write_gateway()
    write_bytes = gateway.write_bytes
    writes = []
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")

    def write(path, payload, **kwargs):
        token = get_active_governance()
        assert token is not None and token.domain == "state_mutation"
        writes.append((path, kwargs["source"]))
        write_bytes(path, payload, **kwargs)

    monkeypatch.setattr(gateway, "write_bytes", write)
    path = tmp_path / "new" / "halting.npz"
    assert head.save(path) is None
    assert writes == [(path, "adaptive_halting.save")]
    assert path.stat().st_mode & 0o777 == 0o600
    loaded = HaltingHead.load(path)
    assert loaded.hidden_size == 4 and loaded.threshold == pytest.approx(0.7)
    np.testing.assert_array_equal(np.asarray(loaded.weight), np.asarray(head.weight))
    np.testing.assert_array_equal(np.asarray(loaded.bias), np.asarray(head.bias))


def test_an_encoder_failure_keeps_the_previous_halting_head(tmp_path, monkeypatch):
    head = _head()
    path = tmp_path / "halting.npz"
    head.save(path)
    previous = path.read_bytes()

    def broken_encoder(destination, **arrays):
        if hasattr(destination, "write"):
            destination.write(b"incomplete archive")
        else:
            Path(destination).write_bytes(b"incomplete archive")
        raise OSError("encoder could not finish")

    monkeypatch.setattr(np, "savez", broken_encoder)
    with pytest.raises(OSError, match="encoder could not finish"):
        head.save(path)
    assert path.read_bytes() == previous
    assert HaltingHead.load(path).hidden_size == 4


def test_a_halting_path_without_an_npz_suffix_keeps_the_numpy_filename(tmp_path):
    path = tmp_path / "halting.head"
    assert _head().save(str(path)) is None
    assert not path.exists()
    assert HaltingHead.load(tmp_path / "halting.head.npz").hidden_size == 4
