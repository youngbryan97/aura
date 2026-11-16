"""A turn-reader save uses its governed writer and preserves a good head on failure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.interiority.what_a_turn_did import TurnReader


def _reader():
    return TurnReader(
        np.array([[1, 2, 3], [-1, -2, -3]], dtype=np.float32),
        np.array([0.5, -0.5], dtype=np.float32), ("joy", "sadness"),
        embedding_model="fixture", report={"held_out_examples": 2},
    )


def test_the_reader_artifact_has_an_internal_governed_writer(tmp_path, monkeypatch):
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    reader = _reader()
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
    path = tmp_path / "new" / "reader.npz"
    assert reader.save(path) == path
    assert writes == [(path, "turn_reader.save")]
    assert path.stat().st_mode & 0o777 == 0o600
    loaded = TurnReader.load(path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.weights, reader.weights)
    np.testing.assert_array_equal(loaded.bias, reader.bias)
    assert loaded.labels == reader.labels and loaded.embedding_model == reader.embedding_model
    assert loaded.report == reader.report
    assert loaded.feelings([1, 0, 0]) == reader.feelings([1, 0, 0])


def test_an_encoder_failure_keeps_the_installed_reader(tmp_path, monkeypatch):
    reader = _reader()
    path = tmp_path / "reader.npz"
    reader.save(path)
    previous = path.read_bytes()

    def broken_encoder(destination, **arrays):
        if hasattr(destination, "write"):
            destination.write(b"incomplete archive")
        else:
            Path(destination).write_bytes(b"incomplete archive")
        raise OSError("encoder could not finish")

    monkeypatch.setattr(np, "savez", broken_encoder)
    with pytest.raises(OSError, match="encoder could not finish"):
        reader.save(path)
    assert path.read_bytes() == previous
    assert TurnReader.load(path) is not None


def test_a_reader_path_without_an_npz_suffix_keeps_the_numpy_filename(tmp_path):
    path = tmp_path / "reader.head"
    assert _reader().save(path) == path
    assert not path.exists()
    assert TurnReader.load(tmp_path / "reader.head.npz") is not None
