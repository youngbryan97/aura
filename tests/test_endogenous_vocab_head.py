from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.brain.llm.endogenous_state import STATE_DIM, layout_digest, semantics_digest
from core.brain.llm.endogenous_vocab_head import (
    HEAD_ARTIFACT_SCHEMA,
    EndogenousVocabHead,
    HeadUnusableError,
)


def _head(*, offset: float = 0.0) -> EndogenousVocabHead:
    values = np.arange(3 * STATE_DIM, dtype=np.float32).reshape(3, STATE_DIM)
    return EndogenousVocabHead(
        weights=(values / 1000.0) + offset,
        bias=np.asarray([0.1, -0.2, 0.3], dtype=np.float32) + offset,
        vocab_size=3,
        layout=layout_digest(),
        semantics=semantics_digest(),
        tokenizer="tokenizer-fixture",
        trained=True,
        report={"held_out_gain": 0.25},
        trained_at=1234.5,
    )


def test_save_and_load_bind_one_exact_weight_generation(tmp_path):
    expected = _head()

    path = expected.save(tmp_path)
    loaded = EndogenousVocabHead.load(tmp_path)

    manifest = json.loads((tmp_path / "vocab_head.json").read_text(encoding="utf-8"))
    payload = path.read_bytes()
    assert manifest["schema"] == HEAD_ARTIFACT_SCHEMA
    assert manifest["weights_bytes"] == len(payload)
    assert manifest["weights_sha256"] == hashlib.sha256(payload).hexdigest()
    np.testing.assert_array_equal(loaded.weights, expected.weights)
    np.testing.assert_array_equal(loaded.bias, expected.bias)
    assert loaded.report == expected.report


def test_load_rejects_tampered_weight_payload_before_numpy_parses_it(tmp_path):
    path = _head().save(tmp_path)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(HeadUnusableError, match="digest does not match"):
        EndogenousVocabHead.load(tmp_path)


def test_load_rejects_manifest_from_another_generation(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _head(offset=0.0).save(first)
    _head(offset=1.0).save(second)
    (second / "vocab_head.json").write_bytes((first / "vocab_head.json").read_bytes())

    with pytest.raises(HeadUnusableError, match="digest does not match"):
        EndogenousVocabHead.load(second)


def test_load_rejects_legacy_unbound_manifest(tmp_path):
    path = _head().save(tmp_path)
    manifest_path = tmp_path / "vocab_head.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("schema")
    manifest.pop("weights_sha256")
    manifest.pop("weights_bytes")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert path.exists()

    with pytest.raises(HeadUnusableError, match="does not bind"):
        EndogenousVocabHead.load(tmp_path)


def test_save_rejects_a_gateway_receipt_for_different_bytes(tmp_path, monkeypatch):
    import core.runtime.file_write_gateway as gateway_module

    class ForgedGateway:
        def write_bytes_batch(self, entries, *, source):
            paths = tuple(str(entry.path.parent.resolve() / entry.path.name) for entry in entries)
            return SimpleNamespace(
                transaction_id="forged",
                paths=paths,
                sha256=tuple((path, "0" * 64) for path in paths),
            )

    monkeypatch.setattr(gateway_module, "get_file_write_gateway", lambda: ForgedGateway())

    with pytest.raises(HeadUnusableError, match="receipt does not match"):
        _head().save(tmp_path)
