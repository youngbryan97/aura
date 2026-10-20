"""Each entry of her kNN store keeps the text of the token it predicts: the encoder decodes it, and a store written
without it is repaired from the ids it kept and put back, never over a store that is active."""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


class _Tokenizer:
    def decode(self, ids):
        return {7: " the", 8: ",", 9: " "}.get(int(ids[0]), f"<{ids[0]}>")


def test_her_encoder_gives_the_ingest_each_tokens_text():
    from core.brain.nonparametric_generation import MLXEncoder
    from core.brain.nonparametric_ingest import _decode_token

    encoder = object.__new__(MLXEncoder)
    encoder.tok = _Tokenizer()
    assert _decode_token(encoder, 7) == " the"
    assert _decode_token(encoder, 8) == ","


def _quarantined(tmp_path, ids):
    meta = tmp_path / "nonparametric_memory_5120_abc.meta.json.unusable-20261019-065956"
    keys = tmp_path / "nonparametric_memory_5120_abc.keys.npy.unusable-20261019-065956"
    meta.write_text(json.dumps({"token_ids": ids, "tokens": [""] * len(ids), "dim": 5120}))
    keys.write_bytes(b"keys")
    return meta, keys


def test_a_store_without_token_text_is_repaired_and_restored(tmp_path):
    from tools.repair_nonparametric_tokens import repair

    meta, keys = _quarantined(tmp_path, [7, 8] * 40)
    report = repair(meta, _Tokenizer().decode, apply=True)

    assert report["usable_before"] == 0 and report["usable_after"] == 80 and report["passes"]
    restored = json.loads((tmp_path / "nonparametric_memory_5120_abc.meta.json").read_text())
    assert restored["tokens"][:2] == [" the", ","] and restored["token_ids"][:2] == [7, 8]
    assert (tmp_path / "nonparametric_memory_5120_abc.keys.npy").read_bytes() == b"keys" and not keys.exists()
    assert list(tmp_path.glob("*.before-token-repair-*"))  # the quarantined meta, copied first


def test_it_never_overwrites_an_active_store_or_restores_one_still_below_the_bar(tmp_path):
    from tools.repair_nonparametric_tokens import repair

    meta, _keys = _quarantined(tmp_path, [9] * 80)  # only whitespace: still unrecallable
    with pytest.raises(SystemExit):
        repair(meta, _Tokenizer().decode, apply=True)
    meta, _keys = _quarantined(tmp_path, [7] * 80)
    (tmp_path / "nonparametric_memory_5120_abc.meta.json").write_text("{}")
    with pytest.raises(SystemExit):
        repair(meta, _Tokenizer().decode, apply=True)
