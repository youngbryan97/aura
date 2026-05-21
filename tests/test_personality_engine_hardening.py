from __future__ import annotations

import stat
from types import SimpleNamespace

from core.brain.personality_engine import _IDENTITY_KEY_BYTES, PersonalityEngine


def build_engine(tmp_path):
    engine = object.__new__(PersonalityEngine)
    engine.key_file = tmp_path / "identity.key"
    engine.seal_file = tmp_path / "identity.seal"
    engine.soul = SimpleNamespace(version="test", intensities={}, protocols={})
    return engine


def test_identity_key_is_persisted_and_reused(tmp_path):
    engine = build_engine(tmp_path)

    key = engine._load_or_generate_key()

    assert len(key) == _IDENTITY_KEY_BYTES
    assert engine.key_file.read_bytes() == key
    assert stat.S_IMODE(engine.key_file.stat().st_mode) == 0o600
    assert engine._new_key_generated is True
    assert engine._identity_key_persistent is True

    reloaded = build_engine(tmp_path)

    assert reloaded._load_or_generate_key() == key
    assert reloaded._new_key_generated is False
    assert reloaded._identity_key_persistent is True


def test_invalid_identity_key_is_quarantined_and_replaced(tmp_path):
    engine = build_engine(tmp_path)
    engine.key_file.parent.mkdir(parents=True, exist_ok=True)
    engine.key_file.write_bytes(b"short")

    key = engine._load_or_generate_key()

    assert len(key) == _IDENTITY_KEY_BYTES
    assert engine.key_file.read_bytes() == key
    assert list(tmp_path.glob("identity.key.invalid.*"))


def test_corrupt_persona_json_is_quarantined(tmp_path):
    engine = build_engine(tmp_path)
    path = tmp_path / "profile.json"
    path.write_text("{broken", encoding="utf-8")

    assert engine._load_json_object(path, label="profile data") is None
    assert not path.exists()
    assert list(tmp_path.glob("profile.json.invalid.*"))


def test_missing_identity_seal_initializes_during_trusted_bootstrap(tmp_path):
    engine = build_engine(tmp_path)
    engine.secret_key = b"a" * _IDENTITY_KEY_BYTES
    engine._new_key_generated = True

    assert engine._verify_cryptographic_seal() is True
    assert engine.seal_file.exists()

    engine._new_key_generated = False
    assert engine._verify_cryptographic_seal() is True
