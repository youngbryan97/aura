"""Resolving a catalogue path reads state; its governed writer creates it."""
from __future__ import annotations

import json
import threading

from core.consciousness.absorbed_voices import PERSIST_FILENAME, AbsorbedVoices


def test_resolving_and_loading_a_missing_catalogue_creates_nothing(tmp_path):
    folder = tmp_path / "new" / "catalogue"
    voices = AbsorbedVoices(storage_dir=folder)
    assert voices.voice_count() == 0
    assert not folder.exists()

    voices.add_voice("fixture", sample_text="A stored perspective")
    voices.save()

    assert json.loads((folder / PERSIST_FILENAME).read_text())["fixture"]["voice_id"] == "fixture"
    assert AbsorbedVoices(storage_dir=folder).get_voice("fixture") is not None


def test_a_background_save_carries_its_own_governance_scope(tmp_path, monkeypatch):
    from core.governance_context import governance_runtime_active

    folder = tmp_path / "catalogue"
    voices = AbsorbedVoices(storage_dir=folder)
    voices.add_voice("fixture", sample_text="A stored perspective")
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    assert governance_runtime_active()
    errors = []

    def save():
        try:
            voices.save()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=save, daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive(), "the catalogue writer did not finish"
    assert not errors
    saved = folder / PERSIST_FILENAME
    assert saved.is_file(), "the internal writer was refused under governance enforcement"
    assert AbsorbedVoices(storage_dir=folder).get_voice("fixture") is not None
