"""A changed voice path cannot inherit an earlier presentation result."""
import hashlib
import json

import pytest

from core.organism.model_validation_voice import CHECKS, SOURCES, measured_voice
from core.organism.nothing_measured import NothingMeasured


def test_voice_measurement_expires_with_the_measured_source(tmp_path):
    with pytest.raises(NothingMeasured):
        measured_voice(tmp_path)
    sources = {}
    for name in SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = tmp_path / "docs/evidence"
    evidence.mkdir(parents=True)
    receipt = evidence / "voice-polish-example.json"
    receipt.write_text(json.dumps({"schema": "aura.voice.presentation_measurement.v1", "ok": True,
                                   "sources": sources, "checks": list(CHECKS)}))
    assert measured_voice(tmp_path)
    (tmp_path / SOURCES[0]).write_text("changed")
    with pytest.raises(NothingMeasured):
        measured_voice(tmp_path)
