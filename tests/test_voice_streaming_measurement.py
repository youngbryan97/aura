"""A transport receipt cannot serve as a live latency measurement."""
import hashlib
import json

import pytest

from core.organism.model_validation_voice import (
    STREAM_CHECKS,
    STREAM_SOURCES,
    measured_voice_streaming,
)
from core.organism.nothing_measured import NothingMeasured


def test_no_receipt_does_not_count_as_a_passing_measurement(tmp_path):
    with pytest.raises(NothingMeasured):
        measured_voice_streaming(tmp_path)


def test_receipt_expires_when_its_transport_or_tests_change(tmp_path):
    hashes = {}
    for name in STREAM_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('measured source')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = tmp_path / 'docs/evidence'
    evidence.mkdir(parents=True)
    receipt = {'schema': 'aura.voice.streaming_measurement.v1', 'sources': hashes,
               'checks': sorted(STREAM_CHECKS), 'ok': True}
    path = evidence / 'voice-streaming-example.json'
    path.write_text(json.dumps(receipt))
    assert measured_voice_streaming(tmp_path)
    receipt['checks'].remove('cancelled and foreign previews rejected')
    path.write_text(json.dumps(receipt))
    assert measured_voice_streaming(tmp_path) is False
    receipt['checks'] = sorted(STREAM_CHECKS)
    path.write_text(json.dumps(receipt))
    (tmp_path / 'core/brain/llm/reply_previews.py').write_text('changed source')
    with pytest.raises(NothingMeasured):
        measured_voice_streaming(tmp_path)
