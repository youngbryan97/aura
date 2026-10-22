"""Changing the gradient invalidates the browser measurement that described it."""
import hashlib
import json

import pytest

from core.organism.model_validation_presentation import GRADIENT_SOURCES, measured_gradient
from core.organism.nothing_measured import NothingMeasured


def test_gradient_measurement_requires_current_source(tmp_path):
    with pytest.raises(NothingMeasured): measured_gradient(tmp_path)
    sources = {}
    for name in GRADIENT_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = tmp_path / 'docs/evidence'
    evidence.mkdir(parents=True)
    (evidence/'ambient-gradient-example.json').write_text(json.dumps({
        'ok':True,'sources':sources,'checks':['pixels change over time','reduced motion stops movement',
                                            'voice mode stops background work']}))
    assert measured_gradient(tmp_path)
    (tmp_path/GRADIENT_SOURCES[0]).write_text('changed')
    with pytest.raises(NothingMeasured): measured_gradient(tmp_path)
