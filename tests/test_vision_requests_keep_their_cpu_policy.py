"""A requested CPU policy is acknowledged before OCR runs."""
import os
from types import SimpleNamespace

import pytest

from core.perception.vision_request_compute import (
    _cpu_vision_request_invariant,
    configure_vision_request,
)

pytestmark = pytest.mark.unit


def test_a_default_request_keeps_its_platform_choice(monkeypatch):
    monkeypatch.setenv('AURA_VISION_CPU_ONLY', '0')
    assert configure_vision_request(object()) is False


def test_the_declared_policy_reaches_the_request(monkeypatch):
    monkeypatch.setenv('AURA_VISION_CPU_ONLY', '1')
    received = []
    request = SimpleNamespace(setUsesCPUOnly_=received.append, usesCPUOnly=lambda: bool(received and received[-1]))
    assert configure_vision_request(request) is True
    assert received == [True]


@pytest.mark.parametrize('candidate', [object(), SimpleNamespace(setUsesCPUOnly_=lambda value: None),
                                    SimpleNamespace(setUsesCPUOnly_=lambda value: None, usesCPUOnly=lambda: False)])
def test_missing_or_refused_acknowledgement_prevents_processing(candidate):
    with pytest.raises(RuntimeError, match='acknowledge'):
        configure_vision_request(candidate, cpu_only=True)


def test_cpu_vision_has_a_registered_request_contract():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _cpu_vision_request_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'cpu_vision_requires_request_acknowledgement')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.external
def test_opted_in_native_ocr_reads_generated_words_with_cpu_acknowledgement(monkeypatch, tmp_path):
    if os.environ.get('AURA_RUN_CPU_VISION') != '1':
        pytest.skip('explicit native CPU OCR test was not requested')
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from core.perception import vision_request_compute as policy
    from core.perception.what_the_pixels_show import _read_anew

    monkeypatch.setenv('AURA_VISION_CPU_ONLY', '1')
    original, confirmed = policy.configure_vision_request, []

    def configure(request):
        result = original(request)
        confirmed.append(bool(request.usesCPUOnly()))
        return result

    monkeypatch.setattr(policy, 'configure_vision_request', configure)
    image = Image.new('RGB', (360, 100), 'white')
    ImageDraw.Draw(image).text((20, 20), 'READY 42', font=ImageFont.load_default(size=48), fill='black')
    regions = _read_anew(np.ascontiguousarray(np.asarray(image)[..., ::-1]))
    words = ''.join(str(region['text']).replace(' ', '') for region in regions).upper()
    assert confirmed == [True] and 'READY42' in words
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    folder = Path(os.environ['AURA_CPU_VISION_EVIDENCE']) if os.environ.get('AURA_CPU_VISION_EVIDENCE') else tmp_path
    with local_internal_governed_scope('cpu_vision_verification'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(folder, source='cpu_vision_verification')
        gateway.write_text(folder / 'words.json', json.dumps({'cpu_acknowledged': confirmed, 'regions': regions}, indent=2),
                           source='cpu_vision_verification')
