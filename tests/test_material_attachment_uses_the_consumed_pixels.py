"""The ordinary observer's pixel receipt survives its publication boundary."""
import numpy as np
import pytest

from core.perception.a_visual_reference_in_the_picture import _digest
from core.perception.pixels_consumed_by_the_observer import consumed_pixels
from core.perception.play_observation_without_holding_the_loop import _detached_state
from core.perception.seeing_play_elsewhere import _brought_up_to_date
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


def test_the_observer_keeps_the_exact_source_rgb_bytes_across_worker_publication():
    picture = np.random.default_rng(781).integers(0, 255, (64, 96, 3), np.uint8)
    observer = WhatMoves()
    observer.see(picture, 4.)
    receipt = observer.pixels_consumed
    assert receipt == consumed_pixels(picture, 4.) and receipt.sha256 == _digest(picture)
    state = _detached_state(observer)
    mirror = WhatMoves()
    _brought_up_to_date(mirror, state)
    assert mirror.pixels_consumed == receipt
    picture[0, 0] = 0
    assert receipt.sha256 != _digest(picture)
    assert mirror.pixels_consumed == receipt
    observer.see(picture, 4.1)
    assert observer.pixels_consumed.at == 4.1 and observer.pixels_consumed.sha256 == _digest(picture)


@pytest.mark.parametrize('picture,at', [
    (np.zeros((16, 16, 3)), 1.), (np.zeros((16, 16, 4), np.uint8), 1.),
    (np.zeros((0, 16, 3), np.uint8), 1.), (np.zeros((16, 16, 3), np.uint8), True),
    (np.zeros((16, 16, 3), np.uint8), float('nan')),
])
def test_unsupported_arrays_and_times_cannot_supply_pixel_custody(picture, at):
    assert consumed_pixels(picture, at) is None


def test_the_registered_claim_executes_the_consumed_pixel_canary():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'observers_keep_their_consumed_pixel_receipt')
    assert check.predict(None) is True
