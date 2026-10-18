"""Discovery spreads over pictures; each admitted strip is measured on current pixels."""
import numpy as np
import pytest

from core.perception import how_full_a_bar_is as bars

pytestmark = pytest.mark.unit


def _picture(length=90):
    picture = np.zeros((160, 240, 3), np.uint8)
    picture[30:38, 20:20+length] = (80, 140, 230)
    return picture


def test_discovery_does_not_run_the_complete_scan_inside_one_picture(monkeypatch):
    def unexpected(_picture):
        raise AssertionError('a whole-frame strip scan blocked a picture')

    monkeypatch.setattr(bars, '_strips', unexpected)
    reader = bars.BarsOnTheScreen()
    reader.read(_picture(), 1.)
    assert reader._discovery is not None and not reader._discovery.done


def test_each_step_owns_its_capture_and_visits_only_its_row_allowance():
    picture = _picture()
    discovery = bars._StripDiscovery(picture)
    picture[:] = 0
    found = []
    while not discovery.done:
        before = discovery.next_y
        found.extend(discovery.step())
        assert 0 < discovery.next_y-before <= 2*bars.DISCOVERY_ROWS_PER_PICTURE
    assert len(found) == 1 and (found[0].start, found[0].end) == (20, 110)


@pytest.mark.parametrize('current_length', [0, 60])
def test_a_delayed_candidate_needs_current_pixels_and_cannot_import_an_old_change(current_length):
    reader = bars.BarsOnTheScreen()
    reader.read(_picture(), 1.)
    assert not reader.strips
    reader.read(_picture(current_length), 1.1)
    if not current_length:
        assert not reader.strips
    else:
        strip, = reader.strips
        assert (strip.start, strip.end, strip.longest) == (20, 80, 60)
        assert strip.changes == strip.unsteady == 0 and strip.levels == [(1.1, 60)]


@pytest.mark.parametrize('changed', ['surface', 'epoch', 'shape', 'clock'])
def test_a_changed_capture_cancels_pending_candidate_work(changed):
    class CapturedAt(float):
        surface, epoch = 'one', 1

    reader = bars.BarsOnTheScreen()
    reader.read(_picture(), CapturedAt(1.))
    previous = reader._discovery
    picture, at = _picture(), CapturedAt(1.1)
    if changed == 'surface':
        at.surface = 'another'
    elif changed == 'epoch':
        at.epoch = 2
    elif changed == 'shape':
        picture = np.zeros((120, 240, 3), np.uint8)
    else:
        at = CapturedAt(.9)
    reader.read(picture, at)
    assert reader._discovery is not previous and not reader.strips


def test_established_bars_are_still_sampled_on_every_picture():
    reader = bars.BarsOnTheScreen()
    reader.strips = [bars.Bar(33, 8, (80, 140, 230), 20, 110, longest=90)]
    for n in range(4):
        reader.read(_picture(90-n*3), 1.+n*.1)
    assert len(reader.strips[0].levels) == 4


@pytest.mark.parametrize('thickness', [1, 100])
def test_current_colour_on_an_old_row_does_not_substitute_for_current_strip_shape(thickness):
    reader = bars.BarsOnTheScreen()
    reader.read(_picture(), 1.)
    current = np.zeros((160, 240, 3), np.uint8)
    current[33:33+thickness, 20:110] = (80, 140, 230)
    reader.read(current, 1.1)
    assert not reader.strips


def test_incremental_discovery_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert bars._bar_discovery_budget_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'bar_discovery_keeps_time_for_the_next_picture')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
