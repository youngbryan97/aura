"""Keep the latest observation distinct from the last words ever read."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.perception.observed_transfer import freeze_screen
from core.verify import invariant


def keep_latest_reading(held: dict, reading: Mapping[str, Any]) -> None:
    """A wordless or unavailable reading cannot keep old text labelled as current."""
    frame = freeze_screen(reading)
    previous = held.get('_latest_capture_at')
    if (frame.capture_at is not None and type(previous) in (int, float) and frame.capture_at < previous):
        return
    if frame.capture_at is not None:
        held['_latest_capture_at'] = frame.capture_at
    ok = reading.get('ok') is True
    text = ' '.join(str(reading.get('text') or '').split()) if ok else ''
    observation = dict(status='words_read' if text else 'no_words_read' if ok else 'unavailable',
                       text=text, capture_at=frame.capture_at, capture_epoch=frame.epoch,
                       surface=list(frame.surface), bounds=list(frame.bounds) if frame.bounds is not None else None,
                       pixels=frame.picture is not None)
    if not ok:
        observation['error'] = str(reading.get('error') or reading.get('refused_because') or 'no usable reading')[:300]
    held['last_seen'], held['last_observation'] = text, observation
    if text:
        held['last_read_words'] = dict(observation)


@invariant('perception.handoff_words_belong_to_the_latest_reading', scope='perception',
           owner='core/perception/the_latest_reading_at_handoff.py', observational=False)
def _latest_reading_invariant() -> tuple:
    held = {}
    keep_latest_reading(held, dict(ok=True, text='Instructions', capture_at=1.))
    keep_latest_reading(held, dict(ok=True, text='', capture_at=2.))
    assert held['last_seen'] == '' and held['last_observation']['status'] == 'no_words_read'
    assert held['last_read_words']['text'] == 'Instructions' and held['last_read_words']['capture_at'] == 1.
    keep_latest_reading(held, dict(ok=True, text='Old words', capture_at=1.5))
    assert held['last_seen'] == ''
    keep_latest_reading(held, dict(ok=False, error='capture refused'))
    assert held['last_observation']['status'] == 'unavailable' and held['last_seen'] == ''
    keep_latest_reading(held, dict(ok=True, text='An earlier capture', capture_at=1.5))
    assert held['last_observation']['status'] == 'unavailable'
    return ()
