"""A private image crosses the reader pipe with its actual final capture identity."""
from __future__ import annotations

import io
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception import eyes_of_their_own as eyes
from core.perception.observed_transfer import ScreenReading, freeze_screen


def _reading(**extra):
    return {"ok": True, "text": "visible", "layout": [], "capture_at": 12.5,
            "_capture_epoch": "final-capture", "_settled": False, **extra}


def _wire(**extra):
    reading = _reading(**extra)
    picture = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    reading[eyes._PIXEL_ENVELOPE] = eyes._picture_to_send(picture, reading)
    return reading, picture


def test_private_pipe_round_trip_preserves_exact_immutable_pixels_and_capture():
    wire, picture = _wire()
    restored = eyes._restore_reading(json.loads(json.dumps(wire)))
    assert isinstance(restored, ScreenReading)
    snapshot = freeze_screen(restored)
    assert snapshot.picture.data == picture.tobytes()
    assert (snapshot.capture_at, snapshot.epoch) == (12.5, "final-capture")
    picture[:] = 255
    assert snapshot.picture.data != picture.tobytes()
    assert eyes._PIXEL_ENVELOPE not in restored
    assert json.loads(json.dumps(restored)) == _reading()
    assert not hasattr(dict(restored), "picture_snapshot")


def test_a_missing_private_capture_is_unknown_without_global_pixel_fallback():
    restored = eyes._restore_reading(_reading())
    assert freeze_screen(restored).picture is None


@pytest.mark.parametrize("change", [
    {"version": True}, {"version": 2}, {"height": True}, {"height": 0}, {"height": 961},
    {"width": 0}, {"width": 961}, {"channels": 4}, {"data": ""}, {"data": "!" * 96},
    {"capture_at": 15.0}, {"capture_at": float("nan")}, {"capture_at": True},
    {"capture_epoch": "previous-capture"}, {"capture_epoch": ""}, {"extra": "unrecognized"},
])
def test_malformed_or_mismatched_private_pixels_are_rejected(change):
    wire, _ = _wire()
    wire[eyes._PIXEL_ENVELOPE].update(change)
    with pytest.raises(ValueError):
        eyes._restore_reading(wire)


@pytest.mark.parametrize("envelope", [None, [], "bytes"])
def test_an_explicit_but_invalid_envelope_is_not_an_old_reader(envelope):
    with pytest.raises(ValueError):
        eyes._restore_reading({**_reading(), eyes._PIXEL_ENVELOPE: envelope})


@pytest.mark.parametrize("capture", [{"capture_at": None}, {"capture_at": float("inf")},
                                    {"_capture_epoch": ""}])
def test_capture_identity_is_required_from_the_take_instead_of_stamped_at_reply(capture):
    with pytest.raises(ValueError):
        eyes._picture_to_send(np.zeros((2, 2, 3), np.uint8), _reading(**capture))


def test_large_frame_is_aspect_preserving_and_bounded():
    image = np.zeros((1200, 1800, 3), np.uint8)
    wire = _reading()
    wire[eyes._PIXEL_ENVELOPE] = eyes._picture_to_send(image, wire)
    restored = eyes._restore_reading(wire)
    assert (restored.picture_snapshot.height, restored.picture_snapshot.width) == (640, 960)
    assert len(json.dumps(wire).encode("utf-8")) < eyes._MAX_IPC_BYTES


def test_features_and_private_pixels_use_one_final_picture(monkeypatch):
    from core.perception import how_a_place_looks, keys_drawn_on_screen, shapes_that_look_pressable

    image = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    observed = []

    def shapes(picture, *, apart_from):
        assert picture is image
        observed.append(("shapes", picture.tobytes()))
        return [{"x": 0.0, "y": 0.0, "width": 0.2, "height": 0.2}]

    def look(picture, region):
        assert picture is image
        observed.append(("look", picture.tobytes()))
        return [0.1, 0.2]

    def keys(picture):
        assert picture is image
        observed.append(("keys", picture.tobytes()))
        return [{"key": "space"}]

    monkeypatch.setattr(shapes_that_look_pressable, "pressable_shapes", shapes)
    monkeypatch.setattr(how_a_place_looks, "look_of", look)
    monkeypatch.setattr(keys_drawn_on_screen, "keys_drawn", keys)
    window = SimpleNamespace(owner="Drawing", number=42, bounds=(10, 20, 100, 80))
    sent = eyes._reading_to_send(image, _reading(), window=window, over=(0.1, 0.25, 0.8, 0.75),
                                 still=True, wait=True, looked_took=0.25)
    restored = eyes._restore_reading({**sent, "_settled": False})
    assert all(data == restored.picture_snapshot.data for _, data in observed)
    assert [name for name, _ in observed] == ["shapes", "look", "keys"]
    assert sent["shapes"][0]["look"] == [0.1, 0.2]
    assert sent["keys_drawn"] == [{"key": "space"}]
    assert sent["capture_at"] == 12.5
    assert sent["surface_id"] == "window:Drawing:42"
    assert sent["bounds"] == [20, 40, 70, 40]


def test_ipc_decode_preserves_pixels_without_touching_task_loop_state(monkeypatch):
    from core.perception import where_the_words_point

    remembered = []
    monkeypatch.setattr(where_the_words_point, "remember_the_still_picture",
                        lambda picture, **context: remembered.append((picture.copy(), context)))
    wire, image = _wire(_settled=True, surface_id="window:Drawing:42", bounds=[20, 40, 70, 40])
    restored = eyes._restore_reading(wire)
    assert remembered == []
    snapshot = restored.picture_snapshot
    assert np.array_equal(np.frombuffer(snapshot.data, dtype=np.uint8).reshape(snapshot.height, snapshot.width, 3), image)
    assert restored["surface_id"] == "window:Drawing:42"
    assert restored["bounds"] == [20, 40, 70, 40]


def test_parent_refuses_invalid_custody_without_leaking_wire_bytes(monkeypatch):
    wire, _ = _wire()
    wire["_capture_epoch"] = "another"
    child = SimpleNamespace(_has_answered=True)
    monkeypatch.setattr(eyes, "_CHILD", child)
    monkeypatch.setattr(eyes, "_start_them", lambda: True)
    monkeypatch.setattr(eyes, "_ask", lambda *args: deepcopy(wire))
    monkeypatch.delenv("AURA_EYES_IN_THIS_PROCESS", raising=False)
    assert eyes.look_through_them(SimpleNamespace(owner="Drawing", number=42)) is None


def test_reader_bounds_the_reply_before_parsing(monkeypatch):
    monkeypatch.setattr(eyes, "_MAX_IPC_BYTES", 64)
    child = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("x" * 65 + "\n"))
    assert eyes._ask(child, {"number": 42}, 0.5) is None


def test_non_mapping_reply_is_not_a_reading():
    child = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("[]\n"))
    assert eyes._ask(child, {"number": 42}, 0.5)["ok"] is False


def test_the_capture_custody_invariant_is_pure_and_registered():
    assert eyes._child_capture_custody_invariant() == ()
