"""A thing nothing names is found by how it looks: an example picture, or a colour."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.perception.finding_it_by_look import looks_like, of_colour


def _a_pattern(seed: int, size: int = 96) -> np.ndarray:
    """Something with corners in it, the kind of thing features find."""
    roll = np.random.default_rng(seed)
    blocks = np.kron(roll.random((8, 8)), np.ones((size // 8, size // 8))) * 255.0
    picture = np.dstack([blocks, np.roll(blocks, 3, axis=1), np.roll(blocks, 5, axis=0)]).astype(np.uint8)
    cv2.circle(picture, (size // 3, size // 2), size // 6, (0, 0, 255), 2)
    cv2.line(picture, (0, 0), (size - 1, size - 1), (255, 255, 255), 2)
    return picture


def _a_frame(seed: int) -> np.ndarray:
    roll = np.random.default_rng(seed)
    return (cv2.GaussianBlur(roll.random((360, 640, 3)), (0, 0), 6) * 255).astype(np.uint8)


def test_an_example_is_found_where_it_is_even_bigger():
    example = _a_pattern(1)
    frame = _a_frame(2)
    placed = cv2.resize(example, (125, 125))
    frame[150:275, 400:525] = placed
    found = looks_like(frame, example, name="the carving")
    assert found is not None and found["text"] == "the carving"
    assert found["center_x"] == pytest.approx(462.5 / 640, abs=0.03)
    assert found["center_y"] == pytest.approx(212.5 / 360, abs=0.04)


def test_what_is_not_there_is_not_found():
    assert looks_like(_a_frame(3), _a_pattern(1), name="the carving") is None


def test_a_colour_is_found_where_it_is():
    frame = np.full((360, 640, 3), 128, np.uint8)
    frame += np.random.default_rng(4).integers(0, 20, frame.shape, dtype=np.uint8)
    frame[100:160, 50:130] = (30, 30, 220)  # red, in blue-green-red order
    found = of_colour(frame, "red", name="the red house")
    assert found is not None and found["text"] == "the red house"
    assert found["center_x"] == pytest.approx(90 / 640, abs=0.01)
    assert found["center_y"] == pytest.approx(130 / 360, abs=0.01)


def test_a_colour_that_is_not_there_is_not_found():
    frame = np.full((360, 640, 3), 128, np.uint8)
    frame[100:160, 50:130] = (30, 30, 220)
    assert of_colour(frame, "blue") is None
    assert of_colour(np.full((90, 160, 3), 128, np.uint8), "red") is None


def test_a_colour_is_found_where_opencv_is_refused(monkeypatch):
    """Aura's main process refuses OpenCV, and a trip to the red door runs there."""
    import builtins

    real_import = builtins.__import__

    def refusing(name, *args, **kwargs):
        if name.split(".")[0] == "cv2":
            raise ImportError("OpenCV import is blocked in Aura's primary macOS process")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refusing)
    frame = np.full((120, 200, 3), 128, np.uint8)
    frame[20:60, 30:90] = (30, 30, 220)
    found = of_colour(frame, "red", name="the red door")
    assert found is not None and found["center_x"] == pytest.approx(60 / 200, abs=0.01)
    assert looks_like(frame, frame[20:60, 30:90], name="the carving") is None
