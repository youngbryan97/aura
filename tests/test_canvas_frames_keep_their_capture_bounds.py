"""Canvas pictures retain the interval and canvas owner measured by their capture call."""
from __future__ import annotations

import copy
import json
import pickle
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception import frames_as_they_are_drawn as frames
from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime

pytestmark = pytest.mark.unit

CLIP = {"x": 0, "y": 0, "width": 20, "height": 12}
CAPTURE = {"version": "canvas-capture-v1", "surface": "document/observed-canvas", "epoch": 4,
           "sequence": 8, "width": 20, "height": 12, "source_at": 25.}


@pytest.mark.asyncio
async def test_actual_capture_call_is_bracketed_before_decode_and_screen_validation(monkeypatch):
    clock = SimpleNamespace(at=10.)
    monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock.at))
    scene = {"source": "canvas-paint-v1", "complete": True, "width": 20, "height": 12,
             "objects": [], "texts": []}

    async def evaluate(script, clip):
        assert script == frames._THE_CANVAS_NOW and clip == CLIP
        clock.at += .1
        return {"pixels": "encoded", "capture": CAPTURE.copy(), "scene": scene,
                "sounds": {"now": 25000., "heard": [{"at": 24500., "kind": "tone", "sig": "sine 440"}]}}

    async def decode(function, encoded):
        assert function is frames._from_data_url and encoded == "encoded"
        clock.at += .2
        return np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)

    async def validate(picture, clip):
        clock.at += 1.
        return True

    monkeypatch.setattr(frames, "run_interactive_cpu", decode)
    reader = frames.CanvasFrames(SimpleNamespace(evaluate=evaluate))
    reader._observing = True
    monkeypatch.setattr(reader, "_shows_what_the_screen_shows", validate)
    picture, at = await reader.look(CLIP)
    assert isinstance(at, float) and isinstance(at, MeasuredFrameTime)
    assert isinstance(at, frames.CanvasFrameTime)
    assert at == at.lower == 10. and at.upper == at.arrived == pytest.approx(10.1)
    assert (at.source, at.source_at, at.surface, at.epoch, at.sequence) == (
        "canvas-bracket", 25., CAPTURE["surface"], 4, 8)
    assert clock.at == pytest.approx(11.3)
    assert picture.drawing_scene == scene
    assert [(sound.kind, sound.sig, sound.at) for sound in picture.sounds] == [("tone", "sine 440", 9.5)]


@pytest.mark.parametrize("change", [
    {"version": "other"}, {"surface": ""}, {"surface": None}, {"epoch": 0}, {"epoch": True},
    {"sequence": 0}, {"sequence": True}, {"width": 21}, {"height": 13}, {"width": True},
    {"source_at": float("nan")}, {"source_at": True}, {"source_at": -1.},
])
def test_missing_or_mismatched_capture_metadata_has_no_measured_lower_bound(change):
    at = frames._canvas_capture_time({**CAPTURE, **change}, 10., 10.1, (12, 20, 3))
    assert isinstance(at, MeasuredFrameTime) and not isinstance(at, frames.CanvasFrameTime)
    assert at == 10. and at.lower is None and at.upper == at.arrived == 10.1
    assert at.source == "canvas-receipt" and at.source_at is None


@pytest.mark.parametrize("capture", [None, "encoded", {}, {"surface": "canvas"}])
def test_legacy_pixel_responses_do_not_gain_capture_custody(capture):
    at = frames._canvas_capture_time(capture, 10., 10.1, (12, 20, 3))
    assert at.lower is None and at.epoch == 0 and not hasattr(at, "surface")


@pytest.mark.parametrize("began,finished", [(10.1, 10.), (-1., 10.), (10., float("nan"))])
def test_invalid_host_brackets_stay_unknown(began, finished):
    assert frames._canvas_capture_time(CAPTURE, began, finished, (12, 20, 3)).lower is None


def test_a_measured_call_bracket_does_not_require_a_remote_clock_origin():
    at = frames._canvas_capture_time({**CAPTURE, "source_at": None}, 10., 10.1, (12, 20, 3))
    assert isinstance(at, frames.CanvasFrameTime) and at.lower == 10. and at.source_at is None


def test_canvas_bounds_and_owner_are_immutable_across_private_worker_serialization():
    original = frames._canvas_capture_time(CAPTURE, 10., 10.1, (12, 20, 3))
    for at in (original, copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
        assert isinstance(at, frames.CanvasFrameTime) and vars(at) == vars(original)
        for attribute, value in (("upper", 12.), ("surface", "another"), ("sequence", 9), ("epoch", 5)):
            with pytest.raises(AttributeError):
                setattr(at, attribute, value)
        with pytest.raises(AttributeError):
            del at.surface
    assert frames._canvas_capture_bounds_invariant() == ()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to exercise the capture script")
def test_capture_script_follows_the_actual_canvas_document_and_observed_extent():
    harness = r"""
const take = require('vm').runInNewContext(process.argv[1], globalThis);
let random = 0, source = 25000;
Object.defineProperty(globalThis, 'crypto', {configurable:true,
  value:{getRandomValues(values) { values.fill(++random); return values; }}});
Object.defineProperty(globalThis, 'performance', {configurable:true, value:{now() { return source++; }}});
const rectangle = {left:0, top:0, width:320, height:200};
const makeCanvas = () => ({tagName:'CANVAS', isConnected:true, width:320, height:200,
  getBoundingClientRect() { return rectangle; }, toDataURL() { return 'pixels'; }});
let canvas = makeCanvas();
global.document = {querySelectorAll() { return [canvas]; }};
global.window = {__auraDrawingObservation:{snapshot(c) { return {same:c === canvas}; }},
  __auraSoundsHeard:{take() { return {heard:[]}; }}};
const clip = {x:0, y:0, width:320, height:200};
const first = take(clip), second = take(clip);
canvas.width = 640;
const resized = take(clip);
canvas.isConnected = false; canvas = makeCanvas();
const replaced = take(clip);
global.document = {querySelectorAll() { return [canvas]; }};
const navigated = take(clip);
delete window.__auraCanvasCaptureCustody;
Object.defineProperty(globalThis, 'crypto', {configurable:true,
  value:{getRandomValues() { throw new Error('custody unavailable'); }}});
const unknown = take(clip);
process.stdout.write(JSON.stringify({first, second, resized, replaced, navigated, unknown}));
"""
    result = subprocess.run([shutil.which("node"), "-e", harness, frames._THE_CANVAS_NOW],
                            capture_output=True, text=True, timeout=10, check=True)
    samples = json.loads(result.stdout)
    first, second, resized, replaced, navigated = (samples[key]["capture"] for key in
                                                   ("first", "second", "resized", "replaced", "navigated"))
    assert first["surface"] == second["surface"] == resized["surface"]
    assert first["epoch"] == second["epoch"] < resized["epoch"] < replaced["epoch"]
    assert replaced["surface"] != first["surface"] and navigated["surface"] != replaced["surface"]
    assert [first["sequence"], second["sequence"], resized["sequence"], replaced["sequence"]] == [1, 2, 3, 4]
    assert first["source_at"] == 25. and second["source_at"] == 25.001
    assert resized["width"] == 640 and navigated["sequence"] == 1
    assert all(sample["scene"] == {"same": True} and sample["sounds"] == {"heard": []}
               for sample in samples.values())
    assert samples["unknown"]["pixels"] == "pixels" and samples["unknown"]["capture"] is None
