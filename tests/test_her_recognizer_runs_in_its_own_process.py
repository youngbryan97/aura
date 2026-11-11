"""Her voice mode's recognizer runs in its own process (core/voice/duplex/parakeet_process.py).

LIVE 11 November 2026: her runtime died of a segmentation fault twice in one day, each time inside a Parakeet decode's
mx.eval on an executor thread of her main process. On her real model the process started warm in 1.0 s and decoded a
spoken sentence exactly in 120 to 185 ms.
"""
from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest

from core.voice.duplex.config import AsrConfig
from core.voice.duplex.parakeet_process import ParakeetProcess, RecognizerRefusedError

FAKE = textwrap.dedent('''
    import base64, json, sys, time
    import numpy as np
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        audio = np.frombuffer(base64.b64decode(request["audio"]), dtype="<f4")
        if audio.size == 3:
            print(json.dumps({"error": "ValueError: too short"}), flush=True)
        elif audio.size == 4:
            time.sleep(5)
        elif audio.size == 5:
            sys.exit(0)
        else:
            print(json.dumps({"text": f"{audio.size} samples of {request['repo']}",
                              "tokens": [[" heard", 0.0, 0.5]]}), flush=True)
''')


@pytest.fixture
def recognizer(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE)
    process = ParakeetProcess(["repo-a", "repo-a"], python=sys.executable, worker=worker)
    process.start(timeout_s=30)
    yield process
    process.close()


def test_a_decode_goes_to_the_process_and_comes_back_with_its_timings(recognizer):
    assert recognizer.repos == ("repo-a",)
    text, tokens = recognizer.decode(np.zeros(16000, np.float32), "repo-a")
    assert text == "16000 samples of repo-a" and tokens == ((" heard", 0.0, 0.5),)


def test_a_refused_decode_leaves_the_process_running(recognizer):
    with pytest.raises(RecognizerRefusedError):
        recognizer.decode(np.zeros(3, np.float32), "repo-a")
    assert recognizer.alive
    assert recognizer.decode(np.zeros(8, np.float32), "repo-a")[0] == "8 samples of repo-a"


def test_a_late_or_lost_answer_closes_it_so_it_cannot_become_the_next_one(recognizer):
    with pytest.raises(TimeoutError):
        recognizer.decode(np.zeros(4, np.float32), "repo-a", timeout_s=0.5)
    assert not recognizer.alive
    with pytest.raises(RuntimeError, match="not running"):
        recognizer.decode(np.zeros(8, np.float32), "repo-a")


def test_a_process_that_exits_mid_decode_is_closed(recognizer):
    with pytest.raises(RuntimeError, match="exited"):
        recognizer.decode(np.zeros(5, np.float32), "repo-a")
    assert not recognizer.alive


def test_the_backend_never_loads_parakeet_into_her_runtime(monkeypatch):
    from core.voice.duplex import streaming_asr

    started, closed = [], []

    class _Process:
        alive = True

        def __init__(self, repos):
            self.repos = repos

        def start(self):
            started.append(self.repos)

        def decode(self, audio, repo):
            return "hello there", ((" hel", 0.0, 0.2), ("lo", 0.2, 0.4), (" there", 0.4, 0.8))

        def close(self):
            closed.append(True)

    import core.voice.duplex.parakeet_process as parakeet_process

    monkeypatch.setattr(parakeet_process, "ParakeetProcess", _Process)
    monkeypatch.setattr(streaming_asr, "_parakeet_available", lambda: True)
    monkeypatch.setitem(sys.modules, "parakeet_mlx", None)  # importing it here would raise
    backend = streaming_asr._WhisperBackend(AsrConfig())
    monkeypatch.setattr(backend, "_acquire_model_lane", lambda: (type("L", (), {"set_preemptible": lambda s, v: True,
                                                                                   "release": lambda s, **k: None})(), False))
    text, words = backend.transcribe_words(np.zeros(16000, np.float32), AsrConfig().final_model)
    assert text == "hello there" and [word for word, *_ in words] == ["hello", "there"]
    assert len(started) == 1
    backend.shutdown()
    assert closed == [True]
