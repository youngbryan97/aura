"""Her voice as it forms, measured through her own speech worker: first sound, gaps, and sameness with a whole sentence.

Starts tools/voice/original_speech_worker.py on her prepared voice, asks for each sentence twice (whole, then streamed)
and reports: when the first piece arrived; whether any piece arrived after the audio before it had finished playing,
if playback starts on the first piece; and how close the streamed sentence is to the whole one at her normal pace,
after matching level (the streamed gain is learned, the whole gain is per sentence).

    AURA_VOICE_ORIGINAL_PYTHON=~/.aura/voice-lab/qwen-env/bin/python python tools/voice/prove_speech_as_it_forms.py
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SENTENCES = (
    "I hear you clearly.",
    "The house on the hill can wait a minute, tell me what you want to change first.",
    "Four.",
    "That one surprised me, honestly, because I expected the opposite.",
)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from core.runtime.flags import env_str

    owner = "tools/voice/prove_speech_as_it_forms.py"
    python = env_str("AURA_VOICE_ORIGINAL_PYTHON", default=str(Path.home() / ".aura/voice-lab/qwen-env/bin/python"),
                     description="Python that runs her original voice model", owner=owner)
    manifest = Path(env_str("AURA_VOICE_MANIFEST", default=str(ROOT / "data/voice_models/aura-kiwi-c/manifest.json"),
                            description="Manifest of the voice this proof speaks with", owner=owner))
    worker_log = env_str("AURA_PROOF_WORKER_LOG", default=os.devnull,
                         description="Where this proof's speech worker writes its stderr", owner=owner)
    if not manifest.is_file():
        manifest = Path.home() / ".aura/live-source/data/voice_models/aura-kiwi-c/manifest.json"
    worker = subprocess.Popen([python, "-u", str(ROOT / "tools/voice/original_speech_worker.py"), str(manifest)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(worker_log, "w"),
                              env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
    lines = iter(worker.stdout.readline, b"")

    def receive() -> dict:
        message = json.loads(next(lines))
        if message.get("error"):
            raise RuntimeError(message["error"])
        return message

    try:
        ready = receive()
        print("worker ready; streams:", ready.get("streams"))
        if not ready.get("streams"):
            return 1
        failures = 0
        for speed in (1.0, 1.08):
            for text in SENTENCES:
                worker.stdin.write((json.dumps({"text": text, "speed": speed}) + "\n").encode())
                worker.stdin.flush()
                whole = np.frombuffer(base64.b64decode(receive()["audio"]), dtype="<f4")
                asked = time.perf_counter()
                worker.stdin.write((json.dumps({"text": text, "speed": speed, "stream": True}) + "\n").encode())
                worker.stdin.flush()
                pieces, arrivals, rate = [], [], 24000
                while True:
                    message = receive()
                    rate = int(message["sample_rate"])
                    pieces.append(np.frombuffer(base64.b64decode(message["audio"]), dtype="<f4"))
                    arrivals.append(time.perf_counter() - asked)
                    if not message.get("more"):
                        break
                first = next((t for t, p in zip(arrivals, pieces, strict=True) if len(p)), arrivals[-1])
                played_from, queued, late = None, 0.0, []
                for t, p in zip(arrivals, pieces, strict=True):
                    if not len(p):
                        continue
                    if played_from is None:
                        played_from = t
                    elif t > played_from + queued:
                        late.append(round(t - (played_from + queued), 3))
                    queued += len(p) / rate
                streamed = np.concatenate(pieces)
                line = (f"speed {speed}: first sound {first:.2f} s, whole sentence {arrivals[-1]:.2f} s for "
                        f"{len(streamed) / rate:.2f} s of speech in {sum(1 for p in pieces if len(p))} pieces; "
                        f"late pieces {late or 'none'}")
                if speed == 1.0:
                    skip = int(.32 * rate)  # the onset: the streamed window saw more of her reference before it
                    n = min(len(whole), len(streamed)) - skip
                    a, b = streamed[skip:skip + n].astype(np.float64), whole[skip:skip + n].astype(np.float64)
                    scale = float(np.dot(a, b) / max(np.dot(a, a), 1e-12))
                    snr = 10 * np.log10(np.sum(b ** 2) / max(np.sum((scale * a - b) ** 2), 1e-12))
                    line += f"; same as whole at {snr:.1f} dB (level ratio {1 / scale:.3f})"
                    failures += snr < 40
                failures += bool(late)
                print(f"{text[:40]!r:44} {line}")
        return 1 if failures else 0
    finally:
        worker.stdin.close()
        worker.terminate()
        worker.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
