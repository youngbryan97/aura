"""Own a prepared original voice in an isolated local MLX speech worker."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import selectors
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.Voice.Original")


class OriginalSpeech:
    name = "qwen_original"

    def __init__(self, manifest: Path, python: Path) -> None:
        self.manifest = manifest
        self.python = python
        self.process: subprocess.Popen | None = None
        self.voices = frozenset({"aura_velvet"})
        self._buffer = bytearray()

    def _receive(self, timeout: float) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Original speech worker timed out")
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError("Original speech worker exited")
                self._buffer.extend(data)
                if len(self._buffer) > 16 * 1024 * 1024:
                    raise ValueError("Original speech worker response exceeded its bound")
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer = bytearray(rest)
        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(str(result["error"])[:400])
        return result

    def load(self) -> bool:
        try:
            settings = json.loads(self.manifest.read_text())
            root = Path(settings["model_directory"])
            reference = Path(settings["reference_audio"])
            if not self.python.is_file() or not root.is_absolute() or not reference.is_absolute():
                raise ValueError("Original speech requires prepared absolute local paths")
            if not (root / "model.safetensors").is_file():
                raise ValueError("Original speech model weights are absent")
            if hashlib.sha256(reference.read_bytes()).hexdigest() != settings["reference_sha256"]:
                raise ValueError("Original voice reference changed")
            worker = Path(__file__).resolve().parents[3] / "tools/voice/original_speech_worker.py"
            environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            self.process = subprocess.Popen(
                [str(self.python), "-u", str(worker), str(self.manifest)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=0, env=environment,
            )
            if self._receive(90).get("ready") is not True:
                raise RuntimeError("Original speech worker did not become ready")
            return True
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Original speech unavailable: %s", exc)
            self.close()
            return False

    def synthesize(self, text: str, spec: Any) -> tuple[np.ndarray, int]:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Original speech worker is not loaded")
        request = json.dumps({"text": text, "speed": float(spec.speed)}) + "\n"
        self.process.stdin.write(request.encode())
        result = self._receive(45)
        audio = np.frombuffer(base64.b64decode(result["audio"], validate=True), dtype="<f4").copy()
        if not np.isfinite(audio).all():
            raise ValueError("Original speech returned nonfinite audio")
        return audio, int(result["sample_rate"])

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()


class SpeechWorkerLease:
    """Reap the worker before its admitted memory can be reused."""

    def __init__(self, lease: Any, engine: OriginalSpeech) -> None:
        self.lease, self.engine = lease, engine

    def set_preemptible(self, value: bool) -> bool:
        return self.lease.set_preemptible(value)

    def release(self, **kwargs: Any) -> bool:
        self.engine.close()
        return self.lease.release(**kwargs)
