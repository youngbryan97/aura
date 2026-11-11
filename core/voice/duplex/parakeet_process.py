"""Her voice mode's speech recognizer, run in its own process (tools/voice/parakeet_worker.py).

LIVE 11 November 2026: her runtime died of a segmentation fault twice in one day, each time inside a Parakeet decode's
``mx.eval`` on an executor thread of her main process, beside the other code there that uses MLX. The recognizer now
holds its own process, as her voice does (original_speech.py): a fault there ends a decode, not her.

Calls block; the caller (streaming_asr's backend) makes them on a worker thread under its usage lock, one at a time.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import selectors
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger("Aura.Voice.AsrProcess")

_WORKER = Path(__file__).resolve().parents[3] / "tools/voice/parakeet_worker.py"
# A reply larger than this is not a transcript; the bound keeps a confused worker from filling her memory.
_REPLY_BOUND = 4 * 1024 * 1024


class RecognizerRefusedError(RuntimeError):
    """The process answered with an error for this decode; the pipe is still in step, so it keeps running."""


class ParakeetProcess:
    """One recognizer process: started once, asked one decode at a time, closed on any failure."""

    def __init__(self, repos: Sequence[str], *, python: str | None = None, worker: Path | None = None) -> None:
        self.repos = tuple(dict.fromkeys(str(repo) for repo in repos if repo))
        self.python = python or sys.executable
        self.worker = worker or _WORKER
        self.process: subprocess.Popen | None = None
        self._buffer = bytearray()

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, timeout_s: float = 180.0) -> None:
        """Start it and wait until its models are loaded and warm; raise when it cannot be."""
        if self.alive:
            return
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        self.process = subprocess.Popen(
            [self.python, "-u", str(self.worker), *self.repos],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0, env=environment,
        )
        try:
            if self._receive(timeout_s).get("ready") is not True:
                raise RuntimeError("her recognizer process did not become ready")
        except (OSError, RuntimeError, ValueError, TimeoutError):
            self.close()
            raise
        logger.info("👂 her recognizer runs in its own process (pid %s): %s", self.process.pid, ", ".join(self.repos))

    def decode(self, audio: np.ndarray, repo: str, *, timeout_s: float = 30.0) -> tuple[str, tuple[tuple[str, float, float], ...]]:
        """Text and timed tokens for ``audio`` (float32 at 16 kHz). A failure closes the process: a reply that came
        late must never be read as the next decode's."""
        if not self.alive or self.process is None or self.process.stdin is None:
            raise RuntimeError("her recognizer process is not running")
        samples = np.ascontiguousarray(np.asarray(audio, dtype="<f4"))
        request = json.dumps({"audio": base64.b64encode(samples.tobytes()).decode(), "repo": str(repo)}) + "\n"
        try:
            self.process.stdin.write(request.encode())
            reply = self._receive(timeout_s)
        except RecognizerRefusedError:
            raise
        except (OSError, RuntimeError, ValueError, TimeoutError):
            self.close()
            raise
        tokens = tuple((str(text), float(start), float(end)) for text, start, end in reply.get("tokens") or ())
        return str(reply.get("text") or ""), tokens

    def _receive(self, timeout_s: float) -> dict:
        assert self.process is not None and self.process.stdout is not None
        deadline = time.monotonic() + timeout_s
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("her recognizer process did not answer in time")
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError("her recognizer process exited")
                self._buffer.extend(data)
                if len(self._buffer) > _REPLY_BOUND:
                    raise ValueError("her recognizer's reply exceeded its bound")
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer = bytearray(rest)
        reply = json.loads(line)
        if reply.get("error"):
            raise RecognizerRefusedError(str(reply["error"])[:400])
        return reply

    def close(self) -> None:
        process, self.process = self.process, None
        self._buffer = bytearray()
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
