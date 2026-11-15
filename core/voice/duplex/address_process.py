"""Her address reader's process (tools/voice/address_worker.py): started once, asked one reading at a time.

The reader is a small chat model that models both sides of a conversation, which her cortex no longer does (see the
worker's docstring). It runs in a process of its own for the reason her recognizer does (parakeet_process.py): MLX on a
thread of her main process beside other MLX code ended her runtime twice on 11 November 2026.
"""
from __future__ import annotations

import json
import logging
import os
import selectors
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Voice.AddressProcess")

_WORKER = Path(__file__).resolve().parents[3] / "tools/voice/address_worker.py"
# A reply larger than this is not a list of token log-probabilities; the bound keeps a confused worker from filling
# her memory.
_REPLY_BOUND = 4 * 1024 * 1024

#: The model her address reader reads with, by name under her models directory.
ADDRESS_MODEL = "Qwen2.5-1.5B-Instruct-4bit"


def address_model_path() -> Path:
    from core.brain.llm.model_registry import get_models_dir

    return Path(get_models_dir()) / ADDRESS_MODEL


class AddressReaderRefusedError(RuntimeError):
    """The process answered with an error for this reading; the pipe is still in step, so it keeps running."""


class AddressProcess:
    """One reader process: started once, asked one reading at a time, closed on any failure."""

    def __init__(self, model: str | Path | None = None, *, python: str | None = None,
                 worker: Path | None = None) -> None:
        self.model = str(model or address_model_path())
        self.python = python or sys.executable
        self.worker = worker or _WORKER
        self.process: subprocess.Popen | None = None
        self._buffer = bytearray()

    @property
    def name(self) -> str:
        return Path(self.model).name

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, timeout_s: float = 120.0) -> None:
        """Start it and wait until its model is loaded and warm; raise when it cannot be."""
        if self.alive:
            return
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        self.process = subprocess.Popen(
            [self.python, "-u", str(self.worker), self.model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0, env=environment,
        )
        try:
            ready = self._receive(timeout_s)
            if ready.get("ready") is not True:
                raise RuntimeError("her address reader did not become ready")
        except (OSError, RuntimeError, ValueError, TimeoutError):
            self.close()
            raise
        logger.info("👂 her address reader runs in its own process (pid %s): %s, holding %s", self.process.pid,
                    self.name, ready.get("memory") or "an unreported amount")

    def read(self, text: str, words: str, scope: str, *, timeout_s: float = 30.0) -> list[float]:
        """The log-probability of each token of ``words`` after ``text``. A failure closes the process: a reply that
        came late must never be read as the next reading's."""
        if not self.alive or self.process is None or self.process.stdin is None:
            raise RuntimeError("her address reader is not running")
        request = json.dumps({"text": str(text), "words": str(words), "scope": str(scope)}) + "\n"
        try:
            self.process.stdin.write(request.encode())
            reply = self._receive(timeout_s)
        except AddressReaderRefusedError:
            raise
        except (OSError, RuntimeError, ValueError, TimeoutError):
            self.close()
            raise
        return [float(v) for v in reply.get("logprobs") or ()]

    def _receive(self, timeout_s: float) -> dict:
        assert self.process is not None and self.process.stdout is not None
        deadline = time.monotonic() + timeout_s
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("her address reader did not answer in time")
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError("her address reader exited")
                self._buffer.extend(data)
                if len(self._buffer) > _REPLY_BOUND:
                    raise ValueError("her address reader's reply exceeded its bound")
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer = bytearray(rest)
        reply = json.loads(line)
        if reply.get("error"):
            raise AddressReaderRefusedError(str(reply["error"])[:400])
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


_PROCESS: AddressProcess | None = None
_LOCK = checked_lock("voice.address_process")


def read_in_her_address_process(text: str, words: str, scope: str) -> Sequence[float] | None:
    """One reading from her address reader, started on first use; None when it cannot be had. Blocking: call it off
    the event loop. Readings are taken one at a time."""
    global _PROCESS
    with _LOCK:
        try:
            if _PROCESS is None:
                _PROCESS = AddressProcess()
            if not _PROCESS.alive:
                _PROCESS.start()
            return _PROCESS.read(text, words, scope)
        except AddressReaderRefusedError as exc:
            logger.info("her address reader refused a reading: %s", exc)
            return None
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation("voice_duplex.address_process", exc,
                               action="judged address without her address reader's reading", severity="warning")
            return None


def warm_her_address_process() -> bool:
    """Start her address reader if it is not running, so the first utterance of a session does not pay its load.
    Blocking: call it off the event loop."""
    global _PROCESS
    with _LOCK:
        try:
            if _PROCESS is None:
                _PROCESS = AddressProcess()
            _PROCESS.start()
            return True
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation("voice_duplex.address_process", exc,
                               action="her voice session judges address without her address reader", severity="warning")
            return False


def her_address_reader_name() -> str:
    return Path(str(address_model_path())).name


def close_her_address_process() -> None:
    global _PROCESS
    with _LOCK:
        if _PROCESS is not None:
            _PROCESS.close()
            _PROCESS = None


__all__ = [
    "ADDRESS_MODEL",
    "AddressProcess",
    "AddressReaderRefusedError",
    "address_model_path",
    "close_her_address_process",
    "her_address_reader_name",
    "read_in_her_address_process",
    "warm_her_address_process",
]
