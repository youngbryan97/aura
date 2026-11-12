"""Own a prepared reference voice in an isolated local MLX speech worker."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import selectors
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.Voice.Original")


class OriginalSpeech:
    name = "qwen_original"

    def __init__(self, manifest: Path, python: Path, engine: str = "qwen_original") -> None:
        self.manifest = manifest
        self.python = python
        self.name = engine
        self.process: subprocess.Popen | None = None
        self.voices = frozenset({"aura_velvet"})
        self.streams = False  # her worker sends a sentence in pieces as it forms (her_speech_as_it_forms.py)
        self._buffer = bytearray()
        self._lock = checked_lock("voice.tts.reference_worker", rank=LockRank.LEAF)
        self._loaded_once = self._retired = False

    def _worker(self) -> subprocess.Popen:
        """Her worker for the next request, started again if it died or was stopped since it was loaded.

        A worker stopped by a timeout or killed under memory pressure left her mute until her voice mode was closed
        and opened again; now the next sentence waits for a new one. Never before her first load, never after close.
        """
        process = self.process
        if process is not None and process.stdin is not None and process.poll() is None:
            return process
        if not self._loaded_once or self._retired:
            raise RuntimeError("Original speech worker is not loaded")
        self.close(for_good=False)
        started = time.monotonic()
        logger.warning("🗣️ her voice worker stopped mid-session; starting a new one")
        if not self.load() or self.process is None:
            raise RuntimeError("Original speech worker is not loaded and did not start again")
        logger.info("🗣️ her voice worker is back after %.1f s", time.monotonic() - started)
        return self.process

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
            if self.name not in {"qwen_original", "voxcpm2"} or settings.get(
                "engine", "qwen_original",
            ) != self.name:
                raise ValueError("Prepared voice engine does not match the selected engine")
            root = Path(settings["model_directory"])
            reference = Path(settings["reference_audio"])
            voice_id = settings.get("voice_id", "aura_velvet")
            if not isinstance(voice_id, str) or not re.fullmatch(
                r"[a-z][a-z0-9]{0,23}_[a-z0-9][a-z0-9_]{0,39}", voice_id,
            ):
                raise ValueError("Prepared voice identifier is invalid")
            if not self.python.is_file() or not root.is_absolute() or not reference.is_absolute():
                raise ValueError("Original speech requires prepared absolute local paths")
            index = root / "model.safetensors.index.json"
            if index.is_file():
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if not shards or any(
                    not isinstance(shard, str) or Path(shard).name != shard
                    or not shard.endswith(".safetensors") or not (root / shard).is_file()
                    for shard in shards
                ):
                    raise ValueError("Prepared speech model shards are absent or invalid")
            elif not (root / "model.safetensors").is_file():
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
            ready = self._receive(90)
            if ready.get("ready") is not True:
                raise RuntimeError("Original speech worker did not become ready")
            self.streams = ready.get("streams") is True
            self.voices = frozenset({voice_id})
            self._loaded_once = True
            return True
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Original speech unavailable: %s", exc)
            self.close(for_good=False)
            return False

    def synthesize(self, text: str, spec: Any) -> tuple[np.ndarray, int]:
        # Backchannels and replies share this pipe. A request must receive its
        # own audio before the next request can use the worker.
        with self._lock:
            process = self._worker()
            request = json.dumps({"text": text, "speed": float(spec.speed)}) + "\n"
            try:
                process.stdin.write(request.encode())
                result = self._receive(45)
            except (OSError, RuntimeError, ValueError):
                # A timed-out response cannot become the next sentence.
                self.close(for_good=False)
                raise
        audio = np.frombuffer(base64.b64decode(result["audio"], validate=True), dtype="<f4").copy()
        if not np.isfinite(audio).all():
            raise ValueError("Original speech returned nonfinite audio")
        return audio, int(result["sample_rate"])

    def synthesize_pieces(self, text: str, spec: Any, deliver: Any, stopped: Any = None, behind: bool = False) -> None:
        """Hand ``deliver(samples, rate, more)`` each piece of the sentence as her worker sends it.

        The pipe is held for the whole sentence, as for ``synthesize``: its pieces must all arrive before another
        request can use the worker. Each piece has the per-response bound, so a worker that stops mid-sentence is
        closed rather than left to send its tail into the next sentence. When ``stopped()`` turns true (she was cut
        off) the worker is told to stop making the sentence, and its last message is still read.
        """
        with self._lock:
            process = self._worker()
            request = json.dumps({"text": text, "speed": float(spec.speed), "stream": True,
                                  "behind": bool(behind)}) + "\n"
            try:
                process.stdin.write(request.encode())
                more, asked_to_stop = True, False
                while more:
                    if not asked_to_stop and stopped is not None and stopped():
                        process.stdin.write(b'{"stop": true}\n')
                        asked_to_stop = True
                    result = self._receive(45)
                    more = result.get("more") is True
                    if isinstance(result.get("timing"), dict):
                        logger.info("🗣️ her voice made a sentence: %s", result["timing"])  # real_time < 1: it ran late
                    audio = np.frombuffer(base64.b64decode(result["audio"], validate=True), dtype="<f4").copy()
                    if not np.isfinite(audio).all():
                        raise ValueError("Original speech returned nonfinite audio")
                    deliver(audio, int(result["sample_rate"]), more)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                self.close(for_good=False)
                raise

    def close(self, *, for_good: bool = True) -> None:
        self._retired = self._retired or for_good
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
