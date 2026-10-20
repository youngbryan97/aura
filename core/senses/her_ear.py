"""Her ear on what one application plays: its sound as it plays, heard and written down as words, in real time.

When she watches something with a person on their screen, the video plays in their own browser, and she hears it as
they do: from the computer's sound, not from a page's captions. ScreenCaptureKit gives the sound of one application
alone (SCContentFilter including only it), so she hears the video and not her own voice, which other processes play,
nor any other application. Parakeet-TDT 0.6B (NVIDIA; mlx-community's MLX weights, already on this machine) writes
it down as it comes, streaming: each stretch of words is written once the model has settled it, with when it began.

Run as its own process, so that its model may use the GPU, which her app's main process keeps MLX off by design
(core/runtime/desktop_boot_safety.py):

    python -m core.senses.her_ear --app com.google.Chrome     JSON lines on stdout: {"at": s, "words": "..."}

``HerEar`` starts it and reads what it hears. Nothing is recorded to disk; the sound is dropped once written down.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.HerEar")

__all__ = ["HerEar", "SAMPLE_RATE", "ASR_MODEL"]

SAMPLE_RATE = 16_000
ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
#: How much sound is handed to the model at a time, in seconds: its streaming step.
STEP_S = 1.0
#: The model's attention context in encoder frames (80 ms each), left and right. Words are settled once their right
#: context has been heard: the default 256 is 20 seconds behind what plays, and nothing of a short clip is settled at
#: all (19 October 2026, a 7.5-second sentence came back empty). 32 is 2.6 seconds behind.
CONTEXT = (256, 32)


def _capture(bundle: str, on_audio: Any) -> Any:
    """Start ScreenCaptureKit on the sound of the application ``bundle``; ``on_audio`` gets float32 mono arrays."""
    import CoreMedia
    import dispatch
    import numpy as np
    import objc
    import ScreenCaptureKit
    from Foundation import NSObject

    found: dict[str, Any] = {}
    ready = threading.Event()

    def have(content: Any, error: Any) -> None:
        found["content"], found["error"] = content, error
        ready.set()

    ScreenCaptureKit.SCShareableContent.getShareableContentWithCompletionHandler_(have)
    if not ready.wait(10) or found.get("content") is None:
        raise RuntimeError(f"what is on the screen could not be read: {found.get('error')}")
    content = found["content"]
    apps = [a for a in content.applications() if str(a.bundleIdentifier() or "") == bundle]
    if not apps:
        raise RuntimeError(f"{bundle} is not running")
    sound_of = ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_includingApplications_exceptingWindows_(
        content.displays()[0], apps, [])
    how = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
    how.setCapturesAudio_(True)
    how.setSampleRate_(SAMPLE_RATE)
    how.setChannelCount_(1)
    how.setExcludesCurrentProcessAudio_(True)
    # The picture is not wanted here; the smallest the stream allows, once a second.
    how.setWidth_(2)
    how.setHeight_(2)
    how.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, 1))

    output = objc.protocolNamed("SCStreamOutput")

    class Listening(NSObject, protocols=[output]):
        def stream_didOutputSampleBuffer_ofType_(self, _stream: Any, buffer: Any, kind: Any) -> None:  # noqa: N802 - its selector
            if kind != ScreenCaptureKit.SCStreamOutputTypeAudio:
                return
            block = CoreMedia.CMSampleBufferGetDataBuffer(buffer)
            if block is None:
                return
            size = CoreMedia.CMBlockBufferGetDataLength(block)
            status, data = CoreMedia.CMBlockBufferCopyDataBytes(block, 0, size, None)
            if status == 0 and data:
                on_audio(np.frombuffer(bytes(data), dtype=np.float32).copy())

    stream = ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(sound_of, how, None)
    listening = Listening.alloc().init()
    queue = dispatch.dispatch_queue_create(b"aura.her_ear", None)
    added, why = stream.addStreamOutput_type_sampleHandlerQueue_error_(listening, ScreenCaptureKit.SCStreamOutputTypeAudio, queue, None)
    if not added:
        raise RuntimeError(f"the sound could not be listened to: {why}")
    started: dict[str, Any] = {}
    began = threading.Event()

    def on_start(error: Any) -> None:
        started["error"] = error
        began.set()

    stream.startCaptureWithCompletionHandler_(on_start)
    if not began.wait(10) or started.get("error") is not None:
        raise RuntimeError(f"the sound could not be listened to: {started.get('error')}")
    return stream, listening  # both held, or the stream's delegate is collected


def _asr_weights() -> str:
    """Where the recognition model's weights are on this machine, else its name to be fetched."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(ASR_MODEL, local_files_only=True)
    except (ImportError, OSError, ValueError):
        return ASR_MODEL


def listen(bundle: str, *, out: Any = sys.stdout, at_most_s: float | None = None, weights: str = ASR_MODEL) -> None:
    """Hear the application's sound and write each settled stretch of words as a JSON line until stopped."""
    import mlx.core as mx
    import numpy as np
    from parakeet_mlx import from_pretrained

    model = from_pretrained(weights)
    pending: list[Any] = []
    lock = threading.Lock()

    def on_audio(samples: Any) -> None:
        with lock:
            pending.append(samples)

    held = _capture(bundle, on_audio)
    began = time.monotonic()
    written = 0
    print(json.dumps({"listening": bundle}), file=out, flush=True)
    with model.transcribe_stream(context_size=CONTEXT) as heard:
        while at_most_s is None or time.monotonic() - began < at_most_s:
            time.sleep(STEP_S)
            with lock:
                chunk = np.concatenate(pending) if pending else None
                pending.clear()
            if chunk is None or not len(chunk):
                continue
            heard.add_audio(mx.array(chunk))
            settled = heard.finalized_tokens[written:]
            if settled:
                written += len(settled)
                words = "".join(token.text for token in settled).strip()
                if words:
                    print(json.dumps({"at": round(settled[0].start, 2), "words": words}), file=out, flush=True)
    del held


@dataclass
class HerEar:
    """Her ear as a process: started on an application, read for what was heard since last asked."""

    bundle: str
    process: Any = None
    heard: list[dict[str, Any]] = field(default_factory=list)
    listening: bool = False
    why_not: str = ""
    _reader: Any = None
    _answered: Any = None

    async def start(self, *, ready_s: float = 60.0) -> bool:
        """Start listening; True once the ear reports it hears the application."""
        from core.governance_context import local_internal_governed_scope
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        try:
            # Its own receipt: hearing one application of the person's, at their asking (core/skills/watching_the_screen.py).
            with local_internal_governed_scope("core.senses.her_ear", domain="environment_action",
                                               constraints={"effect": "the sound of one application, heard",
                                                            "app": self.bundle, "user_visible": False}):
                # A model process, declared as one with its weights named, so her model lanes count its memory.
                self.process = await get_subprocess_gateway().spawn_async(
                    [sys.executable, "-m", "core.senses.her_ear", "--app", self.bundle, "--model", _asr_weights()],
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, source="core.senses.her_ear",
                    accelerator_capability="model")
        except (OSError, RuntimeError, ValueError, AttributeError, TypeError) as why:
            self.why_not = f"the ear did not start: {why}"
            return False
        self._answered = asyncio.Event()
        self._reader = asyncio.ensure_future(self._read())
        try:
            await asyncio.wait_for(self._answered.wait(), timeout=ready_s)
        except TimeoutError:
            pass
        if not self.listening:
            self.why_not = self.why_not or "the ear did not hear the application"
        return self.listening

    async def _read(self) -> None:
        stream = getattr(self.process, "stdout", None)
        while stream is not None:
            line = await stream.readline()
            if not line:
                break
            try:
                said = json.loads(line)
            except ValueError:
                continue
            if "listening" in said or "error" in said:
                self.listening = "listening" in said
                self.why_not = str(said.get("error") or "")
                self._answered.set()
            elif said.get("words"):
                self.heard.append(said)
        self._answered.set()

    def since(self) -> list[dict[str, Any]]:
        """What was heard since the last time this was asked."""
        came, self.heard = self.heard, []
        return came

    async def stop(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except TimeoutError:
                self.process.kill()
        if self._reader is not None:
            self._reader.cancel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True, help="the bundle identifier of the application to hear")
    parser.add_argument("--seconds", type=float, help="stop after this long (for a test)")
    parser.add_argument("--model", default=ASR_MODEL, help="the recognition model's weights: a folder or its name")
    args = parser.parse_args()
    try:
        listen(args.app, at_most_s=args.seconds, weights=args.model)
    except RuntimeError as why:
        print(json.dumps({"error": str(why)}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
