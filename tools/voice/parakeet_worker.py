"""Her voice mode's speech recognizer in a process of its own: Parakeet on MLX, one decode at a time over stdin/stdout.

LIVE 11 November 2026: her runtime died of a segmentation fault twice in one day (about 01:12 and at 09:49), each time
with a Parakeet decode inside ``mx.eval`` on an executor thread of her main process, beside the other code there that
uses MLX. Here MLX runs on this process's one thread and nothing else (core/voice/duplex/parakeet_process.py).

    python -u tools/voice/parakeet_worker.py <repo> [<repo> ...]

Each request line is {"audio": base64 little-endian float32 at 16 kHz, "repo": ...}; each answer is
{"text": ..., "tokens": [[text, start_s, end_s], ...]} or {"error": ...}. The repos named on the command line are
loaded and run once on a second of silence before {"ready": true}, so weights and kernels are never paid inside a turn.

MLX keeps the buffers a decode frees, for reuse, up to its cache limit, and with none set that is most of the
machine. LIVE 14 November 2026 this process held 34 GB after warming a 0.6B model, and her 27B cortex was refused a
worker ("model_load_headroom 21.2GB < required 24.0GB"). The cache is held to the limit her own MLX workers use
(core/runtime/desktop_boot_safety.compute_mlx_cache_limit), and what it holds after warming is sent with its ready.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    output = sys.stdout

    def send(message: dict) -> None:
        output.write(json.dumps(message) + "\n")
        output.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            import numpy as np
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel

            from core.runtime.desktop_boot_safety import compute_mlx_cache_limit

            mx.set_cache_limit(compute_mlx_cache_limit(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")))

            models: dict[str, object] = {}

            def decode(audio: np.ndarray, repo: str) -> tuple[str, list[list]]:
                model = models.get(repo)
                if model is None:
                    model = models[repo] = from_pretrained(repo)
                mel = get_logmel(mx.array(np.asarray(audio, dtype=np.float32)), model.preprocessor_config)
                results = model.generate(mel)
                result = results[0] if results else None
                if result is None:
                    return "", []
                tokens = [[str(getattr(token, "text", "") or ""), float(getattr(token, "start", 0.0) or 0.0),
                           float(getattr(token, "end", 0.0) or 0.0)] for token in (getattr(result, "tokens", ()) or ())]
                return str(getattr(result, "text", "") or ""), tokens

            for repo in sys.argv[1:]:
                decode(np.zeros(16000, dtype=np.float32), repo)
            mx.clear_cache()
            held = {"active_gb": round(mx.get_active_memory() / float(1 << 30), 2),
                    "cached_gb": round(mx.get_cache_memory() / float(1 << 30), 2),
                    "peak_gb": round(mx.get_peak_memory() / float(1 << 30), 2)}
        send({"ready": True, "repos": sys.argv[1:], "memory": held})
        for line in sys.stdin:
            try:
                if len(line) > 16 * 1024 * 1024:
                    raise ValueError("decode request too large")
                request = json.loads(line)
                audio = np.frombuffer(base64.b64decode(request["audio"], validate=True), dtype="<f4")
                if not np.isfinite(audio).all():
                    raise ValueError("audio is not finite")
                with contextlib.redirect_stdout(sys.stderr):
                    text, tokens = decode(audio, str(request["repo"]))
                send({"text": text, "tokens": tokens})
            except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
                send({"error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001 - the parent learns why its recognizer could not start, then this exits
        send({"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
