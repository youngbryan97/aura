"""Local PCM transport for a prepared reference voice in the speech environment."""
from __future__ import annotations

import base64
import contextlib
import json
import sys
from pathlib import Path


def main() -> None:
    output = sys.stdout

    def send(message):
        output.write(json.dumps(message) + "\n")
        output.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            import numpy as np
            from mlx_audio.tts.utils import load_model

            settings = json.loads(Path(sys.argv[1]).read_text())
            temperature = float(settings.get("temperature", .7))
            if not .05 <= temperature <= 1.2:
                raise ValueError("Speech temperature must be between 0.05 and 1.2")
            model = load_model(settings["model_directory"])

            def generate(text, speed=1.0):
                chunks = [np.asarray(r.audio) for r in model.generate(
                    text, ref_audio=settings["reference_audio"], ref_text=settings["reference_text"],
                    lang_code="English", temperature=temperature, max_tokens=500, stream=False,
                )]
                if not chunks:
                    raise RuntimeError("Original speech produced no audio")
                audio = np.concatenate(chunks).astype(np.float32)
                if abs(speed - 1.) > .01:
                    import librosa

                    audio = librosa.effects.time_stretch(audio, rate=max(.85, min(1.15, speed)))
                return audio

            mx.random.seed(73619)
            import librosa

            librosa.effects.time_stretch(np.zeros(4096, dtype=np.float32), rate=1.01)
            generate("I'm here.")
        send({"ready": True})
        for line in sys.stdin:
            try:
                if len(line) > 32768:
                    raise ValueError("Speech request too large")
                message = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    audio = generate(str(message["text"]), float(message.get("speed", 1.)))
                send({"audio": base64.b64encode(audio.astype('<f4').tobytes()).decode(),
                      "sample_rate": int(model.sample_rate)})
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                send({"error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        send({"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
