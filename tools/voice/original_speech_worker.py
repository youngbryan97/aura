"""Local PCM transport for a prepared reference voice in the speech environment."""
from __future__ import annotations

import base64
import contextlib
import json
import sys
from pathlib import Path


def generation_options(settings: dict) -> dict:
    """Use the conditioning API that produced the prepared audition."""
    engine = settings.get("engine", "qwen_original")
    if engine == "voxcpm2":
        steps = settings.get("inference_timesteps", 10)
        if type(steps) is not int or not 1 <= steps <= 50:
            raise ValueError("Speech diffusion steps must be an integer from 1 to 50")
        cfg = float(settings.get("cfg_value", 2.))
        if not 2. <= cfg <= 5.:
            raise ValueError("Speech guidance must be between 2 and 5")
        return {
            "prompt_audio": settings["reference_audio"],
            "prompt_text": settings["reference_text"].rstrip() + " ",
            "inference_timesteps": steps, "cfg_value": cfg, "max_tokens": 300,
        }
    if engine != "qwen_original":
        raise ValueError("Unsupported prepared speech engine")
    temperature = float(settings.get("temperature", .7))
    if not .05 <= temperature <= 1.2:
        raise ValueError("Speech temperature must be between 0.05 and 1.2")
    streaming = settings.get("decoder_streaming", False)
    if type(streaming) is not bool:
        raise ValueError("Speech decoder_streaming must be a boolean")
    return {
        "ref_audio": settings["reference_audio"], "ref_text": settings["reference_text"],
        "lang_code": "English", "temperature": temperature, "max_tokens": 500,
        "stream": streaming, "streaming_interval": .4,
    }


def prepare_model(model, settings: dict, mx, nn) -> None:
    """Keep acoustic weights intact while reducing the language-layer cost."""
    bits = settings.get("language_model_bits")
    compile_solver = settings.get("compile_solver", False)
    if type(compile_solver) is not bool:
        raise ValueError("Speech compile_solver must be a boolean")
    if bits is not None and (type(bits) is not int or bits != 8):
        raise ValueError("Prepared speech supports only 8-bit language layers")
    if (bits is not None or compile_solver) and settings.get("engine") != "voxcpm2":
        raise ValueError("Language-layer optimization requires VoxCPM2")
    if compile_solver:
        model.feat_decoder.solve_euler = mx.compile(model.feat_decoder.solve_euler)
    if bits is not None:
        nn.quantize(model.base_lm, group_size=64, bits=bits)
        nn.quantize(model.residual_lm, group_size=64, bits=bits)
        mx.eval(model.parameters())
        mx.clear_cache()


def main() -> None:
    output = sys.stdout

    def send(message):
        output.write(json.dumps(message) + "\n")
        output.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            import mlx.nn as nn
            import numpy as np
            from mlx_audio.tts.utils import load_model

            settings = json.loads(Path(sys.argv[1]).read_text())
            options = generation_options(settings)
            seed = settings.get("seed")
            if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 32):
                raise ValueError("Speech seed must be an unsigned 32-bit integer")
            mx.set_memory_limit((10 if settings.get("engine") == "voxcpm2" else 8) * 2 ** 30)
            mx.set_cache_limit(128 * 2 ** 20)
            model = load_model(settings["model_directory"])
            prepare_model(model, settings, mx, nn)

            def generate(text, speed=1.0):
                if seed is not None:
                    # A prepared audition must survive warm-up and preceding
                    # utterances without choosing another random delivery.
                    mx.random.seed(seed)
                chunks = []
                for result in model.generate(text, **options):
                    mx.eval(result.audio)
                    chunks.append(np.asarray(result.audio))
                if not chunks:
                    raise RuntimeError("Original speech produced no audio")
                audio = np.concatenate(chunks).astype(np.float32)
                rate = int(model.sample_rate)
                output_rate = settings.get("output_sample_rate", rate)
                if type(output_rate) is not int or output_rate not in {24000, 48000}:
                    raise ValueError("Speech output rate must be 24000 or 48000 Hz")
                if output_rate != rate:
                    import math
                    from scipy.signal import resample_poly

                    divisor = math.gcd(rate, output_rate)
                    audio = resample_poly(audio, output_rate // divisor, rate // divisor).astype(np.float32)
                    rate = output_rate
                # Match audition loudness without changing its register or pace.
                target = float(settings.get("output_rms", 0.))
                if target:
                    if not 0.01 <= target <= .2:
                        raise ValueError("Speech output RMS must be between 0.01 and 0.2")
                    rms = float(np.sqrt(np.mean(audio ** 2)))
                    peak = float(np.max(np.abs(audio)))
                    audio *= min(target / max(rms, 1e-8), .85 / max(peak, 1e-8))
                if abs(speed - 1.) > .01:
                    import librosa

                    audio = librosa.effects.time_stretch(audio, rate=max(.85, min(1.15, speed)))
                return audio, rate

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
                    audio, rate = generate(str(message["text"]), float(message.get("speed", 1.)))
                send({"audio": base64.b64encode(audio.astype('<f4').tobytes()).decode(),
                      "sample_rate": rate})
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                send({"error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        send({"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
