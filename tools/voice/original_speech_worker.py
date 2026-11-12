"""Local PCM transport for a prepared reference voice in the speech environment."""
from __future__ import annotations

import base64
import contextlib
import json
import sys
import time
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


class _Requests:
    """Her worker's stdin as lines, readable without waiting: a stop can arrive while a sentence is being made."""

    def __init__(self, fd: int = 0) -> None:
        self._fd, self._buffer, self._closed = fd, bytearray(), False

    def _fill(self, timeout: float | None) -> None:
        import os
        import select

        if self._closed or not select.select([self._fd], [], [], timeout)[0]:
            return
        data = os.read(self._fd, 65536)
        if not data:
            self._closed = True
        self._buffer.extend(data)

    def _take(self) -> str | None:
        if b"\n" not in self._buffer:
            return None
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer = bytearray(rest)
        return line.decode("utf-8", "replace") + "\n"

    def __iter__(self):
        while True:
            line = self._take()
            while line is None and not self._closed:
                self._fill(None)
                line = self._take()
            if line is None:
                return
            yield line

    def stop_asked(self) -> bool:
        """A stop waiting in the pipe, taken from it; any other line stays for the main loop."""
        self._fill(0)
        if b"\n" not in self._buffer:
            return False
        line, _, rest = self._buffer.partition(b"\n")
        try:
            asked = json.loads(line).get("stop") is True
        except (ValueError, AttributeError):
            asked = False
        if asked:
            self._buffer = bytearray(rest)
        return asked


def main() -> None:
    output = sys.stdout
    requests = _Requests()

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

            # Her voice as it forms (her_speech_as_it_forms.py): VoxCPM2 only, whose decoder is causal.
            streams = settings.get("engine") == "voxcpm2" and int(model.sample_rate) % int(
                settings.get("output_sample_rate", model.sample_rate)) == 0
            if streams:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from her_speech_as_it_forms import (
                    Level,
                    next_body_steps,
                    prompt_patches,
                    receptive_patches,
                    speak_as_it_forms,
                )

                lookback = receptive_patches(model.audio_vae.decoder, model.patch_size)
                context = prompt_patches(model, options["prompt_audio"])
                level = Level(float(settings.get("output_rms", 0.)))
                timing = {"decode_s_per_patch": 0.0, "gen_s_per_patch": 0.0, "patches_per_char": 0.0}
                # Bryan, 11 November 2026: adaptive. Her full steps while her voice keeps ahead of the listener; the
                # manifest's busy steps for a sentence's body when it measured that it could not (her GPU shared with
                # her cortex writing a long reply). Without busy steps in the manifest, always full.
                full_steps = int(options["inference_timesteps"])
                busy_steps = settings.get("busy_inference_timesteps")
                if type(busy_steps) is not int or not 1 <= busy_steps < full_steps:
                    busy_steps = None
                steps = {"body": full_steps, "busy_per_full": None}

            def generate_as_it_forms(text, speed, deliver, behind=False):
                if seed is not None:
                    mx.random.seed(seed)
                sentence = speak_as_it_forms(
                    model, text, options, mx=mx, prompt_context=context, lookback=lookback,
                    out_rate=int(settings.get("output_sample_rate", model.sample_rate)), speed=speed, level=level,
                    decode_s_per_patch=timing["decode_s_per_patch"], gen_s_per_patch=timing["gen_s_per_patch"],
                    expected_patches=round(len(text) * timing["patches_per_char"]), send=deliver,
                    # A reply whose listener already waited between sentences: her busy steps for the rest of it,
                    # so her voice finishes each sooner and her thought gets the GPU back sooner (speech_hold.py).
                    body_steps=(busy_steps if behind else steps["body"]) if busy_steps else None,
                    onset_steps=full_steps, busy_steps=busy_steps,
                    should_stop=requests.stop_asked)
                timing["decode_s_per_patch"] = sentence.decode_s_per_patch
                timing["gen_s_per_patch"] = sentence.gen_s_per_patch
                if busy_steps and not sentence.stopped and sentence.patches > 2:
                    sentence.timing(time.perf_counter())
                    steps["body"] = next_body_steps(sentence.body_steps or steps["body"], full=full_steps, busy=busy_steps,
                                                    real_time=sentence.real_time, busy_per_full=steps["busy_per_full"])
                if text and sentence.patches and not sentence.stopped:
                    per_char = sentence.patches / len(text)  # her own pace of speech, sentence by sentence
                    known = timing["patches_per_char"]
                    timing["patches_per_char"] = per_char if not known else 0.7 * known + 0.3 * per_char
                return sentence

            mx.random.seed(73619)
            import librosa

            librosa.effects.time_stretch(np.zeros(4096, dtype=np.float32), rate=1.01)
            generate("I'm here.")
            if streams:
                # Her first streamed sentence learns her decode cost and loudness before anyone hears one; with busy
                # steps, one at each step count measures what the busy steps cost against the full.
                paces = {}
                for body in ((full_steps, busy_steps) if busy_steps else (full_steps,)):
                    steps["body"] = body
                    paces[body] = generate_as_it_forms("I'm here, and I can hear you.", 1.0,
                                                       lambda audio, more, timing=None: None).gen_s_per_patch
                if busy_steps and paces.get(full_steps):
                    steps["busy_per_full"] = paces[busy_steps] / paces[full_steps]
                steps["body"] = full_steps
        send({"ready": True, "streams": bool(streams)})
        for line in requests:
            try:
                if len(line) > 32768:
                    raise ValueError("Speech request too large")
                message = json.loads(line)
                if message.get("stop") is True:
                    continue  # a stop that came after its sentence had ended: nothing to answer
                if message.get("stream") is True and streams:
                    rate = int(settings.get("output_sample_rate", model.sample_rate))

                    def deliver(audio, more, timing=None, rate=rate):
                        message = {"audio": base64.b64encode(np.asarray(audio, dtype='<f4').tobytes()).decode(),
                                   "sample_rate": rate, "more": bool(more)}
                        if timing:
                            message["timing"] = timing
                        send(message)

                    with contextlib.redirect_stdout(sys.stderr):
                        generate_as_it_forms(str(message["text"]), float(message.get("speed", 1.)), deliver,
                                             behind=message.get("behind") is True)
                    continue
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
