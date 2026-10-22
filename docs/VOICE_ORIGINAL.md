# Aura's original voice

Aura Velvet starts with an original synthetic studio recording. The design
model received a description of register, texture, pacing and articulation.
It received no actor name or actor recording. The generated recording then
conditions the local Qwen3-TTS Base model so subsequent sentences can reuse
that voice. Its provenance and reference hash live beside the prepared WAV.

The speech worker uses its own Python environment. This avoids replacing
MLX or Transformers underneath the resident Cortex. Install
`requirements/voice-original.txt` there. Model preparation downloads weights;
voice calls use absolute local paths with Hugging Face offline mode enabled.
The manifest supplies `model_directory`, `reference_audio`,
`reference_sha256` and `reference_text`.

The desktop settings for this installation are:

```
AURA_VOICE_TTS_ENGINE=qwen_original
AURA_VOICE_ORIGINAL_PYTHON=/Users/bryan/.aura/voice-lab/qwen-env/bin/python
AURA_VOICE_ORIGINAL_MANIFEST=/Users/bryan/.aura/live-source/data/voice_models/aura-velvet/manifest.json
AURA_VOICE_TREATMENT=veil
```

`clean` removes the effect. `veil` adds a light enclosure resonance;
`veil_deep` increases it. The treatment retains the sample count and does
not shift pitch. This is audio processing, independent of the speaker model.

The runtime reserves 8 GB before starting the speech worker. The measured
full-precision candidate peaked at 5.70 GB of MLX memory. The worker is reaped
before that reservation is released, including shutdown during a model load.
If the original engine cannot load, the existing Kokoro fallback remains.

Current delivery sends complete clauses. The native model's streaming
first-packet measurement is not the desktop's time to first audio. The
production adapter took 2.54 and 2.63 seconds to synthesize the two recorded
validation clauses. Voice preference is subjective; these measurements do
not establish that listeners prefer it or that conversational latency matches
ChatGPT Voice.

The chat gradient uses a 160 by 120 colour field with a 500 ms refresh
interval. Lean mode reduces that to 2000 ms. Hidden pages and focused voice
mode stop the timer; reduced-motion settings retain a still image. It uses
no blur filter and no animation at display refresh rate.
