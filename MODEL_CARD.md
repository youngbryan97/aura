# Aura Model Card

Aura uses Apple MLX for local language model inference. The table lists
the configured roles. The local model path or promoted artifact record
selects the weights for a particular installation.
[Model registry](core/brain/llm/model_registry.py)

## Language and vision lanes

| Role | Source-configured model | What it does |
|---|---|---|
| Cortex | `Aura-Cortex` | Foreground conversation and reasoning |
| Deep | `Aura-Cortex` by default | Uses the Cortex unless a distinct specialist is configured and admitted |
| Vision | `Aura-Cortex` | Image-conditioned requests through the vision path |
| Brainstem | `Ternary-Bonsai-2-27B-mlx-2bit` | Background work, subject to lane admission |
| Reflex/fallback | `Qwen2.5-1.5B-Instruct-4bit` | Auxiliary fallback role; compatibility and admission are checked separately |

The published experiments used different artifacts. Check the recorded
model and steering or adapter identity when comparing results.
[Model roster](docs/MODEL_ROSTER.md) · [Hardware profiles](HARDWARE_PROFILES.md)

## Speech recognition

The duplex recognizer tries Parakeet in a separate process. It can fall
back to MLX Whisper or CPU Whisper if the preferred path is unavailable.
NVIDIA lists 25 European languages for its Parakeet v3 model. These are
upstream model capabilities. Aura's actual speech accuracy depends on
its installed adapter and recordings.
[Recognizer](core/voice/duplex/streaming_asr.py) ·
[NVIDIA model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

The earlier decode timing used a 12.4-second synthetic speech sample and
five warm runs. Its medians were 166 ms for Parakeet, 193 ms for Whisper
small.en, and 317 ms for Whisper large-v3-turbo. These timings cover one
recognition step. [Recorded comparison](docs/MODEL_ROSTER.md#speech-to-text)

## Embeddings

[The embedding implementation](core/memory/embedding_model.py) selects
`Qwen/Qwen3-Embedding-0.6B` with 384 output dimensions. The source distinguishes
its 32,768-token native window from a 1,024-token operational view and a
2,048-token padded-work budget. Overlapping views cover longer source text;
the system does not need to encode an entire long page in one large GPU call.
These are source defaults, not a new performance measurement.

## Report a model result

Record the checkpoint, adapter, revision, request budget, controls, and
uncertainty. Aura's artifact fingerprint identifies the metadata used
by the profiler. The experimental model and steering in the ISC report
study are documented with that run.
[Artifact profiler](core/brain/llm/model_artifact_profile.py) · [ISC results](docs/ISC_STATUS.md)
