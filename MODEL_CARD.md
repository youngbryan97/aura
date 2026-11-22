# Aura Model Card

<!-- reader-navigation:start -->
> **In plain English:** The language model supplies text generation. Steering changes selected internal calculations using a measured direction and a controlled strength.
>
> **Why it matters:** An internal state should affect the intended part of a response while preserving competence and remaining tied to the measured model and basis.
>
> **In everyday use:** A change in internal condition may alter how a reply is formed; a channel measurement must show the change rather than relying on the reply sounding different.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#models) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#models) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#steering) · [Document map](docs/DOCUMENTATION_INDEX.md)
> [Glossary](docs/GLOSSARY.md) · [Formulas with worked examples](docs/FORMULAS.md)
<!-- reader-navigation:end -->

Status: Guide · Configuration and capability wording reconciled 25 October 2026

Aura uses local model lanes for conversation, vision, background work and
speech. A lane is a role, not proof that another copy of the model is loaded.
The values below describe the source configuration. A running installation
can differ; its artifact identity and request receipts establish what served
a particular result.

## Language and vision lanes

| Role | Source-configured model | What it does |
|---|---|---|
| Cortex | `Aura-Cortex`, the role bound to the fused/persona 27B artifact | Foreground conversation and reasoning |
| Deep | `Aura-Cortex` by default | Uses the Cortex unless a distinct specialist is configured and admitted |
| Vision | `Aura-Cortex` | Image-conditioned requests through the vision path |
| Brainstem | `Ternary-Bonsai-2-27B-mlx-2bit` | Background work, subject to lane admission |
| Reflex/fallback | `Qwen2.5-1.5B-Instruct-4bit` | Auxiliary fallback role; compatibility and admission are checked separately |

The role defaults are in [LLMConfig](core/config.py) and
[the registry](core/brain/llm/model_registry.py). The active descriptor binds
an exact artifact; a model's folder name alone does not identify its weights,
quantization or capabilities. The [roster](docs/MODEL_ROSTER.md) records the
historical choices and their bounded measurements.

The Cortex is not restricted to text-only input: `vision_model` uses the same
role. Image support still requires a compatible artifact, correctly encoded
image input and an operating vision path. Speech recognition and speech
synthesis use separate adapters; sending an audio file to the text generator
is not a substitute for those adapters.

## Resource limits

The target host has 64 GB of unified memory. This does not authorise keeping
all lanes warm or starting a second full Cortex. The
[lane controller](core/runtime/model_lane_control.py) and
[admission calculation](core/brain/lane_admission.py) account for the host's
available budget. See [hardware profiles](HARDWARE_PROFILES.md).

Quantization and context capacity are artifact properties. This card does not
assign every Cortex build four-bit weights or a universal 32,768-token window.
The [artifact profiler](core/brain/llm/model_artifact_profile.py) and
[registry context reader](core/brain/llm/model_registry.py) read the configured
artifact. A native maximum context is distinct from a safe live request budget.

A distinct deep specialist also needs its comparative qualification certificate
and trust root. Downloading a 72B model or assigning `AURA_DEEP_MODEL` does not
establish permission or sufficient memory to serve it.

## Speech recognition

[The duplex implementation](core/voice/duplex/streaming_asr.py) prefers
Parakeet when installed, then supports Whisper fallbacks. The
[NVIDIA model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
describes v3 as multilingual, covering 25 European languages. Actual language
quality in Aura needs testing on the chosen adapter, recordings and hardware.

The earlier local timing comparison used 12.4 seconds of known speech and
five warm runs: Parakeet 166 ms, Whisper small.en 193 ms and large-v3-turbo
317 ms. These are decode timings on that sample, not conversational latency,
a general accuracy result or a guarantee under a loaded host. The
[roster](docs/MODEL_ROSTER.md#speech-to-text) retains the measurement scope.

## Embeddings

[The embedding implementation](core/memory/embedding_model.py) selects
`Qwen/Qwen3-Embedding-0.6B` with 384 output dimensions. The source distinguishes
its 32,768-token native window from a 1,024-token operational view and a
2,048-token padded-work budget. Overlapping views cover longer source text;
the system does not need to encode an entire long page in one large GPU call.
These are source defaults, not a new performance measurement.

The replaced MiniLM encoder's limit was 256 **tokens**, not 256 words. The
[original model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and the historical tokenizer comparison explain why word-sized chunks could
be truncated. The reported 77% loss concerned a measured 1,122-token example;
it was not a measurement of every document.

## Evidence and limitations

A model can confabulate, retrieve the wrong record or misread an image. A
safeguard or a checked input contract does not guarantee correctness. Model
training-data claims need the relevant model card and the local training
manifest; this project card does not certify that every foundation or persona
training source excluded personal data.

For a result, retain artifact identity, adapter/basis, source revision,
request budget, controls and uncertainty. A structural artifact fingerprint
commits the metadata it hashes; it is not a cryptographic digest of every
weight byte. [ISC status](docs/ISC_STATUS.md) distinguishes the experimental
27B report result from qualification of a desktop installation.
