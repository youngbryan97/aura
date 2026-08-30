# Aura Model Card

*Lane defaults below are `core/config.py:LLMConfig` (verified 2026-08-01).
Override any of them with `AURA_MODEL`, `AURA_DEEP_MODEL`,
`AURA_BRAINSTEM_MODEL`, `AURA_FALLBACK_MODEL`, or the nested
`AURA_LLM__*` form.*

## Primary Model (Cortex)

| Field | Value |
|-------|-------|
| **Role** | Primary reasoning and conversation |
| **Architecture** | Transformer LLM (27B parameters, fused Qwen3.8-27B / `Aura-Cortex`, migrated from historical 32B) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | MLX fused native weights (`training/fused-model/active.json`); legacy 8-bit/4-bit profiles supported |
| **Context Window** | 8192–262,144 tokens (configurable) |
| **Inference** | Local, on-device |
| **Fine-tuning** | Promoted fused LoRA delta (`training/fused-model/active.json`) as the live Cortex without a re-quantize |

### Intended Use
Primary model for all user-facing conversation, reasoning, tool planning,
and complex cognitive tasks.

### Limitations
- May confabulate when knowledge is insufficient
- Context window limits multi-turn reasoning depth
- 4-bit quantization trades precision for memory efficiency
- Cannot process images or audio natively

### Ethical Considerations
- Model weights are publicly available base models
- No private/personal data in training
- Prompt injection mitigations applied at runtime layer

---

## Deep Model (Solver)

| Field | Value |
|-------|-------|
| **Role** | Deep-reasoning hot-swap tier for hard problems |
| **Architecture** | Transformer LLM (72B parameters, Qwen2.5-72B-Instruct) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Inference** | Local, on-device |

### Intended Use
Hot-swapped in for the deepest reasoning passes on 64GB-class desktops. It is
the highest-capacity local lane but the slowest (~84s/pass), so it is not the
default foreground model — the 32B Cortex handles standard turns and the Solver
is promoted only when a problem warrants it. Auto-detected/enabled via
`AURA_DEEP_MODEL`.

---

## Background Model (Brainstem)

| Field | Value |
|-------|-------|
| **Role** | Background maintenance, classification, lightweight tasks |
| **Architecture** | Transformer LLM (9B parameters, Qwen3.5-9B) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Context Window** | 4096 tokens |
| **Inference** | Local, on-device |
| **Reasoning mode** | Explicitly controlled |

Qwen3.5-9B replaced Qwen2.5-7B on 2026-08-12. Nothing is keyed to this tier's
weights — verified before the swap that no draft, contrastive-amateur, or
speculative-decoding path references the Brainstem — so the generation gap was
free to close. The Reflex lane below could *not* move for exactly that reason.
See [docs/MODEL_ROSTER.md](docs/MODEL_ROSTER.md).

### Intended Use
Background tasks: memory consolidation, classification, health probes,
maintenance reasoning. Never used for user-facing responses in production mode.

### Limitations
- Reduced reasoning capability compared to primary
- Not suitable for complex multi-step reasoning
- Background-only; foreground lane isolation prevents interference

---

## Reflex Model

| Field | Value |
|-------|-------|
| **Role** | Fast reflex lane: sub-second acknowledgements, routing, guards |
| **Architecture** | Transformer LLM (1.5B parameters, Qwen2.5-1.5B-Instruct) |
| **Runtime** | MLX on Apple Silicon |
| **Quantization** | 4-bit (MLX native) |
| **Inference** | Local, on-device |

### Intended Use
The lowest-latency local tier. Handles reflexive turns and lightweight
routing/guard decisions when the 32B Cortex is warming or contended, so the
conversation lane can answer immediately instead of waiting on the heavy
model. Never used for substantive full-mind replies.

---

## Speech-to-Text Model

| Field | Value |
|-------|-------|
| **Role** | Primary ASR; serves both duplex stages (480 ms partials and the final) |
| **Architecture** | Parakeet TDT 0.6B v3 (`parakeet_mlx`) |
| **Runtime** | MLX on Apple Silicon |
| **Inference** | Local, on-device. Audio does not leave the machine |
| **Fallback** | `faster_whisper` on CPU |

Replaced a two-stage Whisper configuration on 2026-08-12. Measured on this
host over 12.4 s of real speech, median of 5 warm runs: Parakeet **166 ms** vs
`whisper-small.en` 193 ms (the model it replaced on partials) vs
`whisper-large-v3-turbo` 317 ms (finals). One streaming decode is cheaper than
the incumbent partial, so both stages share one model-lane lease with no
accuracy sacrificed on partials.

### Limitations
- English-focused; the cited accuracy figure (6.32% vs 7.83% English WER) is
  from published benchmarks, not a local measurement — the local sample scored
  0% WER for all candidates and could not discriminate.

---

## Embedding Model

| Field | Value |
|-------|-------|
| **Role** | Semantic memory retrieval — the dense half of hybrid scoring |
| **Architecture** | `Qwen/Qwen3-Embedding-0.6B` |
| **Dimensions** | 384 |
| **Inference** | Local, on-device |

Replaced `all-MiniLM-L6-v2` on 2026-08-12. MiniLM declared a 256-token window
against an 800-word ingestion chunk — 1,122 tokens through its own tokenizer,
so **77% of every full chunk never reached the encoder**, silently, because
tokenizer truncation logs nothing. On documents whose distinguishing sentence
sits past token 256, MiniLM scored 1/4 on tail retrieval (chance) against
Qwen3's 3/4, at 10.7 vs 20.2 ms/query. Chunk size is now *derived* from the
encoder's declared window rather than fixed.

### Limitations
- Roughly 2× the per-query latency of the model it replaced.
- Signal and null score populations overlap on the calibration sample, so the
  admission threshold is calibrated against measured nulls
  (`core/memory/retrieval_calibration.py`) rather than asserted.

---

## Model Verification

Model identity is **measured from the artifact**, not inferred from its path.
`core/brain/llm/model_artifact_profile.py` exists because footprint,
minimum-headroom, deadline, cache-residency, and identity decisions used to be
derived from spoofable path substrings (`"72b"`, `"cortex"`, `"zenith"`) — a
renamed heavy checkpoint inherited light-model budgets, and an unrelated path
containing `"32b"` inherited a 20 GB reservation.

At load time the runtime reads:

- `model.safetensors.index.json` → `metadata.total_parameters` and
  `metadata.total_size` (exact weight bytes),
- `config.json` → architecture shape (parameter count is estimated from this
  when index metadata is absent),
- the safetensors file listing (names + sizes).

From those it derives a cached `ModelArtifactProfile` carrying a SHA-256
**fingerprint over config bytes, index metadata, and the weight-file
listing**. This is an identity binding, *not* a weight hash — hashing 20 GB on
every admission check is not viable, and the card should not imply otherwise.
The fingerprint changes whenever the artifact's declared shape changes.

The profile records *which* evidence produced it, so a receipt can distinguish
measured truth from a naming-convention fallback (used only when the artifact
is absent, as in tests and pre-download paths).

Memory footprint is then validated against the hardware profile before the
load is admitted; `AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB` sets the floor below
which a 32B load is refused outright.
