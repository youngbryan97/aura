# Model roster

Status: Guide · Reviewed against the tree 2026-08-16

Every model Aura loads, which lane it serves, and the measurement that put it
there. Lanes churn, so the reason for each choice is recorded here.

The authoritative values are `core/config.py` (`LLMConfig`) and
`core/brain/llm/model_registry.py` (declared flags with defaults). Where this
page and the code disagree, the code is right and this page is a bug.

---

## Language model lanes

The tri-cameral architecture, tuned for an M5-class Apple Silicon Mac with
64 GB unified memory.

| Lane | Model | Config key | Env override | Role |
|---|---|---|---|---|
| **Cortex** (Tier 2) | `Aura-Cortex` (`Qwen3.8-27B` fused, `qwen3_5`) | `fast_model` | `AURA_MODEL` | Daily interaction, primary conversation lane. Handles nearly everything |
| **Solver** (Tier 3) | `Qwen2.5-72B-Instruct-4bit` (override) / `Aura-Cortex` (default) | `deep_model` | `AURA_DEEP_MODEL` | Deep reasoning, hot-swapped specialist when requested |
| **Brainstem** (Tier 1) | `Qwen3.5-9B-4bit` | `chat_model` | `AURA_BRAINSTEM_MODEL` | Heartbeat, telemetry, background tasks. Lazy-loaded |
| **Reflex** | `Qwen2.5-1.5B-Instruct-4bit` | — | `AURA_FALLBACK_MODEL` | CPU emergency fallback |
| **Vision** | `Aura-Cortex` (`Qwen3.8-27B`) | `vision_model` | — | Pinned to the Cortex build so vision and conversation share one identity |
| **Last resort** | rule-based | — | — | Static responses that cannot fail |

Sampling contract: `temperature` is bounded `[0.0, 2.0]`. The ceiling is not a
taste call — above 2.0 the softmax is effectively uniform and "temperature"
stops naming anything.

All language-model inference is local. Legacy `api_fast` and `api_deep`
labels remain accepted at compatibility boundaries, but resolve to the local
fast and local deep lanes respectively. A remote-only request returns
`remote_model_provider_removed`; it cannot register or select an off-host
model endpoint. Local distillation uses the Solver and resident Cortex as
teacher lanes.

### Why the Brainstem could move and the Reflex could not

Qwen3.5-9B replaced Qwen2.5-7B on 2026-08-12, with explicit reasoning-mode
control. The Reflex lane stayed on Qwen2.5-1.5B in the same pass, and the
asymmetry is the interesting part:

> The 1.5B is the **speculative draft and contrastive amateur** for the Cortex.
> It is therefore locked to the Cortex's own distribution — a draft model from
> a different family proposes tokens the target rejects, and a contrastive
> amateur from a different family measures the wrong contrast.
>
> Nothing is keyed to the Brainstem tier's weights. Verified before the swap:
> no draft, amateur, or contrastive path references the Brainstem, so the
> generation gap was free to close.

Recorded in `core/brain/llm/model_registry.py` beside the flag declaration.

### Hardware honesty

The 27B/32B Cortex wants about 18–20 GB of GPU RAM and will not negotiate. On
lower-memory machines the hardware auditor rejects heavy weights
as real-time heartbeat tiers; use the 9B or 1.5B lanes there. Claiming heavy
heartbeat latency on a machine that cannot hold it would be a benchmark run on
hardware nobody has.

---

## Speech to text

| Role | Model | Where |
|---|---|---|
| **Primary** | `parakeet-tdt-0.6b-v3` (`parakeet_mlx`) | `core/voice/duplex/streaming_asr.py` |
| **Fallback** | `faster_whisper` (CPU) | `core/senses/voice_engine.py` |

Both duplex stages — 480 ms partials and the final — now run **one streaming
model**. The previous design ran two Whisper models on purpose: `small.en` for
partials and `large-v3-turbo` once for the final, because Whisper is not a
streaming model and paying the accurate model's cost on every partial would
contend with the resident 32B for no accuracy that mattered. The entire
LocalAgreement-2 stable-prefix apparatus existed to work around that same fact.

Parakeet TDT removes the trade. Measured on this host with real speech and a
known transcript (locally synthesised — nothing left the machine), median of
5 warm runs over 12.4 s of audio:

| Model | Latency | Stage it replaces |
|---|---|---|
| `parakeet-tdt-0.6b-v3` | **166 ms** | both |
| `whisper-small.en` | 193 ms | partials |
| `whisper-large-v3-turbo` | 317 ms | finals |

One Parakeet decode is cheaper than the incumbent *partial* and about half the
incumbent *final*: one model-lane lease, one set of weights resident, no
accuracy sacrificed on partials.

**A discarded first benchmark.** The first pass used synthetic noise and
produced nonsense for Whisper.
Whisper pads every input to 30 s and its cost is bound by tokens emitted, so on
noise it hallucinated variable-length output and the timings were
non-monotonic — 3 s of audio measured slower than 10 s. Discarded and redone on
real speech. All three models then scored 0% WER on the clean local sample,
which is too easy to discriminate, so the accuracy case (6.32% vs 7.83% English
WER) is cited from published figures rather than claimed as a local
measurement.

---

## Embeddings

| | |
|---|---|
| **Model** | `Qwen/Qwen3-Embedding-0.6B` at 384 dimensions |
| **Where** | `core/memory/embedding_model.py` (`REPO_ID`) |
| **Replaced** | `all-MiniLM-L6-v2`, on 2026-08-12 |

### The defect

`all-MiniLM-L6-v2` declared `max_seq_length: 256`. The ingestion path chunked
at a fixed 800 words (`black_hole_vault.py`, `ingestion_loop.py`;
`rag.chunk_text` at 500). Measured through the model's own tokenizer, an
800-word chunk is **1,122 tokens — so 77% of every full chunk never reached the
encoder.** Nothing logged it: tokenizer truncation is silent by design, and
both constants were individually sane.

The damage was in the blend. `retrieve_memories` mixes dense cosine 60/40 with
lexical TF-IDF. The lexical half read whole chunks; the dense half read the
first quarter; and the heavier weight sat on the half that saw less.

**Both halves were fixed.** Chunk size is no longer a constant at all — it is
*derived* from the encoder's declared window via
`embedding_model.chunk_for_embedding`, so a whole page is normally one chunk
and one memory is one vector. An explicit `chunk_size` is still honoured for
callers with a real reason to want fine-grained passages, but it is clamped to
what the encoder can actually read, so an explicit request can never
reintroduce silent truncation.

### The measurement

Four ~800-word documents whose distinguishing sentence sits past token 256
(M5 Pro, 2026-08-12):

| Model | Tail retrieval | Latency |
|---|---|---|
| `all-MiniLM-L6-v2` | 1/4 | 10.7 ms/query |
| `Qwen3-Embedding-0.6B` @384 | **3/4** | 20.2 ms/query |

MiniLM ranked the *same* document first for all four queries — four identical
prefixes, so it tied and broke arbitrarily. **1/4 is chance.**

Vectors from both models have 384 entries, so the store shape is unchanged.
Qwen3-Embedding conditions the query embedding on a natural-language task
instruction; that wire format is documented on the model card and implemented
in `embedding_model.py`.

### The calibration that came with it

`core/memory/retrieval_calibration.py` records that the two models need
different admission thresholds, and why a threshold cannot simply be carried
over:

```
all-MiniLM-L6-v2      null mean +0.060 (max +0.132)   0.01 admits 5/6 nulls
Qwen3-Embedding@384   null mean +0.260 (max +0.415)   0.01 admits 6/6 nulls
```

On the same sample, Qwen3-Embedding's signal and null populations *overlap*.
The threshold is calibrated against measured nulls rather than asserted.

---

## Known stale config fields

Recorded rather than quietly deleted, because each is a real statement about
the repo:

| Field | State |
|---|---|
| `LLMConfig.whisper_model = "small.en"` | No reader resolves the ASR model through this field. The Whisper fallback takes its size from `core/senses/voice_engine.py` defaults (default `"base"`) or `core/voice/duplex/streaming_asr.py` |
| `LLMConfig.embedding_model = "nomic-embed-text"` | **Dead.** No code reads it; the real embedding model is `REPO_ID` in `core/memory/embedding_model.py`. The name refers to a model this repository does not load |

---

## Gates

```bash
make model-lane-contract-audit   # lane declarations vs what loads
make model-load-audit            # every model-load path enters one admission transaction
make integration-liveness        # the declared integrations import and expose their attrs
```

`core/runtime/integration_liveness.py` is the registry that names each
integration, what it powers, and whether it is user-facing — including which
ASR engine is primary and which is fallback.
