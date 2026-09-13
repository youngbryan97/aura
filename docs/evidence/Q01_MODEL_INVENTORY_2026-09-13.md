# Q01 model inventory, 2026-09-13

Produced by `tools/model_inventory.py` and held by `tests/test_model_inventory.py`.
Re-run either to refresh; nothing here is typed by hand.

## Live roles

Every role the runtime serves, resolved the way the runtime resolves it —
`get_model_path` for the language lanes, the module constants for perception
and voice — with the facts its consumers read: the checkpoint's own geometry
and quantization from `config.json`, the context window from
`get_context_window_evidence`, the declared MLX worker footprint from
`_declared_mlx_worker_footprint_gb`, and the chat template's digest.

| role | model | on disk | GB | geometry | context | declared GB | template | consumer |
|---|---|---|---|---|---|---|---|---|
| cortex | `Aura-Cortex` | yes | 15.15 | 64 layers, hidden 5120, 4-bit | 262144 | 21.32 | `c3cf9e34abf4f9e3` | core/brain/llm/mlx_client.py (resident worker) |
| brainstem | `Qwen3.5-9B-4bit` | yes | 5.98 | 32 layers, hidden 4096, 4-bit | 262144 | 7.65 | `a4aee8afcf2e0711` | core/brain/llm/mlx_client.py (fallback lane) |
| reflex | `Qwen2.5-1.5B-Instruct-4bit` | yes | 0.88 | 28 layers, hidden 1536, 4-bit | 32768 | 2.81 | `d5495a1e5db06111` | core/brain/llm/mlx_client.py (reflex lane) |
| deep | — | optional, unset | — | — | — | — | — | deep solver lane |
| embedding | `Qwen/Qwen3-Embedding-0.6B` | yes | 1.21 | 28 layers, hidden 1024, —-bit | — | 1.6 | `87a2728cb8dc9fe4` | core/memory/vector_memory_engine.py (in-process, torch) |
| vision | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | yes | 3.11 | 36 layers, hidden 2560, 4-bit | 262144 | 4.88 | `3636d0f0bd6bef02` | core/brain/llm/mlx_vision_client.py |
| asr_partial | `mlx-community/parakeet-tdt-0.6b-v3` | yes | 2.51 | not a language model | — | — | `—` | core/voice/duplex/config.py |
| asr_final | `mlx-community/parakeet-tdt-0.6b-v3` | yes | 2.51 | not a language model | — | — | `—` | core/voice/duplex/config.py |

The deep solver lane is optional and unset: `AURA_DEEP_MODEL` is None and the
lane is gated on an attested certificate. No 72B is on this machine and nothing
asks for one.

A declared footprint below the checkpoint's size could never admit it; the
test holds `declared >= size` for every language lane. The brainstem was
declared at a flat 4GB until 2026-09-13 (b6a9a35fe); every class now reads its
own checkpoint.

## On disk and named by nothing

8 entries, 34.7GB. Listed, not deleted: a
checkpoint is the owner's to remove. The tool re-derives this list from the
code, so an entry that a future module names leaves it on its own.

| checkpoint | where | GB | note |
|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | hub | 0.0 | empty shell |
| `Qwen/Qwen2.5-32B-Instruct-GGUF` | hub | 0.0 | empty shell |
| `Qwen/Qwen2.5-72B-Instruct-GGUF` | hub | 0.0 | empty shell |
| `Qwen/Qwen2.5-7B-Instruct-GGUF` | hub | 0.0 | empty shell |
| `Systran/faster-whisper-base` | hub | 0.15 |  |
| `mlx-community/Qwen3-32B-4bit` | hub | 18.45 |  |
| `mlx-community/Qwen3.8-27B-4bit` | hub | 16.08 |  |
| `mlx-community/Qwen3.8-27B-MTP-4bit` | hub | 0.0 | empty shell |

- `mlx-community/Qwen3-32B-4bit` is the cortex the 27B replaced. Nothing
  resolves to it.
- `mlx-community/Qwen3.8-27B-4bit` in the hub cache is a second copy of the
  base the fused cortex was built from; the live base is
  `~/.aura/models/Qwen3.8-27B-4bit-3e6447f082e8`, named by the fused model's
  manifest and kept for re-fusion.
- The four `*-GGUF` directories are empty shells from the retired llama.cpp
  runtime, whose removal `tests/test_retired_external_runtime.py` and
  `tests/test_retired_cloud_residue.py` hold.
- `Systran/faster-whisper-base` predates the MLX ASR models the duplex voice
  lane names.

## Retired providers

The external-server cortex and the llama.cpp runtime are retired and pinned
retired: `core/brain/llm/retired_external_runtime.py`, with
`test_retired_external_runtime.py` (7), `test_retired_cloud_residue.py` and
`test_mlx_retired_handle_liveness.py` — 12 tests, green on 2026-09-13. The
only mention of `llama_cpp` left in production code is the subprocess
gateway's deny list, which is where a retired name belongs.
