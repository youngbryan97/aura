# Installation

## Requirements

- macOS with Apple Silicon. Bryan's tracked target is M5-class with 64 GB RAM.
- Python 3.12+
- 32 GB RAM at minimum. Bryan's tracked deployment target is an M5-class
  Apple Silicon Mac with 64 GB unified memory, which has room for the 32B
  Cortex plus the 7B Brainstem on demand.

## Setup

```bash
git clone https://github.com/youngbryan97/aura.git
cd aura

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

```bash
# Full stack with web UI
python aura_main.py --desktop

# Headless (background cognition only, no UI)
python aura_main.py --headless

# Philosophy/proof stream: live substrate, phi, affect, and Will receipts
python aura_main.py --philosophy
```

Once the server is up, the UI lives at `http://localhost:8000`.

## First boot

First boot takes 30–60 seconds while Metal compiles shaders and the local
model initializes. If the model weights aren't on disk yet, the initial
download can take 5–10 minutes. Subsequent boots are much faster once the
cache is warm.

State loads from SQLite on boot. If there's nothing saved, Aura starts
fresh. The 7B Brainstem isn't loaded at boot — it's lazy so the 32B Cortex
gets the memory it wants (~5 GB difference).

## Optional: fine-tune personality

```bash
# Generate training data
python training/build_dataset.py

# Fine-tune the LoRA adapter (10–30 min)
python -m mlx_lm lora \
  --model mlx-community/Qwen2.5-32B-Instruct-4bit \
  --train \
  --data training/data \
  --adapter-path training/adapters/aura-personality \
  --num-layers 16 \
  --batch-size 1 \
  --iters 600 \
  --learning-rate 1e-5
```

If the adapter ends up at `training/adapters/aura-personality/`, the next
boot picks it up automatically.

## Environment variables (optional)

| Variable | Default | What it does |
|----------|---------|--------------|
| `AURA_HOST` | `127.0.0.1` | Bind address |
| `AURA_PORT` | `8000` | Port |
| `AURA_LORA_PATH` | auto-detected | Path to the LoRA adapter directory |
| `AURA_MODEL` | `Qwen2.5-32B-Instruct-8bit` | Primary Cortex model |
| `AURA_DEEP_MODEL` | auto-detected (72B) | Solver model for deep reasoning |
| `AURA_BRAINSTEM_MODEL` | `Qwen2.5-7B-Instruct-4bit` | Fast fallback |
| `AURA_FALLBACK_MODEL` | `Qwen2.5-1.5B-Instruct-4bit` | CPU emergency fallback |
| `AURA_LOCAL_BACKEND` | `llama_cpp` | `mlx` or `llama_cpp` |
| `AURA_SUBSTRATE_PRIMARY` | `1` | Try substrate token readout before transformer fallback |
| `AURA_SUBSTRATE_DIM` | `64` | Continuous substrate dimension, clamped to 16-512 |
| `AURA_ONLINE_LORA` | `1` | Enable governed reflection-to-LoRA update attempts |
| `AURA_ROOT` | auto-detected | Project root |
| `AURA_SAFE_BOOT_DESKTOP` | `0` | Set to `1` for a lightweight boot |
| `AURA_ENV` | `development` | Use `production` inside Docker |

## Docker

```bash
# Full stack: Aura + Redis + Celery worker
docker-compose up -d

# Tail logs
docker-compose logs -f aura
```

The image is based on `python:3.12-slim`, runs as a non-root user, and
includes Redis for task queuing and Celery for background work. Health
checks hit `/api/health`.

## Troubleshooting

- **Out of memory.** Close other apps, or drop to a smaller model. The
  32B 8-bit needs ~20 GB of GPU RAM. On lower-memory machines set
  `AURA_MODEL=Qwen2.5-7B-Instruct-4bit`.
- **Model won't load.** Make sure `mlx-lm` is installed
  (`pip install mlx-lm`). On the llama.cpp backend, the GGUF files
  should be in `models_gguf/`.
- **Port in use.** Kill stray Aura processes: `pkill -f aura_main`.
- **Model load hangs.** Only one model loads at a time through the GPU
  semaphore. If it's stuck, check for zombie MLX worker processes.
- **Backend choice.** `AURA_LOCAL_BACKEND=mlx` for MLX (recommended on
  Apple Silicon), `llama_cpp` for GGUF.
