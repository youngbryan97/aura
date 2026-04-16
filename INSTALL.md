# Installation

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.12+**
- **32 GB RAM** minimum (64 GB recommended for 32B Cortex 8-bit + 7B Brainstem)

## Setup

```bash
# Clone
git clone https://github.com/youngbryan97/aura.git
cd aura

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
# Full stack with web UI
python aura_main.py --desktop

# Headless (background cognition only)
python aura_main.py --headless
```

The web UI is at `http://localhost:8000` once the server starts.

## First Boot

First boot takes longer as Metal shaders compile and the local LLM model initializes (~30-60 seconds). If models are not yet downloaded, initial download may take 5-10 minutes depending on network speed. Subsequent boots are faster as models and shaders are cached.

Aura loads her state from SQLite on boot. If no state exists, she creates a fresh one. The 7B Brainstem model loads on demand, not at boot, to save ~5GB RAM for the 32B Cortex.

## Optional: Fine-tune personality

```bash
# Generate training data
python training/build_dataset.py

# Fine-tune LoRA adapter (~10-30 min)
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

The adapter is automatically loaded on next boot if present at `training/adapters/aura-personality/`.

## Environment Variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `AURA_HOST` | `127.0.0.1` | Server bind address |
| `AURA_PORT` | `8000` | Server port |
| `AURA_LORA_PATH` | auto-detected | Path to LoRA adapter directory |
| `AURA_MODEL` | `Qwen2.5-32B-Instruct-8bit` | Primary Cortex model |
| `AURA_DEEP_MODEL` | auto-detected (72B) | Solver model for deep reasoning |
| `AURA_BRAINSTEM_MODEL` | `Qwen2.5-7B-Instruct-4bit` | Fast fallback model |
| `AURA_FALLBACK_MODEL` | `Qwen2.5-1.5B-Instruct-4bit` | Emergency CPU reflex model |
| `AURA_LOCAL_BACKEND` | `llama_cpp` | Backend: `mlx` or `llama_cpp` |
| `AURA_ROOT` | auto-detected | Project root directory |
| `AURA_SAFE_BOOT_DESKTOP` | `0` | Set to `1` for lightweight boot |
| `AURA_ENV` | `development` | Environment (`production` for Docker) |

## Docker Deployment

```bash
# Full stack: Aura + Redis + Celery worker
docker-compose up -d

# View logs
docker-compose logs -f aura
```

The Docker setup uses `python:3.12-slim`, runs as a non-root user, and includes Redis for task queuing and Celery for background processing. Health checks are configured on the `/api/health` endpoint.

## Troubleshooting

- **Out of memory**: Reduce model size or close other apps. The 32B 8-bit model needs ~20 GB GPU RAM. Set `AURA_MODEL=Qwen2.5-7B-Instruct-4bit` for lower-memory machines.
- **Model not loading**: Check that `mlx-lm` is installed: `pip install mlx-lm`. For llama.cpp backend, ensure GGUF files are in `models_gguf/`.
- **Port in use**: Kill any existing Aura process: `pkill -f aura_main`
- **GPU semaphore contention**: Only one model loads at a time. If a load hangs, check for zombie MLX worker processes.
- **Backend selection**: Set `AURA_LOCAL_BACKEND=mlx` for MLX or `AURA_LOCAL_BACKEND=llama_cpp` for llama.cpp (GGUF). MLX is recommended on Apple Silicon.
