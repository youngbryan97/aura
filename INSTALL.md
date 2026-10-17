# Installation

## Requirements

- macOS on Apple Silicon (M-series chip).
- Python 3.12+.
- 32 GB RAM minimum; 64 GB recommended.

64 GB is the recommended hardware setup. The target machine is an Apple Silicon Mac with 64 GB of unified memory, which has enough room to run the primary 27B model (Cortex) alongside the compressed 27B fallback model (Brainstem, 8.6 GB) on demand. Aura works with 32 GB RAM, but you will need to switch to smaller models, and response times will be slower than the benchmark numbers quoted in this repository (which were measured on 64 GB hardware).

## Setup

```bash
git clone https://github.com/youngbryan97/aura.git
cd aura

make setup        # creates .venv, installs requirements/core.txt + requirements/dev.txt
```

`make setup-prod` is the strict production installer: it stops immediately if any dependency fails to install rather than silently continuing with degraded features. Use this variant if you plan to run Aura unattended.

Manual equivalent:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements/core.txt
pip install -r requirements/dev.txt      # test + lint tooling
```

Requirements are split into separate files by feature: `requirements/core.txt` (core runtime), `dev.txt` (tests and linting), `ml.txt` (training pipelines), `senses.txt` (camera, screen, and audio capture), and `voice.txt` / `voice-high-fidelity.txt` (speech). Install only the packages you plan to use.

For reproducible installs pinned to verified file hashes, use the lockfile (regenerate with `pip-compile --allow-unsafe --generate-hashes --output-file=requirements_lock.txt requirements.txt`):

```bash
pip install --require-hashes -r requirements_lock.txt
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

Once the server is running, open the web interface at `http://localhost:8000`.

### All boot modes

| Flag | Mode |
|------|------|
| `--desktop` | Desktop GUI (the standard way to run Aura) |
| `--headless` | API server only; runs background processes without a GUI |
| `--server` | API server mode |
| `--cli` | Interactive terminal console |
| `--gui-window` | Open a GUI window connected to an already-running server |
| `--watchdog` | Watchdog supervisor process to keep Aura running |
| `--philosophy` | Stream internal state logs (substrate, phi, emotion, and decision receipts) as JSON lines |
| `--skeletal` | Start quickly by skipping heavy optional subsystems (useful for troubleshooting) |
| `--profile minimal` | Named startup profile |
| `--stop` | Stop a running Aura instance |
| `--reboot` | Cleanly shut down and restart |
| `--host` / `--port` | Network address and port to bind to (default: `127.0.0.1:8000`) |

### Operational subcommands

Installing the package provides the `aura` command-line tool (`aura = aura_main:main`), which includes maintenance commands:

```bash
aura doctor                  # pre-boot self-check: python, sqlite, mlx, data dir,
                             # atomic-writer round-trip
aura doctor --bundle         # redacted diagnostics tarball for incident triage
aura conformance             # schema + integrity sweep
aura verify-state            # cross-subsystem state coherence
aura verify-memory           # memory facade integrity
aura rebuild-index           # vector index rebuild
aura backup / restore / migrate
aura chaos                   # fault-injection smoke
aura plugin                  # plugin management
```

## First boot

The first startup takes 30–60 seconds while Apple Metal compiles GPU shaders and loads the local model into memory. If the model files are not yet on disk, allow an extra 5–10 minutes for them to download. After the first run, model weights and shaders are cached, and subsequent startups are much faster.

System state is loaded from a local SQLite database. If no database exists, Aura starts with a clean slate.

The secondary model (Brainstem) does not load at boot time. It loads only when needed (lazy loading) so the primary model (Cortex) gets all available memory. On machines where free RAM is tight, keeping the Brainstem's 8.6 GB out of memory until requested ensures the main Cortex model can start reliably.

## Optional: fine-tune personality

```bash
# Generate training data
python training/build_dataset.py

# Fine-tune the LoRA adapter (10–30 min)
python -m mlx_lm lora \
  --model models/Aura-Cortex \
  --train \
  --data training/data \
  --adapter-path training/adapters/aura-personality \
  --num-layers 16 \
  --batch-size 1 \
  --iters 600 \
  --learning-rate 1e-5
```

If the trained adapter is saved to `training/adapters/aura-personality/`, Aura will automatically detect and load it on the next startup.

## Environment variables (optional)

Aura manages configuration using Pydantic Settings (`core/config.py:AuraConfig`). All settings can be set with environment variables prefixed by `AURA_`, using double underscores for nested fields (for example, `AURA_LLM__MLX_DEEP_MODEL_PATH` sets `llm.mlx_deep_model_path`). You can also set these in a `.env` file in the project root. Unrecognized `AURA_*` variables are ignored without warning, so run `aura doctor` to verify your configuration if a setting does not seem to take effect.

There is no `AURA_HOST` environment variable; use the `--host` command-line flag instead (default: `127.0.0.1`). Set `AURA_INTERNAL_ONLY=1` to reject any network requests coming from outside your local computer.

| Variable | Default | What it does |
|----------|---------|--------------|
| `--port PORT` | `8000` | Port to run the local API on. This is a flag for `launch_aura.sh`, not an environment variable (`launch_aura.sh` sets `AURA_PORT` itself before reading the environment, so exporting it has no effect). |
| `AURA_INTERNAL_ONLY` | from security profile | Set to `1` to reject requests from outside localhost. |
| `AURA_API_TOKEN` | unset | Secret token required to authenticate API requests. |
| `AURA_LORA_PATH` | auto-detected | Path to the directory containing fine-tuned LoRA weights. |
| `AURA_MODEL` | `Aura-Cortex` | Main language model (`Aura-Qwen3.8-27B-persona`, a persona fine-tune of the fused Qwen3.8-27B). |
| `AURA_DEEP_MODEL` | unset (deep work runs on the Cortex) | A second local model for hard reasoning, admitted only with an attested qualification certificate (`AURA_DEEP_SPECIALIST_CERTIFICATE`); see `docs/OPERATOR_GUIDE.md`. |
| `AURA_BRAINSTEM_MODEL` | `Ternary-Bonsai-2-27B-mlx-2bit` | Background lane (heartbeat, telemetry, background tasks), lazy-loaded: 2-bit quantized Qwen3.5 27B, requiring 8.6 GB RAM (replaced 9B on 2026-09-20; see `docs/evidence/TERNARY_BONSAI_2_27B_2026-09-17.md`). |
| `AURA_FALLBACK_MODEL` | `Qwen2.5-1.5B-Instruct-4bit` | Emergency CPU fallback model. Kept in the same model family as Cortex to ensure consistent output style during speculative decoding. |
| `AURA_LOCAL_BACKEND` | `mlx` | Local inference engine (Apple MLX framework). Production Aura always uses this backend. |
| `AURA_BROWSER_STEALTH` | `1` | Whether her browser hides that it is automated: playwright-stealth, a borrowed user agent, and the automation flag switched off. `0` browses as what it is. A site that refuses automated visitors should be met with `0`; see `docs/BROWSER_PURSUIT.md`. |
| `AURA_SUBSTRATE_PRIMARY` | `1` | Check continuous neural state before falling back to standard text tokens. |
| `AURA_SUBSTRATE_DIM` | `64` | Dimension of the continuous neural state vector (clamped between 16 and 512). |
| `AURA_ONLINE_LORA` | `1` | Allow Aura to update its own weights through background reflection. |
| `AURA_ROOT` | auto-detected | Base directory of the project. |
| `AURA_SAFE_BOOT_DESKTOP` | `0` | Set to `1` to start with a lightweight desktop interface. |
| `AURA_MODE` | `production` | Operating mode: `safe` turns off tool use and autonomous actions; `dev` enables developer features and self-modification; `production` is the default. See `core/runtime/mode.py` for full details. |
| `AURA_LOG_DIR` | `~/.aura/logs` | Directory where log files are written. **Always change this when running tests** so test runs do not overwrite production logs. |
| `AURA_LATENT_CORTEX` | off | Enable recursive latent-space thinking. |
| `AURA_STRICT_RUNTIME` | `0` | Immediately stop on minor errors instead of logging them and continuing. |
| `AURA_GOVERNANCE_MODE` | profile default | Safety policy level applied to consequential tool actions. |
| `AURA_PROCESS_RSS_LIMIT_GB` | derived from total RAM | Maximum physical RAM (Resident Set Size) the main process is allowed to use. Capped by safe limits unless specifically overridden. |
| `AURA_MLX_MEMORY_LIMIT_GB` | derived from total RAM | Maximum memory limit for the Apple MLX memory allocator. |
| `AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB` | `24.0` | Minimum free RAM (in GB) required before loading a large model. |

### Debugging entry points

| Variable | What it does |
|----------|--------------|
| `AURA_PASS_BISECT_LIMIT=N` | Run only the first N thinking phases. Useful for binary searching which stage caused a problem in an answer. |
| `AURA_PASS_TRACE=1` | Print the name of each cognitive phase to the console as it runs. |
| `AURA_TEST_MODE` / `AURA_TESTING` | Enable test mode to disable external side effects. |

## Docker

```bash
# Full stack: Aura + Redis + Celery worker
docker-compose up -d

# Tail logs
docker-compose logs -f aura
```

The Docker image is built on `python:3.12-slim`, runs under a non-root user for security, and includes Redis and Celery for background work. Health checks query `/api/health`.

## Troubleshooting

- **Out of memory.** Close other applications, or switch to a smaller model. The 27B model requires around 18–20 GB of unified memory. On machines with less memory, explicitly set `AURA_MODEL=Qwen3.5-9B-4bit` in advance rather than waiting for an out-of-memory error during inference.
- **Model won't load.** Verify that `mlx-lm` is installed (`pip install mlx-lm`). Both the desktop app and backend runtime rely on the internal Apple MLX engine to run models locally.
- **Port in use.** Shut down the existing instance cleanly with `python aura_main.py --stop`, which saves state and revokes temporary security tokens. Avoid running `pkill -f aura_main`, as this will abruptly terminate an active Aura instance mid-operation.
- **Model load hangs.** Aura loads only one model into GPU memory at a time. If loading appears frozen, check for lingering background MLX worker processes.
- **Backend choice.** Keep `AURA_LOCAL_BACKEND=mlx` on Apple Silicon. Running MLX in-process allows Aura's internal state, memory, and emotion models to directly guide the language model without extra latency.
