# Installation

## Requirements

- macOS on Apple Silicon.
- Python 3.12+.
- 32 GB RAM to run it at all. 64 GB to run it the way it's built.

That second line is the honest one. The tracked target is an M5-class Mac
with 64 GB unified memory, which has room for the 27B Cortex plus the 9B
Brainstem on demand. At 32 GB it works, but you're downshifting model lanes
and you should not expect the latency numbers quoted elsewhere in this repo
— those were measured on the 64 GB machine.

## Setup

```bash
git clone https://github.com/youngbryan97/aura.git
cd aura

make setup        # creates .venv, installs requirements/core.txt + requirements/dev.txt
```

`make setup-prod` is the fail-closed variant: no fallback installs, so a
missing dependency fails the install rather than degrading it silently. Prefer
it for anything you intend to run unattended.

Manual equivalent:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements/core.txt
pip install -r requirements/dev.txt      # test + lint tooling
```

Requirements are split by concern: `requirements/core.txt` (runtime),
`dev.txt` (tests, lint), `ml.txt` (training lanes), `senses.txt` (camera,
screen, audio capture), and `voice.txt` / `voice-high-fidelity.txt`. Install
only the lanes you intend to use.

For reproducible, supply-chain-pinned installs use the hashed lock
(regenerate with `pip-compile --allow-unsafe --generate-hashes
--output-file=requirements_lock.txt requirements.txt`):

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

Once the server is up, the UI lives at `http://localhost:8000`.

### All boot modes

| Flag | Mode |
|------|------|
| `--desktop` | Desktop GUI (the normal way to run Aura) |
| `--headless` | API server only, background cognition, no GUI |
| `--server` | API server mode |
| `--cli` | Interactive console |
| `--gui-window` | Open a GUI window against an already-running server |
| `--watchdog` | Watchdog / keep-alive supervisor |
| `--philosophy` | Stream substrate/phi/affect/Will receipts as JSONL |
| `--skeletal` | Bypass heavy subsystems (fast boot for triage) |
| `--profile minimal` | Named boot profile |
| `--stop` | Stop a running instance |
| `--reboot` | Force cleanup and restart |
| `--host` / `--port` | Bind address (default `127.0.0.1:8000`) |

### Operational subcommands

Installing the package exposes the `aura` console script (`aura = aura_main:main`),
which carries the maintenance verbs:

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

First boot takes 30–60 seconds while Metal compiles shaders and the local
model comes up. If the weights aren't on disk yet, add 5–10 minutes for the
download. After that the cache is warm and boots are quick.

State loads from SQLite. Nothing saved, and she starts fresh.

The 9B Brainstem does not load at boot. That's deliberate — it's lazy so
the 27B Cortex gets the memory it wants, and that's about 5 GB of
difference on a machine where 5 GB decides whether the Cortex loads at all.

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

If the adapter ends up at `training/adapters/aura-personality/`, the next
boot picks it up automatically.

## Environment variables (optional)

Configuration is a `pydantic-settings` model (`core/config.py:AuraConfig`)
with `env_prefix="AURA_"` and `env_nested_delimiter="__"`. Any field on a
sub-config is therefore reachable from the environment — for example
`AURA_LLM__MLX_DEEP_MODEL_PATH` sets `llm.mlx_deep_model_path`. A `.env` file
in the project root is read the same way. Unknown `AURA_*` variables are
ignored, so a typo fails silently — check `aura doctor` output if a setting
does not seem to take.

There is no `AURA_HOST`: the bind address comes from the `--host` flag
(default `127.0.0.1`). Set `AURA_INTERNAL_ONLY=1` to reject non-localhost
requests outright.

| Variable | Default | What it does |
|----------|---------|--------------|
| `--port PORT` | `8000` | Bind the local API to PORT. This is a flag on `launch_aura.sh`, not an environment variable — the script assigns `AURA_PORT` itself before reading your environment, so exporting it has no effect |
| `AURA_INTERNAL_ONLY` | from security profile | `1` rejects non-localhost requests |
| `AURA_API_TOKEN` | unset | Bearer token for the local API |
| `AURA_LORA_PATH` | auto-detected | Path to the LoRA adapter directory |
| `AURA_MODEL` | `Aura-Cortex` | Primary Cortex model (fused Qwen3.8-27B) |
| `AURA_DEEP_MODEL` | auto-detected (72B) | Solver model for deep reasoning |
| `AURA_BRAINSTEM_MODEL` | `Qwen3.5-9B-4bit` | Fast fallback (replaced Qwen2.5-7B on 2026-08-12) |
| `AURA_FALLBACK_MODEL` | `Qwen2.5-1.5B-Instruct-4bit` | CPU emergency fallback. Locked to the Cortex family — it is also the speculative draft and contrastive amateur, so it cannot drift from the Cortex distribution the way the Brainstem could |
| `AURA_LOCAL_BACKEND` | `mlx` | Internal MLX runtime. Live Aura always uses this path. |
| `AURA_SUBSTRATE_PRIMARY` | `1` | Try substrate token readout before transformer fallback |
| `AURA_SUBSTRATE_DIM` | `64` | Continuous substrate dimension, clamped to 16-512 |
| `AURA_ONLINE_LORA` | `1` | Enable governed reflection-to-LoRA update attempts |
| `AURA_ROOT` | auto-detected | Project root |
| `AURA_SAFE_BOOT_DESKTOP` | `0` | Set to `1` for a lightweight boot |
| `AURA_MODE` | `production` | Runtime posture. `safe` disables autonomy and tools, `dev` is the only mode permitting self-modification. `core/runtime/mode.py` holds the capability matrix |
| `AURA_LOG_DIR` | `~/.aura/logs` | Log sink. **Set this for anything test-like** so you never write into the live instance's logs. |
| `AURA_LATENT_CORTEX` | off | Enable the recursive latent-cortex lane |
| `AURA_STRICT_RUNTIME` | `0` | Fail closed on degradations that would otherwise be recorded and survived |
| `AURA_GOVERNANCE_MODE` | profile default | Governance posture for consequential actions |
| `AURA_PROCESS_RSS_LIMIT_GB` | derived from total RAM | Hard RSS ceiling for the main process. A value you set is still capped by the safe-boot ceiling unless unsafe limits are explicitly allowed |
| `AURA_MLX_MEMORY_LIMIT_GB` | derived from total RAM | MLX allocator ceiling, capped the same way |
| `AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB` | `24.0` | Refuse a heavy Cortex load below this much free memory (legacy env var name) |

### Debugging entry points

| Variable | What it does |
|----------|--------------|
| `AURA_PASS_BISECT_LIMIT=N` | Run only the first N cognitive phases — binary-search N to find which phase ruined an answer |
| `AURA_PASS_TRACE=1` | Announce each cognitive phase as it runs |
| `AURA_TEST_MODE` / `AURA_TESTING` | Test posture: no live side effects |

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

- **Out of memory.** Close other apps, or drop to a smaller model. The 27B
  Cortex wants about 18–20 GB of GPU RAM and will not negotiate. On a smaller
  machine set `AURA_MODEL=Qwen3.5-9B-4bit` and take the smaller
  lane on purpose rather than discovering it under load.
- **Model won't load.** Check that `mlx-lm` is installed
  (`pip install mlx-lm`). Desktop and runtime operation both use the
  internal MLX Cortex — that's what lets the substrate steer the live model
  path instead of talking at it.
- **Port in use.** Stop the running instance cleanly with
  `python aura_main.py --stop`, which drains receipts and revokes capability
  tokens. Avoid a blanket `pkill -f aura_main` — on a machine that is already
  running Aura, that kills the live instance mid-tick along with the stray one.
- **Model load hangs.** Only one model loads at a time through the GPU
  semaphore. If it's stuck, check for zombie MLX worker processes.
- **Backend choice.** Keep `AURA_LOCAL_BACKEND=mlx` for live Aura. The desktop
  Cortex is the in-process MLX lane so Aura's substrate, memory, affect, and
  response gates steer the same model path that speaks to the user.
