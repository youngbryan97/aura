# Aura Hardware & Model Profiles

What hardware Aura runs on, and — the part that matters — what you're
allowed to claim from a run on each one.

Those two things are usually kept apart. They shouldn't be. A benchmark on
an 8 GB laptop with mocked models and a benchmark on a 64 GB machine with
the resident 27B are not the same measurement, and a number carried from
the first to the second is just a number. Each profile below lists its
allowed claims and its disallowed ones explicitly, so a result knows what
hardware produced it.

## 1. No-Model / Dev Profile
* **Target Hardware**: Standard laptop (e.g. Intel/M1 MacBook Air), 8GB RAM.
* **Required Models**: None (mocks/stubs only).
* **Memory/Compute**: Minimal resource requirement.
* **Allowed Claims**:
  - `governed runtime` (static verification only)
  - `production-sealed` (static gate validation)
* **Disallowed Claims**: All empirical claims, including `operational volition`, `autonomous agency`, `emergent intelligence`, `DNU AGI`, `synthetic cognitive entity`.
* **Tests That Can Run**:
  - `python -m compileall`
  - `pytest --collect-only`
  - Strict Flagship Readiness check
  - Production Surface Lint check
  - Static Enterprise/Readiness gates
* **Tests That Are Blocked**: All live capability runs, agent loop tests, longevity soak, and model-dependent tests.

---

## 2. CI / Proof-Short Profile
* **Target Hardware**: Virtualized CI Runner (e.g. GitHub Actions standard runner), 2-4 vCPUs, 7-14GB RAM.
* **Required Models**: Light local MLX-compatible models for bounded proof runs.
* **Memory/Compute**: Bounded.
* **Allowed Claims**:
  - `governed runtime` (receipt verification on light runs)
  - `persistent memory` (local persistent memory writes)
  - `operational volition` (bounded Will Decision receipt logging)
  - `production-sealed`
* **Disallowed Claims**: `emergent intelligence`, `external real-world validation`, `DNU AGI`, `AGI-candidate`, `mature RSI`, `synthetic cognitive entity`.
* **Tests That Can Run**:
  - All unit/integration tests (`pytest`)
  - Bounded Agency Emergence proof runs (with local/mocked LLMs)
  - Bounded Longevity soak (`proof_short` profile)
* **Tests That Are Blocked**: Full 100-task DNU AGI suite, multi-hour longevity soak, high-capacity model-reasoning evaluations.

---

## 3. Local Apple Silicon Profile
* **Target Hardware**: Mac Studio / MacBook Pro, M5 Pro or better, 64GB+ Unified Memory. `core/config.py` and `core/runtime.py` both name M5 Pro 64 GB as the budget the tri-cameral tiers are sized against.
* **Required Models**: the in-process MLX tiers —
  - Cortex (`Aura-Cortex` / fused Qwen3.8-27B, foreground)
  - Brainstem (`Qwen3.5-9B-4bit`, background)
  - Reflex (`Qwen2.5-1.5B-Instruct-4bit`, fast lane)
  There is no separate coder model: `core/brain/llm/local_code_model.py` runs code generation on Aura's own lane with persona steering bypassed, because steering corrupts symbolic output.
* **Memory/Compute**: High-throughput CPU/GPU memory bandwidth.
* **Allowed Claims**:
  - `governed runtime`, `persistent memory`, `causal internal state`, `affect steering`, `System 2 planning/search`, `self-repair`
  - `operational volition`, `autonomous agency`, `entity-in-a-box behavior`
  - `experience-adjacent functional indicators`
* **Disallowed Claims**: `DNU AGI`, `AGI-candidate`, `external real-world validation` (requires independent high-horizon evaluation), `indefinite autonomy`.
* **Tests That Can Run**:
  - Local model-aware agency emergence batteries
  - Local sandbox/boxed entity suites
  - Medium-duration longevity soak (e.g., `local_4h`)
* **Tests That Are Blocked**: Multi-day longevity soak (e.g., `local_72h`) and high-horizon external validation.

---

## 4. Local High-Memory Profile
* **Target Hardware**: High-memory Apple Silicon (e.g. M5 Ultra 192 GB+), or dedicated workstation with 128GB+ System RAM (note: MLX inference requires Apple Silicon).
* **Required Models**: Aura MLX 27B/72B lane artifacts. An optional local reasoning solver is fetched with `scripts/fetch_models.py --reasoning-solver`; the supported aliases are `r1-qwen32b`, `r1-qwen32b-8bit`, `qwq32b`, `qwq-32b`, and `deepseek-r1-qwen32b`.
* **Memory/Compute**: Massive local GPU memory allocation.
* **Allowed Claims**: Same as Local Apple Silicon, plus:
  - `emergent intelligence` (locally evaluated on larger distributions)
* **Disallowed Claims**: `DNU AGI`, `AGI-candidate`, `indefinite autonomy`.
* **Tests That Can Run**:
  - Heavy local model reasoning runs
  - Local System 2 search rollouts
  - Longer longevity soak (e.g., `local_24h`)
* **Tests That Are Blocked**: Third-party benchmark gates that exceed local compute capacity.

---

## 5. Live Hardware / Browser Profile
* **Target Hardware**: Dedicated robotic/embodied system or developer workstation with full system access and live web interface hooks.
* **Required Models**: Local Cortex, Solver, Brainstem, and Reflex lanes.
* **Memory/Compute**: Unconstrained host access.
* **Allowed Claims**: Bounded by authorization/compliance profiles.
* **Disallowed Claims**: `mature RSI` (unless sandboxed with rollback), subjective consciousness.
* **Tests That Can Run**:
  - Live browser-use and OS-control validation
  - Physical or simulation co-presence integration
* **Tests That Are Blocked**: Bounded by environment safety profiles and authority filters.
