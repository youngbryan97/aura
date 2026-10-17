# Aura Hardware & Model Profiles

This guide explains what hardware Aura runs on and—most importantly—what claims you can validly make about a test run on each machine.

Hardware specs and test results belong together. Running a benchmark on an 8 GB laptop with simulated (mocked) models is completely different from running it on a 64 GB machine with a real 27-billion-parameter model in memory. A test score from the laptop proves almost nothing about real-world performance on the workstation. Each hardware profile below spells out which claims you are allowed to make and which are forbidden, so every test result stays tied to the hardware that produced it.

## 1. No-Model / Dev Profile
* **Target Hardware**: Standard laptop (such as an Intel or M1 MacBook Air) with 8 GB of RAM.
* **Required Models**: None (uses simulated mocks and stubs only).
* **Memory/Compute**: Minimal resource requirement; runs easily on standard consumer hardware.
* **Allowed Claims**:
  - `governed runtime` (static code verification only)
  - `production-sealed` (static gate validation)
* **Disallowed Claims**: All empirical claims based on live model execution, including `operational volition` (independent decision-making), `autonomous agency`, `emergent intelligence`, `DNU AGI`, or `synthetic cognitive entity`.
* **Tests That Can Run**:
  - `python -m compileall`
  - `pytest --collect-only`
  - Strict Flagship Readiness check
  - Production Surface Lint check
  - Static Enterprise/Readiness gates
* **Tests That Are Blocked**: All live capability runs, agent loop tests, longevity soak tests (extended continuous runs), and model-dependent tests.

---

## 2. CI / Proof-Short Profile
* **Target Hardware**: Cloud CI virtual runner (such as a standard GitHub Actions runner) with 2–4 virtual CPUs and 7–14 GB of RAM.
* **Required Models**: Lightweight local models compatible with Apple's MLX framework for short, bounded proof runs.
* **Memory/Compute**: Bounded.
* **Allowed Claims**:
  - `governed runtime` (receipt verification on lightweight runs)
  - `persistent memory` (local persistent memory writes to disk)
  - `operational volition` (bounded Will Decision receipt logging)
  - `production-sealed`
* **Disallowed Claims**: Any claim of high-level autonomy or general intelligence, including `emergent intelligence`, `external real-world validation`, `DNU AGI`, `AGI-candidate`, `mature RSI` (recursive self-improvement), or `synthetic cognitive entity`.
* **Tests That Can Run**:
  - All unit/integration tests (`pytest`)
  - Bounded Agency Emergence proof runs (using local or mocked LLMs)
  - Bounded Longevity soak (`proof_short` profile)
* **Tests That Are Blocked**: Full 100-task DNU AGI suite, multi-hour longevity soak, and high-capacity model-reasoning evaluations.

---

## 3. Local Apple Silicon Profile
* **Target Hardware**: Mac Studio or MacBook Pro with an M5 Pro chip or better, and 64 GB+ Unified Memory. Both `core/config.py` and `core/runtime.py` specify an M5 Pro with 64 GB as the standard hardware budget that Aura's three model tiers are sized against.
* **Required Models**: Three in-process models running via Apple's MLX framework:
  - Cortex (`Aura-Cortex` / fused Qwen3.8-27B, handles foreground conversation and complex tasks)
  - Brainstem (`Ternary-Bonsai-2-27B-mlx-2bit`, Qwen3.5 27B at two bits, 8.6 GB; handles background reflection and processing)
  - Reflex (`Qwen2.5-1.5B-Instruct-4bit`, handles fast routine responses)
  There is no separate coder model: `core/brain/llm/local_code_model.py` runs code generation on Aura's primary model lane with persona steering bypassed (because emotional or conversational steering corrupts code syntax).
* **Memory/Compute**: High-throughput unified memory bandwidth shared between CPU and GPU.
* **Allowed Claims**:
  - `governed runtime`, `persistent memory`, `causal internal state`, `affect steering`, `System 2 planning/search` (deliberate multi-step reasoning), and `self-repair`
  - `operational volition`, `autonomous agency`, and `entity-in-a-box behavior` (independent goal pursuit within a sandbox)
  - `experience-adjacent functional indicators` (measurable internal state metrics)
* **Disallowed Claims**: `DNU AGI`, `AGI-candidate`, `external real-world validation` (requires independent evaluations across long time horizons), or `indefinite autonomy`.
* **Tests That Can Run**:
  - Local model-aware agency emergence batteries
  - Local sandbox/boxed entity suites
  - Medium-duration longevity soak (such as `local_4h`, a 4-hour continuous run)
* **Tests That Are Blocked**: Multi-day longevity soak (such as `local_72h`, a 72-hour continuous run) and high-horizon external validation.

---

## 4. Local High-Memory Profile
* **Target Hardware**: High-memory Apple Silicon (such as an M5 Ultra with 192 GB+ Unified Memory), or a dedicated workstation with 128 GB+ System RAM (note: MLX model inference requires Apple Silicon).
* **Required Models**: Aura MLX 27B or 72B lane artifacts. An optional local reasoning solver can be downloaded using `scripts/fetch_models.py --reasoning-solver`; supported model names include `r1-qwen32b`, `r1-qwen32b-8bit`, `qwq32b`, `qwq-32b`, and `deepseek-r1-qwen32b`.
* **Memory/Compute**: Very large local memory pool available to the GPU.
* **Allowed Claims**: Same as Local Apple Silicon, plus:
  - `emergent intelligence` (locally evaluated on larger distributions)
* **Disallowed Claims**: `DNU AGI`, `AGI-candidate`, or `indefinite autonomy`.
* **Tests That Can Run**:
  - Heavy local model reasoning runs
  - Local System 2 search rollouts
  - Longer longevity soak (such as `local_24h`, a 24-hour continuous run)
* **Tests That Are Blocked**: Third-party benchmark gates that exceed local compute capacity.

---

## 5. Live Hardware / Browser Profile
* **Target Hardware**: Dedicated robotic system, physical device, or developer workstation with full operating system access and live web browser automation hooks.
* **Required Models**: Local Cortex, Solver, Brainstem, and Reflex model tiers.
* **Memory/Compute**: Unconstrained host access.
* **Allowed Claims**: Strictly bounded by authorization and compliance profiles.
* **Disallowed Claims**: `mature RSI` (recursive self-improvement, unless strictly sandboxed with automatic rollback) and subjective consciousness.
* **Tests That Can Run**:
  - Live browser automation and operating system control validation
  - Physical interaction or simulation co-presence integration
* **Tests That Are Blocked**: Bounded by environment safety profiles and authority filters.
