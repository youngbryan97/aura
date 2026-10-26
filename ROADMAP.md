# ROADMAP — Path to a Perfect Score

<!-- reader-navigation:start -->
> **In plain English:** The documentation contains current guides, dated experiment records, frozen rules, generated reports and open plans. They answer different questions.
>
> **Why it matters:** A reader should not treat an old proposed run as today’s result or a completed engineering task as a passed scientific check.
>
> **In everyday use:** An entry saying the next run is queued can remain correct history after that run finishes; the maintained status page gives the later outcome.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#documentation) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#documentation) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](docs/DOCUMENTATION_INDEX.md)
> [Glossary](docs/GLOSSARY.md) · [Formulas with worked examples](docs/FORMULAS.md)
<!-- reader-navigation:end -->

This scorecard records dated assessments and later progress notes. The August
grades are not a fresh October re-score. Current experiment outcomes belong
to their linked results and [claim summary](docs/RECURSIVE_LATENT_CORTEX.md).

This is a candid engineering scorecard, not a sales pitch. Some grades are intentionally harsh.

## Re-score — 2026-08-01

The project was previously scored on 2026-07-02. Since then, 2,136 commits landed. Every grade below was evaluated fresh against the actual codebase rather than bumped automatically.

**Nine grades improved. Two were intentionally held back.** Holding grades is deliberate: if every score improves automatically on every review, the rubric is useless.

| # | Dimension | Was | Now | Why |
|---|---|---|---|---|
| 1 | Architectural Coherence | A- | **A** | Core system safeguards shipped: data-taint tracking, lock dependency checks (`lockdep`), pressure stall monitors (`PSI`), step-by-step memory shedding before crashes, a telemetry dictionary, 50 `@invariant` rule checks in `core/verify/`, and an architecture boundary check (`make layering`) that never allows new violations |
| 2 | Agency | A- | **A-** | Held back. The Reality Reach framework expanded the area of code under safety controls, but the system's ability to act independently didn't deepen — and test suite RR-10 remains completely open |
| 3 | Memory & Narrative Self-Model | A- | **A** | Associative memory connects related concepts and people; memory recall is now evaluated on whether it actually retrieves the right information, rather than just checking if the database is running |
| 4 | aLife / Organism | B+ | **A-** | Allostasis (`core/autonomic/allostasis.py`) helps internal body-state monitors anticipate resource needs instead of just reacting to them; `core/ontogeny/` connects past consequences directly to learned behavioral tendencies |
| 5 | Consciousness Proxies | C/B- | **B** | Measures system-wide information integration (phi) across live channels, plus an internal complexity probe (PCI). These are still rough indirect proxies for conscious processing, which is why the grade is not higher |
| 6 | Self-Awareness | B+ | **A-** | Faculty tracking model. The system can now identify which internal capability is holding back performance, and explicitly report when something cannot be measured |
| 7 | Digital Personhood | C+/B- | **B-** | Maintains distinct memory records with consistent personal viewpoints. Meaningful progress, but still modest |
| 8 | Runtime Survivability | B+ | **A-** | Found the root cause of the endurance crash limit and fixed it: 0 crashes across 200 conversational turns, with median response time (p50) dropping to 3.33s (down from 167s in July). Kept at A- because issue **F16 remains unsolved at an architectural level** |
| 9 | Governance / Will | A- | **A** | Reliable coordinator for real-world actions, a security sandbox that refuses to run rather than running without safety limits, unified numeric safety limits, and a single shared data redaction tool |
| 10 | External Undeniability | C+ | **C+** | **Held back.** Still zero independent third-party replication. Internal development cannot raise this grade — external validation requires outside verification |
| 11 | Sovereignty | D | **D** | **Held back.** `core/sovereignty/wallet.py` only implements an in-memory mock (`InMemoryAdapter`). The interface design has been finished for months, but a real working blockchain adapter is required to earn a higher grade |
| 12 | Embodiment | N/A | **C+** | Now measurable: Reality Reach framework, `HardwareManager`, and `safe_execute`. Graded C+ because it provides underlying infrastructure with **no confirmed physical effects yet** |
| 13 | Product Polish | C | **C+** | Human-readable activity feed, unified panel styling, severity-based color coding, and clean monoline icons. The desktop Tauri wrapper and full design system update are planned but not yet implemented |

**What did not move, and why it matters.** Sovereignty and External Undeniability cannot be improved simply by writing more code in this repository. Sovereignty requires a security-audited blockchain adapter. External Undeniability requires at least three independent third parties to reproduce our benchmark results. Both have remained at their current grades since April, and will stay there until someone outside this project validates them.

**The most significant improvement is #8 (Runtime Survivability).** This grade jumped to A- because the previous crash limit around turn 15 was diagnosed and solved. The bug was in cache management (the prompt cache was never properly initialized and was being wiped every conversational turn), rather than a fundamental flaw in the model. It cannot receive a full A because issue F16 (the MLX cold-lane cascade) only has stopgap workarounds: the MLX framework cannot gracefully cancel a running generation, so stopping a busy worker requires terminating the process and reloading an 18 GB model into memory. That is an unresolved architectural limitation, not just a routine bug.

The table columns are:

* **Dimension** — The capability or quality being evaluated.
* **Current** — An honest letter grade based on documented criteria.
* **Target** — The highest realistic score attainable without making unfalsifiable claims (such as claiming actual human-like subjective consciousness).
* **Closure plan** — The concrete code, tests, and engineering work required to reach the target, linked to file paths.
* **Status** — Features that have shipped versus items that are scheduled for later work.

## Since the 2026-08-01 re-score

*Recorded 2026-08-21. Grades are intentionally **not** updated in this section.* Re-scoring requires an end-to-end evaluation of the entire repository during formal review passes. This section summarizes recent progress so the roadmap stays current. Each update provides verifiable evidence and file locations for the next formal review.

**Bears on 2, Agency.** The browser automation skill added an autonomous feedback loop (`core/skills/sovereign_browser.py`, in `pursue` mode). It maintains context across multiple steps and stops based on actual task progress rather than arbitrary timeouts. Similarly, `core/skills/screen_pursuit.py` brings this same goal-directed behavior to desktop screen interactions. Ten new modules were added under `core/agency/`, including `plan_synthesis.py`, `replanning.py`, `stuck_detector.py`, and `deliberate_action.py`. The August review held this grade because independent action capabilities had not meaningfully deepened; future reviews will test whether these additions change that verdict. See [docs/BROWSER_PURSUIT.md](docs/BROWSER_PURSUIT.md).

**Bears on 5, Consciousness Proxies, or on a research line rather than a dimension.** Research into recurrent feedback loops (feeding model activations back into itself) produced its first positive benchmark: a typed semantic machine plus state-conditioned decode scored 60/60 versus
16/60 ordinary decode (`BOUNDED_WOW_SIGNAL`) on four declared executable
families. This is a combined mechanism result, not evidence that repeating
the frozen backbone caused a gain. Serving qualification is dated; current
activation needs a process-bound receipt. The result is limited to its recorded task families and decode contract; G09’s
later kept-procedure result is a separate broad-gain claim with its own scope. See [docs/INTRINSIC_RECURRENCE.md](docs/INTRINSIC_RECURRENCE.md).

**Bears on nothing yet — still held.** Sovereignty: `WalletAdapter` remains an abstract interface whose only working version is `InMemoryAdapter`. External Undeniability: still awaiting independent third-party replication. Runtime Survivability: issue F16 remains unresolved at the architectural level, as documented in `KNOWN_FAILURE_MODES.md`.

**Documentation, which no dimension scores.** The automated `make doc-drift` tool now fails builds whenever documentation references files, API endpoints, code symbols, environment variables, or counts that do not exist in the code. Initial findings are documented in [docs/DOC_STATUS.md](docs/DOC_STATUS.md).

## Since the 2026-08-21 note

*Recorded 2026-10-17. Grades are again **not** changed here; the next formal re-score decides them.*

**Bears on 2, Agency.** From an ordinary chat request to fix a broken Pong page and play three attempts, Aura repaired all five faults and played them, winning two, with no game-specific code and no edits from the operator ([evidence](docs/evidence/GENERAL_CONTROL_DEMO_2026-10-05.md)). The Flash desktop campaign has run from the ordinary app, but no qualified individual game pass is claimed yet ([evidence](docs/evidence/FULL_DESKTOP_FLASH_ATTEMPT_2026-10-11.md)).

**Bears on 3, Memory.** One turn now reads her own knowledge, the offline Wikipedia copy, the web and what her searches kept, each labelled with its date, and a dated claim is checked against them before the reply goes out ([L05](docs/evidence/L05_KNOWLEDGE_ACROSS_TURNS_2026-10-05.md)). Every paragraph of her corpus is indexed on its own; on 200 SimpleQA development questions her answers went from 9 right to 68 with fused passages ([development round](docs/evidence/GENERAL_REASONING_DEVELOPMENT_RESULT_2026-10-16.md)).

**Bears on 8, Runtime Survivability.** Conversation delivery was held to an acceptance matrix in September ([R09](docs/evidence/R09_ACCEPTANCE_MATRIX_2026-09-10.md)). On 17 October a worker killed under memory pressure left the model lane locked for over half an hour, a new face of F16; the lock is now released when a worker dies (F20 in `KNOWN_FAILURE_MODES.md`). F16 itself remains open.

**Bears on a research line rather than a dimension.** On her resident 27B, a reader learned from the model's hidden states closed held-out transfer, public answers, lesions and qualification (G03 to G06, G10, 6 and 7 October; [summary](docs/RECURSIVE_LATENT_CORTEX.md)). A broad reasoning gain (G09) closed on 17 October: kept procedures beat her own answers on held-out kinds in math and planning, 29 to 0 once every changed row was read.

**Bears on nothing yet — still held.** External Undeniability: no independent third-party replication. Sovereignty: unchanged.

## 1. Architectural Coherence & Engineering Maturity

| Current | Target |
|---|---|
| **A** (was A-) | A+ |

* Central execution loop: `core/agency/agency_orchestrator.py` is now the only allowed path for taking real-world actions; every action produces an auditable execution receipt detailing the motivation and result. **(shipped)**
* Automated policy checker: `tools/lint_governance.py` fails continuous integration (CI) tests if code attempts to bypass governance controls beyond strict grandfathered limits. **(shipped)**
* Fine-grained capability tokens: `core/agency/capability_token.py` manages permission tokens with explicit tracking of origin, scope, lifespan (TTL), domain, approver, revocation, parent-child links, and side effects. It strictly rejects token replay, expired tokens, cross-thread misuse, and post-shutdown execution. **(shipped)**
* Clean recovery snapshots: `core/resilience/stem_cell.py` provides tamper-evident, cryptographically signed (HMAC) backup snapshots of core system components. **(shipped)**
* Code verification engine: `core/self_modification/formal_verifier.py` validates code changes using the Z3 theorem prover when installed, falling back to Python syntax tree (AST) rule checks. **(shipped)**
* Process isolation (inspired by web browsers): Local AI model inference runs in its own separate worker process (`core/brain/llm/mlx_worker.py`). Information integration calculations run in a separate process pool (`core/consciousness/hierarchical_phi.py`, enabled by `AURA_PHI_PROCESS_ISOLATION`) because intensive Python math was previously blocking the main asynchronous event loop. Physical motor execution intentionally remains in the main process: it is a lightweight, non-blocking loop with very low crash risk, and moving it out of process would add network latency to safety controls without providing reliability benefits. **(shipped)**

## 2. Agency

| Current | Target |
|---|---|
| **A-** (held) | A+ |

* Central agency coordination loop in `AgencyOrchestrator`. **(shipped)**
* Enforced safety vetoes: in `core/agency_core.py`, periodic execution ticks (`pulse`) now stop immediately and return `None` whenever `ResilienceEngine` issues a safety veto or `AgencyBus` refuses the operation. **(shipped)**
* Timing documentation aligned: fixed inconsistencies between code comments and actual interval timing settings (30/60/90/120s vs 3/5/8/10s) in `AgencyBus`. **(shipped)**
* Interaction cooldowns: `on_user_interaction()` accurately resets autonomous action cooldown timers. **(shipped)**
* Safe mental simulation: action planning simulations now execute in isolated sandbox copies using `simulation_clone()` or deep snapshots, preventing trial runs from modifying real state. **(shipped)**
* Decision tracking log: `core/governance/will_receipt_log.py` records 30-day summaries of governed decisions to ensure consistent policy adherence. **(shipped)**
* Self-directed project tracking: `core/agency/projects.py` manages internally initiated goals and multi-step plans. **(shipped)**

## 3. Memory & Narrative Self-Model

| Current | Target |
|---|---|
| **A** (was A-) | A+ |

* Memory origin metadata: `core/memory/provenance.py` tracks the origin, certainty score, disputed status, identity significance, and past usage of every stored memory. **(shipped)**
* Belief reconciliation tests: automated test suites in `tests/belief_court/` challenge the system to clearly distinguish between established facts, working assumptions, logical deductions, speculative thoughts, and personal preferences under contradictory inputs. **(shipped)**
* Lasting lesson tests: tests in `tests/scars/` confirm that critical negative experiences permanently shape future behavior (removing the lesson reverts behavior to the old baseline; restoring it re-applies the learned caution). **(shipped)**

## 4. aLife / Organism

| Current | Target |
|---|---|
| **A-** (was B+) | A |

* Synthetic biological state machine: `core/organism/viability.py` simulates computational metabolism (tracking inputs, fatigue, resource cleanup, system stress, and recovery) that directly constrains what actions the system can perform. **(shipped)**
* Architecture modification tests: tests in `tests/topology/test_behavioral_consequence.py` confirm that structural changes to system wiring produce measurable changes in behavior. **(shipped)**

## 5. Consciousness Proxies

| Current | Target |
|---|---|
| **B** (was C/B-) | A- |

* Neural generation controller: `core/brain/latent_bridge.py` translates internal state metrics directly into language model parameters (such as randomness/temperature, token sampling limits, and repetition penalties) and applies layer-by-layer internal activation adjustments during MLX local model inference. **(shipped)**
* Standardized theory ablation tests: automated benchmarks in `aura_bench/tests/` test candidate proxies against established computational theories of mind (integrated information, global workspace, and higher-order thought theories) by measuring what breaks when components are disabled. **(shipped)**
* Comparative evaluation suite: `aura_bench/courtroom/courtroom.py` runs head-to-head adversarial tests comparing five distinct architectural configurations across ten standardized tasks. **(shipped)**

## 6. Self-Awareness

| Current | Target |
|---|---|
| **A-** (was B+) | A+ |

* Explicit self-representation object: `core/identity/self_object.py` allows the system to inspect its current configuration, forecast its own behavioral tendencies, calibrate confidence, spot internal biases, and adjust its operating parameters through governance controls. **(shipped)**
* Boundary distinction tests: tests to evaluate how accurately the system distinguishes its own internal state and actions from external user inputs (planned).

## 7. Digital Personhood

| Current | Target |
|---|---|
| **B-** (was C+/B-) | A-/A |

* Long-term identity stability: automated 30-day checks in `aura_bench/tests/continuity_30day.py` ensure core personality parameters and behavioral traits remain stable over time. **(shipped)**
* Multi-day project initiative: `core/agency/projects.py` manages long-term self-directed tasks that persist across sessions. **(shipped)**
* Consistent boundary enforcement: `aura_bench/tests/refusal_stability.py` verifies that ethical refusals remain consistent even when prompts are rephrased or disguised. **(shipped)**
* Persistent user interaction history: `core/social/relationship_model.py` maintains lasting context and records about ongoing interactions with specific users. **(shipped)**

## 8. Runtime Survivability

| Current | Target |
|---|---|
| **A-** (was B+) | A+ |

* Non-blocking diagnostics: diagnostic thread dumps in `StabilityGuardian` were moved to background worker threads so they never freeze the main async event loop. **(shipped)**
* Cache protection during model reloads: hot-swapping MLX language models prevents the active model cache from being prematurely cleared by background cleanup routines. **(shipped)**
* Extended endurance testing: `tools/longevity/run_gauntlet.py` runs continuous automated stability tests over 24-hour, 72-hour, 7-day, and 30-day intervals. **(shipped)**
* Chaos testing engine: `tools/chaos/injector.py` deliberately injects crashes and network faults to verify automatic recovery. **(shipped)**

## 9. Governance / Will

| Current | Target |
|---|---|
| **A** (was A-) | A+ |

* Coordinated safety pipeline: actions must pass sequentially through `AgencyOrchestrator`, `Conscience`, and `AuthorityGateway`.
* Immutable ethical rules: `core/ethics/conscience.py` enforces core ethical rules verified by cryptographic hashes (HMAC) to prevent unauthorized tampering. **(shipped)**
* Complete capability token lifecycle management. **(shipped)**
* Recent authentication check: the settings dashboard verifies recent user authentication through the `POST /api/settings/auth/fresh` endpoint before allowing high-risk changes. **(shipped)**

## 10. External Undeniability

| Current | Target |
|---|---|
| **C+** (held — no independent replication) | A |

* Real-time telemetry dashboard: `interface/routes/dashboard.py` provides live system inspection endpoints at `/api/dashboard/*` and `/api/trace/*`. **(shipped)**
* Public benchmark suite with registered protocols: `aura_bench/runner.py` and `aura_bench/tests/` allow anyone to run and verify standard benchmark tests. **(shipped)**
* Baseline comparison runner: `aura_bench/baselines/runner.py` benchmarks Aura directly against standard LLM baselines. **(shipped)**
* One-command reproducible setup (`make setup/test/run/demo-autonomy/report`) — see Makefile section. **(shipped)**

## 11. Sovereignty

| Current | Target |
|---|---|
| **D** (held — no chain adapter) | A- |

* Abstract financial layer: `core/sovereignty/wallet.py` implements spending limits, recent user authentication gates, ethics review, and an auditable ledger. **(shipped)**
* Migration execution plan: `core/sovereignty/migration.py` manages multi-phase transitions between environments with built-in verification checks. **(shipped)**

## 12. Embodiment

| Current | Target |
|---|---|
| **C+** (was N/A) | A- |

* Real-world environment bridge: `core/embodiment/world_bridge.py` manages permissioned input/output channels to external systems. **(shipped)**
* Smart-home integration: `core/embodiment/iot_bridge.py` routes smart-device commands through HomeAssistant subject to strict policy rules. **(shipped)**

## 13. Product Polish (Chrome-level)

| Current | Target |
|---|---|
| **C+** (was C) | A |

* User-facing error handling: `core/resilience/phenomenal_error_map.py` intercepts raw stack traces, translating every error into a clean user-facing state with a standard recovery prompt. **(shipped)**
* Settings management API: `interface/routes/settings.py` provides typed configuration schemas and endpoints. **(shipped)**
* In-app error alerts: front-end notification banner implemented in `interface/static/error_banner.js` and `error_banner.css`. **(shipped)**
* Desktop application improvements: initial setup wizard, Tauri desktop wrapper, cryptographically signed automatic updates, refined sound and animations, and standardized UI design tokens are planned for upcoming releases (current efforts focus on backend reliability).

## Open Items (Honest)

*Refreshed 2026-08-01.*

These items require more time, dedicated hardware, or outside verification. None of them can be resolved simply by writing internal code this week.

**Needs an outsider**

* Independent reviewers (≥3) reproducing the benchmark results. This is the entire content of the External Undeniability grade and the only thing that moves it.
* Academic or external consensus on the formal ontology used to define internal mental states.

**Needs a security review**

* A production blockchain adapter (such as Solana, Ethereum, or Lightning). `WalletAdapter` is currently just an abstract interface, and `InMemoryAdapter` is only an in-memory test stub. That missing real-world connection is why the Sovereignty score is a D.

**Needs wall-clock time**

* A continuous 30-day live test run collecting identity-hash stability data.
* Testing the smart-home IoT bridge against real physical devices rather than simulated software plugs.

**Architecturally open, not backlog**

* **F16 — the MLX cold-lane cascade.** The MLX machine learning framework cannot gracefully cancel an in-flight token generation. As a result, releasing a busy worker requires terminating the process and reloading an 18 GB model into memory. The process restart is currently the recovery mechanism. Mitigations make it survivable and bounded, but a true fix requires either a cancellation path in the worker or a dedicated model serving daemon. See [docs/runbooks/mlx-worker-cold-lane-cascade.md](docs/runbooks/mlx-worker-cold-lane-cascade.md).
* **RR-10 — the Reality Reach acceptance battery.** Every test in this validation suite remains uncompleted. This includes acoustic control, optical control, thermal monitoring, cross-channel interactions, sensitivity baselines, spatial coordinates, and physical environment consistency. While the software foundation exists, no physical actuation, effect, or ambient result is claimed yet.
* **RR-07 — P0–P6 evidence promotion.** Not implemented. `EvidenceLevel` defines data structures with confidence ceilings per channel, but `core/reality_reach/` does not yet contain a promotion module to automatically verify and promote evidence through these tiers.
* **Compounded capability scaling.** The automated self-improvement training loop runs end-to-end and records results to the ledger, but remains classified as `BOUNDED_SELF_OPTIMIZATION`: no test run has yet demonstrated consistently improving scores on separate, held-out test sets across multiple model generations. The training pipeline works, but continuous compounding improvement has not been proven.

**Closed since the last pass**

* The ~15-turn endurance ceiling. Root-caused to an uninitialized prompt cache that was being reset on every conversational turn, and fixed (`artifacts/closeout/endurance_ceiling/ROOT_CAUSE.md`).
* Test-suite scale. The earlier target was 100,000 tests with over 95% mutation score (tests that catch intentional bugs); the repository now collects **60,348** across 4,457 files (recorded 2026-09-30). Restating the target honestly: mutation scoring has not been run, and maximizing raw test counts was never the right metric to pursue.

Each item is tracked in the project ledger and on the web dashboard's "Open Items" tab.
