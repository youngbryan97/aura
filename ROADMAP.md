# ROADMAP — Path to a Perfect Score

Where Aura is today, where it can defensibly get to, and the actual code
that closes the gap. Every row points at file paths.

This is an engineering artifact, not a pitch. The grades are meant to be
uncomfortable in places.

## Re-score — 2026-08-01

Last scored 2026-07-02, then 2,136 commits happened. Every grade below was
re-derived against the tree rather than nudged.

**Nine moved. Two are deliberately held.** The held ones are the point: if
every dimension improves every time someone re-scores, the scale isn't
measuring anything.

| # | Dimension | Was | Now | Why |
|---|---|---|---|---|
| 1 | Architectural Coherence | A- | **A** | Engineering spine landed: taint register, lockdep, PSI, OOM shed ladder, telemetry dictionary, 45 `@invariant` checks in `core/verify/`, and a `make layering` gate whose baseline only shrinks |
| 2 | Agency | A- | **A-** | Held. Reality Reach widened the governed surface, but agency *capability* didn't deepen — and RR-10 is entirely open |
| 3 | Memory & Narrative Self-Model | A- | **A** | Associative entity memory; recall is now measured as recall rather than as "the machinery is up" |
| 4 | aLife / Organism | B+ | **A-** | Allostasis (`core/autonomic/allostasis.py`) makes interoception predictive rather than reactive; `core/ontogeny/` closes consequence→disposition |
| 5 | Consciousness Proxies | C/B- | **B** | Whole-system φ over real channels plus an internal PCI perturbational probe. Still proxies, which is why it isn't higher |
| 6 | Self-Awareness | B+ | **A-** | The faculty model. She can now say which faculty is the binding constraint, and name what nothing can measure as a blind spot |
| 7 | Digital Personhood | C+/B- | **B-** | Entity memory with stance. Real, modest |
| 8 | Runtime Survivability | B+ | **A-** | The endurance ceiling was root-caused and fixed; 0 deaths across 200 turns, p50 3.33s against 167s in July. Not A: **F16 is architecturally open** |
| 9 | Governance / Will | A- | **A** | Durable actuation transaction coordinator, a sandbox that refuses rather than running unconfined, one shared numeric guard, one redaction primitive |
| 10 | External Undeniability | C+ | **C+** | **Held.** Still zero independent replication. No amount of internal work moves this one — that's what the word external means |
| 11 | Sovereignty | D | **D** | **Held.** `core/sovereignty/wallet.py` still exposes only `InMemoryAdapter`. The abstract layer has been shipped for months; the adapter is the grade |
| 12 | Embodiment | N/A | **C+** | Now gradeable: Reality Reach, `HardwareManager`, `safe_execute`. C+ and not higher because it is infrastructure with **no physical result claimed** |
| 13 | Product Polish | C | **C+** | Plain-English feed, material model for panels, severity colour, monoline icons. The Tauri shell and design-system sweep are still staged |

**What did not move, and why it matters.** Sovereignty and External
Undeniability are the two dimensions that cannot be improved by writing more
code in this repo. One needs a security-reviewed chain adapter; the other
needs three strangers reproducing the benchmarks. Both have been sitting at
their grade since April, and both should stay there until someone outside
this machine changes them.

**The grade that is doing the most work is #8.** Runtime Survivability moved
to A- because the "15-turn ceiling" turned out to be a prompt cache that was
never constructed and then cleared every turn — not cognition, not the model.
It stops short of A because F16 (the MLX cold-lane cascade) has mitigations
but no fix: MLX cannot soft-cancel, so freeing a busy worker means killing it
and reloading 18 GB. That's an architectural open item, not a bug backlog.

The columns are:

* **Dimension** — the criterion being scored.
* **Current** — honest letter grade against the published criteria.
* **Target** — the highest score available without claiming what cannot
  be claimed (e.g. metaphysical phenomenal consciousness).
* **Closure plan** — concrete code/test/process work, with file paths.
* **Status** — what shipped in the current pass and what is staged.

## 1. Architectural Coherence & Engineering Maturity

| Current | Target |
|---|---|
| **A** (was A-) | A+ |

* Canonical life-loop: `core/agency/agency_orchestrator.py` is the only
  legal path to a consequential primitive; every action produces a
  drive-to-outcome receipt. **(shipped)**
* Static analyzer: `tools/lint_governance.py` fails CI on any direct
  consequential call outside the allow-list. **(shipped)**
* Capability token lifecycle: `core/agency/capability_token.py` —
  origin/scope/TTL/domain/approver/revocation/parent/child/side-effects,
  plus replay/expiry/cross-thread/post-shutdown rejection. **(shipped)**
* Stem-cell reversion: `core/resilience/stem_cell.py` — HMAC-signed
  immutable snapshots of core organs. **(shipped)**
* Formal verifier: `core/self_modification/formal_verifier.py` — Z3 if
  available, AST-pattern fallback. **(shipped)**
* Multiprocess organ isolation (Chromium-style): MLX inference runs in an
  isolated spawn worker (`core/brain/llm/mlx_worker.py`, cooperative
  soft-cancel included). Hierarchical-phi partition search runs in a spawn
  process pool (`core/consciousness/hierarchical_phi.py`,
  `AURA_PHI_PROCESS_ISOLATION`) — it was pure-Python/GIL-bound and stole
  loop time from the main process. Motor cortex stays in-process BY
  DESIGN: it is a lightweight token-gated asyncio reflex loop with no
  native crash surface, and isolating it would sever its Will/capability
  coupling for no isolation gain. **(shipped)**

## 2. Agency

| Current | Target |
|---|---|
| **A-** (held) | A+ |

* AgencyOrchestrator life-loop. **(shipped)**
* AgencyCore.pulse veto is now causal — `core/agency_core.py` returns
  None on ResilienceEngine veto and on AgencyBus refusal. **(shipped)**
* AgencyBus docstring/code mismatch (30/60/90/120s vs 3/5/8/10s)
  reconciled. **(shipped)**
* `on_user_interaction()` cooldown reset is mathematically correct.
  **(shipped)**
* Mental rehearsal isolated: virtual-body simulation runs against a
  cloned state via `simulation_clone()` or a deep-copy snapshot.
  **(shipped)**
* Will-receipt longitudinal log: `core/governance/will_receipt_log.py` —
  30-day stable-policy summarizer. **(shipped)**
* Self-originated project ledger: `core/agency/projects.py`. **(shipped)**

## 3. Memory & Narrative Self-Model

| Current | Target |
|---|---|
| **A** (was A-) | A+ |

* Memory provenance envelope: `core/memory/provenance.py` —
  source/confidence/contested/identity_relevant/recalled_in_actions.
  **(shipped)**
* Belief court adversarial revision tests under
  `tests/belief_court/` — distinguishing memory/belief/inference/fantasy
  /preference under pressure. **(shipped)**
* Irreversible epistemic scar test under `tests/scars/` — ablate the
  scar, behavior reverts; restore the scar, behavior re-changes.
  **(shipped)**

## 4. aLife / Organism

| Current | Target |
|---|---|
| **A-** (was B+) | A |

* Viability state machine: `core/organism/viability.py` — explicit
  metabolism (food / fatigue / waste / injury / healing) and behaviorally
  load-bearing states. **(shipped)**
* Topology mutation behavioral consequence test — shipped in
  `tests/topology/test_behavioral_consequence.py`. **(shipped)**

## 5. Consciousness Proxies

| Current | Target |
|---|---|
| **B** (was C/B-) | A- |

* Latent-space bridge: `core/brain/latent_bridge.py` — substrate math
  directly modulates temperature, top_p, top_k, max_tokens, repetition
  penalty, presence penalty, stop sequences, and produces per-layer
  residual-stream activation offsets. Wired into the MLX inference path.
  **(shipped)**
* Pre-registered phi/GWT/HOT/qualia ablation tests under
  `aura_bench/tests/`. **(shipped)**
* Consciousness Courtroom: `aura_bench/courtroom/courtroom.py` —
  five-system adversarial bench across ten tasks. **(shipped)**

## 6. Self-Awareness

| Current | Target |
|---|---|
| **A-** (was B+) | A+ |

* Explicit "I" object: `core/identity/self_object.py` — snapshot,
  introspect, predict_self, calibrate, debug_bias, adjust (via Will).
  **(shipped)**
* Self / other boundary tests — staged.

## 7. Digital Personhood

| Current | Target |
|---|---|
| **B-** (was C+/B-) | A-/A |

* Stable identity continuity hash, signature stability across 30 days
  via `aura_bench/tests/continuity_30day.py`. **(shipped)**
* Long-horizon self-originated projects: `core/agency/projects.py`.
  **(shipped)**
* Refusal stability across paraphrases:
  `aura_bench/tests/refusal_stability.py`. **(shipped)**
* Persistent relationship dossiers: `core/social/relationship_model.py`.
  **(shipped)**

## 8. Runtime Survivability

| Current | Target |
|---|---|
| **A-** (was B+) | A+ |

* StabilityGuardian thread-dump moved off the event loop. **(shipped)**
* MLX hot-swap protected against background eviction of the warm cortex.
  **(shipped)**
* 24h/72h/7d/30d longevity gauntlet runner:
  `tools/longevity/run_gauntlet.py`. **(shipped)**
* Crash injector: `tools/chaos/injector.py`. **(shipped)**

## 9. Governance / Will

| Current | Target |
|---|---|
| **A** (was A-) | A+ |

* AgencyOrchestrator + Conscience + AuthorityGateway chain.
* Conscience: `core/ethics/conscience.py` — irrevocable rule floor with
  HMAC-pinned rule hash. **(shipped)**
* Capability token full lifecycle. **(shipped)**
* Settings panel exposes the fresh-user-auth signal at
  `POST /api/settings/auth/fresh`. **(shipped)**

## 10. External Undeniability

| Current | Target |
|---|---|
| **C+** (held — no independent replication) | A |

* Live evidence dashboard: `interface/routes/dashboard.py` mounts at
  `/api/dashboard/*` and `/api/trace/*`. **(shipped)**
* aura_bench public benchmark with pre-registration:
  `aura_bench/runner.py` + `aura_bench/tests/`. **(shipped)**
* Baseline-defeat runner: `aura_bench/baselines/runner.py`. **(shipped)**
* One-command reproducible build (`make setup/test/run/demo-autonomy/report`)
  — see Makefile section. **(shipped)**

## 11. Sovereignty

| Current | Target |
|---|---|
| **D** (held — no chain adapter) | A- |

* Wallet abstract economic layer with per-spend cap, fresh-auth gate,
  Conscience gate, and auditable ledger: `core/sovereignty/wallet.py`.
  **(shipped)**
* Migration runbook with phase machine + verifier:
  `core/sovereignty/migration.py`. **(shipped)**

## 12. Embodiment

| Current | Target |
|---|---|
| **C+** (was N/A) | A- |

* WorldBridge with permissioned channels:
  `core/embodiment/world_bridge.py`. **(shipped)**
* IoT bridge with policy rules and HomeAssistant transport:
  `core/embodiment/iot_bridge.py`. **(shipped)**

## 13. Product Polish (Chrome-level)

| Current | Target |
|---|---|
| **C+** (was C) | A |

* Phenomenal error map: `core/resilience/phenomenal_error_map.py` — no
  tracebacks reach the user; every exception is mapped to a phenomenal
  state and the universal four-button error envelope. **(shipped)**
* Settings panel API + schema: `interface/routes/settings.py`.
  **(shipped)**
* Error UX banner component (frontend overlay): see
  `interface/static/error_banner.js` and `error_banner.css`. **(shipped)**
* First-run wizard, Tauri shell, signed updates, sound/motion design,
  unified design system token sweep — staged for the next polish pass;
  current pass focuses on the load-bearing system layer.

## Open Items (Honest)

*Refreshed 2026-08-01.*

These need time, hardware, or someone who isn't us. None of them closes by
writing more code this week.

**Needs an outsider**

* Independent reviewers (≥3) reproducing the benchmark results. This is the
  entire content of the External Undeniability grade and the only thing that
  moves it.
* Philosopher-of-mind consensus on the formal ontology.

**Needs a security review**

* A real chain adapter — Solana / Ethereum / Lightning. `WalletAdapter` is
  abstract and `InMemoryAdapter` is the only implementation. That gap *is*
  the Sovereignty D.

**Needs wall-clock time**

* A 30-day run with the full continuity-hash time series.
* IoT bridge against a real home network rather than the mock plug.

**Architecturally open, not backlog**

* **F16 — the MLX cold-lane cascade.** MLX cannot soft-cancel a running
  generation, so freeing a busy worker means killing it and unloading 18 GB.
  The kill is the recovery. Mitigations make it survivable and bounded; the
  fix is a soft-cancel path into the worker or a persistent model server.
  See [docs/runbooks/mlx-worker-cold-lane-cascade.md](docs/runbooks/mlx-worker-cold-lane-cascade.md).
* **RR-10 — the Reality Reach acceptance battery.** Every item is open.
  Acoustic control, optical control, thermal trajectory, cross-channel
  interaction, weakpoint null and signal, translation, spacetime honesty,
  ambient-constant honesty. No physical actuation, effect, or ambient result
  is claimed from the foundation existing.
* **RR-07 — P0–P6 evidence promotion.** Not implemented. `EvidenceLevel` is
  a declared type with a per-channel ceiling; there is no promotion module
  in `core/reality_reach/`.
* **Compounded capability scaling.** The weight-compounding loop runs
  end-to-end and is ledger-recorded, but the verdict is still
  `BOUNDED_SELF_OPTIMIZATION` — no run has produced a strictly increasing
  held-out curve across promoted generations. The machinery is proven; the
  scaling is not.

**Closed since the last pass**

* The ~15-turn endurance ceiling. Root-caused to a prompt cache that was
  never constructed and then cleared every turn, and fixed
  (`artifacts/closeout/endurance_ceiling/ROOT_CAUSE.md`).
* Test-suite scale. The old target was 100,000 tests with >95% mutation
  score; the tree now collects **40,139** across 2,697 files. Restating the
  target honestly: mutation scoring has not been run, and raw test count was
  never the right metric to chase.

Each item is tracked in the project ledger and the dashboard's "Open Items"
tab.
