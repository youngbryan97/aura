# ROADMAP — Path to a Perfect Score

This document is the live, code-grounded map from where Aura is today to
the highest defensible score across every dimension reviewers have
flagged. It is an engineering artifact, not a marketing pitch.

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
| A- | A+ |

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
* Multiprocess organ isolation (Chromium-style): see
  `core/embodiment/world_bridge.py` shell-sandbox for the pattern; the
  next pass extracts MLX worker, motor cortex, and phi_core into
  separate IPC processes. **(staged)**

## 2. Agency

| Current | Target |
|---|---|
| A- | A+ |

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
| A- | A+ |

* Memory provenance envelope: `core/memory/provenance.py` —
  source/confidence/contested/identity_relevant/recalled_in_actions.
  **(shipped)**
* Belief court adversarial revision tests under
  `tests/belief_court/` — distinguishing memory/belief/inference/fantasy
  /preference under pressure. **(staged)**
* Irreversible epistemic scar test under `tests/scars/` — ablate the
  scar, behavior reverts; restore the scar, behavior re-changes.
  **(staged)**

## 4. aLife / Organism

| Current | Target |
|---|---|
| B+ | A |

* Viability state machine: `core/organism/viability.py` — explicit
  metabolism (food / fatigue / waste / injury / healing) and behaviorally
  load-bearing states. **(shipped)**
* Topology mutation behavioral consequence test — staged in
  `tests/topology/`. **(staged)**

## 5. Consciousness Proxies

| Current | Target |
|---|---|
| C / B- | A- |

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
| B+ | A+ |

* Explicit "I" object: `core/identity/self_object.py` — snapshot,
  introspect, predict_self, calibrate, debug_bias, adjust (via Will).
  **(shipped)**
* Self / other boundary tests — staged.

## 7. Digital Personhood

| Current | Target |
|---|---|
| C+/B- | A-/A |

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
| B+ | A+ |

* StabilityGuardian thread-dump moved off the event loop. **(shipped)**
* MLX hot-swap protected against background eviction of the warm cortex.
  **(shipped)**
* 24h/72h/7d/30d longevity gauntlet runner:
  `tools/longevity/run_gauntlet.py`. **(shipped)**
* Crash injector: `tools/chaos/injector.py`. **(shipped)**

## 9. Governance / Will

| Current | Target |
|---|---|
| A- | A+ |

* AgencyOrchestrator + Conscience + AuthorityGateway chain.
* Conscience: `core/ethics/conscience.py` — irrevocable rule floor with
  HMAC-pinned rule hash. **(shipped)**
* Capability token full lifecycle. **(shipped)**
* Settings panel exposes the fresh-user-auth signal at
  `POST /api/settings/auth/fresh`. **(shipped)**

## 10. External Undeniability

| Current | Target |
|---|---|
| C+ | A |

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
| D | A- |

* Wallet abstract economic layer with per-spend cap, fresh-auth gate,
  Conscience gate, and auditable ledger: `core/sovereignty/wallet.py`.
  **(shipped)**
* Migration runbook with phase machine + verifier:
  `core/sovereignty/migration.py`. **(shipped)**

## 12. Embodiment

| Current | Target |
|---|---|
| N/A | A- |

* WorldBridge with permissioned channels:
  `core/embodiment/world_bridge.py`. **(shipped)**
* IoT bridge with policy rules and HomeAssistant transport:
  `core/embodiment/iot_bridge.py`. **(shipped)**

## 13. Product Polish (Chrome-level)

| Current | Target |
|---|---|
| C | A |

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

These items require time, infrastructure, or external cooperation that
cannot be condensed into one engineering pass:

* 30-day actual run with full continuity-hash time series.
* Independent reviewers (≥3) reproducing benchmark results.
* Philosopher-of-mind consensus on the formal ontology.
* A 100,000-test suite with mutation-test scores >95%.
* IoT bridge against the user's specific home network.
* Real crypto wallet adapter (Solana / Ethereum / Lightning) — the
  abstract layer is shipped; the adapter requires a security review.

Each of those is captured as a tracked issue in the project ledger and
the dashboard's "Open Items" tab.
