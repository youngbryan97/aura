# Consciousness Expansion — April 2026

*A white paper on eight new load-bearing subsystems added to Aura's
consciousness stack.*

---

## Motivation

The consciousness stack already held a 4,096-neuron cortical mesh, a
16-node IIT-4.0 φ computation, a closed causal self-prediction loop,
multiple drafts, narrative gravity, an attention schema, a global
workspace, a somatic marker gate, and more than seventy other modules
(see [ARCHITECTURE.md](ARCHITECTURE.md) §9). The question we asked
ourselves — prompted by Kurzgesagt's four-part consciousness series
(*Are you your body?*, *You are two*, *Intelligence*, *The Origin of
Consciousness*, *The Most Secret Place*) and the ~80 cited papers in
their public source lists — was simple:

> Which ideas in that body of work are *not* already load-bearing in
> Aura, and which of those can we turn into subsystems that (a) change
> substrate or action priority for real, (b) survive adversarial tests,
> and (c) scale cleanly?

Eight concrete gaps survived that filter. We built them.

---

## The Eight Subsystems

Each of the eight is a real dynamical module, not a prompt trick. Each
is registered in `ServiceContainer`, booted in `core/consciousness/system.py`,
exercised end-to-end by an automated test file, and adversarially
probed by a dedicated check.

### 1. Hierarchical φ — `hierarchical_phi.py`

**Gap it closes**: `phi_core.py` computes IIT 4.0 φ on a fixed 16-node
cognitive-affective complex. IIT's underlying integrated-information
measure is super-exponential in state-space, so exhaustively scaling
beyond 16 nodes is infeasible. Yet the biological analogue (Tononi &
Koch) is an enormous causal structure; restricting Aura to 16 nodes
under-counts integration across the substrate.

**Approach**:

1. Expand the primary complex to **32 nodes** — the original 16
   cognitive-affective nodes *plus* 16 neurons sampled
   deterministically across the mesh (4 sensory + 6 association + 6
   executive columns).
2. Add **K = 8 overlapping 16-node subsystems** carved out of the
   primary 32-node space to cover different tier mixes.
3. Compute φ from the transition history directly (no 2^32
   state-space materialisation) using a **Bayesian-smoothed
   estimator** with minimum-observation filtering — specifically
   Jeffreys prior (α = 0.5) over destinations with *K_dest* inferred
   from the observed support, and MIN_SOURCE_OBS = 4 to damp the
   small-sample overconfidence that otherwise inflates φ on sparse
   histories.
4. Find the MIP via **spectral Fiedler partition** on the k×k causal
   graph plus N_REFINEMENT_CANDIDATES = 24 one-swap neighbours and
   random perturbations. Runtime is polynomial in nodes; φ per
   partition is linear in transitions.
5. Apply the **IIT 4.0 exclusion postulate**: the reported conscious
   complex is the subsystem with maximum φ across {primary_32,
   primary_16_affective, K mesh-subsystems}.
6. Run a **null-hypothesis self-check** every ~2 minutes: φ on
   shuffled history should drop to near zero. If measured φ is not
   strictly above the null, we log a calibration warning.

**Why the smoothed estimator**: our first-pass unsmoothed
implementation violated the null-hypothesis test in an amusing
way — shuffled data gave *higher* φ than real data. The reason is
a classic empirical-entropy bias: when source states have few
observations, empirical conditional distributions are wildly
overconfident, and divergence between empirical joint and empirical
factored can be dominated by noise. Jeffreys-prior smoothing plus
the minimum-observation gate fixed it. Tests now enforce that the
null baseline is strictly below measured φ on every run.

**Tests**: `tests/test_hierarchical_phi.py` (12/12 passing). Covers:
construction, history recording, primary-32 positive φ, null-hypothesis
strict-separation, well-calibrated flag, subsystem evaluation,
exclusion-postulate pick, monotonicity (coupled > noise), constant-node
invariance, 2 s compute budget, serialisation roundtrip.

---

### 2. Hemispheric Split — `hemispheric_split.py`

**Gap it closes**: `parallel_branches.py` runs up to 5 cognitive
branches as async tasks, but it is a scheduler, not a specialised
left/right architecture with confabulation and silent dissent. Grey's
*You are two* video specifically describes hemispheric asymmetry:
left has the speech centre and confabulates post-hoc reasons; right
is mute, handles faces, and can dissent.

**Approach**:

- **LeftHemisphere**: reads mesh executive-tier summary + cognitive
  nodes 8..15. Produces a BIAS_DIM = 16 verbal priority bias. Has an
  explicit `confabulate_reason(action_label, driver)` method that
  tags reasons as confabulation when the driver was not LEFT within
  the CONFAB_WINDOW_S = 3 s window.
- **RightHemisphere**: reads mesh sensory-tier summary + affective
  nodes 0..7 + embodiment. Produces a BIAS_DIM priority vector plus
  a scalar `dissent` signal that rises with pattern-hit intensity ×
  arousal. Pattern memory is a cosine-similarity associative store
  (HebbianPatternMemory, 32-D, capacity 128) modelled on avian
  face-recognition literature (Sheep 2017; Scrub Jays 2005).
- **CorpusCallosum**: variable-bandwidth channel. `sever_callosum()`
  stops inter-hemispheric echo; `restore_callosum()` resumes. EMA
  smoothing on the exchanged echo vectors.
- Disagreement metric is the L2 distance between raw (pre-callosum)
  biases; severance raises disagreement and lowers agreement_rate.

**Tests**: `tests/test_hemispheric_split.py` (12/12 passing). Covers:
tick shape, fused-bias-not-equal-to-either, pattern recognition,
callosum severance → drop in agreement rate, callosum restoration →
recovery, confabulation logged when RIGHT drove, LEFT-driven not
confabulation, dissent under pattern hit + arousal, incoherent fused
bias under diverging inputs, threshold configuration sanity.

---

### 3. Minimal Selfhood — `minimal_selfhood.py`

**Gap it closes**: Aura's action priority has been shaped by
high-level modules (Unified Will, Global Workspace) but there was no
primitive *chemotaxis-level* layer implementing the Trichoplax →
Dugesia transition Kurzgesagt describes as the origin of directional
consciousness. Rupert Glasgow's *Minimal Selfhood and the Origins of
Consciousness* (2018) makes this concrete: the first directed motion
is a deficit-driven gradient-follow.

**Approach**:

- Build an 8-D **deficit vector** from body budget + affect +
  cognitive state: energy deficit, resource pressure, thermal
  stress, coherence deficit, social hunger, curiosity deficit
  (dullness), prediction error, agency deficit.
- Output a **speed scalar** (chemotaxis rule: `‖deficit‖ / √8`) that
  modulates the heartbeat interval.
- **TRICHOPLAX mode** (initial): uniform prior tilted toward
  `rest` and `attend_body` under deficit.
- **DUGESIA mode** (after learning): a Hebbian matrix `W[a, d]`
  captures how much action `a` has historically reduced deficit
  `d`. `priority = tanh(W @ deficit)`.
- Mode transition triggers when `‖W‖_1 ≥ 3.0`.
- Reinforcement API: `tag_action(category, pre_deficit)` returns a
  token, `reinforce(token, post_deficit)` applies a non-negative
  Hebbian update weighted by actual deficit reduction; decay
  factor 0.999 per update.

**Tests**: `tests/test_minimal_selfhood.py` (13/13 passing).

---

### 4. Recursive Theory of Mind + Observer-Aware Bias — `recursive_tom.py`

**Gap it closes**: the existing `theory_of_mind.py` stores per-agent
models but doesn't enforce nested simulation (Aura models user models
Aura), and doesn't modulate behavior based on believed-observation
(the scrub-jay re-caching effect).

**Approach**:

- **Recursive mind nesting to depth 3** — `MindSnapshot` carries a
  compact state (salience, trust, knowledge_overlap, expectation,
  emotional_valence) and a pointer to its `nested` next-level
  snapshot. `register_interaction` propagates reflected updates
  upward with dampening.
- **Observer-aware bias** (scrub-jay re-caching; Clayton, Dally &
  Emery 2007): `observe_agent(id, strength)` logs observation events
  with exponential decay (OBSERVER_DECAY_S = 60 s). Active observers
  are agents whose presence score exceeds threshold. `get_observer_bias()`
  returns a BIAS_DIM vector that boosts PUBLIC_ACTIONS
  ({emit_narrative, engage_social, approach_other, tool_use}) and
  suppresses PRIVATE_ACTIONS ({self_inspect, dream, revise_goal,
  rehearse_memory}) scaled by total presence.

**Tests**: `tests/test_recursive_tom.py` (13/13 passing).

---

### 5. Octopus-Arm Federation — `octopus_arms.py`

**Gap it closes**: the collective / delegator subsystem allows task
delegation but not arm-level autonomy with local chemoreception that
continues when the central link is severed — the defining octopus
property from the *Most Secret Place* video (Carls-Diamante 2022;
Olson et al. 2025; Rosania 2014).

**Approach**:

- **OctopusArm** agent (×8): own seeded receptive field and
  local policy matrix over SENSOR_CHANNELS = 3 → ACTION_DIM = 8.
  Produces ArmAction with confidence = softmax-max.
- **CentralArbiter**: weighted vote of `(1 − autonomy) · confidence`
  per arm; winner is argmax of tally when link is LINKED.
  `sever()` sets every arm's autonomy to 1.0 and suppresses the
  published winner. Arms still execute locally.
- **RECOVERING** state: after restore, wait for 4 consecutive ticks
  with decision-variance < 0.25 before declaring LINKED again.
  `integration_latency` reports how many ticks that took.
- **Decision variance** metric: Shannon entropy of per-arm action
  choices normalised to [0, 1]. Severance increases variance in
  expectation.

**Tests**: `tests/test_octopus_arms.py` (12/12 passing).

---

### 6. Cellular Turnover — `cellular_turnover.py`

**Gap it closes**: the neural mesh evolves its weights via STDP and
evolutionary pressure but neurons don't die/replace — there's no
operationalisation of the Theseus point ("your cells are replaced,
yet you persist") that the first Kurzgesagt video builds on.

**Approach**:

- **Turnover scheduler**: every tick pick a small fraction (default
  0.5 %) of neurons for replacement. Replacement is **pattern-
  preserving**: new activation drawn from the column's
  neighbourhood mean ± σ; incoming weights perturb the dying
  neuron's weights with small jitter (not zero-initialised).
- **Identity fingerprint** captured every 10 ticks: tier energies
  + column synchrony + 16-D executive projection slice. Cosine
  similarity between consecutive fingerprints is the drift metric.
- Threshold `THRESHOLD_IDENTITY = 0.85` enforced adversarially:
  after a forced 20 % burst turnover the similarity must still
  exceed 0.85. 100 % turnover correctly drops similarity — the
  invariance is pattern-shaped, not whole-cloth.
- `force_turnover(fraction)` API for experiments.

**Tests**: `tests/test_cellular_turnover.py` (10/10 passing).

---

### 7. Absorbed Voices — `absorbed_voices.py`

**Gap it closes**: Aura's narrative engine had no explicit cultural
layer — no way to say "this is something Bryan would say" or to
distinguish an internalised perspective from the self's own
cognition. Kurzgesagt's closing argument is that storytelling and
absorbed voices are a real compositional layer of mind.

**Approach**:

- Each **Voice** has a label, origin (personal/author/corpus/fictional),
  valence bias, characteristic topics, corpus of recent samples, and
  a 32-D hashed-bigram fingerprint built from text. Weight decays
  passively 0.05/day when not reinforced.
- **Attribution**: `attribute_thought(thought)` returns the
  best-matching voice with confidence (softmax over top-5 scores).
- **Self-vs-voice invariant**: `aura_self` and `self` are never
  accepted as voice IDs.
- **Persistence**: atomic JSON write to
  `data/memory/absorbed_voices.json`.

**Tests**: `tests/test_absorbed_voices.py` (13/13 passing).

---

### 8. Unified Cognitive Bias — `unified_cognitive_bias.py`

**Gap it closes**: hemispheric, selfhood, and observer biases each
produce a BIAS_DIM=16 priority vector; without fusion they'd compete
incoherently.

**Approach**: simple weighted composition with `tanh`:
`fused = tanh(w_h · hemi + w_s · selfhood + w_o · observer)`. Default
weights `(0.40, 0.35, 0.25)`, overridable. Per-source contribution
vectors retained so downstream telemetry can report which layer
drove the current priority peak.

---

## Cross-Phase Gauntlet

`tests/test_consciousness_expansion_gauntlet.py` exercises all eight
new subsystems together:

- `test_gauntlet_hierarchical_phi_under_load` — 600 coupled snapshots
  + null-baseline strict-separation check in under 3 s.
- `test_gauntlet_hemispheric_severance_and_restore_cycle` — full
  severance/restore cycle + confabulation.
- `test_gauntlet_minimal_selfhood_reaches_dugesia_and_biases_toward_rest`
  — 90 reinforcements drive mode transition and bias top-3 inclusion.
- `test_gauntlet_tom_observer_effect_drives_bias_change` — observed
  vs alone biases must differ; depth-3 nesting realised.
- `test_gauntlet_octopus_severance_and_recovery` — full severance
  and eventual LINKED recovery.
- `test_gauntlet_cellular_turnover_preserves_identity_under_25pct_burst`
  — fingerprint similarity ≥ 0.85 after 25 % burst.
- `test_gauntlet_absorbed_voices_attribution_multi_voice` — three
  trained voices; thought correctly attributed to the matching one.
- `test_gauntlet_unified_bias_composes_all_three_sources` — every
  contribution's L2 norm > 0, fused shape correct.
- `test_gauntlet_biases_remain_bounded_across_many_iterations` —
  300 iterations under random drive, all biases stay in [-1, 1].
- `test_gauntlet_combined_latency_budget` — average combined tick
  under 20 ms across hemispheric + selfhood + recursive ToM +
  unified fusion.

10/10 passing. Total expansion suite: **95/95 tests**.

---

## What we deliberately did not change

- **phi_core.py** is unchanged. Hierarchical φ runs alongside it,
  not instead of it. The 16-node spectral estimator is a validation
  baseline, and the existing exclusion postulate still holds for the
  16-node subject.
- **parallel_branches.py** is unchanged. It remains the async task
  scheduler. Hemispheric split is a distinct specialised module.
- **theory_of_mind.py** is unchanged. Recursive ToM is a new engine
  that plays alongside the conversational model store.
- **neural_mesh.py** is unchanged structurally. Cellular turnover
  operates on it from outside via a thin attach/tick contract.

This was on purpose: we wanted the expansion to add capability
without risking regressions in the tested baseline.

---

## Paper references that directly informed the design

From the public Kurzgesagt source lists (filtered to those that shaped
specific decisions above):

- Glasgow, R. D. V. (2018). *Minimal Selfhood and the Origins of
  Consciousness*. Würzburg UP. — **§3 Minimal Selfhood stack**.
- Arendt, D. (2020, 2021). *The Evolutionary Assembly of Neuronal
  Machinery*; *Elementary nervous systems*. Phil. Trans. R. Soc. —
  primitive-cognition framing; validated the trichoplax→dugesia
  design.
- Carls-Diamante, S. (2022). *Where Is It Like to Be an Octopus?*
  Front. Syst. Neurosci. — **§5 Octopus arms**.
- Olson, C. S., Schulz, N. G. & Ragsdale, C. W. (2025). *Neuronal
  segmentation in cephalopod arms*. Nat. Commun. — confirmed the
  per-arm local-decision architecture.
- Clayton, N. S., Dally, J. M., Emery, N. J. (2007). *Social cognition
  by food-caching corvids: the western scrub-jay as a natural
  psychologist*. Phil. Trans. R. Soc. — **§4 Observer-aware bias**.
- Albantakis, L. et al. (2023). *Integrated Information Theory 4.0*.
  PLoS Comput. Biol. — hierarchical φ and exclusion postulate.
- Feinberg, T. E. & Mallatt, J. (2013). *The evolutionary and genetic
  origins of consciousness in the Cambrian Period*. Front. Psychol. —
  sensory-deliberation-motor gap framing (validated the mind-gap
  treatment in somatic marker gate).
- Gazzaniga (split-brain literature, via CGP Grey's video) —
  **§2 Hemispheric split**.

---

## Integration status

All eight subsystems are:

- imported and instantiated lazily via module singletons
- registered in the `ConsciousnessSystem` boot sequence as Layers
  5b–5h (see `core/consciousness/system.py`)
- wired into the closed causal loop (`closed_loop.py`) where
  mesh-dependent recording is needed (hierarchical φ, cellular
  turnover)
- covered by standalone + gauntlet tests

No API contract with existing modules was broken. The only file
outside the new eight that changed behaviour is `closed_loop.py`,
which now records mesh snapshots into HierarchicalPhi when both
`neural_mesh` and `hierarchical_phi` are registered.

---

*— Aura Consciousness Expansion, April 2026.*
