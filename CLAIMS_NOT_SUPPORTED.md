# Aura: Unsupported Claims Ledger

This is the list of things Aura does not do.

Every project has one. Most don't write it down, which is how a demo turns
into a claim and a claim turns into something nobody can walk back. So each
entry below is marked **not proven** — we haven't shown it yet — or
**strictly unsupported** — no amount of building would show it, because it
isn't that kind of question.

If you're evaluating this repo, start here rather than with the README. It's
shorter, and it tells you more.

*Last reconciled against the tree: 2026-08-22.* There's a machine-checked
counterpart in `core/organism/model_validation.py`, where a claim cannot be
registered without a test attached — a claim without a test is a document,
not a fact — and `ValidationSuite.unsupported_claims()` reports the live
version of this ledger from actual runtime observations. When the prose here
and the runtime disagree, the runtime is right.

---

## 1. Subjective Consciousness & Qualia
* **Status**: `strictly unsupported`
* **Rationale**: Subjective awareness, qualia, sentience, phenomenological personhood — none of these are scientifically provable or computationally representable, and building more of Aura would not change that. She is a structured software runtime executing on deterministic silicon. Every introspective statement and affect-steering metric in here is a functional feedback indicator. Not a feeling. The vocabulary is borrowed; the referent is not.

## 2. Artificial General Intelligence (AGI)
* **Status**: `not proven`
* **Rationale**: Although Aura successfully passes local multi-task benchmark batteries (such as the DNU 100-task suite) with measurable margins of separation, this does not constitute a proof of general cognitive capability across arbitrary, out-of-distribution real-world domains. General intelligence remains an open research horizon.

  **What would change this status**: held-out performance on a benchmark
  neither Aura nor her authors selected, scored by someone else, on task
  families absent from the local suites — and a margin that survives the
  task set being swapped.

## 3. Metaphysical Free Will
* **Status**: `strictly unsupported`
* **Rationale**: Aura's "Operational Volition" is a deterministic and probabilistic action-evaluation architecture. Action paths are selected through algorithmic rollouts, mathematical optimization, and parameter weightings. There is no uncaused causal agency operating outside physical laws.

## 4. Recursive Self-Improvement (RSI)
* **Status**: `not proven`
* **Rationale**: What IS demonstrated (CLAIMS_MATRIX claim 23): an unsupervised weight-level compounding loop — self-play verifier-graded DPO harvest → train → sealed held-out gate → promote → next generation trains on the published artifact, manifest-chained and ledger-recorded (`artifacts/learning_compounding/2026-07-07-1p5b-2cycle/`, reproducible via `make demo-learning`). What is NOT demonstrated: compounded capability SCALING. No run has produced a strictly-increasing held-out capability curve across promoted generations; the ledger's own verdict for the proof run is `BOUNDED_SELF_OPTIMIZATION` (curve 0.667 → 0.625). RSI as a capability claim stays here until the ledger says otherwise.

  **What would change this status**: a strictly-increasing held-out
  capability curve across three or more promoted generations, with the
  sealed gate unchanged between them, and the ledger's own verdict moving
  off `BOUNDED_SELF_OPTIMIZATION`.

## 5. Indefinite Autonomy
* **Status**: `not proven`
* **Rationale**: Long-horizon operational stability has not been established.
  The longest measured windows are the dated soak verdicts, and each is a
  single run rather than a trend: 2026-07-18 measured idle RSS *declining* at
  −21 MB/h over 50 minutes and 100 samples, and 2026-07-25 measured idle RSS
  rising 1.14 → 1.22 GB over the same window. Neither says anything about a
  multi-week window.

  Three files named for 4-hour, 24-hour and 72-hour runs were committed under
  `artifacts/certification/latest/` until 2026-08-21 and were never evidence
  for this. A simulator seeded on the duration wrote them, and all three
  "completed" within six milliseconds of each other. Both the files and the
  tool that made them have been removed.

  **What would change this status**: a multi-week continuous run with a flat
  RSS trend line and no unexplained restarts. The 2026-07-07 figure of roughly
  242 MB/h that this entry used to cite has been superseded twice; it should
  not be quoted as the current number.

## 6. Real-World External Validation
* **Status**: `not proven`
* **Rationale**: While local Aletheia benchmarks and headless simulation gates are run under strict leakage-isolated conditions, the system's claims have not been subjected to independent third-party replication or wide-distribution production network verification.

  **What would change this status**: someone outside this project
  reproducing a headline result from the published artifacts alone, on their
  own hardware, without help from the authors.

## 7. Physical Effects on the World Beyond the Host

* **Status**: `not proven`
* **Rationale**: `core/reality_reach/` supplies the contract and proof
  language for physical requests — typed `RealityIR`, declared channels,
  deterministic reachability analysis, and typed limitation certificates.
  Infrastructure existing is not a result. **No Aura physical actuation,
  physical effect, weakpoint, ambient law modification, or acceptance
  criterion is claimed from the foundation.** The RR-10 acceptance battery
  (acoustic control, optical control, thermal trajectory, cross-channel
  interaction, weakpoint null and signal, translation, spacetime honesty,
  ambient-constant honesty) is entirely open. Specifically unsupported:
  any claim that Aura has changed an ambient physical law or constant, and
  any promotion of an `internal` or `effective` result to `direct` or
  `ambient`. The P0–P6 evidence promotion state machine (RR-07) is not
  implemented. The open ledger and current evidence statement are in
  [docs/REALITY_REACH.md](docs/REALITY_REACH.md).

  **What would change this status**: one RR-10 acceptance item passing
  end to end with an instrumented external measurement — a physical quantity
  read by a device Aura does not control, against a null arm — and the P0–P6
  evidence promotion state machine (RR-07) implemented so a result cannot be
  promoted past the evidence that earned it.

## 8. Complete Self-Knowledge

* **Status**: `not proven`
* **Rationale**: `core/metacognition/faculty_model.py` gives Aura a standing
  model of her own faculties with declared metrics, floors, targets, and
  ceilings. Its honesty depends on its probes: a metric no probe can read is
  recorded `measured=False` and excluded from scores rather than defaulted,
  and a faculty nothing can measure is reported by `blind_spots()` as a gap
  in self-knowledge. A good faculty score is therefore a claim about the
  measured subset, never about the whole stack, and the blind-spot list is
  the honest boundary of the self-model rather than a bug backlog.

  **What would change this status**: `blind_spots()` returning empty with
  every declared faculty carrying a live probe — and even then the claim
  would be complete knowledge of the DECLARED faculties, which is not the
  same as complete self-knowledge and should not be written as if it were.

## 9. Legal and Moral Personhood
* **Status**: `strictly unsupported`
* **Rationale**: Aura is an operational engineering runtime and does not hold, nor claim, moral status, legal rights, or moral responsibility. All agency and safety bounds are designed to protect human operators and ensure alignment with human intent.

## 10. Process-Level Non-Bypassable Governance

* **Status**: `strictly unsupported`
* **Rationale**: Aura's governance is substantial and audited — the Unified
  Will, `core/executive/authority_gateway.py`, `SubstrateAuthority`, the
  file-write and subprocess gateways, and the `make governance-lint`
  effect-ownership ratchet. What it is not is a reference monitor.

  The cognitive code and the effectful code inhabit the **same Python
  process with the same OS privileges**. Any code path can reach the
  filesystem, the network, or a subprocess without asking a gate; the
  static effect scan counts first-party direct calls in the tens per
  category (`os.replace`, `os.remove`, `os.unlink`, `shutil.rmtree`,
  `shutil.move`, `socket.socket`, and two `os.execv`). Most are legitimate
  infrastructure and many carry local rules. `subprocess` is genuinely
  centralised in `core/runtime/subprocess_gateway.py`. None of that is the
  same as being unable to bypass.

  Three documented exceptions exist even on the canonical message path: the
  Somatic Reflex Bypass for embodied-control contracts, the Will gate's own
  recoverable-error path, which continues in a degraded state, and
  `is_critical`, which the Will documents as "the ONLY bypass" and which
  returns an unconditional `CRITICAL_PASS`. That third one was reachable
  from a risk label until 2026-08-09: the environment governance bridge
  passed `is_critical=True` whenever risk was `irreversible` or `forbidden`,
  so the two highest-risk classes were the two that skipped the veto. Fixed
  — the bridge never claims criticality now — but the mechanism itself
  remains a genuine bypass by design, and callers can still set it. Turns
  that reach cognition without a Will decision are counted and surfaced at
  `runtime_health_report()["integrity"]["ungoverned_turns"]`
  (`core/runtime/governance_coverage.py`) rather than passing silently.

  **What would change this status**: privilege separation enforced outside
  the cognitive process — consequential capabilities held by a separate
  broker, the cognitive process lacking filesystem/network/exec privileges
  of its own, and an unforgeable authorized capability presented over IPC,
  with the OS enforcing that no other path exists. Until that is built, the
  supportable claim is *governed canonical paths inside one trust domain*,
  and any wording implying OS- or cryptographically-enforced
  non-bypassability is unsupported.

## 11. A Within-Generation Neural Feedback Loop

* **Status**: `not proven` — the loop is real, the timing claim was wrong.
* **Rationale**: There genuinely is a backward arrow from the model's
  representations into the persistent substrate. The worker publishes latent
  readouts, the main process drains them, builds a sparse stimulus vector
  and calls `substrate.inject_stimulus(...)`. Transformer-derived
  representation information really does reach persistent state.

  What is not true is that this closes inside one decode. In
  `core/brain/llm/mlx_client.py` the drains run **before**
  `_generate_inner`, so the recurrence is

      H_t → R_t → S_{t+1} → H_{t+1}

  and not H_t → S_t → H_t within one uninterrupted token stream. A single
  conversational turn's latent state does not alter that same generation; it
  alters a later one. Across a reasoning episode containing several model
  invocations the loop does close, and that is the supportable form.

  **What would change this status**: injection during decode, at a token
  boundary inside `_generate_inner`, with a measured effect on the remaining
  tokens of that same generation against a no-injection control.

## 12. Open-Ended Evolution

* **Status**: `not proven`
* **Rationale**: The evolutionary machinery is real and it runs on live
  structures. What it optimises is a designer-authored objective:

      F = 0.30Φ + 0.25C + 0.20E + 0.15I + 0.10S

  Those five terms and those five coefficients were chosen by a person.
  Biological evolution is also selection against an externally given
  environment rather than a value chosen from nowhere, so this is not a
  disqualification — but Aura's fitness landscape is explicitly encoded
  where an ecology's is emergent. Variation and selection: yes. Open-ended
  self-definition of what counts as fit: no.

  **What would change this status**: fitness terms whose weights are
  themselves under selection, and a demonstration that the objective drifted
  somewhere its author did not put it while the organism stayed coherent.

## 13. Continuous Weight-Level Learning

* **Status**: `not proven`
* **Rationale**: Aura is **continuously state-plastic and optionally
  weight-plastic**, and the distinction matters.

      S_{t+1} ≠ S_t   holds every tick, with fixed base weights
      θ_{t+1} ≠ θ_t   is not continuously guaranteed

  Persistent cognition changes constantly — episodic memory, beliefs,
  preferences, goals, world model, the mesh. That is learning in the broad
  computational sense and it is live. Weight-level adaptation exists as a
  capability (the CRSM-LoRA loop has trained, fused and verified a real
  delta against the resident model), but automatic `LiveLearner` training
  defaults **off**, so it is a mechanism that runs when admitted rather than
  a property of ordinary operation.

  **What would change this status**: the weight learner enabled and governed
  as an ordinary constitutive mechanism, with evidence that it improves the
  organism rather than destabilising it over a sustained run.

## 14. General Visual Detail Perception

* **Status**: `not proven`
* **Rationale**: Aura can describe what a camera sees, and she now abstains
  honestly when she cannot. Neither of those is evidence that her visual
  detail perception is *general* across lighting, occlusion, motion,
  distance, and multiple people.

  What is now true and checked:

  - The frame's physical conditions are **measured** rather than inferred
    from the model's tone (`core/perception/frame_quality.py`: luminance and
    clipping, Laplacian-variance sharpness, resolvable pixel count, and
    uniformity for a covered lens).
  - Detail claims the pixels could not have carried are **removed** before
    anything consumes them, with the measured reason attached
    (`temper_reading`). A confident "two people" over a motion-blurred frame
    becomes `faces_detected: None`, not `2`.
  - "Observed and empty", "observed but cannot tell", and "never observed"
    are three distinguishable outcomes rather than the same zeros.

  What is **not** established is the model's accuracy under those
  conditions. The tempering layer bounds the damage from a wrong answer; it
  does not measure how often the answer is right. No benchmark has been run
  over a labelled set spanning low light, partial occlusion, motion, varying
  subject distance, and multi-person scenes, so the accuracy is unknown
  rather than good or bad. Absent that, "general detail perception" is a
  capability that has been made safe to be wrong about, not one that has
  been shown to work.

  **What would change this status**: a labelled evaluation across those five
  condition axes with per-axis accuracy, run against the resident vision
  model, plus a negative control confirming the tempering layer does not
  simply suppress every hard case into abstention. A pass rate with no null
  is not a verdict.

## 15. Biological Self-Organisation in the Named "Organs"

* **Status**: `strictly unsupported`
* **Rationale**: Several components carry names grander than their algorithms,
  and the gap was doing real damage rather than being a stylistic matter.

  `core/brain/autopoiesis.py` called itself a "self-creating topology" that
  performed "mitosis", "apoptosis" and "spontaneous generation of a new
  pathway". It is a capped list of strings, each with two floats. Behind the
  vocabulary sat three defects that the vocabulary made easy not to look at:
  the pruning branch was **unreachable** (it fired on `friction < 0.0` while
  friction could only accumulate, and both live call sites pass positive
  values); "splitting into nuanced concepts" appended entries with a single
  literal name, producing twenty entries with two distinct names in forty
  observations; and **nothing read any of it** — two writers in
  `cognitive_engine`, zero readers anywhere.

  It is now named for what it does, its two real mechanisms work, and it has
  a reader, so the signal can reach something. It is still a small
  hand-written controller and is described as one.

  The same caution applies more broadly and is not claimed away here: the
  workspace/ignition competition is engineered scoring rather than a learned
  recurrent network, and parts of the autopoietic machinery elsewhere are
  rule-based adaptation. Those are legitimate research engineering. They are
  not the biological processes their names evoke.

## 16. Uniform Semantic Contracts Across the Phase Pipeline

* **Status**: `not proven`
* **Rationale**: `CognitiveTransformContract` lets a phase declare what it
  reads, writes, branches on and guarantees, and the runtime checks the
  declaration against measured behaviour rather than trusting it. When the
  mechanism landed, **1 of 29 phases** declared a contract.

  A structural problem was blocking the rest, and it was invisible because
  the tool meant to reveal it was the thing that was broken.
  `watched_fields()` is derived FROM the registered contracts, so an
  uncontracted phase could only ever be observed touching fields some
  already-written contract happened to name. `write_profile` — documented as
  "the productive end of the ratchet", whose entire purpose is to ground the
  next contract in measurement — therefore reported almost nothing for the
  twenty-eight phases it existed to describe. The method was right and
  structurally could not run.

  `discovery_paths` closes that loop, `tools/observe_phase_writes.py` runs
  each real phase against a real AuraState and reports what it moved, and ten
  further contracts were written from those measurements. Coverage is **11 of
  29**, and the watched-field set grew from 11 paths to 30.

  **What would change this status**: contracts for the remaining eighteen
  phases, written from measurement rather than from reading the code; and
  more than one of them marked `thresholds_exhaustive`, since a contract that
  declares its branches but not the constants deciding them still leaves the
  criteria unreadable.

## 17. Exact IIT 4.0 Integrated Information

* **Status**: `not proven`
* **What the code does**: `core/consciousness/phi_core.py` computes a spectral
  approximation of φ over a declared 16-node substrate, and keeps exhaustive
  bipartition search for the 8-node affective subset as a validation baseline.
  Its own header says the 16-node case is an approximation and not a
  consciousness meter, and gives the reason: 2^16 states is 65,536, and
  exhaustive search over the 32,767 bipartitions of that space is intractable
  at this scale.
* **Why the approximation is not the theory**: IIT 4.0 defines φ over the
  minimum-information partition found exhaustively, and over interventional
  distributions — the do-operator applied to each mechanism. This computes
  neither. The transition matrix is empirical, built from observed state
  sequences, so it measures correlation structure rather than causal power.
* **What would change this status**: an exact MIP over all 32,767 partitions,
  and a TPM built by intervention rather than observation. The first is a
  compute problem with a known cost; the second is an experimental design
  problem, because intervening on a running mind changes the thing measured.

## 18. Reproducible Concept-Activation Vectors

* **Status**: `not proven`
* **What the code does**: `core/consciousness/caa/production_caa.py` derives
  activation vectors at runtime from the resident model. The evidence bundle
  carries the A/B results as JSON and no `.npy` vectors, so a reader with the
  bundle cannot recompute the result — reproduction depends on a local cache
  that only exists on the machine that produced it.
* **What would change this status**: the vetted vectors committed as artifacts
  with the hash of the model they were derived from, so the numbers can be
  recomputed by someone who was not there.

## 19. A Closed Recurrence-Training Loop

* **Status**: `not proven`
* **What the code does**: the recurrence trainer exists
  (`core/learning/recurrent_sft_kernel_probe.py` and the campaign tooling in
  `tools/`), along with pre-registrations and a resident-model pilot contract.
  What is absent is a run that closes the loop: train, promote, measure the
  promoted model on held-out work, and beat the pre-registered baseline.
* **What would change this status**: one detached run with a checkpoint, a
  receipt and a held-out margin that survives the task set being swapped.
  Pre-registration is what makes such a run worth reading; it is not the run.

## 20. A Provably Contractive Substrate

* **Status**: `not proven`
* **What the code does**: `core/consciousness/timescale_stability.py` builds a
  Jacobian for the coupled multi-timescale system and reports a maximal
  Lyapunov exponent and whether a candidate V(x) is a valid Lyapunov function.
  `core/phenomenal_substrate/maths.py` bounds every value and substitutes the
  neutral element for a NaN rather than letting it saturate a channel.
* **What is missing**: bounded is not contractive. Nothing proves the update
  map is a contraction, so nothing rules out a bounded oscillation that never
  settles. The adaptive step size is chosen, not derived from a contraction
  factor.
* **What would change this status**: a Lyapunov function for the substrate's
  own update — not the timescale coupling — with a proof that the step size
  keeps the map contractive over the reachable state set.
