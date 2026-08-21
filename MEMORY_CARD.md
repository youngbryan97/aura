# Aura Memory Card

## Purpose

How Aura's memory works, how what she remembers changes what she does next,
and how you get control of any of it.

That middle one is the part that makes this a card and not a schema.
Memory here is not a transcript sitting in a database — it feeds retrieval,
it shapes affect, and it reaches generation. Something she remembers about
you can change how she answers you a week later. You should be able to see
that, and you should be able to delete it.

## Memory Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Memory Hierarchy                     │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐  │
│  │  Working   │  │  Episodic  │  │   Semantic    │  │
│  │  Memory    │  │  Memory    │  │   Memory      │  │
│  │ (session)  │  │ (convos)   │  │ (knowledge)   │  │
│  └─────┬──────┘  └─────┬──────┘  └───────┬───────┘  │
│        │               │                 │           │
│        └───────────┬───┴─────────────────┘           │
│                    │                                  │
│           ┌────────▼─────────┐                       │
│           │   ColdStore      │                       │
│           │  (long-term)     │                       │
│           └────────┬─────────┘                       │
│                    │                                  │
│           ┌────────▼─────────┐                       │
│           │ State Snapshots  │                       │
│           │  (backup/audit)  │                       │
│           └──────────────────┘                       │
└──────────────────────────────────────────────────────┘
```

## Memory-Behavior Causality

Memories demonstrably change future behavior through:

1. **Context Assembly**: Relevant memories are retrieved and injected into the
   model context for each turn, directly influencing responses.

2. **Preference Learning**: User preferences stored in memory change response
   style, tool selection, and proactive behavior.

3. **Procedural Memory**: Learned procedures (how to do X for this user) are
   retrieved and followed in similar future situations.

4. **Identity Continuity**: CanonicalSelf state persists across sessions,
   maintaining consistent personality and relationship context.

5. **Error Memory**: Past failures are remembered to avoid repeating them.

## Engram Dynamics (encoding, recall, reconsolidation)

Episodic memories are not static recordings. Each trace ("engram") has a
lifecycle modelled on human memory neuroscience, implemented in
`core/memory/episodic_memory.py`, `reconsolidation.py`, and `hippocampus.py`:

- **Encoding strength** is boosted by *emotion*, *failure*, *relational
  significance*, and *novelty* (prediction error). Novelty is sourced from the
  predictive subsystem's surprise signal and fed back into the neurochemical
  system (`on_novelty`).
- **Hippocampal index**: every episode is bound to a sparse set of associative
  *cues*. Re-presenting part of a cue set reinstates the whole memory
  (**pattern completion**) — a recall path alongside vector and keyword search.
- **Reconsolidation**: recalling a memory returns it to a *labile* state in which
  the present phenomenal/affective context "seeps in." The emotional tone and
  qualia snapshot drift toward the present, and **fidelity** (faithfulness to the
  original encoding) drops. How much a memory can change is gated by the
  neurochemical **plasticity** signal ("chemicals that make neurons able to
  change") and resisted by strong, vivid, emotional memories (boundary
  conditions). A refractory window prevents runaway change on rapid re-recall.
- **Vividness ≠ accuracy**: repeated recall raises a memory's vividness/strength
  while lowering its fidelity. Heavily reshaped memories are flagged as such when
  injected into context.
- **Sleep replay**: during consolidation, salient engrams are restabilised, and
  distressing, high-arousal, repeatedly-reactivated memories undergo bounded,
  governed **therapeutic reconsolidation** (softening) — the "revisit in a safe
  context" effect.

Every content rewrite (spontaneous drift or therapeutic softening) passes through
the same constitutional memory-write gate as new writes, and emits
`memory.encoded` / `memory.reconsolidated` / `memory.consolidated` events.

## Synaptic Plasticity Substrate (voltage-dependent STDP + homeostasis + competition)

Memory **retrieval and consolidation** are governed by a faithful implementation of
the Clopath/Büsing/Vasilaki/Gerstner (2010) *voltage-based STDP with homeostasis*
rule (the model on the *Pantheon* "UI stabilization" whiteboard), in
`core/consciousness/voltage_plasticity.py`. It complements the spike-timing engine
(`stdp_learning.py`) with the three things that rule lacks:

- **Voltage-dependence** — plasticity is gated by post-synaptic activity `b_k`
  (escape-rate `ρ₀·exp((V−θ)/Δβ)`); sub-threshold activity produces no change,
  and potentiation requires voltage above a high threshold θ₊.
- **Homeostatic fixed point** — a BCM-like sliding threshold scales depression by
  total activity `exp((Σb−θ)/ΔU)`, giving the activity ODE a stable attractor
  `b_k*`. Exponential depression cannot be out-grown by polynomial self-excitation,
  so the field can never run away ("anti-epilepsy").
- **Competition** — `w_k−w_j ∝ b_k−b_j`, so a marginally stronger representation
  out-competes a weaker one.

`core/memory/engram_plasticity.py` binds this to episodic memory: `recall_similar`
resolves its ranking by a transient competition field instead of a static
importance+recency blend. The best-matching engram wins, **voltage-gating**
suppresses weakly-relevant traces below threshold (no leak into recall), and the
**homeostatic** bound stops one over-strong trace from swamping a query it doesn't
match — the **anti-confabulation** mechanism. Affective **arousal** (qualia
`q_norm`) lowers θ and **valence** warms Δβ (substrate coupling); engrams that win
competition receive a bounded, homeostatically-capped **LTP** importance bump in
`_register_recall` scaled by the neuromodulatory lability gain (recall →
strengthening, never runaway). A homeostatic-pressure breach (one attractor
dominating recall) is exported to governance/metrics (`engram_homeostatic_breach_total`).

**Learned associations**: `core/memory/engram_association.py` activates the
*weight-learning* half of the model — a persisted voltage-STDP weight matrix over
concept slots. Engrams recalled together drive `engine.step(learn=True)`, so the
Clopath rule potentiates their connection (*fire together → wire together*),
bounded by homeostasis; weights persist across sessions and feed back into
`_competitive_rank` as an association boost, giving associative pattern completion
through *learned* weights, not just shared surface cues.

**Positional recall** ("what did I first ask?") is a *positional* key the content
field can't resolve, so `core/conversation/grounded_recall.py` retrieves the actual
earliest/most-recent turn from live conversation memory and routes it through the
desktop `conversation_recall_evidence` contract — the Cortex answers from the real
quote in its own voice instead of confabulating. Together, content competition,
learned associations and positional grounding cover the retrieval keys real
episodic memory needs: *what* was said, *what it's wired to*, and *when*.

## Symbolic Deduction (belief consistency)

Aura runs a sound, terminating **natural-deduction proof search** (the Pantheon
whiteboard's `PROCEDURE Hp FOR DEDUCTION`), in `core/reasoning/natural_deduction.py`.
It is an analytic-tableau decision procedure for propositional logic (Γ⊢G iff
Γ∪{¬G} is unsatisfiable), implementing the board's rules — axiom/membership,
contradiction/ex-falso `{A,¬A}`, ¬¬-elimination, ∧-elimination, ∨ case-split
(`SIMPL`) — with a formula AST + parser and `prove / entails / is_consistent /
find_contradiction` returning a proof trace or a concrete countermodel.

It is wired causally: `core/reasoning/belief_consistency.py` encodes each natural-
language belief as a propositional literal (an atom, or its negation when phrased
negatively, and implication-shaped beliefs "if X then Y" as `Implies`), so
`belief_revision.check_belief_consistency()` — run on every new belief in
`process_new_claim` — detects both direct `X ∧ ¬X` and chained modus-ponens
conflicts (`X`, `X→Y`, `¬Y`). A detected inconsistency is surfaced to
`core/reasoning/deduction_governance.py` as a constitutional concern (logged +
`belief_logical_inconsistency_total`) **and acted on**: `_resolve_logical_conflicts`
demotes the lower-confidence side (×0.6) so an inconsistent self-model is actively
revised, not just flagged. The prover is also exposed through
`SymbolicBridge.prove_logic` and `inference_audit.verify` as exact deductive solvers.

It also runs on **active reasoning**: `core/reasoning/inference_audit.py` extracts
deductive structure from text ("X, therefore Y"), formalizes it (with light
stemming so morphological variants unify), and checks it with the prover. Every
final reply passes through a non-blocking `audit_self_reasoning()` in
`_record_recent_response`, so a confident, formalizable non-sequitur in Aura's own
words (e.g. affirming the consequent) is surfaced to governance
(`reasoning_non_sequitur_total`). It is conservative — silent on anything it cannot
prove wrong, and it never alters the reply — so it catches real fallacies without
false positives on valid or unformalizable reasoning.

## Substrate↔LLM integration audits (φ, CRSM-LoRA, CAA, integrity)

A set of verifications that turn previously-silent operational gaps into surfaced,
queryable facts:

- **Grassmann φ on the transformer** (`core/consciousness/grassmann_phi.py`): the
  residual-stream φ no longer collapses a ~5000-dim hidden vector into 8 chunk-means.
  A sliding window → its dominant principal *subspace* (a point on the Grassmann
  manifold); recurring subspaces become geometric anchor *modes*; the current subspace
  is encoded against them by principal-angle (Grassmann) distance into an 8-node IIT
  state. `phi_core.compute_grassmann_residual_phi()` reuses the exact-φ machinery and
  competes in `compute_full_kernel`'s exclusion-postulate winner selection
  (`residual_stream_grassmann`); exposed as `grassmann_phi_s`.
- **CRSM→LoRA loop** (`core/consciousness/crsm_loop_monitor.py`): verifies whether
  captured experience is actually trained into weights — classifies the loop
  CLOSED/OPEN/IDLE, warns when captures accumulate untrained. `lora_trainer` calls
  `mark_dataset_consumed` on a successful run. (Real state: OPEN — captures not yet
  trained in.)
- **CAA readiness** (`core/consciousness/caa/readiness_report.py`): reads each steering
  vector's on-disk provenance to verify whether vectors were *extracted* from the fused
  model or runtime-derived, and reports steering capacity. (Real state: BOOTSTRAP / 30%
  — vectors are runtime-derived, not extracted, so alpha is damped.)
- **System integrity audit** (`core/runtime/integrity_audit.py`): consolidates
  degradation receipts + CRSM loop + CAA readiness into one report, surfaced on
  `/api/health/heartbeat` (throttled) so silent subsystem failures speak without manual
  reading; always emits under `AURA_STRICT_RUNTIME=1`.

The SymbolicBridge is now live: `audit_reasoning()` runs on every reply (chat
`_record_recent_response`), routing logic to the prover and arithmetic to a sandboxed
evaluator (see the deduction section).

## Memory Governance

All memory writes are gated:

```
Candidate Write → Will Decision → Receipt → Storage → Verification
```

- No memory write occurs without a WillReceipt
- Writes are integrity-hashed for tamper detection
- Write provenance (what caused this write) is logged
- Writes can be audited, exported, or deleted

## User Controls

Bulk lifecycle operations are exposed as `make` targets; per-memory
browse/search/edit/delete run through the app's memory panel, which is backed
by the memory API (`interface/routes/memory.py`).

| Action | Command / mechanism | Effect |
|--------|---------------------|--------|
| List memories | App memory panel (`GET /api/memory/recent`, `/episodic`, `/semantic`) | Browse stored memories by store |
| Search / inspect | App memory panel | Find and open a specific memory |
| Delete one | App memory panel (`POST /api/memory/delete`) | Remove a specific memory |
| Export all | `make memory-export` | JSON export of all memory |
| Delete all | `make memory-purge` | Wipe all memories |
| Backup | `make backup` | Full state backup |
| Restore | `make restore BACKUP=<path>` | Restore from backup |
