# Aura: The Language Substrate & Generality Programme

**Status**: Living document · Verified against the tree 2026-08-29  
**Scope**: Everything Aura does to learn, represent, and compose new abstractions  
**Honest boundary**: What is measured, what is aspirational, where the walls are

---

## The Question

> Can a system — running a frozen local checkpoint, on one machine, with no cloud
> calls and no money — learn a new representation from experience, invent a
> reusable abstraction, apply it in an unrelated domain, and then use the fact
> that it did this to get better at doing it again?

That question has four parts, and each has a different answer today.

---

## Part 1: Relation Induction — learn a new representation from experience

**Code**: [`core/cognition/relation_language.py`](../core/cognition/relation_language.py)  
**Test**: [`test_a_relation_learned_in_one_world_helps_in_another.py`](../tests/test_a_relation_learned_in_one_world_helps_in_another.py)  
**Status**: ✅ ESTABLISHED

The `RelationLanguage` learns structured rules from examples — transformations
over sequences — without the request naming what it wants. Given paired
observations (input → output), it constructs candidate relations from a
parameterised family, validates them on held-out transitions, refuses them when
they don't compress, and admits them to a persistent library.

### What holds (each backed by a named test)

| Ref | Claim | Evidence |
|---|---|---|
| A1 | Language insufficiency is detected, not forced | `language_is_sufficient` |
| A2 | Shape inferred from examples, not requested by label | `_possible_sources`, `_forms_that_fit` |
| A3 | Validated on held-out transitions | `held_out` |
| A4 | Must compress — substitution tables refused, noise invents nothing | Substitution tables, noise scores zero |
| A5 | Works over structured states (words, colours, records, grids): 12/12 each | Scored identically across representations |
| A6 | Composition: "mirror then rotate" is one shape nobody wrote | 20/100 battery problems unreachable without composition |
| B1–B4 | Transfer across worlds and representations | Measured with cost-of-wrong-prior null |
| B5 | Higher-order: 3-deep unreachable → reachable after learning 2-deep | `known_forms` |
| B6 | Persistence across processes | Shapes are structured programs, round-trip through JSON |
| B7 | Refactoring: common sub-structure extracted, reaches what winners cannot | `RelationLanguage.refactor` |
| B8–B9 | Live in the runtime, demonstrated across consecutive turns | Live 2026-08-28 |

### The metalanguage question

The families the system searches over — `mirror`, `offset`, `exchange`,
`grouping`, `affine`, `compose` — are **human-designed**. The search within
them is genuine induction; the vocabulary itself is not.

This is not a gap unique to Aura. The ILP literature established that
unconstrained predicate invention is intractable ([Turning 30: New Ideas in
ILP](https://arxiv.org/pdf/2002.11002)); metarules are what made it workable.
ARC grounds its priors in Spelke's Core Knowledge. Aura's basis covers
geometry/topology and number from that set but **omits objectness and agency**,
which predicts the two battery failures exactly.

**Current state**: The fitted affine-modulo operation space `f(i) = (a*i + b) mod m`
subsumes the manual primitives and generates dozens of novel ones automatically.
Grouping (objectness primitive #1) added — predicted failure went 0/10 → 10/10
with no other shape moving (101 → 111 of 120). Full objecthood (cells as objects
with properties) remains open.

### The ablation

The ablation that matters ([`test_whether_an_abstraction_is_downstream_of_experience.py`](../tests/test_whether_an_abstraction_is_downstream_of_experience.py)):
the learned library passes, the authored families do not. Of 120 problems:
composition is worth 20, the learned library 24–27, the prior nothing measurable.

### What is NOT established

| Ref | Gap |
|---|---|
| D2 | Battery scored WITH the foundation model for comparison |
| C7 | A second concept formed from a different failure signature |
| 1.1f | The basis families themselves being principled rather than author-chosen |

---

## Part 2: Recursive Latent Cortex — think longer on hard problems

**Code**: [`core/brain/llm/latent_cortex/`](../core/brain/llm/latent_cortex/) (156 modules)  
**Code**: [`core/learning/intrinsic_recurrence.py`](../core/learning/intrinsic_recurrence.py)  
**Doc**: [`RECURSIVE_LATENT_CORTEX.md`](RECURSIVE_LATENT_CORTEX.md)  
**Status**: Mixed — frozen loop REFUTED; trained intrinsic recurrence BOUNDED_WOW_SIGNAL

### The question

> A frozen 32B checkpoint is a fixed-depth pipeline — 64 layers, once, per token.
> Can you make it *think longer* without changing a stored weight?

### The answer, in three acts

**Act 1: The frozen loop (REFUTED).** The original RLC seeded thought slots
beside the prompt and recurred middle layers over them. The answer tokens
still traversed all 64 layers exactly once — only the slots were recurred.
Preregistered campaign (seed committed first, n=24/family, Holm-corrected):
slot causality REFUTED at n=72; all 7 factorial ablation arms REFUTED. The
frozen loop does not help; at 1.5B scale it hurts.

**Act 2: Intrinsic recurrence (BOUNDED_WOW_SIGNAL).** A different mechanism:
the answer's own token stream re-enters the middle block `T` times, so a
64-layer checkpoint runs 160 layers deep at T=4 with the same weights. The
controller is trained on typed traces against a verifiable opcode vocabulary.

Results on the 32B (CP566): trained controller 60/60, ordinary decode 16/60,
matched wire 7, coefficient lesion 5, wrong-state 0. 44 gains, 0 regressions,
paired one-sided exact *p* = 5.7 × 10⁻¹⁴. Adjudicated `BOUNDED_WOW_SIGNAL`.

**Act 3: Cross-generation portability (CP1011).** The gain survived migration
to a fused Qwen3.8-27B. Treatment 60/60, ordinary decode 0/60, matched wire
6/60, coefficient lesion 4/60, wrong-state 0/60. *p* = 8.67 × 10⁻¹⁹.
Independent verification replayed all 300 journal rows.

### What is bounded

The gain is on **four named executable families** (coding, calibration,
misleading premise, scientific inference) at specific task depths. It is not:
- Open-domain reasoning
- Frontier performance
- Static fusion
- Consciousness evidence
- Authorized for ordinary chat (`ordinary_chat_authorized=false`)

### The serving path

[`core/brain/llm/semantic_neural_serving.py`](../core/brain/llm/semantic_neural_serving.py):
refuses to serve unless the activation record says `active_by_default` and
matches the active model descriptor. Admission is by an answer-blind parser
over the public task grammar — unsupported language never acquires the model lane.
Current package: `rlc-27b-recovery-05346acd618d1c925f16`.

---

## Part 3: Endogenous Language — the state's own voice

**Code**: [`core/brain/llm/endogenous_*.py`](../core/brain/llm/) (12 modules)  
**Doc**: [`ENDOGENOUS_LANGUAGE_PATHWAY.md`](ENDOGENOUS_LANGUAGE_PATHWAY.md)  
**Status**: ✅ ESTABLISHED (pathway architecture), partial (live measurement)

A trained, causal path from Aura's cognitive state into the transformer's
output distribution:

```
z_Aura → 74 named dimensions → Δlogits over model vocabulary → language
```

### The first fit on Aura's own turns (2026-08-25)

- **27B persona lane (118 turns)**: `no_verdict_corpus_too_repetitive` — 39
  distinct replies, 41 "ready", 37 bare commas. No head written.
- **9B utility lane (1,629 turns)**: `content_bearing`. Held-out gain 0.0208
  nats overall, 0.0389 on rare tokens. 962 fitted, 390 held out, no reply on
  both sides. Refit on permuted states: 3e-05 (three orders of magnitude below).

### The paired recovery test

On 491 held-out turns, her own state favoured her own words in **54.0%**
(sign test p = 0.043). Above chance, close to the line, and modest. The corpus
is pinned at a recorded-at boundary; evidence:
[`docs/evidence/endogenous_language/paired_recovery_9b.json`](evidence/endogenous_language/paired_recovery_9b.json).

### z_Aura coverage

Of 74 named dimensions: 48 present, 25 varying, **50 pinned at one value**
across 1,629 turns. Each dead dimension was a reader naming an organ or key
no writer publishes. Fixed: substrate now reads `conscious_substrate`; three
collinear pairs separated; floor held by
[`test_the_state_is_not_mostly_dead.py`](../tests/test_the_state_is_not_mostly_dead.py).

### What is NOT measured

No generation has been biased by the substrate — no head has been admitted to
a decode loop. The pathway reports `no_head:no trained head on disk` on all 51
generations it has seen. A held-out likelihood gain ≠ a measured difference
between two generations under two states.

---

## Part 4: The Ghost Substrate — continuity across substrate swaps

**Code**: [`core/ghost/`](../core/ghost/) (6 modules)  
**Doc**: [`GHOST_SUBSTRATE.md`](GHOST_SUBSTRATE.md)  
**Status**: ✅ ESTABLISHED

| Organ | What it does |
|---|---|
| `causal_integration.py` | System-Φ: cross-subsystem influence, feedback recurrence, minimum-partition MI |
| `ghost_line.py` | Append-only hash-linked chain of self-pattern frames (tamper-evident) |
| `ghost_hack_guard.py` | Identity-attack defence: 5 categories, refuses silent self-mutation |
| `provenance.py` | Stand Alone Complex: "did I think this, or was I made to think it?" |
| `ghost.py` | Facade: `GhostSnapshot` → `ghost_strength` |

The Ghost fills the gap between per-forward-pass integration (`grassmann_phi`)
and continuity across moments and substrate swaps. Weight compounding calls
`Ghost.on_substrate_change()` on every promoted fusion. Restart extends the
chain rather than forking it.

---

## Part 5: Whole-System Φ & Inner Light — honest integration measurement

**Code**: [`core/consciousness/integrated_information.py`](../core/consciousness/integrated_information.py),
[`core/consciousness/inner_light/`](../core/consciousness/inner_light/)  
**Docs**: [`WHOLE_SYSTEM_PHI.md`](WHOLE_SYSTEM_PHI.md),
[`INNER_LIGHT_TEST.md`](INNER_LIGHT_TEST.md)  
**Status**: ✅ ESTABLISHED (as instruments, NOT as consciousness claims)

### Whole-System Φ

Continuous Gaussian dynamics over all live channels. Exact MIP by Queyranne's
algorithm (O(n³), verified against brute force). Grain discovery via causal
emergence. Surrogate nulls (coupling-destroying).

**Checked-in measurement**: 30.1 min, 2 Hz, 3,960 samples, 8 channels.
Φ̂ = 0.0175 nats, z = 26.3 against null. Integration established.
PCI 0.087 vs sham 0.0.

### Inner Light Test

Four neuroscience markers (differentiation, integrated complexity, criticality,
ignition) scored against 6 negative controls. Only the intact system is 4/4;
strongest surrogates reach 3/4. The claim is a **conjunction**, not a score.

> [!IMPORTANT]
> No number this subsystem emits is presented as a consciousness measurement.
> This is enforced in code (`PhiEstimate.claim`) and in tests.

---

## Part 6: Capability & Competence — the frontier-general arc

**Code**: [`core/brain/verifiers/foundry.py`](../core/brain/verifiers/foundry.py)  
**Doc**: [`FRONTIER_GENERAL_ARC.md`](FRONTIER_GENERAL_ARC.md)  
**Status**: P1 shipped; P2–P5 open

The reframe: the goal is **frontier-general Aura** (the organism), not
frontier weights (impossible — pretraining compute gap is orders of magnitude).

Three levers: test-time compute scaling (reasoning amplifier exists), movable
verifier boundary (Verifier Foundry P1 shipped), compounding without collapse
(CRSM→LoRA loop exists).

The v5 evidence protocol requires Ed25519 trust roots, commit/reveal
challenges, sealed evaluation, and 5+ independently challenged runs for a
trend claim. **No v5 Aura-model result or frontier trend is claimed.**

---

## Part 7: Novel Abstraction from Failure

**Doc**: [`GENERALITY_TODO.md`](GENERALITY_TODO.md) §C  
**Status**: C1–C6 ✅ ESTABLISHED; C7 OPEN

From 422 dated failure notes across 146 files, a concept was formed from what
failures share (running the code, not reading the prose). It named 121
patterns, pointed at 4 live unobserved defects, and the repair changed behaviour:
violations 246 → 110 → 24 → 20.

**Honest boundary**: the repair was hand-written (C7: synthesised repair is
the real frontier, and it is named but not built).

---

## Part 8: The Unification Reading — what is still missing

From an external reading of the codebase (documented in GENERALITY_TODO.md §U):

| Ref | Gap | What It Means |
|---|---|---|
| U1 | No `why_did_this_fail` that returns a *level* | Learning mechanisms are entry points, not subscribers |
| U2 | Each mechanism is boxed in a constrained substrate | Each works; none can escape its own language |
| U3 | The synthesiser has no input | `log_gap` had no caller — confirmed and joined (recording only) |
| U4 | Authority when representations disagree | The new architectural question |
| U5 | A minimal causal spine | Giant functions are the dominant complexity source |
| U6 | Counterfactual ablations, not consciousness scores | Lesion studies > scalar Φ |
| U7 | Every subsystem earns its existence by ablation | Cathedral accumulation is the stated risk |
| U8 | The test that would settle it | Repeated autonomous improvement across domains |

---

## The Current Scorecard

| Capability | Status | Bounded To |
|---|---|---|
| Learn a new representation from examples | ✅ Established | Parameterised families (authored metalanguage) |
| Transfer learned representations across domains | ✅ Established | Same family, different data types |
| Compose learned representations | ✅ Established | Depth limited by library + refactoring |
| Persist representations across restarts | ✅ Established | JSON-serialised structured programs |
| Extend compute at inference (trained recurrence) | ✅ Bounded WOW | 4 named executable families only |
| Map internal state to language | ✅ Established | Linear head; no generation biased yet |
| Maintain identity across substrate swaps | ✅ Established | Hash-chained ghost line |
| Measure system integration | ✅ Established | As instrument, not consciousness claim |
| Form concepts from failure patterns | ✅ Established | Hand-written repairs |
| Frontier-general reasoning | 🔴 Not claimed | V5 protocol ready; no result yet |
| Action-side abstractions | ✅ Established | Composition, repetition, branching and recovery over given actions |
| Autonomous capability improvement loop | 🔴 Open | U8 |
| Open-domain recurrent reasoning | 🔴 Not authorized | `ordinary_chat_authorized=false` |

---

## Sources

- [DreamCoder](https://arxiv.org/pdf/2006.08381) · [Stitch / top-down library learning](https://arxiv.org/pdf/2211.16605)
- [Turning 30: New Ideas in ILP](https://arxiv.org/pdf/2002.11002) · [On the Measure of Intelligence](https://arxiv.org/pdf/1911.01547)
- [Spelke, Core Knowledge](https://www.harvardlds.org/wp-content/uploads/2017/01/spelke2000-1.pdf)
- [Thoroughman & Shadmehr, motor primitives](https://www.researchgate.net/publication/12277176)
- [Long-term symbolic learning in Soar and ACT-R](https://www.academia.edu/18277101)
