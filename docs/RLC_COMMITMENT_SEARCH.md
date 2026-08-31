# Commitment search: what to do when depth stopped working

Status: **measured. +11.3 points on 160 paired tasks (p=0.044), by REJECTION SAMPLING. Prompt-conditioned exclusion was measured and REFUTED.**
Owner: `core/brain/llm/latent_cortex/commitment_*.py`, `sequential_exclusion.py`

## The problem, stated as physics

Every latent-recurrence result in this codebase has been negative, and in
retrospect all of them were predictable from one measurement:

```
cos(pass1, pass2) = 0.9994
```

Iterating a fixed operator on its own output is a contraction. It reaches
its fixed point in one step, so depth 8 computes what depth 1 computed.
More depth on a contraction is a longer identity function. The frozen-loop
refutation, vanilla beating every latent arm, and "recurrence itself is the
damage" are all the same finding seen from different angles.

The question that survives is why chain-of-thought works, since it
demonstrably does. The usual answer — "more computation" — cannot be right,
because the latent loop is also more computation and buys nothing.

**A chain of thought is a sequence of irreversible commitments.** Emitting a
token collapses a distribution; every later pass conditions on a decision
that cannot be unmade, and the hypothesis space monotonically shrinks. That
is an information-theoretic act, not an arithmetic one. Latent recurrence
bought computation *without commitment* — it carries a superposition forward
and smooths it, and averaging a superposition is exactly the contraction we
measured.

## The commitment ratchet

A token-free commitment cannot be a vector: a vector can be blended, and
anything blendable can be un-decided. It is a **constraint** — a discrete
proposition the answer must satisfy, with four properties a latent state
lacks.

| property | meaning |
|---|---|
| discrete | committing is a decision, not a blend |
| irreversible | within an episode it cannot be retracted |
| narrowing | it removes admissible answers, and the amount is **measured** |
| checkable | a deterministic function decides satisfaction |

Integrity rule: **committing a contradiction is refused.** A device that
admits ¬P after P is a pile, not a ratchet, and a pile does not narrow.

Honesty rule: narrowing is measured against a live candidate pool or
reported as `unmeasured`. It is never asserted. A commit that narrows
nothing is refused with a receipt, so the caller cancels the pass instead of
spending a forward pass on an unchanged problem — the 0.9994 identity step,
made structurally impossible.

Exception, deliberately: a requirement the **prompt stated** ("answer in one
word", "in kilometres") commits even when it eliminates no current
candidate. Its value is that it stays true on pass six, when a model several
passes deep has quietly stopped honouring it. Those commit as unmeasured and
never contribute to a measured-narrowing claim.

## Sequential exclusion — the part with a theorem

The specific commitment with the highest prior probability of paying,
because it assumes nothing about the model.

Let the answer distribution be `p` over answer set `A`, correct set `A*`
with mass `p*`.

```
i.i.d. best-of-N:              P = 1 − (1 − p*)^N
after refutations of mass m_k: P(draw k+1 correct) = p*/(1 − m_k)  ≥  p*
```

Every factor is no larger than the i.i.d. one and strictly smaller once any
mass is removed. **Exclusion dominates i.i.d. for every N, every
distribution, every p\*.** That is a statement about renormalising a measure
after removing mass — not a claim about a checkpoint, which is why it does
not need the checkpoint to internalise anything.

### Why it is large here — measured, not argued

The dominance scales with how peaked `p` is, and this system had measured
its own peakedness twice without naming it as such: `cos = 0.9994`, and
"collapse is cheapest". Both say the sampler keeps redrawing the same
answer. But neither was ever turned into the number that matters, because
branch candidate texts are worker-private and no campaign artifact retained
them — so the premise behind every "more branches" decision was untested.

Measured 2026-08-09, Qwen2.5-1.5B-Instruct-4bit, 8 short-answer tasks,
8 i.i.d. draws each at temperature 0.7
(`artifacts/rlc/commitment_search/peakedness_qwen1p5b_20260809.json`,
reproduce with `tools/measure_candidate_peakedness.py`):

| | measured |
|---|---|
| mean peakedness (Herfindahl) | **0.516** |
| distinct answers | **25 of 64 draws** |
| expected distinct per 8 i.i.d. draws | **2.58** |

**Best-of-8 is best-of-2.6.** Five of eight passes re-derive an answer
already examined. One task ("how many sides does a hexagon have") returned
the identical answer all eight times — peakedness 1.0, the point-mass case,
seven wasted passes out of eight.

That is the complete explanation for why more branches and more depth have
bought so little, and it is now a measurement rather than an inference.

Exclusion turns the peak from a liability into an asset: the bigger the
mode, the more one refutation removes.

| p(wrong mode)=0.70, p*=0.05, N=8 | success |
|---|---|
| i.i.d. | 33.7% |
| sequential exclusion | 92.2% |

Same checkpoint, same eight passes. Asserted in
`tests/test_sequential_exclusion_dominates_iid.py`, not in prose.

### The three premises, all instrumented

A null result must be diagnostic, not mysterious.

- **soundness** — the verifier must only refute incorrect answers. A refuted
  correct answer excludes the truth permanently. `gold_exclusions` — one is
  a defect report, not a metric.
- **compliance** — the model must honour an exclusion. If it redraws the
  excluded answer, exclusion is nominal and no gain can appear. Measured per
  draw.
- **support** — the correct answer must be reachable at all. If `p* = 0`, no
  search policy helps. Exposed as the oracle ceiling.

### It predicts its own gain

`predict_distinct_advantage` computes expected **distinct** candidates for
both policies from a pilot sample alone. That needs no knowledge of `p*`, so
it is checkable in the same run that produces the outcome. Matching
prediction means the mechanism is understood rather than correlated; missing
it names which premise broke.

## Where it is wired

| seam | file | effect |
|---|---|---|
| RLC episode | `core/brain/llm/latent_cortex/engine.py` `_build_episode_ratchet` | refuted branches become rejection-set entries; stated requirements and unanimous agreement remain prompt-visible commitments |
| repair redraw | `core/brain/llm/latent_cortex/engine.py` local-repair loop | exact repeats are rejected before decomposition or verification and receive up to three bounded generations |
| live response lane | `core/brain/reasoning_revision_gate.py` `deliberate_best_of` | verifier-call-bounded best-of-N samples **without replacement**; excluded text never enters the prompt |
| standalone policy | `core/brain/llm/latent_cortex/sequential_exclusion.py` | verifier calls, generations, compliance and rejected redraws are independently receipted |
| receipts | `EpisodeReceipt.commitment_ratchet` | commitments, refusals, measured narrowing |
| operator view | `core/brain/llm/latent_cortex/commitment_telemetry.py` | `rlc.duplicate_passes` goes RED at 4 — best-of-8 behaving like best-of-2, visible at last |

### On blindness

`deliberate_best_of` runs blind passes by design, and that blindness buys
something real: a pass that sees a plausible prior answer can rationalise
toward it instead of solving. **That hazard applies only to unrefuted
candidates.** A refuted one is removed, not offered. So blindness survives
exactly where it was paying, and coverage improves where it was costing.
Unchecked and undecided verdicts never exclude.

## How to refute it

```bash
python tools/run_commitment_ablation.py --tasks tasks.jsonl --draws 8
```

Five arms. Exit 0 SUPPORTED, 1 REFUTED, 2 INCONCLUSIVE.

- `vanilla` — the floor.
- `depth_only` — same extra passes, conditioned on nothing. Isolates the
  claim from "more compute".
- **`shuffle` — the arm built to kill it.** Same constraints, same count,
  same context cost, permuted across steps. The claim is that a commitment
  narrows the problem for the passes that *follow* it; if shuffle matches
  real, ordering carries no information and the mechanism is not the
  mechanism. `adjudicate()` returns REFUTED and says so in those words.
- `random` — same vocabulary, no evidence. Predicts *worse* than vanilla: a
  wrong commitment is irreversible too.
- `oracle` — ceiling. Not deployable; it answers whether the constraint
  channel can carry a gain at all, so a null in the real arm is attributable
  to extraction rather than to the idea.

Missing arms are always INCONCLUSIVE. There is no combination of absent
comparisons that returns SUPPORTED.

The null hypothesis is printed with every verdict, so nobody has to
reconstruct what would have counted as failure after seeing the numbers:

> the ratchet's score is explained by extra passes and extra prompt text,
> not by the order in which commitments constrain later passes

## The end-to-end measurement

Qwen2.5-1.5B-Instruct-4bit, 3-digit multiplication, best-of-6, 4 seeds x 40
tasks = 160 paired tasks, sound non-oracle verifier (recomputation, never
shown the gold answer).
`artifacts/rlc/commitment_search/exclusion_ab_aggregate_20260809.json`

| arm | solved | rate | mean verifier calls |
|---|---|---|---|
| i.i.d. best-of-6 | 77/160 | 48.1% | 3.97 |
| exclusion, **prompt-conditioned** | 75/160 | **46.9%** | 4.2 |
| exclusion, **rejection-sampled** | 95/160 | **59.4%** | **3.19** |

**+11.3 points, z=2.02, p=0.044 — on FEWER verifier calls.**

### The finding that matters: how you implement the restriction IS the result

Prompt-conditioned exclusion lost. Twice, before the difficulty was even
calibrated, and again here. Describing the excluded answers in context does
not restrict `p` — it *perturbs* `p`, adding tokens that anchor on the very
values being excluded. Compliance was 0.43, then 0.64 after the instruction
was rewritten, and distinct coverage FELL in both.

The first version of the conditioning block ended with

> Work within these. They were committed against evidence; reopening them
> repeats work already done.

which is a *convergence* instruction. Told to settle, the model settled —
onto the answers it had been told were wrong. Requirements say "stay inside
this"; exclusions say "go somewhere else". Rendering both under one heading
with one closing instruction inverted the mechanism.

Rejection sampling has none of that. Draw from the unconditioned model,
discard any draw landing in R, redraw. That is `p` restricted to `A \ R`,
renormalised — the theorem, exactly, with no perturbation.

### And the budget unit decided the sign

Charging a rejected draw against the DRAW budget made rejection lose. That
was right: a duplicate of an already-refuted answer costs nothing to verify
when the verifier is arithmetic, so removing it saved nothing and cost a
generation.

The theorem's budget is **verifier calls** — the expensive, bounded resource
in any real deployment, where verifying means running a test, calling a
tool, or paying a model. Denominated correctly, rejection wins and spends
less. The live revision path, standalone policy, and RLC local-repair path now
implement that budget explicitly. They report verifier calls separately from
generations and rejected redraws so the extra model compute stays visible.

### Three calibration failures, all caught by the harness

| band | i.i.d. rate | outcome |
|---|---|---|
| two-digit | 100% | ceiling — zero refutations, nothing to exclude |
| moderate | 91.7% | ceiling |
| **hard** | **45.8%** | **the only band where a policy can show** |
| 3x3-digit | 8.3% | floor — p* ≈ 0, the answer is outside the model's reach |

A ceiling and a floor both look exactly like a null result. `run_exclusion_ab`
returns INCONCLUSIVE for either rather than reporting a policy verdict from a
saturated band, which is how all three were caught instead of being written
up as "no effect".

## What is measured, and what is not

**Measured.** The dominance arithmetic, swept over 400 constructed
distributions with zero counterexamples (registered claim
`sequential_exclusion_dominates_iid_sampling`). And the premise: answer
distributions on a real model are peaked enough that i.i.d. best-of-8
examines ~2.6 distinct answers.

**Also measured now.** A correctness gain end to end: +11.3 points on 160
paired tasks, p=0.044, on fewer verifier calls — by rejection sampling only.

**Not measured.** One model (1.5B), one task family (multiplication), one
difficulty band. p=0.044 is a real effect at the edge of the noise, not a
settled one. Nothing here shows the gain transfers to the resident Cortex (27B), to
reasoning tasks with long answers where "the same answer" needs a semantic
judge rather than string equality, or to a verifier that is expensive enough
to change the trade in the other direction.

What would change the verdict:

* if the SHUFFLE arm matches the real arm, ordering carries no information
  and this is refuted;
* if compliance is low, the model ignores exclusions and the coverage gain
  cannot appear regardless of the arithmetic;
* if the verifier ever refutes a correct answer, the truth is excluded
  permanently and no budget recovers from it.

All three are instrumented. A null here will name its cause.
