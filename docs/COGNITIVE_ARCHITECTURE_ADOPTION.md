# Cognitive architecture adoption: Soar, ACT-R, and a real competition

<!-- reader-navigation:start -->
> **In plain English:** Several concerns can propose work, but attention selects what has influence now and makes selected information available elsewhere.
>
> **Why it matters:** A new question, a recalled event and an old task must not silently become interchangeable sources of authority.
>
> **In everyday use:** Your new question should not be overwritten by a background task that happened to win a previous competition.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#attention) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#attention) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#cognitive-architectures) · [Document map](DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

Status: Guide · Reviewed against the tree 2026-08-13

The companion to [ENGINEERING_ADOPTION.md](ENGINEERING_ADOPTION.md). That page
covers seven waves of *runtime* discipline taken from the Linux kernel, LLVM,
Kubernetes, Chromium, and flight software. This one covers what was taken from
the two mature symbolic cognitive architectures — **Soar** and **ACT-R** — in
August 2026, and the workspace repair that came with them.

The selection criterion is the same throughout the repository: *does this
address a failure Aura has actually had, or an assumption it currently makes
with nothing enforcing it?* In all three cases below, the answer was a specific
measured defect rather than an architectural preference.

| Recorded defect | What was missing | Adopted from |
|---|---|---|
| Tied workspace bids resolved by whoever submitted first | A decision procedure that admits it did not decide | Soar impasses |
| Episodic recency was a constant — every episode scored 1.000000 | A scale-free activation equation with no epoch to go stale | ACT-R base-level activation |
| Deadlocked decisions were re-derived every time | An occasion for learning, with the cost of learning accounted | Soar chunking + the utility problem |

---

## Soar: impasses and chunking

`core/cognition/impasse.py`

Soar's central claim is that learning has exactly one occasion: the
architecture notices it cannot decide, creates a substate to work out what to
do, and compiles the result into a rule so the same deadlock is never worked
through twice. That occasion is an **impasse**.

Aura had decision points that deadlocked and no vocabulary for it. The global
workspace sorted candidates and took `[0]`, so exactly-tied bids resolved by
whoever submitted first — a tie impasse settled by list order, invisibly.
Nothing recorded that a decision had been arbitrary, so nothing could learn
from it and nothing could report how often it happened.

The four impasse types are Soar's, and they are exhaustive over "the decision
procedure did not produce a choice":

| Type | Meaning |
|---|---|
| `TIE` | Several candidates, nothing discriminates |
| `CONFLICT` | Preferences contradict each other |
| `REJECTION` | Every candidate was rejected, or none offered |
| `NO_CHANGE` | A choice was made and nothing moved |

### The utility problem, paid up front

The standard criticism of chunking — the one Soar spent years on — is that
learned rules are not free. Each adds match cost to every subsequent decision,
so a system that learns indiscriminately gets *slower* as it gets more
experienced, and the slowdown is invisible because each individual chunk looks
like a win.

So a chunk is retained only while it pays, on a per-use expected value:

```
EV = p_correct · cost_saved_per_use − match_cost
```

`ChunkStore.prune` retracts everything with `EV <= 0`. That is a derived
condition, not a tuned threshold — a chunk whose expected value is negative is
costing more than it saves, by definition. The accounting is what makes
chunking safe to leave switched on.

The second criticism is **over-generalisation**: a chunk firing in situations
its originating episode did not cover, giving a confident wrong answer. That is
why `ChunkStore.record_outcome` exists and why `p_correct` is *measured* rather
than assumed. A chunk that fires often and is wrong has a negative EV and is
retracted by the same rule that catches the expensive ones.

Above the exact chunker sits a Tier 2 generalization layer, wired into live
deliberation with the seam method held as a gate.

---

## ACT-R: subsymbolic activation

`core/cognition/actr_activation.py` · `tools/fit_actr_retrieval.py`

### The defect

`EpisodicMemory._recency_score` was this:

```python
min(1.0, max(0.0, ep.timestamp - 1774000000) / 2000000)
```

That is a step function keyed to a hardcoded wall-clock epoch: 0.0 before 2026-03-20, a 23-day ramp, then a flat 1.0
forever. Evaluated on 2026-08-12, an episode from one minute ago and one from
thirty days ago **both scored exactly 1.000000**. The recency term contributed
a constant 0.4 to every candidate and the ranking was importance-only. It could
not have discriminated, and it drifted further from usable every day.

A better constant would not fix it. *Any* absolute-epoch formulation carries
the same bug. ACT-R's base-level equation is scale-free — it depends only on
elapsed time — so it cannot saturate and has no epoch to go stale.

### The equations

```
B_i = ln( Σ_j t_j^-d )                       base-level activation
A_i = B_i + Σ_k W_k · S_ki + P_i + ε         plus context and error terms
P(retrieve) = 1 / (1 + exp(-(A_i - τ) / s))  retrieval probability
T_i         = F · exp(-A_i)                  predicted latency
```

Frequency raises `B_i`, recency weights recent uses more, and the sum
reproduces both the forgetting curve and the spacing effect without either
being modelled separately.

### The fit, and the null

Both equations were fitted against Aura's own measured recall — 6,000 samples,
150 batches of 40 traces, ages from one minute to a year, 0 to 30 rehearsals,
scored on whether each trace actually came back in the ranked top-k. **They
came out differently.**

**The retrieval curve fits.** Maximum likelihood gives `tau = -0.4666`,
`s = 2.0` (`FITTED_PARAMETERS`), with a Brier skill of 0.154 over the base
rate. Modest by construction: activation carries 0.4 of the ranking blend
against 0.6 for importance, so activation alone *should* explain some of recall
and not most of it.

**The latency equation does not transfer, so `F` is not fitted.** Regressing
`ln T` on `-A` across the same samples gives `r² = 0.000037`. There is no
relationship, and no reason there should be: `T = F·e^-A` earns its shape in
ACT-R because retrieval there is a race between activations, whereas Aura's
recall is a ranked scan whose cost tracks how many candidates exist and what
the store does, not how strong the winning trace is.

**`F` is a pure multiplicative scale and would have absorbed any timing
whatsoever.** Fitting it would have produced a confident number with no
mechanism under it. `tools/fit_actr_retrieval.py`
refuses to emit an `F` below an r² of 0.10 for that reason, and a test pins the
null so that if retrieval ever does become activation-driven, someone finds
out.

So Aura can now predict *which* memories return, with fitted parameters and a
reported skill score. It cannot predict *how long* recall takes from
activation — and that is a property of its retrieval architecture, not a gap in
the model.

---

## The global workspace: an actual competition

`core/consciousness/global_workspace.py`

The workspace is the competitive bottleneck — one winner per cognitive tick.
Three defects were fixed together.

**Ties were being decided by the scheduler and reported as a judgement.**
Sorting and taking `[0]` settled ties by arrival order, and worse, it did not
*look* like arrival order: `effective_priority` scales salience by
`(1 - 0.03·age)`, so submission microseconds became a priority difference. Four
sources bidding an identical 0.70 came out ~2e-6 apart.

`_resolve_tie` now applies two rules in order, neither of which can see when a
bid arrived:

1. **Least fatigued wins.** Fatigue already encodes how recently and how often
   a source held the broadcast, so among equals it goes to whoever has waited
   longest. It is the same quantity the competition already uses, so
   tie-breaking cannot pull against arbitration.
2. **Rotate on the tick.** When fatigue is level too — the common case at
   startup, when everything is zero — the tick index selects from the sources
   in a stable order. Deterministic, reproducible, and fair over time, where a
   fixed order would hand every genuine tie to whichever source sorts first,
   forever.

**Winner fatigue makes the competition real.** A win reduces that source's next
effective priority by `_WINNER_FATIGUE = 0.15` — large enough to let a 0.02
priority gap rotate, small enough that a 0.79 gap (urgent vs idle) still
dominates. Accumulated adaptation is capped at two wins' worth.

**The rotation is wider than two, and observable.** The previous behaviour
alternated between a pair; the workspace now rotates across the full contender
set, and the tie count and last tie are exposed rather than inferred.

---

## Related work in the same wave

- **Belief decay.** `core/epistemics/belief_revision.py` — the canonical engine,
  the one that actually boots — gained per-day multiplicative decay toward a
  `DECAY_FLOOR` rather than toward zero, because absence of reinforcement is
  absence of evidence, not evidence of absence. The duplicate at
  `core/cognition/belief_revision.py` was reachable from nothing and has been
  reduced to a tombstone: its docstring had declared four behaviours and only
  one existed.
- **Action value.** Contextual, self-refreshing, and graded, ranked on recorded
  outcomes rather than on how an action is spelled.
- **A cognitive FMEA.** Only infrastructure had a failure-mode registry; see
  [FMEA.md](FMEA.md) (generated — do not hand-edit).
- **Written-down assumptions.** Following seL4's practice, the checks now
  record what they assume rather than leaving it implicit.

## Adding to this

Same bar as [ENGINEERING_ADOPTION.md](ENGINEERING_ADOPTION.md): name the
recorded defect first, then the idea, then the module. An adoption that cannot
name the failure it addresses does not belong on this page — and an equation
borrowed from a cognitive architecture must be *fitted against Aura's own
data*, with the null reported when it doesn't transfer.
