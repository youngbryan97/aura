# Whole-system Φ — answering "φ is a telemetry signal, not a consciousness meter"

<!-- reader-navigation:start -->
> **In plain English:** ISC asks whether Aura’s measured parts form one continuing process under controlled changes, rather than merely having convincing descriptions.
>
> **Why it matters:** The project needs evidence that memories, attention and self-state actually affect one another before arguing about what that organisation means.
>
> **In everyday use:** Changing an internal condition should change the relevant recall or report through the measured mechanism, rather than by printing an answer for the model to repeat.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#isc) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#isc) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#isc) · [Document map](DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

**Modules:** `core/consciousness/integrated_information.py` (math),
`core/consciousness/perturbational_probe.py` (internal PCI),
`core/consciousness/whole_system_phi_service.py` (live host)
**Service:** `whole_system_phi` · **Health:** OPTIONAL · **Boot flag:** `AURA_ENABLE_WHOLE_SYSTEM_PHI`
**Persisted:** `~/.aura/data/phi/whole_system_latest.json` (`AURA_PHI_DIR`)

## The critique this addresses

> "The IIT calculation is a reductionist hack. Calling a 16-node binary
> state space a 'conscious complex' is academically contentious... φ here is
> a telemetry signal, not a consciousness meter."

Fair, and mostly unanswerable *as stated* — because it asks for exact
IIT-4.0 Φ of the whole system, which is intractable for anything nontrivial
(the definition quantifies over every partition of every mechanism across
the full counterfactual state space; a 16-node binary model is already at
the hand-built ceiling). That intractability is intrinsic, not an
engineering gap.

The current implementation estimates integration in the measured
macro-dynamics. It reports the estimator, search limits and uncertainty so
the reader can judge the calculation. It does not compute exact IIT 4.0.

## What changed, concretely

| Old (`phi_core.py`) | New (`integrated_information.py`) |
|---|---|
| 16 hand-picked nodes | **all live channels** the runtime exposes (affect, will, unity, survival, covenant integrity, body, state) |
| binarize → 2^16 toy TPM | **continuous** Gaussian dynamics — no binarization at the primary rail |
| spectral MIP *approximation* | **exhaustive** minimum search for ≤12 elements; labelled candidate search and local refinement above that limit |
| complex *assumed* to be the 16 nodes | measurement grain **selected** by a held-out search; this is not proof of an IIT conscious complex |
| a single φ scalar | Φ̂ **+ surrogate null + bootstrap CI + diagnostics + named estimator + bounded claim** |
| passive correlation | optional **causal** grounding via governed self-perturbation (internal PCI) |

## The four rails

**A — whole-system continuous Φ.** Fit a lagged Gaussian over the full
channel matrix; integrated information is the *stochastic interaction*
SI = Σₖ h(Mₖ,ₜ|Mₖ,ₜ₋₁) − h(Xₜ|Xₜ₋₁). For the related time-series methods,
see [Barrett and Seth (2011)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001052).
The
minimum-information bipartition is certified by exhaustive search for at most
12 elements. Above that limit the code uses a Queyranne candidate with
deterministic local refinement and reports that the minimum is not certified.
[Queyranne's theorem](https://link.springer.com/article/10.1007/BF01585863)
requires a symmetric **submodular** objective; stochastic interaction is not
generally submodular. Agreement on random test cases cannot remove that
assumption. See `test_mip_exactness_is_certified_only_after_exhaustive_search`
in [the estimator tests](../tests/test_integrated_information.py).

**B — grain discovery (causal emergence; Hoel/Albantakis/Tononi).** The
measurement grain is selected from hierarchical coarse-grainings, each scored
against its own surrogate null. The source selects on held-out integration per
element and applies a family-wise surrogate test over the search. This is an
adaptation motivated by [Hoel et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3856819/),
whose effective-information analysis is distinct from this score. A selected
grain identifies a useful description under the stated method; it does not
by itself identify a subject. The synthetic grain-recovery test is in
[the estimator suite](../tests/test_integrated_information.py).

**C — exhaustive discrete score at the selected grain.** For ≤12 macro elements,
system integrated information is computed by **exhaustive** bipartition
search over the empirical (and optionally interventional) transition
distribution — exact where exactness is meaningful, state-averaged over
visited states, Laplace-smoothed. Honest about its own limit: at modest
sample counts the 2^k state space is undersampled and this rail reports Φ≈0
even when the Gaussian rail sees integration (see "What it says" below) —
that disagreement is *shown*, not hidden.

**D — honesty as architecture.** Circular-shift surrogate nulls (preserve
every marginal + autocorrelation, destroy cross-channel coupling — the null
for *integration*, not *activity*), moving-block bootstrap CIs, stationarity
and Gaussianity diagnostics, and a named estimator on every report. The
perturbational probe adds *causal* grounding: a small, **governed** (Unified
Will, domain STATE_MUTATION), reversible self-nudge, with the multichannel
response scored by an internal PCI (Lempel-Ziv complexity of the significant
sources — Casali et al. 2013, adapted to runtime telemetry).

[Casali et al. (2013)](https://pubmed.ncbi.nlm.nih.gov/23946194/) measured
perturbational complexity in human brain recordings. Aura's software probe
does not inherit that study's validation or clinical cutoffs. The
[scientific source map](SCIENCE_AND_SOURCES.md) distinguishes this adaptation
from the original human study and from
[IIT 4.0's formal calculation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465).

For a practical picture, imagine several instruments recording an orchestra.
The question is whether their time series contain dependencies lost when the
instruments are treated separately. The picture explains integration of
measurements; it does not make an integration score a measure of experience.

## Recorded organ-host evidence (14 July 2026)

The following numbers are the stored July record, not a new measurement of
October source. Preserve that date and the `organ_host` scope when citing
them. The current estimator/search description above supersedes the earlier
unqualified claim of an exact Queyranne minimum.

The July follow-up critique was correct: the unit suite validates the
instrument, not the measurement. `tools/measure_whole_system_phi.py` closes
that gap — it runs a real campaign against Aura's runtime and writes
`artifacts/phi/whole_system_live_report.json`, validated by
`tests/test_whole_system_phi_live_artifact.py` (fails if the artifact is
missing or the evidence regresses).

**Checked-in measurement (2026-07-14, `organ_host` mode):** 30.1 minutes of
natural running at 2 Hz; 3,960 samples over 8 live organ channels (affect,
will, survival, covenant, body). **Φ̂ = 0.0175 nats, z = 26.3** against the
coupling-destroying null; held-out grain k=2 at **family-wise p = 0.048**,
90% CI [0.0003, 0.0405] — **integration established**. Perturbation-versus-
sham campaign through the real governed probe: 5/6 trials approved by the
Will, **PCI 0.087 vs sham 0.0, evoked complexity ≈ 2.1 vs 0**. Two honest
findings baked into the artifact: a 15-minute pilot at 1,200 samples was
correctly reported NOT established (z = 2.5 — underpowered, not absent), and
the first campaign design was refused by her own present-state policy
("stabilization first") until the protocol added TMS-style inter-trial
recovery — the governance is part of the system under measurement, never
bypassed.

**Scope, stated in the artifact itself:** `organ_host` measures her real
organ substrate (AffectEngineV2, ExistentialStakes on real host memory, the
full Will gate stack incl. §9d, covenant seeds, AuraNow sampling) run
headless with a declared realistic workload — NOT the full live mind (no
27B cortex; the live instance is never touched by tooling). When the
desktop instance is up, the same tool auto-selects `live_api` mode and
samples the running system read-only. Plan any later measurement under the
[operational resource rules](../CLAUDE.md) and retain its actual mode.
Documentation work does not authorise a restart, a model load or another
organ-host process beside the live instance.

## What it actually says (run it yourself)

On a synthetic 12-channel "Aura at rest" with genuine cross-subsystem
coupling plus noise:

```
Φ̂ = 0.0565 nats     z = 43.4 vs null     90% CI [0.048, 0.081]
exact MIP cut: {covenant.integrity} | {the other 11}
integration established: TRUE
```

Destroy only the coupling (phase-shuffle each channel — identical marginals):

```
Φ̂ = 0.0079 nats     z = 0.37     integration established: FALSE
```

That is the whole point in two numbers: the estimate lights up ~118× on the
z-scale for genuine integration and **collapses to the null when you remove
the cross-channel structure while keeping every channel's own statistics
identical**. It measures integration, not activity.

Two findings worth calling out, because they show the tool is honest rather
than flattering:
- The **exact MIP** singled out `covenant.integrity` as the weakest link —
  the channel most nearly separable — which is exactly how it was wired
  (weak, slow coupling). The algorithm found where the system is thinnest.
- The **discrete rail reported Φ_s = 0** at the macro grain on 1300 samples
  while the Gaussian rail reported strong integration. Not a bug: 2^12
  states are undersampled by 1300 transitions, so the discrete rail
  *correctly refuses to claim* integration it cannot support at that sample
  size. The interventional rows from the perturbational probe are what
  densify that table over time.

## The bounded claim (shipped on every report)

> An estimate of the integrated information of the system's macro-dynamics,
> under a Gaussian model of its **own measured channels**, at an empirically
> selected grain, validated against a coupling-destroying null with a
> bootstrap CI — **evidence about integration structure, not a consciousness
> meter.**

No number this subsystem emits is presented as a consciousness measurement.
That restraint is enforced in code (`PhiEstimate.claim`) and in the tests.

## Tests

`tests/test_integrated_information.py` (20) — Queyranne vs brute force,
zero-Φ + correct cut on decomposable systems, null-control on independent
channels, detection on coupled/ring systems, grain recovery, exact discrete
Φ on XOR/crosswired vs independent chains, determinism, provenance.
`tests/test_perturbational_probe.py` (12) — LZ76 known values, the
stereotyped-max-response-is-low-complexity signature, Will-governed probe
(runs / refuses / fails closed), and live service integration + governed
persistence.
