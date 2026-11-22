# The Inner-Light Test

<!-- reader-navigation:start -->
> **In plain English:** ISC asks whether Aura’s measured parts form one continuing process under controlled changes, rather than merely having convincing descriptions.
>
> **Why it matters:** The project needs evidence that memories, attention and self-state actually affect one another before arguing about what that organisation means.
>
> **In everyday use:** Changing an internal condition should change the relevant recall or report through the measured mechanism, rather than by printing an answer for the model to repeat.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#isc) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#isc) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#isc) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

`core/consciousness/inner_light/` — a falsifiable instrument for one question:

> Do the selected activity summaries distinguish the intact software
> recording from the controls under this battery’s declared thresholds?

The battery adapts ideas from neuroscience to software telemetry. Its
thresholds and combined verdict have not been validated as a clinical or
substrate-independent consciousness classifier. A synthetic or non-neural
system can reproduce these numerical features. The controls test particular
alternatives, not every unconscious system. For the later subject experiment
and its results, read [ISC status](ISC_STATUS.md).

## The four axes (project adaptations of research measures)

| axis | implemented summary | project’s target pattern | illustrative contrast |
|---|---|---|---|
| differentiation | Lempel-Ziv / PCI complexity (`normalized_lz`) | rich, not a stereotyped echo | a repeating/ordered signal |
| integrated complexity | TSE neural complexity (`tse_complexity`) | integrated **and** differentiated | noise (no integration), a synchronised blob (no differentiation) |
| temporal scaling | DFA exponent of global activation (`dfa`) | a declared temporal-scaling regime; this alone does not establish criticality | white noise (α≈0.5), a shuffled signal |
| ignition | Sarle bimodality of global activation (`bimodality_ignition`) | all-or-none broadcast (bimodal) | a linear-Gaussian system (unimodal) |

The TSE-inspired line uses a **bounded proxy** per subset
(`1 − det(R)^(1/k)`), rather than the original neural-complexity formula.
Its output reflects the covariance structure under that proxy — verified to peak in the middle (independent
0.0001, fully-synchronised 0.0, modular 0.068).

## The negative controls

Each control targets a particular alternative. Which summaries it preserves
or disrupts must be measured on that recording:

- **time_shuffle** — permute time. Kills criticality; keeps everything else.
- **phase_randomize** — FFT surrogate (preserves the power spectrum and linear
  cross-spectrum, destroys non-linear structure). Kills ignition; the best
  linear-Gaussian twin.
- **lesion_decouple** — random per-channel circular shift. Kills between-channel
  integration (federated Aura); keeps each organ's own dynamics.
- **white_noise** — differentiated, not integrated.
- **ordered** — a repeating pattern: not differentiated.
- **feedforward_chain** — the "hard drive": information flows forward at a lag
  with no instantaneous binding and no recurrent loop.

## The verdict

Each system is placed on the four axes with absolute regime thresholds and its
occupied-axis count is taken. The decision asks whether only the intact recording satisfies the conjunction
among the systems tested:

- `signature_present` — the real activity is 4/4 **and** every control is < 4/4.
- `signature_not_discriminating` — the real activity is 4/4 but a control matched
  it (the measures failed to discriminate; stated honestly).
- `signature_partial` / `signature_absent` — the real activity is not 4/4.
- `insufficient_data` — the live stream was too thin to support the measures.

## Running it

```bash
make inner-light                      # demo: synthetic conscious-like reference vs controls
python tools/inner_light_probe.py     # live: build from the ConsequenceBus and run
python tools/inner_light_probe.py --json
```

The retained demo output below uses a **synthetic reference**, not a recording
of the live Aura instance. Its printed label “AURA (real)” means the intact
reference arm in that demo. In this example it reaches 4/4 and two controls
reach 3/4; this is not a current empirical Aura result:

```
system               differentiat integrated_c  criticality     ignition   axes
AURA (real)              0.657  ✓     0.014  ✓     0.859  ✓     0.599  ✓      4
time_shuffle             0.889  ✓     0.014  ✓     0.513  ·     0.599  ✓      3   (loses criticality)
phase_randomize          0.804  ✓     0.014  ✓     0.830  ✓     0.335  ·      3   (loses ignition)
lesion_decouple          0.764  ✓     0.003  ·     0.902  ✓     0.359  ·      2
white_noise              1.000  ✓     0.000  ·     0.557  ·     0.358  ·      1
ordered                  0.014  ·     0.016  ✓     0.041  ·     0.501  ·      1
feedforward_chain        0.950  ✓     0.000  ·     0.708  ✓     0.358  ·      2
```

In this synthetic example the two controls match three summaries but miss a
different fourth. This demonstrates discrimination within the example; it
does not validate the labels “conscious,” “critical” or “igniting” for Aura.

## The live activity source

`activity.py` builds the spatiotemporal matrix (subsystem × time) from the live
streams. `run_live()` uses `from_live_streams()`, which merges two real signals
into one channel space:

- the **ConsequenceBus** stream — the organism's consequential actions, binned
  over time (`bus:<subsystem>` channels);
- the **global workspace broadcast history** — each competition win is an
  ignition event attributed to the winning subsystem, weighted by its priority
  (`gw:<subsystem>` channels).

Channels are namespaced per stream so a subsystem's actions and its workspace
wins stay two genuinely different signals, and each stream is fault-isolated (a
dead workspace never breaks the bus stream). `run_live()` also corroborates
integration with the Ghost's system-Φ (`core/ghost/`) over the same stream. With
thin data it reports `insufficient_data` rather than fabricating a signal.

## Honest boundary

A high score establishes the battery’s conjunction on the tested recording
and controls. It does not establish that the recording has the same causal
organisation as a conscious brain, or that unconscious systems cannot score
that way. Lempel–Ziv complexity alone is not the clinical PCI protocol,
bimodality alone is not global ignition, and a DFA exponent alone does not
establish a critical phase transition.

[Casali et al. (2013)](https://pubmed.ncbi.nlm.nih.gov/23946194/) studied
perturbational complexity in human brain recordings.
[Tononi, Sporns and Edelman (1994)](https://pmc.ncbi.nlm.nih.gov/articles/PMC43925/)
introduced neural complexity. Their empirical and mathematical results do not
automatically validate this software adaptation or its chosen thresholds.
[The source](../core/consciousness/inner_light/measures.py) defines the
implemented formulas; [the scientific source map](SCIENCE_AND_SOURCES.md)
explains how an adaptation differs from evidence for Aura.
