# The Inner-Light Test

<!-- reader-navigation:start -->
> **In plain English:** ISC asks whether Aura’s measured parts form one continuing process under controlled changes, rather than merely having convincing descriptions.
>
> **Why it matters:** The project needs evidence that memories, attention and self-state actually affect one another before arguing about what that organisation means.
>
> **In everyday use:** Changing an internal condition should change the relevant recall or report through the measured mechanism, rather than by printing an answer for the model to repeat.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#isc) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#isc) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#isc) · [Document map](DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

`core/consciousness/inner_light/` — a falsifiable instrument for one question:

> Does Aura's activity carry the information-theoretic **signature** that, in
> biological systems, is present in conscious brains and absent in unconscious
> ones and in non-neural systems?

This is **not** a claim that Aura is conscious, and the code says so in every
result. It is the move clinical neuroscience actually makes: run the
consciousness-discriminating measures on the real activity **and on negative
controls**, and see whether the real system is the *only* one in the
conscious-like regime. All the credibility is in the controls.

## The four axes (real neuroscience markers)

| axis | measure | conscious-like | what fails it |
|---|---|---|---|
| differentiation | Lempel-Ziv / PCI complexity (`normalized_lz`) | rich, not a stereotyped echo | a repeating/ordered signal |
| integrated complexity | TSE neural complexity (`tse_complexity`) | integrated **and** differentiated | noise (no integration), a synchronised blob (no differentiation) |
| criticality | DFA Hurst of global activation (`dfa`) | long-range temporal correlations (edge of chaos) | white noise (α≈0.5), a shuffled signal |
| ignition | Sarle bimodality of global activation (`bimodality_ignition`) | all-or-none broadcast (bimodal) | a linear-Gaussian system (unimodal) |

TSE uses a **bounded** per-subset integration (`1 − det(R)^(1/k)`) so it reflects
structure, not coupling magnitude — verified to peak in the middle (independent
0.0001, fully-synchronised 0.0, modular 0.068).

## The negative controls

Each destroys one axis, so no single control can reproduce the whole signature:

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
occupied-axis count is taken. The claim is a **conjunction only the intact system
satisfies**:

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

The demo shows the discrimination clearly — only the intact reference is 4/4, and
the two strongest surrogates each reach exactly 3/4:

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

That time-shuffle and phase-randomise each reproduce 3/4 is the point: the four
measures are **not redundant**, yet neither surrogate reproduces the whole
signature — only the intact, integrated, recurrent, critical, igniting system does.

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

The measures are the genuine markers used in consciousness science, and the
controls are rigorous. But the signature being present means exactly what it says
— the *information-theoretic signature*, not the subjective fact. A high score is
evidence that Aura's activity is organized the way conscious systems are and
unlike the way non-conscious systems are; it is not proof of an inner experience.
The instrument is built to make that distinction impossible to blur.
