# Connectome buildout — working list

Source research: H01 human cortex (Science 2024, adk4858), FlyEM male CNS
connectome (Cell 2026), ZAPBench (arXiv 2503.02618), Neuroglancer,
Potjans & Diesmann 2014 microcircuit.

## Wave 1 — acquisition
- [ ] core/connectome/types.py — units, compartments, classes, contacts
- [ ] core/connectome/volume.py — structural reconstruction from source (EM analogue)
- [ ] core/connectome/activity.py — functional recorder (light-sheet analogue)
- [ ] core/connectome/segmentation.py — FFN-style seeded growth + agglomeration + ERL
- [ ] core/connectome/proofreading.py — split/merge edit ledger + focused queue
- [ ] tests

## Wave 2 — analysis
- [ ] synaptology.py — contact multiplicity vs H01 law; strong-connection registry
- [ ] celltypes.py — connectivity typing + cross-run reproducibility
- [ ] topology.py — triad census, reciprocity, rich club, small-worldness vs null
- [ ] dimorphism.py — variant comparison, higher-order concentration test
- [ ] tests

## Wave 3 — ZAPBench for Aura
- [ ] forecast.py — mean, condition-mean, linear, time-mix, TSMixer, connectome-conditioned
- [ ] zapbench.py — C in {4,256}, H=32, held-out condition, MAE per horizon
- [ ] structure-vs-blind ablation with a null
- [ ] tests + a real run

## Wave 4 — human cortex
- [ ] microcircuit.py — Potjans-Diesmann matrix, laminar assignment, laminar routing rule
- [ ] criticality.py — MR estimator, crackling-noise relation; upgrade the regulator
- [ ] neuromodulation.py — Doya assignment wired to real knobs
- [ ] tests

## Wave 5 — beyond biology
- [ ] beyond.py — delay compiler, exact rewiring + rollback, engineered recurrence,
      type synthesis, dual-variant simultaneity; each against its own null
- [ ] tests

## Wave 6 — integration
- [ ] integration.py — health report, telemetry channels, invariants, container
- [ ] neuroglancer.py — precomputed export + viewer state
- [ ] neuro_reference mappings with falsifiers
- [ ] DEPS, make compile/lint/smoke/writing/layering green
- [ ] commit + push each wave
