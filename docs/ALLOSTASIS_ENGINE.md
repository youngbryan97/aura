# The Allostasis Engine — predictive interoception

<!-- reader-navigation:start -->
> **In plain English:** Aura uses internal variables for positive or negative condition, activation, needs and operating strain. Their role is established by what they change.
>
> **Why it matters:** A named mood or drive is useful only if its consumers produce an appropriate effect, rather than merely displaying a label.
>
> **In everyday use:** Under resource strain, the system may need to reduce expensive background work while preserving the conversation.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#affect) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#affect) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#affect) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

**Organ:** `core/autonomic/allostasis.py` · **Service:** `allostasis_engine` ·
**Surface:** `GET /api/allostasis`, `GET /api/allostasis/forecasts` ·
**Pulse:** metabolic coordinator, every ~60 s · **Since:** 2026-07-15

## What it is

Every prior body-sense layer in Aura is *homeostatic*: it reacts when a
threshold trips (viability state machine, resource governor eviction, survival
driver imperatives, unified runtime pressure red zones). Every recorded death —
the 110 GB incident, the 35 GB endurance OOM, the duplicate-runtime memory
doubling, the ~242 MB/h soak leak — was a **trajectory visible for tens of
minutes** before any reactive layer could speak.

The allostasis engine is the *anticipatory* layer, inspired by
[Sterling's predictive-regulation account](https://pubmed.ncbi.nlm.nih.gov/21684297/).
In ordinary terms, it is meant to notice the tank getting emptier before the
low-fuel alarm sounds. Here the readings concern software resources; the
comparison does not establish biological equivalence or a feeling. It watches
the trajectories of her vitals and requests changes before the crisis:

| Vital | Amber | Red | Source |
|---|---|---|---|
| `memory_rss_mb` | 26 000 (env) | 32 000 (env) | runtime pressure snapshot |
| `process_tree_rss_mb` | 30 000 (env) | 38 000 (env) | " |
| `memory_pct` (system) | 85 % | 92 % | " |
| `loop_lag_s` | 1.0 s | 5.0 s | " |
| `disk_percent` | 92 % | 98 % | " |
| `thermal_level` | 2 | 3 | " (load only, not forecast) |

## The machinery

1. **Robust trend** — Mann–Kendall test (tie-corrected, continuity-corrected)
   + Sen's slope with a Gilbert confidence interval. Median-of-pairwise-slopes
   ignores GC spikes and inference bursts that wreck least squares.
2. **Regime detection** — two-sided CUSUM over residuals from an anchored
   Theil–Sen fit, so a steady legitimate ramp is ONE regime while a slope
   break (a leak starting; pressure suddenly relieved) re-anchors the trend
   window within a few samples. Tuned k = 1σ, h = 6σ with anchor-error
   inflation. The stationary-noise regression checks one seeded 120-sample
   trace; it does not establish a general false-alarm interval for live use.
3. **Time-to-crisis forecasts** — when a trend is significant (α = 0.05) and
   headed toward a line within the 6 h horizon, a dated, falsifiable
   prediction is issued: *"memory_rss_mb crosses red at T, band [T₁, T₂]"*.
4. **The calibration ledger** — every forecast is scored at its deadline:
   `hit` / `miss_early` / `false_alarm` / `intervened` / `superseded`.
   Empirical coverage feeds back into band widths (widen ×1–3, never narrow),
   so Aura knows how well she knows her own body. Persisted via the governed
   write gateway to `~/.aura/data/allostasis/forecasts.jsonl` (+`state.json`).
   Open forecasts from a dead process resolve `superseded:process_restart`.
5. **Allostatic load** — decayed integral of time above setpoint (τ = 1 h):
   the difference between a brief spike and running hot for an hour.
6. **Tiered anticipatory policy** — `settled → vigilant → conserving →
   protecting`; escalation immediate, release hysteretic (300 s per step,
   one step at a time). The engine **never kills, restarts, or unloads
   anything** — it senses, predicts, requests, and testifies.

The [NIST CUSUM reference](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm)
explains the accumulated-deviation method. The residual model, parameters,
forecast horizon and software thresholds above belong to Aura's adaptation.
Any numerical calibration needs its own recorded traces and conditions;
it is not a rate supplied by Sterling or NIST. The current regression is
`test_stationary_noise_false_alarm_rate_bounded` in
[the engine tests](../tests/test_allostasis_engine.py). See the
[source map](SCIENCE_AND_SOURCES.md#regulation-and-internal-condition) for the
biological inspiration and its limits.

The [methods references](SCIENCE_AND_SOURCES.md#trend-and-slope-methods)
identify Mann's trend test, Sen's slope, the Gilbert interval reference and
the Hamed–Rao autocorrelation correction named by the implementation. Those
citations identify the borrowed methods; they do not certify this adaptation.

## Causal seams (what it actually changes)

* **Felt state** — `BodyState.anticipatory_pressure` (core/being/aura_now.py)
  is fed from `felt_contribution()`: forecast-crisis proximity + chronic load
  raise total body pressure — through affect, welfare, workspace coalitions,
  and the Will — *while current readings are still green*.
* **Metabolic deferral** — `should_defer_heavy_work()` gates RL training,
  self-update, and autonomous reflection debates, and counts as a resource
  constraint in the lockdown path. Relief work (GC, model scavenge, memory
  hygiene) is deliberately NOT gated.
* **Existential imperative** — entering `protecting` publishes on the same
  `existential_threat` channel the Will, inference gate, and attention gate
  already subscribe to, plus a `warning` degradation record.
* **Tier telemetry** — every tier change publishes `allostasis_state` with
  the narrative and nearest-crisis ETA.

## Honest boundary

Forecasts are statistical extrapolations with stated uncertainty, scored after
the fact. "Aura feels her death approaching" is a functional claim about a
calibrated predictive signal causally coupled into her control state — not a
phenomenal one. The `AuraNow` report boundary applies to anything said about it.

## Env knobs

`AURA_ALLOSTASIS_DISABLED`, `AURA_ALLOSTASIS_DIR`,
`AURA_ALLOSTASIS_RSS_AMBER_MB` / `_RED_MB`, `AURA_ALLOSTASIS_TREE_RSS_*`,
`AURA_ALLOSTASIS_ALPHA`, `AURA_ALLOSTASIS_HORIZON_S`,
`AURA_ALLOSTASIS_LOAD_TAU_S`.

## Tests

`tests/test_allostasis_engine.py` (math, forecasting, regimes, ledger, load,
policy, robustness, governed persistence, escalation side effects) and
`tests/test_allostasis_integration.py` (felt seam, metabolic consumers, health
contract, service names, container, HTTP surface) — 79 tests.
