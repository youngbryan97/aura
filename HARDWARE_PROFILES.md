# Aura Hardware & Model Profiles

<!-- reader-navigation:start -->
> **In plain English:** Resource management keeps model inference, background work and the operating system within the machine’s real capacity.
>
> **Why it matters:** A useful assistant must avoid exhausting the host or making the conversation unusable while doing background work.
>
> **In everyday use:** A second large model load can disrupt a working session even if each model fits alone. Scheduling and a measured memory budget matter.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#resources) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#resources) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](docs/DOCUMENTATION_INDEX.md)
> [Glossary](docs/GLOSSARY.md) · [Formulas with worked examples](docs/FORMULAS.md)
<!-- reader-navigation:end -->

Status: Guide · Resource and evidence scope reconciled 25 October 2026

Hardware determines which experiment can fit. It does not confer a capability
claim. A passing test supports its stated property on its recorded setup;
more memory alone does not establish agency, reliability or intelligence.

## Choose a setup by the work

| Setup | Suitable work | What it cannot establish alone |
|---|---|---|
| Development without models | Source inspection, static checks, bounded tests with doubles | Model output, live latency or autonomous behavior |
| Small CI runner | Bounded offline contracts under a declared memory budget | Full desktop performance; MLX inference also requires a supported Apple Silicon environment |
| Target 64 GB Apple Silicon host | A resident Cortex with controlled auxiliary work, when admission and existing workloads permit | Safe concurrent loading of all named lanes, or a second full Cortex |
| Higher-memory Apple Silicon host | Larger declared experiments if admitted and qualified | Better answers or mature autonomy without measurements |
| Live browser or physical setup | Observed interaction with the declared environment and independent outcome evidence | General performance from a fixture or a single demonstration |

The [model roster](docs/MODEL_ROSTER.md) lists the configured Cortex,
Brainstem and Reflex roles. Three roles do not require three simultaneously
resident models. Code generation also has a route through the Cortex;
[local_code_model.py](core/brain/llm/local_code_model.py) describes that path.

## Budget the whole process tree

[Lane admission](core/brain/lane_admission.py) derives a model-lane budget from
host memory by default. [Model lane control](core/runtime/model_lane_control.py)
owns durable reservations and lifecycle decisions. An arithmetic fit check
alone is not a reservation, and current free memory alone does not account
for another process's pending model load.

The 64 GB target is a shared CPU/GPU memory pool. Keep the operating system,
conversation KV cache, embedding work, speech and other active campaigns in
the budget. Follow [CLAUDE.md](CLAUDE.md) before any live experiment. Do not
restart, kill or duplicate the resident runtime without current authorisation.

A smaller configured model can support a smaller experiment; it does not
reproduce a result measured on the 27B. A large model's disk size and its live
peak allocation are different measurements.

## Tests and longevity runs

Set a separate `AURA_LOG_DIR`, limit numeric-library threads, bound runtime
and measure the test process tree. Use the chunk runner for broad suites and
choose chunk count from the available budget. Do not run bare `pytest` over
the whole repository or assume the historical six-chunk choice is safe with
the resident model present. See [TESTING](TESTING.md).

Longevity profiles name durations and workloads. A `proof_short` run does not
establish days of continuity. A 64 GB host does not inherently prohibit a
multi-day run; such a run requires an approved resource window, supervision,
recovery and actual duration evidence. [Evaluation guidance](EVALUATE_AURA.md)
and [claim limits](CLAIMS_NOT_SUPPORTED.md) explain what a recorded run supports.
