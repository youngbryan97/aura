# Guide to evaluating Aura

<!-- reader-navigation:start -->
> A test states what will be measured, what it will be compared with, and what counts as a pass.
>
> Source code, a passing test, and a useful result in practice each answer a different question. To see whether memory helps, try the same tasks with memory enabled and disabled. Keep every success and failure.
>
> [Plain-language guide](docs/PLAIN_ENGLISH_GUIDE.md#measurement) · [Terms and examples](docs/VARIABLES_EXPLAINED.md#measurement) · [Evidence and related work](docs/DEMOS_AND_RELATED_WORK.md#measurement)
> [All docs](docs/DOCUMENTATION_INDEX.md) · [Glossary](docs/GLOSSARY.md) · [Worked formulas](docs/FORMULAS.md)
<!-- reader-navigation:end -->

Status: Guide · Commands and evidence scope reviewed 25 October 2026

Start with something you want to check. Does Aura remember a correction?
Can a learned method improve an answer? Is a published number reproducible?
The commands and records below help you choose a check for that question.

The G09 comparison passed its declared math and planning criterion. Tests
of reliable improvement in everyday use (L09) and a matched comparison with
frontier systems (G12) remain open. [Reasoning experiments](docs/RECURSIVE_LATENT_CORTEX.md) ·
[Open claims](CLAIMS_NOT_SUPPORTED.md)

## Prepare an independent checkout

Use Python 3.12. The resident MLX desktop targets Apple Silicon macOS;
some offline checks can run elsewhere when their dependencies permit it.
Linux support for an offline check is not support for the complete MLX app.
Follow [INSTALL](INSTALL.md) and [hardware profiles](HARDWARE_PROFILES.md).

`make setup-prod` creates or uses `.venv` and installs `requirements/core.txt`.
It does not install `requirements_hardened.txt`. Hash-pinned reproduction uses
its declared lock file and environment; a filename alone is not proof that
installed packages match it. Do not update a shared resident environment as
part of reviewing documentation.

## Select bounded checks

Set `AURA_LOG_DIR` to a directory belonging to the review. On a shared host,
check available memory and existing owners first. Start with:

```bash
make smoke
make doc-drift
make claim-constants
make evidence-integrity
```

`make doctor` checks the development environment; `aura doctor --bundle`
collects runtime diagnostics. They are different entry points. For full
offline regression use the chunk runner and memory budget in
[TESTING](TESTING.md), rather than a single process collecting the whole suite.

The checks above do not prove ISC, general intelligence or production uptime.
Record the source commit, command, result and environment beside every claim.

## Certification and live experiments

`make certify` invokes [tools/certify.py](tools/certify.py). Its configured
checks are a selected safe-mode test, boot certification, Aletheia live proof
and architecture ablation. It does not compile or run every test in the repo.
Boot and live-proof producers can launch or contact runtime processes; run
them only in a separately budgeted environment with no collision with the
resident instance. They are not the default document check.

The tool writes `artifacts/certification/latest/` and a verdict under
`artifacts/aletheia/`. The verdict’s `agi_proven`, `consciousness_proven` and
`open_world_autonomy_proven` fields are policy-set false values, not measured
negative findings. The writer emits ordinary JSON; inspect each artifact’s
actual hash/signature fields before claiming cryptographic verification.
Local certification is not third-party certification.

`tools/ablation_runner.py --list` lists its conditions without running a
model. A measured ablation supports the component, task family and controls
in that result; it cannot establish that every component earns its cost.

## Read retained results and corrections together

The fabricated ablation and simulated soak files removed in August are
explained in [the dated reconciliation](docs/DOC_STATUS.md#fabricated-evidence-found-and-removed).
They supply no capability or longevity credit. A retraction sidecar preserves
why an earlier output cannot support its former claim.

The [15 July](docs/SOAK_VERDICT_2026_07_15.md),
[18 July](docs/SOAK_VERDICT_2026_07_18.md) and
[25 July](docs/SOAK_VERDICT_2026_07_25.md) soak verdicts describe their recorded
runs. A short idle sample is not a multi-day stability result. New longevity
work needs the workload, duration and resource limits in
[the soak standard](docs/LONGEVITY_SOAK_STANDARD.md).

For ISC use [current status](docs/ISC_STATUS.md) and
[reproduction](docs/ISC_REPRODUCTION.md). Seed-23 outputs predate subsequent
arm-isolation repairs; no full ISC pass or new carrier campaign is established
by this document review.

## Explain a finding plainly

Say what changed, what comparison was made, and how far the conclusion reaches:
“Kept procedures rescued 29 answers across the tested math and planning
requests without a loss. That passed G09’s rule. It does not establish an
advantage on arbitrary questions or against frontier systems.”

Use [the glossary](docs/GLOSSARY.md), [formula guide](docs/FORMULAS.md) and
[scientific sources](docs/SCIENCE_AND_SOURCES.md) for the terms and assumptions.
