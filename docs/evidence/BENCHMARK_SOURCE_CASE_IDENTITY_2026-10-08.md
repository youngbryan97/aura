# Benchmark source cases, 8 October 2026

The offline benchmark audit found a result-key collision before any new model
evaluation was started. A request's text identifies its content, but does not
identify its occurrence in a source collection.

## Source evidence

The pinned BBEH Mini file has SHA-256
`14e77b3d6be68faa008d268abf53b1f8d2420ffdd762504a304dafc3f8d43026`.
It contains 460 cases and 459 distinct input strings. Zero-based source
positions 14 and 258 contain the same 715-character input, with targets
`ahɨka` and `it was opened`. Their former input-only key was
`541170b4582e970f57a20036`.

The original runner could overwrite one completed case with the other. On
resume, that one file could cause both source cases to be skipped. The G09
draft importer also used an input-key dictionary, which selected the last
reference. This audit establishes a defect in those mechanisms. It does not
establish that an existing saved evaluation row was damaged.

The [publisher's README](https://github.com/google-deepmind/bbeh/blob/main/README.md)
also specifies 460 Mini cases and identifies the official evaluation code.
The collision measurements above come from the pinned local source bytes.

## Repair

`tools/benchmark_case_identity.py` indexes a complete source before selection
or reordering. A unique legacy ID remains unchanged. Repeated IDs receive the
original source ordinal. Each source case carries a type-preserving reference
hash; the catalog checks its ID, reference and supplied provenance together.
This mechanism accepts arbitrary JSON references and has no task-specific
grading rules.

The pinned Mini now has 460 distinct result IDs. Its 458 unique legacy IDs
remain unchanged. The two repeated cases use `--case-14` and `--case-258`,
with their original inputs and targets intact.

G12 validates saved row bindings and decode receipts before importing the
model library. Invalid durations, token counts, verdicts and termination
records cannot stand for completed evaluation. Valid exhausted-budget results
remain completed wrong results, including the historical single-decode
`length` termination. New rows publish through the governed atomic file
gateway without replacing prior evidence.

The G09 BBEH importer uses the same complete source catalog and carries its
ordinal and reference hash into the kept-procedure arm. Comparison rejects
duplicate saved IDs. With a known source, missing counts include cases absent
from both arms. Without a source expectation, completeness is unknown. An
empty expected collection does not certify an unmeasured run.

Prompts, source targets, graders, token allowance and decode settings are
unchanged. Ambiguous legacy rows require explicit recovery; this repair does
not reinterpret or migrate them.

## Validation

The independent regression run passed **49 tests** across source identity,
G12 analysis, MATH-500, batched decoding, the G09 harness and public-answer
protocol. These tests mock model delivery. They cover filtered and resumed
execution, reverse completion order, contradictory references, binding
mismatches, malformed saved decodes, historical budget failures, atomic
no-overwrite publication, G09 provenance and shared omissions.

Smoke passed **165 tests, one skipped**. Lint, compilation, layering and
writing passed. Strict Ruff checks on all six changed Python paths passed.
Governance and independent review results are appended after their completion.

The existing benchmark continuation owns model work. These changes did not
load a model, launch a benchmark, modify its evidence, or signal its processes.
The local recovery supervisor remains prepared and unlaunched; its 26 synthetic
safety tests do not prove an executed watchdog. Flash presentation readiness
still requires later ordinary in-app prompting and observed task completion.

## Final review and governance

Independent read-only review approved the six Python paths, including the
publication gateway and downstream consumers. It found no P0 or P1 defects.

Governance lint remains red with **94 existing baseline regressions**. This
checkpoint changes no files in its effect-inventory roots (`core`,
`interface`, `skills`, `tools/longevity`, `tools/chaos`). The raw report is
`~/.aura/flash-proof-2026-10-08/benchmark-identity-governance.log`.
The ownership baseline was not expanded.
