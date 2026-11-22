# Reviewer packet

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

Status: Guide · Reconciled 25 October 2026

This packet helps a reviewer trace a claim to its implementation and evidence.
Begin with [current ISC status](ISC_STATUS.md), [the source map](SCIENCE_AND_SOURCES.md)
and [unsupported claims](../CLAIMS_NOT_SUPPORTED.md). A link resolving is not
proof that its contents describe the current source or a current live result.

## What you need

Use Python 3.12 and a separate checkout pinned to the result's source revision.
The 64 GB Apple Silicon host is the target for the resident 27B path. Other
hardware can support bounded offline checks or different model experiments;
retain that difference in the result. No fixed install or benchmark duration
is promised.

Before running anything, read [the operational rules](../CLAUDE.md),
[hardware profiles](../HARDWARE_PROFILES.md) and [TESTING](../TESTING.md).
Preserve the live instance and campaigns. Set a separate `AURA_LOG_DIR` and an
explicit time/memory budget. Longevity, chaos injection and model evaluation
are separately planned experiments, not install checks to run in sequence.

## Inspect before executing

| Item | What it supplies |
|---|---|
| [Documentation status](DOC_STATUS.md) and [document map](DOCUMENTATION_INDEX.md) | Current guides versus dated/frozen/generated records |
| [Claim limits](../CLAIMS_NOT_SUPPORTED.md) and [supported claims](../CLAIMS_SUPPORTED.md) | Bounded engineering claims and their evidence requirements |
| [ISC reproduction](ISC_REPRODUCTION.md) | Versioned battery, carrier, content, reports and joins |
| [Reality reach](REALITY_REACH.md#current-evidence) | Physical-claim boundary and its open ledger |
| [Terminology](TERMINOLOGY.md) | What the internal names mechanically mean |
| [Benchmark runner](../aura_bench/runner.py) | Registration and scoring contracts |
| [Authority gateway](../core/executive/authority_gateway.py) | Action authority and receipts |
| [Latent bridge](../core/brain/latent_bridge.py) | Internal state reaching generation controls |

## Choose bounded checks

The [test guide](../TESTING.md) describes smoke and focused contracts. Broader
offline coverage uses `tools/run_test_chunks.py`; choose its chunk count from
host load, rather than copying a fixed six-chunk setting. A full single pytest
process has historically exhausted memory. The recorded test inventory is a
dated source of counts, not a timeless total.

The capability-token, conscience, self-object and belief-court test paths in
the repository provide narrow starting points. Run only the selected checks
under the declared budget. A green lint or contract test checks its property;
it does not establish that every live action was governed or every claim was
independently replicated.

## Interpret results within their scope

The ISC seed-23 outputs and later measurement corrections must be read
together. The 22/24 record is not a completed full pass. The component demo is
not the current ISC experiment. The [demos guide](DEMOS_AND_RELATED_WORK.md)
separates scripts, fixtures, recorded interactions and controlled experiments.

A runner does not establish a 30-day result. Physical interfaces do not
establish a measured physical effect. A method motivated by a consciousness
theory does not by itself settle the experience question. Keep the named
bridge assumptions beside any scientific argument.

The [ROADMAP](../ROADMAP.md) contains programme plans and older assessments;
it is not the sole authority for every current result. Prefer the specific
versioned artifact, source and maintained status page for the claim at issue.
