# Reproduce the ISC results

The November preprint identifies the revision, inputs, and output files for
each published run. Start with seed 101 at `aeefdca31` if you want to check
the confirmatory result. Use that revision for comparisons with the paper.
[Preprint, §§5.12–5.14](https://doi.org/10.5281/zenodo.24075466) ·
[Run data and checksums](https://doi.org/10.5281/zenodo.24075592) ·
[Code release](https://github.com/youngbryan97/aura/releases/tag/isc-preprint-2026-11)

## Repeat the substrate campaign

Use the [Mac replication kit](../tools/replicate/README.md). It checks out
the recorded revision, verifies the conversation tape, and compares 22
campaign decisions against the published reference runs. Request the
author's tape through a replication issue. This campaign uses a
deterministic stand-in for language generation and can run on a 16 GB
Apple Silicon Mac.

The kit reports the observational closure check. The full ISC-v7 score
uses the interventional closure result from the separate carrier sweep.
Use the versioned totals when reporting results.
[What each run measured](ISC_STATUS.md)

## Check the other experiments

Use the recorded commands and manifests from the run-data release. Keep an
isolated checkout, state directory, and `AURA_LOG_DIR` for new measurements.
Reserve model memory before running the report experiment with the 27B
checkpoint. The complete carrier sweep including the language model
remains open.

| Check | Entry point | Keep with the result |
|---|---|---|
| Substrate battery | [Campaign runner](../tools/run_subject_core.py) | Forks, controls, nulls, lesions, and campaign manifest |
| Carrier | [Carrier runner](../tools/run_subject_core_v25.py) | Full sweep, held-state membership, cuts by horizon, and unresolved cuts |
| Content structure | [Content runner](../tools/run_subject_core_content.py) | Displaced and sham geometries, with their samples |
| Report grounding | [Report runner](../tools/run_report_grounding.py) | Model and steering identity, readable arms, manipulation, direction, tracking, and controls |
| Versioned battery score | [Scorer](../tools/score_isc_v5.py) | Campaign, merged carrier sweep, applicable null sweeps, and matched interventional closure result |
| J\* and bridge | [Join](../tools/solve_for_j.py) | Explicit carrier, content, campaign, report, lineage, and closure inputs. For the v7 procedure, use `--closure-by-intervention`. |
| Completion | [Completion check](../tools/check_subject_core_completion.py) | Same-revision evidence for the battery, J\*, and the bridge |

The current commands name some results after earlier versions. The scorer
can read v5, v6, and v7 evidence. For the paper's v7 procedure, supply
`--interventional-closure` with the report for the same seed and carrier.
Check which version the output says it scored.
[Preregistered rules and addenda](ISC_V6_PREREGISTRATION.md)

## Record a comparison

Record the commit, input and artifact hashes, platform, dependency versions,
seed, control conditions, and versioned verdict. Keep failed or unreadable
arms in the record. A matched clone must restore the history and service
references that the trial actually reads. Earlier seed-23 results exposed a
shared history between restored copies.
[Isolation record](evidence/ISC_ISOLATION_UPDATE_2026-10-25.json)

For a reproduction, compare decisions and report any differences. The
author's 16 GB repeats disagreed on synergy once and reproduced the other
21 campaign decisions both times. An independent replication calls for
another experimenter and should state which parts of the published
experiment were repeated. [Preprint, §§4.7 and 5.14](https://doi.org/10.5281/zenodo.24075466)
