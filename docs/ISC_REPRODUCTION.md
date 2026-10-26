# Reproducing and reviewing ISC

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

**25 October isolation update:** the seed-23 outputs predate `dcdcd637c`,
which repairs consequence history shared between experimental arms. Retain
the recorded scores, but recheck paired results under repaired isolation before
presenting them as clean causal confirmation. See
[the source-hashed update](evidence/ISC_ISOLATION_UPDATE_2026-10-25.json).

The later `3e9211d3f` repair also restores the degradation tracker, which
supplies recent faults to threat computation. Verify both consequence history
and degradation history in the isolation pre-flight. The reported sixty-cut
pre-flight is a diagnostic, not a replacement campaign; current status also
records a developing-run failure in the closure estimator.


Status: Runbook · Reconciled 25 October 2026 through `517c4807c`

This is the guide for repeating a measurement and checking where its result
came from. New readers can start with [ISC explained](ISC_EXPLAINED.md) and
[the variables guide](ISC_VARIABLES.md). Read [current evidence](ISC_STATUS.md)
for completed results and open questions. These instructions describe
the measurement routes; they do not schedule another experiment on the Mac
that may already be running a live instance or an experiment.

## Protect the running system

Use a separate checkout at a pinned revision and a separate output directory.
Set `AURA_LOG_DIR` before importing project code. Use the shared Python 3.12
environment on Bryan's host. Keep the live runtime, its port and its state
root out of the experiment.

Do not start another full model beside the resident cortex or a campaign that
owns it. Whole-cortex report grounding needs a reserved model window. A
parallel-worker default is not a safe resource budget: set worker counts,
thread counts, timeouts and memory limits explicitly for the machine that
will run the experiment. Consult [the operational guide](../CLAUDE.md).

Reading this documentation and copying existing small reports requires no
model load, runtime restart, signal to an existing process or campaign rerun.

## Pin what is measured

Record the commit, relevant tree hash, dirty flag, Python and dependencies,
schema, seed, conditions, clocks, intervention design and estimators. Retain
the campaign fingerprint and all refusal reasons.

Use each experiment's actual command from its manifest. The October seed-23
run uses 370 baseline rounds, six trials and three turns per intervention arm.
The carrier sweep's deciding horizon remains two turns under its separate
preregistered design. Those durations serve different measurements.

The code changes frequently. A result at `3dc0b0599` and one at `8f548ed79`
measure different organisms. Combining them can guide discovery; it cannot
produce a same-revision confirmatory result.

The evening `72dd0d703` addendum withdrew the least-domain margin. The later
night addendum at `96487c6f1` declares K*, a numeric margin `0.02 / 1.02`, and
requires **both** observational and interventional closure on the scored seed.
Retain held-state membership, increment-based inputs, estimator diagnostics,
fold results, uncertainty and planted-cause resolution. A missing
interventional input cannot qualify a found carrier. Read the
[current method snapshot](evidence/ISC_K_STAR_METHOD_UPDATE_2026-10-25.json)
and the [scoped explanation](ISC_STATUS.md#the-night-addendum-include-what-the-transition-reads-then-test-both-ways).

The `517c4807c` restore repair also keeps method tables bound to the actual
organs rather than deep-copied frozen instances. Verify this property alongside
unchanged-fork equality: two equally frozen copies can agree while measuring
the wrong process.

## Read the measurement routes

| Purpose | Entry point | Evidence to keep |
|---|---|---|
| Complete battery | [run_subject_core.py](../tools/run_subject_core.py) | Campaign, recording, paired v6 samples, intervention arms, graph, nulls, lesions, rescues, report and manifest |
| Standalone v6 probe | [run_paired_synergy.py](../tools/run_paired_synergy.py) | `paired_synergy.json`, raw/whitened sample arrays and launch provenance; no battery total |
| Carrier sweep | [run_subject_core_v25.py](../tools/run_subject_core_v25.py) or [sharded launcher](../tools/run_subject_core_v25_sharded.py) | Frozen design, grain sufficiency, merged cuts at each horizon, controls and authority blockers |
| Content structure | [run_subject_core_content.py](../tools/run_subject_core_content.py) | Static agreement, design recovery, displaced and sham geometries, samples and verdict |
| Withheld reports | [run_report_grounding.py](../tools/run_report_grounding.py) | Cortex identity, basis, served/readable arms, manipulation, direction and tracking |
| Joined v5/v6 battery reading | [score_isc_v5.py](../tools/score_isc_v5.py) | Both versioned counts and all criterion decisions, with explicit campaign/sweep inputs |
| J* and parity | [solve_for_j.py](../tools/solve_for_j.py) | Explicit carrier, content, campaign, reports and lineage inputs, plus unresolved grounds |

The scorer's filename retains v5 because it joins the v5 partition result.
It also writes and prints the v6 reading when the campaign carries that
evidence. Check `v5_passed`, `isc_v5_on_this_seed`, `v6_passed` and
`isc_v6_on_this_seed`; a bare legacy `passed` count is ambiguous.

`--quick`, `--skip-nulls`, `--skip-lesion` and `--skip-grain` are discovery or
wiring options. An unmeasured requirement cannot be counted as a pass.

## Join explicit completed inputs

The following commands illustrate the joins after the runs finish. The
uppercase path names are placeholders for verified local evidence, not paths
that ship in this repository. Run them in the pinned checkout with its log
directory already set.

```bash
/Users/bryan/.aura/live-source/.venv/bin/python tools/score_isc_v5.py \
  --campaign CAMPAIGN_RUN_DIRECTORY \
  --sweep MERGED_CARRIER_RUN_DIRECTORY

/Users/bryan/.aura/live-source/.venv/bin/python tools/solve_for_j.py \
  --carrier CARRIER_REPORT \
  --content CONTENT_REPORT \
  --campaign CAMPAIGN_REPORT \
  --reports REPORT_GROUNDING_REPORT \
  --state-log LINEAGE_STATE_DATABASE \
  --out-root NEW_BRIDGE_OUTPUT_DIRECTORY
```

Supply `--null-sweep NAME=REPORT` to the scorer for every null that requires
its own sweep. A shard or screen cannot stand in for the merged deciding
sweep. Audit matching commit, seed, configuration and scope yourself as well
as checking each tool's refusals; a file being accepted by a loader does not
prove that all inputs belong to one experiment.

Use explicit files for a published result. `solve_for_j.py` can otherwise
discover recent carrier/content files in its default artifact roots. That is
convenient for exploration and insufficient provenance for a confirmed join.

## Check the scope

Campaign A uses the real offline organs with `DeterministicMind` in place of
language generation. A positive result concerns that measured substrate.
Campaign B must include the real cortex and relevant recurrent state in
comparable forks. A whole-model report experiment establishes only the
reported measurement; it does not retroactively make Campaign A's carrier
cortex-inclusive.

Report grounding needs the real cortex, readable answers and the intended
steering conditions. The bridge reads the run with the printed state readings
withheld. Verify steering-on/off and unrelated controls where required by the
declared design. Inspect `measured` separately from `holds`: missing cortex
answers and a measured failure of tracking are different outcomes.

Record a candidate basis's fusion certificate separately from its installation
verifier and signed serving authority. On 24 October the layer-43 candidate
passed the first and failed the second. A measurement-basis experiment does
not attest that the desktop is serving that candidate.

## Publish enough to disagree

Retain every failed, void and superseded run with its original label. Publish
versioned counts, per-criterion decisions, effect sizes, uncertainty, controls,
sample counts, fingerprints, commands and artifact hashes. State which private
inputs, model weights or local records another person would need. The public
source alone does not include every input required for a whole-Aura replay.

[subject_core_replication_package.py](../tools/subject_core_replication_package.py)
builds a package from its configured battery artifact collection. Review its
actual file list: it does not automatically gather the separate v25, content,
withheld-report and steering-certificate trees. Add those inputs and their
manifests to the published record without implying that a battery-only package
already contains them.

The [25 October snapshot](evidence/ISC_STATUS_2026-10-25.json) is a
checkable summary of selected existing files. Their hashes bind the extracts
to the originals; the extracts do not replace raw trajectories, model inputs
or an independent replication package.

An independent rerun should report disagreements and unresolved criteria. A
minimal alternative architecture should face the same frozen protocol, scope
and controls. Optimizing one synergy statistic or hand-labeling the comparator
unconscious does not test the full conjunction's specificity.

## Joining the current closure evidence

The scorer and J* command now accept `--interventional-closure PATH`, pointing
to the actual measured report for the matching seed, boundary and source.
See [score_isc_v5.py](../tools/score_isc_v5.py) and
[solve_for_j.py](../tools/solve_for_j.py). Supplying a JSON file is not itself
proof of matching provenance or adequate intervention coverage; inspect the
report's manifest, anchors, unchanged twins and tested units. Keep missing,
open and closed results distinct. A new source rule does not rescore the
older committed outputs automatically.

At `5bc357b8c`, the upstream discovery account reports 284 unchanged swaps on
the seed-23 anchors used to develop K*. It also fixes service references and
republish-before-relink ordering during restore. Verify object ownership as
well as matching numbers. A new seed supplies the held-out check; do not use
the discovery account as that confirmation. See [the scoped status](ISC_STATUS.md#later-discovery-check-and-another-restore-repair).
