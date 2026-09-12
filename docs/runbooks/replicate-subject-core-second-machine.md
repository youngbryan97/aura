# Replicate the Subject Core campaign on a second machine

A result that survives only on the machine it was measured on is a result about
that machine. This is the procedure for running the frozen campaign somewhere
else and reading the two runs against each other.

## Before you start

The second machine needs the same source, the same configuration and the same
model. The campaign fingerprint enforces the first two — a run with a different
threshold or a different schema is a different campaign, and
`tools/compare_across_machines.py` refuses to read across them. Nothing enforces
the third, so check it by hand.

Apple Silicon is the reference platform. A second Apple Silicon machine answers
"does this depend on this laptop"; a different architecture answers a wider
question and is the auxiliary robustness test, not the replication.

## Procedure

1. On the second machine, check out the exact commit the first run recorded.

   ```bash
   git rev-parse HEAD    # must equal campaign.commit in the first report
   ```

2. Confirm the tree hashes the same as the run recorded, so the code is the
   code that produced the number.

   ```bash
   python tools/seal_evaluation_run.py --run artifacts/subject_core/run_0NN --check
   ```

3. Run the frozen campaign whole. Do not shorten it: the criteria are a
   conjunction, and a partial run answers a different question.

   ```bash
   caffeinate -dims python tools/run_subject_core.py --rounds 60 --trials 6 \
       --lesion-rounds 24 --out artifacts/subject_core
   ```

4. Copy the second machine's run directory next to the first, and compare.

   ```bash
   python tools/compare_across_machines.py \
       artifacts/subject_core/run_0NN there/run_0MM --json artifacts/subject_core/machines.json
   ```

## How to read the comparison

Two runs on one machine already differ — irreducibility moved 0.028 and the
lesion deficit 0.122 between two of ours. So one number here against one number
there says nothing. The tool reports every measure twice: the spread between
runs on one host, and the spread between hosts.

- **Between smaller than within.** No machine effect on that measure. This is
  the answer you want.
- **Between larger than within.** A machine effect, named in
  `measures_with_a_machine_effect`. If the apparent integrated subject
  disappears under small timing differences, that is the result — record it,
  do not tune around it.
- **`criteria_that_disagreed` is not empty.** The two machines answered a
  preregistered criterion differently. The claim does not hold on the strength
  of one machine.
- **`synergy.passing_only_on_some_runs` is not empty.** The numbers can agree
  while the verdicts do not; a triple that clears its null on one host and not
  the other has not replicated.

To separate the machine from the run you need at least two runs per machine.
With one run each, `within_machine_spread` is empty and every gap reads as
`None` rather than as a machine effect, which is honest and useless.

## If the second machine cannot run it

`tools/cortex_campaign_preflight.py` reports what a machine can and cannot
claim before anything starts. A blocked answer there is a blocked answer for
the replication too.
