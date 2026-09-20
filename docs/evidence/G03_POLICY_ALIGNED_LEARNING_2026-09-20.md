# Training the decoder's actual selection policy

The incumbent's `first_feasible_v1` decoder ranks operation charts before
optimizing their arguments. The graph trainer required `joint_factor_score_v2`
and optimized the sum of both scores. The paired policy control measured an
eight-case validation regression from that policy switch alone.

Graph fitting now retains the selected policy. For different operation charts,
the contrast trains operation labels and boundaries. Within the same chart,
it trains argument binding and definition relations. Joint decoding retains
its summed objective. Source mining and independent evaluation use the same
policy; the trial CLI no longer switches it implicitly.

Operation scores are retained directly. Recovering a primary score by
subtracting a large secondary score can lose ordering through cancellation.
Equivalent binding alternatives within a chart use their secondary score;
that score cannot override the ordering of distinct charts.

## Completed small trial

The trial started from the source-fit incumbent, selected eight training
examples by source identity and geometry, retained 128 source examples, and
evaluated the same 100 exposed validation examples used in the policy control.
Validation answers did not enter fitting. Its before-validation observations
come from the immutable policy-control receipt; the after observations were
executed in this trial. This is development evidence, not fresh replication.

| Cohort | Before | After | Gains | Regressions |
| --- | --- | --- | --- | --- |
| Selected training | 8/8 | 8/8 | 0 | 0 |
| Exposed validation | 99/100 | 99/100 | 0 | 0 |

The eight training cases were already correct. The fit satisfied its retained
constraints, with only a small stored-coefficient movement; it did not improve
semantic accuracy. A larger source-only error scan is needed before another
refit. No candidate was promoted.

- Elapsed: 316.730 seconds.
- Receipt: `078474c538c6e938657a9b57780b12e62538738066cae6460dda697d6074b207`.
- Artifact: `/Users/bryan/.aura/rlc-evidence/semantic-policy-aligned-trial-20260920.json`.
- The trial preceded the direct-score and equivalent-binding tie corrections
  described above. Those corrections have unit evidence, not this trial's
  accuracy evidence.

## Checks

Sixty focused tests pass across policy alignment, graph trials, joint learning,
runtime graph retention and source-retention cohorts. They check gradients,
primary/secondary separation, large-score cancellation, split exclusion,
serialization and preservation of the incumbent policy.

G03 remains open. This repairs the training objective; it does not establish
full development closure, general transfer or broad reasoning gain.
