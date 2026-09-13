# G03 joint definition selection

The opt-in decoder carries competing definition spans into the existing typed
argument graph optimizer. Each register retains its original input/operation
anchor and up to four pointer-ranked local spans. Each selected reference is
scored against its selected definition. Binary label variables prevent one
register from taking different definitions across its uses.

The decoder uses the existing learned pointer, role, proposal and relation
heads. It does not receive source text, expected answers, gold program edges,
or definition annotations at runtime. The default decoder and serving packages
are unchanged. The policy is hash-bound in the transducer receipt.

## Development measurements

| Source validation measure | Frozen parent | Joint candidate | Gains | Regressions |
| --- | ---: | ---: | ---: | ---: |
| Exact program | 304/500 | 405/500 | 114 | 13 |
| Structurally equivalent program | 319/500 | 431/500 | 120 | 8 |
| Exact answer | 346/500 | 444/500 | 105 | 7 |

The preceding operation-pointer candidate scored 377/500 answers. The new
candidate gets all 192 fork/join programs exactly correct. Arithmetic answers
rise from the frozen parent's 97/128 to 117/128, cataphoric from 4/48 to 25/48,
role binding from 24/48 to 39/48, and reserved aliases from 30/48 to 35/48.
The two natural source sets retain 24/24 and 12/12 answers.

On the 48 exposed weave tasks the candidate scores 47/48 programs and answers,
against the preceding candidate's 48/48. The failure selects a mention from a
neighboring clause and changes the branch graph. Coefficient lesion and
hidden-token shuffle each score 0/48 programs and answers. Their complete
receipts are retained, including the shuffle's longer search time.

These cohorts use each input and intermediate once. They do not measure a gain
from the cross-use consistency constraint itself. The gain is from admitting
alternative definition hypotheses to the scored graph. Repeated-use correctness
is covered by solver tests, not claimed as a measured neural transfer result.

A separate candidate combines the corrected fork-definition pointer refit with
the same joint graph. It scores 438/500 source answers and 47/48 weave answers;
its 185/192 fork/join answers trail the 192/192 result above. That refit is not
selected. Neither candidate passes the unchanged no-regression admission rule.

## Verification and retained evidence

`tests/test_semantic_argument_optimization.py` compares small joint charts with
exhaustive enumeration, reverses candidate order, tests inconsistent definitions,
and checks that the opt-in policy reaches the ordinary transducer decoder.
Combined solver/shared-transducer tests: 86 passed. Related relation, definition,
scoring and verification tests: 50 passed. Smoke: 164 passed, one skipped.
Compile, governance lint, layering, writing and touched-file Ruff passed.
Whole-repository lint is not claimed by this record.

The evidence directory is
`artifacts/rlc/semantic_program_joint_definition_dev_20260913/`.
Its receipt binds all measured rows and the two local candidate files.
Primary candidate receipt:
`a0e90470dd6e023c3c8dd71b2366687d670606de1d41247ea5652f696325cbb3`.

These are exposed development comparisons, not fresh replication, broad gain,
backbone recurrence, public-answer quality, frontier performance, or live
activation. G03 remains open. G04-G12 are not closed by this checkpoint.
