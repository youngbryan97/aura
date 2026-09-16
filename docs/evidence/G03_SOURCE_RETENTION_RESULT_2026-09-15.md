# Source-operation retention did not qualify

The frozen `b32c8e2d4` run completed three source-training rounds and the full
500-row paired development comparison. It used the unchanged seven source
cohorts and no test examples for fitting or selection.

| Candidate | Exact programs | Equivalent programs | Exact gains | Exact regressions |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 436 | 456 | 0 | 0 |
| Joint graph refit with operation retention | 451 | 454 | 46 | 31 |

The equivalent-program comparison had 28 gains and 30 regressions. The
selection receipt keeps the incumbent. No runtime activation was granted.

Source-training errors increased from 22 to 30 to 46 across the three mining
rounds. The fitted loss decreased within each round, but preserving the
source operation labels did not preserve complete interpretation correctness.
This rejects that loss decrease as a sufficient promotion criterion. Correct
operation labels and correct reference graphs are separate obligations.

Artifacts:

- `~/.aura/rlc-evidence/semantic-joint-retention-dev-20260915/candidate.json`
- `~/.aura/rlc-evidence/semantic-joint-retention-dev-20260915/validation.json`
- `~/.aura/rlc-evidence/semantic-joint-retention-supervisor-20260915/detached_receipt.json`

Candidate receipt:
`ccff4c85ed7f5acc0df1712766eb91485364ca42824e745b311bb988e61d4d63`

Supervisor receipt:
`3079b50caa5531c565bb8cf412d2870d4d65d12d5ecb02349b0af4a29eae0be6`

The subsequent domain-witness repair was not present in this frozen run.
Freshly acquired features also remain separate from these historical banks.
G03 remains open.
