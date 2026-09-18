# Operation/background competition: rejected candidate

The prior operation classifier saw only annotated operation spans during
training. At runtime it classified pointer proposals, including fragments
and result references. The new opt-in refit gives those source-training
negatives a background class in the same learned distribution.

Runtime keeps background probability in the denominator when scoring an
operation. It never emits background as an opcode and does not hard-reject
an answer when background wins. Typed graph selection remains the existing
path. The refit preserves the source vocabulary, pointer, argument heads,
register-use contract and operation-count calibration.

The implementation and CLI are available for development. No coefficient
is promoted or deployed.

## Source fit and small replay

The source-only fit used 4,110 labeled spans and completed in 3.789 seconds
after source loading. Candidate identity:
`960d8d729c8269a83d31210e94fc820ac7baecf8088a974e0e4ab5883d066e14`.

The replay combined the twelve source-aligned incumbent failures with 48
validation cases chosen by hash across geometries. All are exposed
development data, not prospective replication.

| Candidate | Exact and equivalent | Paired gains | Paired losses |
| --- | ---: | ---: | ---: |
| Incumbent | 48/60 | 0 | 0 |
| Background refit | 45/60 | 0 | 3 |

The candidate repaired none of the twelve failures and regressed on three
other cases. The incumbent remains selected. A full 500-case replay is not
warranted by this result. G03 remains open.

Receipt: `~/.aura/rlc-evidence/semantic-operation-background-20260918/small.json`.
Content receipt:
`804cb7f33e3976a85209392f3fbf561f112b526c25ef4034d8bd578f9c010cd7`.

## Checks

Focused tests: 57 passed. The tests cover source-only fitting, hard-negative
label integrity, probability scoring without answer suppression, receipt
identity, serialization, and coefficient lesions. Smoke: 164 passed, one
skipped. Lint, compile, governance and layering gates passed.

An initial scratch replay passed the wrong keywords to the cohort selector
after fitting. The saved candidate was reused unchanged after correcting
that call. The failed invocation did not produce an evaluation result.
