# Source-anchored program scoring

The historical evaluator compared register numbers without aligning their
source mentions. Equal-valued inputs can exchange register numbers while
all uses remain attached to the same source mentions. Sixteen of the
incumbent's 28 historical failures have this form.

The opt-in `source_anchors_v2` evaluator reuses the training coordinate
alignment. It requires unique matching source spans and preserves each
public input value. Wrong binding between equal-valued source mentions
still fails. Missing anchors and unequal-valued permutations fail. Decoding
receives no annotated operations, bindings, or expected answers.

The default v1 evaluator and its checkpoint identity are unchanged. The v2
identity binds the scoring rule and source anchors; old checkpoint rows
cannot be reused under the new rule. Historical reports are unchanged.

## Completed replay

All 500 exposed development cases were replayed for both candidates:

| Candidate | Source-anchored exact | Equivalent | Paired gains | Paired losses |
| --- | ---: | ---: | ---: | ---: |
| Unchanged incumbent | 488/500 | 488/500 | 0 | 0 |
| Typed chart construction, two retained labels | 488/500 | 488/500 | 0 | 0 |

The selected candidate remains the incumbent. The change from the historical
472/500 is an evaluator correction, not a learning gain. Twelve semantic
failures remain. This is not fresh-transfer evidence and does not close G03.

Receipt: `~/.aura/rlc-evidence/semantic-source-anchored-20260917/validation.json`.
Content receipt:
`78c95ac20d889e8de1ffaa271e29d1e7ecf4e8b627bd484f905356a233ef768d`.
Incumbent identity:
`c3817f75a0b9e8b72c601fbbd97c0494dca423bcb30eac9683b871831701d15a`.
Typed candidate identity:
`5a24ce33d25487ec8196443292f9776eb48f2500bc6e3136605ed9c58a118850`.

The earlier 48-case hash/geometry replay was also unchanged at 48/48.
Replaying all 28 historical failures with reference-backed input alignment
found 16 equivalent and 12 different under both candidates. The typed search
repair has no measured accuracy gain over this incumbent.

## Verification

Focused validation, input-coordinate, source-training and verification
suites: 37 passed. Smoke: 164 passed, one skipped. Lint, compile, governance
and layering gates passed. No runtime model or serving policy was changed.
