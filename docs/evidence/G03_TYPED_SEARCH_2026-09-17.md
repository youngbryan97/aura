# Typed operation search development

Operation beams could fill with integer-only charts even when the public
inputs included a sequence that the learned use contract required consuming.
The new opt-in search keeps prefixes by argument demand and result supply,
as well as arity. It checks per-type register-use bounds before pruning.
These are necessary feasibility conditions, not proofs of task correctness.
No expected answer or annotated operation enters runtime selection.

Exhaustive small-grammar tests preserve valid connected programs and compare
the selected top-k charts with complete enumeration, including overlapping
spans. The policy survives serialization and reaches runtime feasibility.
The focused suite passed 105 tests. Existing candidates keep their policies.

## Small replay

Eight training and eight exposed validation examples were selected by source
hash and geometry. No fitting occurred. Training stayed 8/8 equivalent;
validation improved from 6/8 to 7/8 against the experimental joint decoder.
Retaining two labels and retaining the whole label inventory agreed on these
16 rows. Receipt:
`~/.aura/rlc-evidence/semantic-likelihood-diagnosis-20260917/typed-cohort.json`,
content receipt
`bffac3c4f1185d33b46490e9b2e28dde5eaad9d7248e76354665550ba911dbe2`.

## Stronger control

The wider replay included the original 472/500 incumbent. It was stopped
after seven complete paired rows because joint scoring still regressed on
source `00950779577b2916002706a516c78d8fe449159ac443b4deb04a1703638e7cf9`.
The incumbent was equivalent; both joint variants were different.
The recovered cataphoric case was already correct under the incumbent.
Thus the small improvement above repairs an experimental regression, not a
measured gain over the incumbent. The unrun rows remain unmeasured.

Completed per-arm observations are immutable under
`~/.aura/rlc-evidence/semantic-typed-search-20260917/`. Each binds source,
model identity, elapsed time and semantic outcome. The next comparison keeps
the incumbent's selection rule and changes only typed candidate construction.
G03 remains open. No candidate is promoted or deployed.
