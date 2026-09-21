# Learn paired boundaries in the complete span-set objective

The factor-score diagnostic reproduces every one of the conditional
candidate's fifteen remaining pilot failures. On all fourteen span failures,
the target's operation score is below the selected graph's operation score.
On seven of those fourteen, the argument score favors the correct graph but
does not offset the operation deficit. The remaining case has identical
operation evidence and a reversed division binding. The pointer has no pair
interaction in this candidate.

The diagnostic report under `~/.aura/rlc-evidence/` is
`semantic-factor-score-attribution-20260921/report.json`, receipt
`c4747b50db51f62733a9e91e562cab3d543b5469ef4abb8853c9a720dc0db4e9`.
It performs no learning. Target annotations are used only for attribution.

## Capacity and objective

An additive pointer scores boundaries by `a(start) + b(end)`. For any two
starts and two ends, the sum on matched pairs equals the sum on crossed pairs.
No adjustment of additive coefficients can change that identity. It does not
prove that all current errors require an interaction, but it gives a concrete
capacity limit to test rather than repeatedly refitting the same model.

The existing `LinearPointerHead` already supports a learned diagonal
interaction. The complete span-set trainer can now fit it, opt-in:

    score(s,e) = H[s] @ w_start + H[e-1] @ w_end + bias
                 + sqrt(d) * (H[s] * H[e-1]) @ w_pair - count_penalty

The feature vectors remain frozen. Consequently this score is still affine
in all trained coefficients, and the exact log-partition plus positive
quadratic regularization remains strictly convex. The expectation-minus-target
gradient includes the paired feature. This is not a generalization theorem.

The implementation streams one span-length diagonal at a time. It does not
allocate an examples-by-intervals-by-hidden-width tensor. The interval
partition is unchanged; public-input exclusions remain identical to runtime.
Default fitting still freezes the parent pair coefficients. Learned interactions
use the existing serialized pointer and its existing runtime consumers, not a
new decoder or prompt. Other heads and the count penalty remain fixed.

The command is the existing refit tool with `--objective span_set_pointer`
and `--learn-span-pairs`. Incompatible objectives reject the flag. Tests cover
finite-difference gradients with excluded intervals, exhaustive runtime-score
agreement, the crossed-boundary identity, restoration, source-only fitting,
and unchanged unrelated heads.

The first focused run exposed an invalid test fixture: random hidden vectors
were not unit-normalized as required by the runtime pointer. The fixture was
corrected; no runtime contract was loosened. All 85 focused tests then passed
in 8.16 seconds. Smoke passed 164 with one skip in 44.27 seconds; lint,
compilation, layering, governance lint and writing passed without changing
baselines. The real source-trained pilot remains pending at this checkpoint.
G03 is not closed and no candidate is promoted.
