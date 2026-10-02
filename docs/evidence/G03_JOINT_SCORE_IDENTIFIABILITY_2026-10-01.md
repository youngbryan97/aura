# Joint Role Score Identifiability, 2026-10-01

The factored replay from frozen `9b45e9ff58` completed all sixteen public
operation charts in 29.907505 seconds. It returned a program rather than a
budget refusal, but that program was wrong. Both comparison controls remained
correct. The selected graph added a fourth `mul` operation to the intended
three-operation fork/join computation.

The selected chart's 9,265 scored alternatives had learned scores from
66.033417 to 80.338959. Those absolute additions raised its argument score
to 627.093596. Thirty mention realizations were certified redundant; 9,235
remained in the exact SSA solver. The best/second distinct register-graph
margin was 0.483108. A positive margin did not make the interpretation true.

The report receipt is
`73a84abd87a4abcbb5db5a01197befc1792a71ff5b30d4c16c2cb140a70271cb`.
Independent negative verification returned
`dc23df4d22447fabcef5595a576039a860d3df467587baf20945f8cbe1728c81`.
The contained terminal receipt is
`0f15a70710ae72cb62141f0f12360782d4ccc502219ac6f160295181d0094763`:
108.368418 seconds, return two, empty process group and lineage. Supervisor
12685, child 12688 and inhibitor 12694 were gone on inspection. Artifacts
remain under `semantic-grounded-joint-profile-factored-v2-20261001`.

## Identified Defect

The fitted role-choice loss is invariant to adding a constant to every
candidate logit in a role. Its margin, categorical retention, stationarity
and equivariance terms also cancel such an offset. The graph-choice term
compares assignments of the same role set, so its common offset cancels too.
Thus choice fitting does not identify an absolute per-role energy scale
that can be summed across operation charts with different numbers of roles.

The old integration added raw learned scores to each alternative. A common
positive offset then rewarded every additional role. This defect exists
even when the pointer supplies no information about the correct candidate.
The actual positive offset and unnecessary-operation selection are observed;
a replay must still test whether offset removal repairs this interpretation.

## Declared Conditional Update

For a complete public role-choice pool, let `b_i` be the existing baseline
score and `s_i` the fitted pointer score. With the existing evidence weight
`w`, execution version three supplies:

```text
u_i = b_i + w*s_i - [logsumexp(b + w*s) - logsumexp(b)]
```

This is a categorical likelihood update relative to the baseline choice
distribution. It retains every original alternative before certified mention
reduction. Within a chart, the correction is constant for each required
slot, so it preserves the corrected argument ranking. Across charts, the
partition mass of each baseline slot stays unchanged. A uniform learned
score leaves the baseline exactly unchanged; adding any common finite logit
offset leaves the update unchanged within floating-point precision.

The implementation centers learned logits before scaling to avoid subtracting
large meaningless offsets. No target, desired answer, construction identity
or held-correctness flag determines the correction. It does not tune a new
weight on the exposed request, alter the fitted weights, add a prompt, or
turn an ambiguity margin into a calibrated probability.

The CLI declares `--relation-score-policy conditional_likelihood` alongside
the batched execution mode. Raw scoring remains an explicit comparison mode.
Plans and decoder receipts bind the chosen policy and execution file hashes.
Per-role update receipts expose the complete pool count, learned center and
centered partition correction. Compact diagnostics retain each examined
chart's signature, score and margin rather than only the winning chart.

This repairs an identifiable integration defect. It does not prove that the
learned pointer recognizes the right relationship on unseen constructions,
that the parent discovers every intended graph, or that the mixture cannot
regress. Those still require public-only paired replay and later independent
transfer/control stages. G03 and later gates remain open.

Focused checks passed 88 tests in 30.44 seconds. They check per-role offset
invariance at four finite offsets and four weights, unchanged partition mass,
exact neutrality for uniform evidence, and the unnecessary-operation failure
under a synthetic uniform bias. Native fixture replay exercises the declared
conditional policy with the same selected weights and public-only inputs.
The synthetic example establishes the integration property, not real-language
success. Invalid or incomplete choice evidence is rejected.

The standard gates also passed: smoke 164 passed and one skipped in 56.89
seconds, then lint, compile, governance, layering and writing. No gate
baseline was relaxed. The corrected policy's real replay remains pending.

## Corrected Public Canary

The declared conditional policy completed the same exposed public request
with all sixteen charts in 28.725182 seconds. Parent, global chart and joint
native arms each returned the correct source-anchored program:
`sha256:d333e1fb70c5ff4e6ca71b398e83d04638b0c152e47e2f10d283a16a998f6c56`.
The joint arm had no refusal. Its selected graph contains the intended three
operations, without the extra multiply selected by raw-score execution.
The checkpoint, evidence weight and thirty-second request allowance were
unchanged. The selected graph's ambiguity margin was 5.316511; this is a
solver margin, not a calibrated probability of semantic correctness.

Frozen execution was `198696ceb`. The report receipt is
`e8d2aba9a30934cbae760abe79bd8fbf407ec7824762d1df288d990cfc9d7585`,
with evaluation plan
`776c803952cbeda98eb7ebda0cdd3f7b54101f3d41d6db96c99b2f91ed9e8a61`.
Independent artifact verification returned
`e867089d2f9fe16fd4e7138d3f1e3c1b00f0ad52a53665adbe4f891b261aa396`.
The contained terminal receipt is
`b0dbcaa7e20785c46e9f4a18fc8e133ec7ae33487bc2900345eb4d175838d457`:
106.738468 seconds, return two, no timeout, empty process group and lineage.
Supervisor 14379, child 14382 and inhibitor 14385 were gone. Profile-only
execution returns two and denies advancement even when all arms are correct.
Artifacts remain under `semantic-grounded-joint-profile-conditional-v3-20261001`.

This repairs the observed canary failure. One previously exposed request
does not establish gain, full development coverage or fresh transfer.
The complete twenty-one-request three-arm development screen was then
launched from the same frozen checkout, without a profile-source filter.
Its supervisor plan is
`982523ef7e16a4d9084e19c4dd38795b070fe402001d4b2bb5f0e3bfc3a93f50`
and command is
`4c70c64bca99afa6d179af24b85a2921300f948f0cb69997fc6024628b07b1db`.
Supervisor 15245 owns child 15248 under a 2,400-second outer bound. No fitting
stage is repeated. Results belong under
`semantic-grounded-joint-development-conditional-v3-20261001`.
G03 remains open pending its acceptance evidence.
