# Native-thinking coding diagnostic

Source `beab1bf34340de0d99af32bdc8fbd8b3ae7ce1c7` ran six coding tasks on
the signed resident 27B persona model. The task seed was 2026091402, with two
tasks at each of three difficulties and five counterbalanced arms. The run
used one exclusive standalone model lease and finished in 1,218.703 seconds.
The lease was released. This was not a desktop chat test.

| Arm | Correct | Parsed | Mean decode seconds | Mean generated tokens |
| --- | --- | --- | --- | --- |
| Ordinary model, native thinking | 6/6 | 6/6 | 161.827 | 2,129.833 |
| Treatment | 6/6 | 6/6 | 10.144 | 100.667 |
| Matched wire | 0/6 | 6/6 | 8.995 | 87 |
| Coefficient lesion | 0/6 | 6/6 | 8.263 | 64 |
| Wrong state | 0/6 | 6/6 | 12.222 | 100.667 |

All 30 decodes completed, with one attempt each and no token-limit censoring.
Independent replay reproduced the counts, six treatment-state receipts, and
the 32-event journal chain. There were zero accuracy gains and zero regressions;
the paired exact p-value is 1.0. Admission is false.

The ordinary model returned correct state values and complexity on every case.
One used Theta notation, accepted by the versioned semantic grader. The matched
wire arm skipped native reasoning and made real value errors. It must not stand
in for ordinary model capability. Treatment read computed state; the coefficient
lesion read wrong zero state. These interventions establish dependence of the
serialized answer on state, not an accuracy improvement over native reasoning.

The observed decode latency ratio is about 16, but treatment state computation
is outside that timing. The arms do not use equal reasoning computation, this
small cohort is development data, and it spans only one coding construction.
It establishes neither broad reasoning gain nor frontier performance.

## Retained evidence

`artifacts/closeout/latent_cortex/native_coding_diagnostic_20260914/` contains
byte-identical result and journal copies, plus independent verification.
Result SHA-256:
`6793fab8ec40e911b8ca4b1f7db037e077359106dc113c505a5ad996458fb15e`.
Verification receipt:
`a535879330d6de6426503e6688794668597b4bdb6218ac8bc72245e882e5779a`.

The verifier now permits an explicit negative-result audit. It checks the same
source identity, public outputs, journal, and state replay, but reports
`integrity_verified=true`, `verified=false`, and `admitted=false`. Its default
qualification mode still refuses this result. Negative evidence must remain
replayable without becoming serving authority. G06 and G08 remain open.
