# Literal grammar aliases retain their input identity

The remaining binding-only error in the 54-case conditional-recognition pilot
reverses the operands of integer division. The source says to add 30 and 6,
then divide 78 by that result. Both candidate graphs use the same operations.
The wrong graph's argument score is -4.5040195063; the target scores
-4.9534134333.

The wrong graph binds the token span for ` 78` to the computed sum. The input
anchor is the span for `78`. Only exact anchor equality restricted a literal
mention to its input register, so adding a grammar-admitted leading space
bypassed that restriction. The existing tokenizer-bound grounding grammar
already recognized both forms as the same value.

The new opt-in `token_grammar_aliases_v1` policy uses that same grammar to
bind complete, overlapping forms of a selected input occurrence to its
register. It does not merge distinct occurrences merely because their values
are equal. A larger linguistic phrase containing a value is not a literal
alias. Inputs, coefficients, and ordinary mention inference remain unchanged.
The new policy receives the source token sequence in all argument-assignment
callers: decoding, candidate search, scoring, and training diagnostics. Missing
tokens cannot silently disable it. Older receipts retain exact-anchor behavior.

The diagnostic is `semantic-division-factor-attribution-20260921/report.json`
under `~/.aura/rlc-evidence/`, receipt
`60c2fc9ab8c54a9d970d5d373601597fcc1df5293a98cb0fca92bc649b55a736`.
Its supervisor passed with exit zero and an empty descendant lineage, receipt
`b88323872857dd2d53abdaaecbcae7dd3a1a15751933f5c63fad72b711e91e1a`.
Annotations were used for attribution, not fitting or autonomous decoding.

Tests cover signed integers, sequences, empty sequences, punctuation, distinct
equal-valued inputs, nonliteral phrases, invalid anchors, receipt restoration,
runtime chart ownership, and the ordinary decode and offline scoring callers.
The occurrence check is registered as
`learning.literal_alias_occurrence_identity`. The initial focused regression
passed 181 tests in 38.33 seconds, before adding that registered canary.
With the canary included, all 182 passed in 43.11 seconds.
Smoke passed 164 tests with one skip in 50.49 seconds. Lint, compilation,
governance lint, layering and writing passed without expanding baselines.

Full development replay remains necessary. This mechanism repair does not
establish a new accuracy score, promote a candidate, or close G03.

## Completed paired pilot

The same 54 development cases were replayed in rotating order across four
arms: incumbent, incumbent with grammar identities, conditional recognition,
and conditional recognition with grammar identities. All operations and
input anchors were inferred during decoding; no coefficients were fitted.

The incumbent moved from 30/54 to 31/54 and conditional recognition from
39/54 to 40/54. Each comparison has exactly one gain, the division case above,
and no regressions. The incumbent retained all 16 controls and three source
retention cases. Conditional recognition retained its prior 15/16 controls
and three source cases. The still-failing control was not concealed.

The report is `semantic-literal-identity-pilot-20260921/report.json`, receipt
`1ec5e4208caa67168f607d244d59ddc181e7d0df64e67e6c9d1e4971a4704cea`.
Its supervisor completed in 289.188455 seconds with exit zero and an empty
descendant lineage, receipt
`ff3029537f36a5f2d1899d0b016aff3562714bc8fab8b7ce685177e21e3b7e63`.
The runtime code was frozen at `ee1a54c8cc` for this replay. Subsequent source
integration does not alter that historical identity. The full development
cohort is the next measurement; this pilot does not authorize promotion.
