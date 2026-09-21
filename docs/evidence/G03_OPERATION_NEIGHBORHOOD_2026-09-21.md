# Source-only neighborhood diagnostic

The operation classifier was compared with nearest source-training operation
features. Queries were gold operation spans from 16 exposed validation controls
and the 24 known validation failures. No coefficients were fitted or exported.
Test membership stayed excluded. This diagnostic does not measure end-to-end
decoding or fresh generalization.

| Feature view | Controls, nearest / 40 | Failures, nearest / 48 | Controls, target in top five | Failures, target in top five |
| --- | ---: | ---: | ---: | ---: |
| Contextual mean | 35 | 17 | 38 | 34 |
| Lexical mean | 33 | 29 | 33 | 29 |
| Middle mean | 36 | 34 | 37 | 35 |
| Contextual last | 37 | 22 | 39 | 30 |

The incumbent classifier gets 38/40 control operations and 39/48 failure-cohort
operations right when supplied their gold spans. None of these nearest-example
rules improves that count. Finding the target label among five neighbors does
not supply a rule for choosing it. No retrieval candidate is promoted, and this
result does not prove that the representation cannot support a better scorer.

Immutable report and script:
`~/.aura/rlc-evidence/semantic-operation-neighborhood-20260921/`.
Receipt:
`1156a311b446626a2ff9c71806b9783f1f35b0ce88189f6eaae7aab848dedfa8`.

G03 remains open. The complete source audit still has 24 semantic selection
failures on 500 exposed validation tasks.
