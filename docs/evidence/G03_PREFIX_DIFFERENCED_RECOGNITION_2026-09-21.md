# Prefix-differenced operation recognition

## Capacity diagnostic

The unchanged operation head correctly classifies all 1,720 source-training
operation labels when given annotated boundaries. Its minimum target-versus-
competitor logit margin is 0.7706374269850048 in float64 replay of the stored
float32 parameters and features. Actual float32 prediction agrees on every
development operation. Validation is 1,162/1,192. No additional fit is needed
to demonstrate finite training-label capacity; none was performed in this
diagnostic. This says nothing about autonomous boundary recovery or unobserved
tasks.

Receipt: `e0814c8f496067778d84e5fe45d8307be2278429879e5da9a5bd74da2bbea05d`
at `~/.aura/rlc-evidence/semantic-operation-readout-capacity-20260921/report.json`.

## Controlled feature experiment

The new opt-in `contextual_mean_transition` view concatenates two normalized
vectors: the span's mean final hidden state, and its ending hidden state minus
the state immediately preceding the span. A span starting at zero uses a zero
preceding state. The concatenation is normalized. The mechanism uses neither
target labels nor construction identifiers at prediction time. It does not
claim that subtracting nonlinear hidden states recovers a context-free meaning.

All fits use the same classifier and all 1,720 training operation labels.
Validation labels do not fit the classifier. At annotated validation boundaries:

| Feature | Training | Validation |
| --- | --- | --- |
| Span mean | 1720/1720 | 1163/1192 |
| Ending-minus-prefix state | 1720/1720 | 1152/1192 |
| Mean and ending-minus-prefix | 1720/1720 | 1175/1192 |

Receipt: `a901b2f6158d0397603a0d4a980ffe53a8c178689b5d44957ab2a06f57934b75`
at `~/.aura/rlc-evidence/semantic-operation-transition-probe-20260921/report.json`.
This is exposed-development feature selection, not fresh transfer.

## Autonomous paired pilot

The existing refitter trained each head on all 764 source-training examples.
Sixteen source-identity/geometry-selected validation controls calibrated chart
length; these controls and all 24 known failures were then decoded without
supplying their target boundaries, labels or arguments. Arm order rotates per
example. The solve allowance is 20 seconds per arm. No sealed test case is used.

| Arm | Equivalent programs | Different programs |
| --- | --- | --- |
| Incumbent | 16/40 | 24/40 |
| Span-mean refit | 21/40 | 19/40 |
| Mean-transition refit | 22/40 | 18/40 |

The mean-transition candidate repairs seven known failures and regresses one
previously correct control. It is not promoted. The plain-mean refit repairs
five failures, so the combined feature must not receive credit for all seven.
Full-cohort coverage and fresh transfer remain unmeasured for these candidates.

Receipt: `34a6768e04a99cf346ea772a149437d39dee20d9f9ec4e8eda1caf386b7a3732`
at `~/.aura/rlc-evidence/semantic-transition-context-pilot-20260921/report.json`.

The joint non-overlapping labeled-span trainer now accepts the same opt-in view.
Projection and adjoint use token-level arrays instead of storing every span's
10,240-dimensional feature. Explicit-feature, finite-difference gradient,
zero-state, background-score, serialization and training-isolation tests pass.
The joint fit's outcome is a separate experiment, not included in this result.

No serving defaults or prior evidence identities change. G03 remains open.
