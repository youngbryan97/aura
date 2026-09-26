# Real prefix grouping measurement

The frozen role-relative batch-four fit refused prefix equivalence before
any optimizer update. Its plan and supervision remain intact. A separate
model-active probe reused those exact tokens and the same resident checkpoint.
No thresholds were changed.

| Prefix group | Full split logit error | Single-row supervised error | Single-row inference error | Accepted |
| --- | ---: | ---: | ---: | --- |
| 1 | 0 | 0 | 0 | yes |
| 2 | 0 | 0.06014251708984375 | 0.07048988342285156 | no |
| 4 | 0 | 0.03527069091796875 | 0.04105186462402344 | no |

The existing supervised tolerance is 0.015625. The full split is exact at
each grouping; prefix grouping and subsequent single-row projection are
different numerical paths on the quantized checkpoint. Cached first-row
hidden values differ by up to 7.0 and 5.5 for groups two and four respectively.
These are measured differences, not evidence of worse semantic answers.
The training path must nevertheless reproduce its inference computation.

Probe directory:
`~/.aura/rlc-evidence/semantic-native-prefix-grouping-probe-20260926/`.
Receipt: `5c3343be659782d93fee28a8c7ca423d5201cc78fff747ee82adc9a145e7fa7c`.
Elapsed: 28.01143129099364 seconds after model admission.
The receipt digest was independently recomputed from its fields.

The next role-relative fit uses single-row capture. Its natural requests,
supervision, calibration coverage, selected-logit gate, optimizer schedule,
and token limits are unchanged. The trainer now saves a failed equivalence
receipt before refusing, including the actual differences. The probe and
trainer checks pass 42 focused tests. Neither batching nor this repair grants
an accuracy, transfer, serving, or fusion claim; G03-G05 remain open.
