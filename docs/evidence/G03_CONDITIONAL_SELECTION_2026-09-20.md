# Conditional semantic selection

## Mechanism

The first-feasible decoder ranks operation charts before seeing their argument
evidence. The raw joint decoder compares argument log odds from different local
choice pools. A shared additive offset in one pool preserves every local choice
but can change which operation chart wins.

The opt-in selector now scores each valid graph as:

```
operation_score + definition_attachment_score
  + sum_slots(selected_score - logsumexp(all_slot_scores))
```

For any slot offset `c`, `logsumexp(scores + c) = c + logsumexp(scores)`.
The offset therefore cancels. This is a product of local categorical factors;
it is not the partition function of the globally constrained graph or a
calibrated probability that an interpretation is correct. Definition attachment
keeps its existing factor. The existing typed solver still enforces graph
consistency, register use, and source anchoring.

Training differentiates the same denominator through all captured alternatives.
Its gradient is the selected feature gradient minus the probability-weighted
alternative gradients. Source-target restriction preserves the full denominator.
Runtime charts, graph-retention witnesses, saved numerical problems, and replay
share this score. The policy is receipt-bound; no default or serving lane changes.

## Measurements

One frozen replay inspected the same 16 operation charts per observation with
unchanged coefficients. Labels were used only after candidate selection.

| Selection | Selected source programs | Development validation |
| --- | ---: | ---: |
| First feasible | 16/25 | 99/100 |
| Raw joint | 17/25 | 91/100 |
| Conditional slots | 17/25 | 98/100 |

Artifact: `~/.aura/rlc-evidence/semantic-conditional-selection-20260920/result.json`.
Receipt: `e41f235c031b937a390fef9c72dc215f2c274f7359716f96731ed0f6f16de482`.
Implementation: `160a231fee7436c61e7244423db7fee8ff8d29a431c801b09cde184a158e176b`.
Elapsed: 379.907 seconds. This replay measures the formula before its training
integration. It does not establish complete search or qualification.

The preceding affine-function-coordinate fit reached 25/25 selected source
programs but regressed from 99/100 to 96/100 on development validation.
Artifact: `~/.aura/rlc-evidence/semantic-operation-function-decode-20260920/result.json`.
Receipt: `1b293bdce48f57d835b7462af3b572efcbf1657bcd16131f789a9d04b6e85caa`.
Elapsed: 247.385 seconds. That candidate is not promoted.

## Verification

The first integration suite passed 70 tests. The widened suite passed 111 tests,
including source-only training, score replay, finite-difference derivatives,
shared-offset cancellation, batched gradients, target restriction, and archive
round trips. Neither test result is an accuracy claim. Fresh source fitting and
independent paired decoding remain required. G03 stays open.
