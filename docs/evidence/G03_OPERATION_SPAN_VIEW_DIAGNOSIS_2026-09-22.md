# G03 operation-span view diagnosis

The role-alias fit left two frozen validation role swaps wrong. A source-chart
diagnostic initially appeared to contradict the candidate bank: when supplied
the annotated operation spans, the correct graph beat the highest witnessed
wrong graph by 9.27 and 9.30 score units. That chart is not what ordinary
decoding selected. In the eight-chart answer-blind bank, both selected programs
used shorter operation spans than the source annotations, and neither exact
annotated-span chart appeared. Both banks still reached the correct program.

The diagnostic then rebuilt the first two answer-blind charts, checked their
operation spans against the frozen bank receipt, and only afterward restricted
the graph to the source arguments. Both charts used the source operation types.

| Frozen validation source | Annotated-span margin | Runtime chart 0 margin | Runtime chart 1 margin |
| --- | ---: | ---: | ---: |
| `29d14639...` | +9.27 | -5.98 | -4.32 |
| `805a4c74...` | +9.30 | -8.43 | -6.62 |

Positive margin favors the source binding; negative margin favors a witnessed
different program. On chart 0, the correct-minus-wrong role-factor differences
are -2.80 and -3.90, and the proposal-factor differences are -4.03 and -5.18.
Relation and pointer factors are much smaller on these two charts. The source
chart's positive margin therefore cannot justify changing relation scales or
adding more equal-valued training examples.

The current role and proposal fits use annotated operation spans. Runtime
operation recognition can choose a shorter span with the same operation type,
changing the feature vector seen by those heads. The two observations show
that this view mismatch is sufficient to reverse the local ranking. They do
not prove that it explains every G03 miss. The next candidate should train
against source-labeled, answer-blind runtime operation hypotheses and compare
complete held-out graphs without supplying target labels to decoding.

The diagnostic tool now supports split- and source-bound read-only attribution.
It rejects fitting or capacity-witness selection on validation and test rows.
No fitted candidate or serving change follows from this diagnostic.

Receipts: `~/.aura/rlc-evidence/semantic-role-alias-graph-factors-20260922/`.
