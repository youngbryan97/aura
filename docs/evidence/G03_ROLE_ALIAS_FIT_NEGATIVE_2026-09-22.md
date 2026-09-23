# G03 role-alias fit: no gain

Two source fits used the same seven companion feature cohorts and the same
48 validation and 48 test role-binding observations. The control used 48
role-binding training examples (764 total). The counterfactual added 48
equal-valued selector/adjustment examples to training (812 total). Neither
fit received validation or test answers as training examples. Both remained
development-only.

| Role-binding measure | Control | Counterfactual |
| --- | ---: | ---: |
| Validation exact programs | 40/48 | 40/48 |
| Validation exact public values | 48/48 | 48/48 |
| Test exact programs | 30/48 | 30/48 |
| Test exact public values | 38/48 | 38/48 |

There were no paired changes in program or answer correctness on either
split. All 16 validation observations in construction 4 kept their outcome.
Equal public values sometimes hide a wrong program when two inputs have the
same number; exact program identity is therefore necessary here.

The two originally identified role swaps were still in the candidate bank.
For source `29d1463992b020356a64d415c12987beb460f1bb91e3ec54482642752f087621`,
the correct graph was 4.57 score units behind the wrong graph in the control
and 5.98 behind it after the counterfactual fit. For source
`805a4c748ab7cec8ec4552ae67b742f290bcaf748047b981e724a229129f0dc0`,
the corresponding margins were 6.79 and 8.43. The bounded bank did not
exhaust the full grammar, but these two correct graphs were reached. The
extra examples moved their ranking in the wrong direction.

This is compatible with a training/runtime definition-view mismatch. The
relation tissue fits annotated definition spans, while this candidate selects
jointly among pointer-ranked definition hypotheses. A correct graph can remain
reachable while its score falls below a role-swapped graph. Earlier
bidirectional and learned-attachment policies improved some cohorts and
regressed others; these two misses alone do not isolate a score factor. The
next candidate must train on runtime graph decisions with source-only labels,
then pass paired non-regression and fresh transfer checks.

Receipts:

- Control fit: `~/.aura/rlc-evidence/semantic-role-alias-baseline-fit-20260922/`
- Counterfactual fit: `~/.aura/rlc-evidence/semantic-role-alias-fit-20260922/`
- Paired frozen replay: `~/.aura/rlc-evidence/semantic-role-alias-paired-eval-20260922/`

No serving authority, G03 closure, or broad gain follows from this run.
