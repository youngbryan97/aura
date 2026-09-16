# Fresh-source retained-constraint result

The complete development comparison evaluated 500 validation examples, with
zero test examples used. The incumbent scored 472 exact/equivalent programs;
the constrained refit scored 458. There were seven gains and 21 regressions.
The existing zero-regression selection rule retained the incumbent. No serving
configuration changed and G03 remains open.

## Constraint boundary

The fit retained 9,337 witnessed training inequalities. None of the initially
positive retained margins became nonpositive. However, 26 wrong-or-tied
constraints remained wrong-or-tied after all 20 accepted steps. The minimum
margin remained -19.361361622810367. The optimizer reported
`search_budget_exhausted`, not satisfaction or proven infeasibility.

Retaining training inequalities does not guarantee correct fresh decoding or
validation non-regression. The latent choices used for each inequality were
frozen during its update. These distinctions must remain separate in reports.

The existing path ensemble retains incumbent and challenger outputs, but its
selector must choose correctly. An oracle union of these two validation
outputs would reach 479/500; 21 cases are missed by both. That union uses
validation labels and is a diagnostic ceiling, not an executable selector or
an achieved score. Improving selection alone cannot close this run's gap.

## Artifacts

- Directory: `/Users/bryan/.aura/rlc-evidence/semantic-constrained-fresh-20260915`
- Candidate receipt: `5fcbf1e315b3d5d749a908050df2bbddabb9793201c99136edf451c324585db6`
- Incumbent receipt: `c3817f75a0b9e8b72c601fbbd97c0494dca423bcb30eac9683b871831701d15a`
- `validation.json` contains the complete paired result and selection.
- `candidate.json` retains the constraints, witnesses, and optimization trace.
