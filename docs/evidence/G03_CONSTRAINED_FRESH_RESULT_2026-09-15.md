# Fresh-source retained-constraint result

The complete development comparison evaluated 500 validation examples, with
zero test examples used. The incumbent scored 472 exact/equivalent programs;
the constrained refit scored 458. There were seven gains and 21 regressions.
The existing zero-regression selection rule retained the incumbent. No serving
configuration changed and G03 remains open.

The candidate also enabled joint operation/argument decoding before fitting.
This paired result measures that combined change. A policy-only comparison is
needed to attribute regressions separately to decoding and learned coefficients.

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

## Source-only gradient replay

The worst retained training witness is
`a766ccdc0d8918d0e21c21403e999ffd0b0ed6563896b8d085072bc93cb7f731`.
Replaying it gave margin -19.361361622810364, zero relation gradients,
operation gradient norms below 2.5e-8, and fixed margin
-19.361361622810367. The adjustable heads could not meaningfully change its
argument-mention preference.

The opt-in `--learn-argument-heads` extension differentiates the existing role
and proposal heads using exactly the chart's conditional-log-odds or sigmoid
score. The same witness retains its original total score, but its fixed
component becomes -2.0064874455272275e-6, with argument weight gradient norms
0.62442058548023 and 0.5463680122952014. This is a repaired training connection,
not a new validation result. Pointer and other fixed terms are still fixed.

Forty-nine focused tests passed, including finite differences, runtime score
replay, retained constraints, source-only fitting, and exported-model reload.
The mode remains opt-in and has no serving authority.
Smoke passed 164 tests with one skipped. Lint, compile, layering, governance,
writing and documentation-drift checks passed.

The argument-enabled fit now computes a step from the linearized weighted
deficits, then checks every protected constraint and actual loss after float32
storage. Backtracking rejects unsafe steps. This replaces the fixed maximum
parameter step only in the new opt-in mode. Fifty-eight focused tests passed,
including a 20-point linear deficit repaired within three updates and
conflicting-objective retention under both step policies.

`--compare-fit-start` adds the unchanged pre-fit decoder as a separate
validation candidate. Its purpose is to measure the policy/weight distinction,
not to choose answers with validation labels at runtime.
