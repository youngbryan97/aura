# Factorized recognition: measured result

The two source-trained conditional-label variants each recover 39/54 programs
in the declared development pilot. Both retain all three source cases and
15/16 controls, with 21/35 challenge cases correct. The additional complete
span-set boundary fit changes no semantic outcomes in this cohort. Neither
variant is promoted. This failure-enriched cohort is not a generalization
estimate or the complete development replay.

The report under `~/.aura/rlc-evidence/` is
`semantic-factorized-pilot-20260921/report.json`, receipt
`c1286c422f53121c7e3e5ea7eea28cf6e828c54b28768215e04cba64ebfd4ac3`.
The conditional candidate is
`85822afbb8b62ddde7b1506751460090eb5daf984e8315fdbb267f5df4b35f3a`;
the boundary-refitted candidate is
`adcebdd72d9e1c172cd98bf5b0d0bdfa7987d7b1c6c5c778277e66802d02504e`.
The supervised run finished in 294.66 seconds with empty process lineage.

## Interventions on every remaining failure

All fifteen archived failures reproduce. Fourteen differ in operation spans;
one preserves the spans and operation identities but reverses division
arguments. Supplying annotated boundaries alone repairs all fourteen span
failures. Supplying labels in addition repairs no further case. Arguments
remain inferred in both interventions. These diagnostic repairs are not
autonomous passes and never enter coefficient fitting.

The attribution report is
`semantic-factorized-attribution-20260921/report.json`, receipt
`1d744d4f795b24abad9ba5e2abec04485064942f650c74e5ecaa236cee8da136`.
The underlying factor scores still need attribution: a selected wrong span
does not by itself establish whether its boundary evidence, label evidence,
or downstream argument score caused selection. The next diagnostic records
those terms separately rather than assuming another boundary fit will help.

## Reproducible command path

`tools/refit_semantic_argument_proposals.py --objective operation_views`
now accepts `--conditional-operation-labels`. Tests verify that it reaches
the existing fitter and is rejected with incompatible objectives. Defaults
are unchanged. Training uses the 764 source rows; sixteen exposed validation
controls calibrate the initial chart-length penalty, not classifier weights.
No test rows are used. No serving authority or G03 closure follows.

Verification: 42 focused tests passed in 8.40 seconds. Smoke passed 164 with
one skip in 86.19 seconds. Lint, compilation, layering, governance lint and
writing passed without changing baselines.
