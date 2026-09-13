# G03 Definition Supervision

The source-complete definition-pointer refit is rejected. It scored 407/500
source-validation answers against 346/500 for the frozen parent, with 97 gains
and 36 regressions. Exposed weave answers fell from the operation-only
candidate's 48/48 to 12/48 (9 exact programs). No serving change was made.

The candidate and results are retained under
`artifacts/rlc/semantic_program_clean_pointer_dev_20260912/`, bound by
`definition-pointer-evidence-receipt.json`.

## Correction To The Definition Diagnosis

The preceding diagnostic's gold definition targets include fallback anchors.
`project_register_definition_spans` substitutes literal and operation spans
when a corpus example has no explicit register-definition annotations. The
feature bundle retains those substituted spans without their origin. Therefore
an anchor target does not establish that the source has no symbolic name.

The weave generator writes input names and intermediate alias clauses but does
not populate `register_definition_spans`. All 48 retained examples carry
projected anchor fallbacks. The earlier description of these targets as
anchor-based supervision must not be read as independently annotated semantics.
The localization comparisons remain reproducible diagnostics of those targets,
not a proof that choosing every supplied span would recover source meaning.

A source-trained binary probe classified anchor versus non-anchor targets from
the pooled anchor hidden state. It used 4,024 training registers, including 456
non-anchor labels, and no validation outcomes during fitting. On weave it
predicted 522/528 registers as non-anchor while every supplied target was an
anchor fallback. This is not evidence that the probe lacks semantic competence;
the comparison is confounded by missing definition annotations. The probe is
not admitted or integrated.

The next repair must preserve explicit versus fallback annotation provenance
and provide real definition supervision before using these labels to train a
selector. Historical bundles and verdicts remain unchanged. G03 remains open.

## Explicit Annotation Replay

The weave builder now offers explicit input-name and intermediate-alias spans.
Its default preserves the historical corpus. The annotated variant has a
separate materialization kind. Feature record v3 stores annotation origin;
v1 and v2 remain readable without inventing provenance. Training conversion
retains the origin, and the relation diagnostic reports it.

CPU reprojection checked every source hash and token ID against the frozen
48-example bundle. It attached annotations under a separate receipt without
changing the hidden states, original records, or split membership. This is
exposed development evidence, not fresh replication or training.

With explicit definitions, both candidates recover 470/480 argument links.
Runtime-selected spans recover 450/480 for the operation-only candidate and
286/480 for the rejected definition refit. The refit changed definition
selection, not relation coefficients. Evidence is in
`explicit-definition-diagnostic.json`, SHA-256
`5890f47d15ca094e8756946a65bb412d0fcb8ca43ebc0a07296807191b6b93f6`.

Focused tests: 101 passed. Smoke: 164 passed, one skipped. Compile,
governance-lint, layering, and writing pass. The aggregate lint gate reports
69 findings outside this patch, including import placement after module
extractions. This checkpoint does not claim aggregate lint closure.
