# Joint transition-feature and span-set learning

The [prefix-differenced operation view](G03_PREFIX_DIFFERENCED_RECOGNITION_2026-09-21.md)
was integrated into the existing exact labeled non-overlap objective. Training
used all 764 source-training examples and no validation targets. The initial
head's chart-length calibration had used 16 exposed validation controls; these
remain development material, not fresh evidence. Training resets length penalty
to zero and learns labels and boundaries together against every permitted span.

The fit converged in 74 iterations and 183.369 seconds. Objective loss changed
from 9.490540356895412 to 0.6407982695281956 after float32 export. Loss reduction
alone does not establish improved decoding.

The counterbalanced pilot independently decoded the same 16 controls and all
24 known incumbent failures, without target annotations at inference. Each
arm had a 20-second solve allowance.

| Arm | Equivalent | Different |
| --- | --- | --- |
| Incumbent | 16/40 | 24/40 |
| Positive-span mean-transition head | 22/40 | 18/40 |
| Joint labeled-span mean-transition head | 29/40 | 11/40 |

The joint candidate preserves all 16 incumbent-correct controls and repairs
13 of its 24 known failures. It does not dominate the positive-span candidate
on every row. This is a selected exposed-development pilot, not a powered
transfer result, an estimate of full-cohort accuracy or a no-regression proof.

- Candidate: `c0d1f75be726c79ba1654aed201b43915d5bf4ead218c10c44c87889512940c8`
- Report: `~/.aura/rlc-evidence/semantic-transition-joint-pilot-20260921/report.json`
- Receipt: `ac57836b949ea3c3dbfd486e9b714021b259b39ec0901672041d3e8d29041e22`

The frozen plan, exact script, fitted candidate and individual observations are
retained beside the report. No test examples were used. The complete 764-train /
500-validation audit is a separate run, with no further parameter changes.

Verification: 70 focused feature, gradient, serialization and isolation tests;
68 transducer, graph-learning, background and pointer regression tests; smoke
164 passed and one skipped. Targeted Ruff, compile, layering, governance and
writing checks passed. Neither serving authority nor G03 closure is granted.
