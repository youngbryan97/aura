# Public composition diagnostic

Measured source: `d004e08ccff63f87be8b745e63b676c980fd7b63`.
The exclusive model evaluation completed 48 decodes in 494.811 seconds on the
resident 27B persona checkpoint. The previous steering evaluation was interrupted
with Bryan's authorization; all 80 trained vector file hashes were preserved.
This was a standalone model measurement, not desktop serving validation.

Retained evidence:
`artifacts/closeout/latent_cortex/public_composition_diagnostic_20260914/`.
The result, journal, and independent verification are unchanged copies. The
separate semantic audit does not replace their original grades.

| Arm | Strict correct | Semantic correct | Semantic parsed |
| --- | --- | --- | --- |
| Treatment | 8/8 | 8/8 | 8/8 |
| Ordinary base | 0/8 | 0/8 | 6/8 |
| Matched wire | 0/8 | 0/8 | 6/8 |
| Additive lesion | 0/8 | 0/8 | 6/8 |
| Multiplicative lesion | 0/8 | 0/8 | 6/8 |
| Wrong state | 0/8 | 0/8 | 8/8 |

All 48 decodes reached a public contract or EOS. None hit the 4096-token ceiling.
The independent verifier reproduced the scores and journal chain. Its paired
one-sided exact p-value is 0.00390625; this eight-task diagnostic is not powered
broad-transfer replication.

Reading the answers exposed a scoring distinction: ordinary output often used
a JSON fence without the required final marker. The audit accepts one plain,
fenced, or marker-prefixed JSON object and ignores key order. It preserves integer
typing and rejects duplicate keys, multiple answers, extra prose, malformed
escaping, and missing fields. No incorrect value is repaired. The strict ordinary
parse count rises from 1 to 6 under this diagnostic, but exact accuracy stays zero.

## Limits that constrain the next experiment

- The query names custom operations without defining ratio-choice and ratio-band
  categories. The executor knows these definitions; an ordinary model cannot be
  expected to infer their arbitrary labels and thresholds. A reasoning comparison
  needs a public, answer-independent operation specification shared by all arms.
- Native thinking is disabled in this version of the composition protocol.
- Every task uses one construction with different literals. This does not test
  held-out constructions or natural-language binding.
- Treatment receives computed result state. Correct serialization does not prove
  hidden-state internalization, static fusion, or independent model execution.
- The two coefficient lesions failed to produce state, so their model prompts
  reduce to the matched-wire control. This demonstrates loss of the executor's
  contribution, not selective disruption inside an otherwise identical state channel.

No serving authority changed. G04 through G12 remain separate obligations.

Validation: 20 focused tests passed, including a replay against the installed
model identity and retained journal. Smoke: 163 passed, one skipped, one failed.
The failure remains the existing G10 resident-manifest drift alarm; no bound
source file drift was reported.
