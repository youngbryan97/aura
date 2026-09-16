# Corrected full-cohort procedure reuse

The frozen `1f2fc6c401` evaluator completed the unchanged 500-row validation
cohort through the existing procedure registry. The relation-only candidate
was unchanged. It received no target program, task answer or verifier trace.

| Measurement | Result |
| --- | ---: |
| Accepted interpretations | 500/500 |
| Proved-equivalent interpretations | 471/500 |
| Unique parameterized procedures | 283 |
| Fresh-value probes | 15,860 |
| Lowering mismatches | 0 |
| Infrastructure execution errors | 0 |
| Defined task probes answered correctly | 12,677/13,127 |
| Matched domain rejections | 2,509 |
| Task value/domain mismatches | 674 |

Domain rejections are not counted as answers. The 674 task mismatches include
450 wrong answers where the requested program was defined and 224 probes
where the requested program was undefined but the selected program returned
a value. The zero lowering-mismatch result means execution preserved the
selected program on these probes, not that selection was correct.

Report:
`~/.aura/rlc-evidence/semantic-procedure-reuse-v2-dev-20260915/report.json`

Report receipt:
`d2002523c8b6bf2ed3e9a62740e2bf1cc068e7f2d0876b68920d14cf8432c300`

Supervisor receipt:
`451fb5571b183d15ee32efa1709090837e57d0774e04249bc354cece281c37f1`

The supervisor recorded exit zero and 165.286099 seconds. Finite fresh-value
probes do not establish new-wording transfer, equivalence outside the proved
class, live authority or broad reasoning gain. S06 and G03 remain open.

## Domain witnesses during learning

The same investigation found that the graph-learning comparator discarded a
defined/undefined distinction along with infrastructure failures. It now
admits a domain witness only when the independent primitive interpreter and
the universal floor agree: a successful exact value on one side, and an
expected `Stuck` domain rejection on the other. Each witness keeps both
compiled receipts and any successful execution receipt.

Fuel exhaustion, worker errors, receipt errors and disagreement between the
reference and floor remain inconclusive. Two undefined probes do not prove
equivalence. Focused tests exercise these distinctions. The running source
retention experiment remains frozen on its original comparator; this change
does not alter that experiment or its historical receipts.
