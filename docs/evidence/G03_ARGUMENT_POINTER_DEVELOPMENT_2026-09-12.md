# G03 Argument Pointer Development

This continues the semantic compiler programme. It does not measure repeated
backbone layers, frontier reasoning, runtime fusion, or live activation.

## Calibration Rejection

The operation-pointer candidate reached 377/500 executed answers on original
source validation against 346/500 for the frozen parent, with 24 regressions.
The existing shared path-quality calibrator was then fitted on those source
validation outcomes and evaluated on the original 500 source-test examples.
Those test examples are now consumed for calibration, not fresh replication.

The challenger scored 143/250 on the tuning half against 138/250 for the parent,
and 162/250 on the admission half against 146/250. The quality scorer failed
independent calibration: expected calibration error 0.18069311. Its Brier score
was 0.26779568 against the fit-constant baseline 0.272077. No selector or serving
package was admitted. The rejection and individual outcomes are retained in
`artifacts/rlc/semantic_program_clean_pointer_dev_20260912/`, hash-bound by
`path-calibration-evidence-receipt.json`.

A separate gold-span diagnostic compared prefix-eligible and all-type-eligible
register competition on 12 source-validation examples in each of four families.
No previously correct decision was lost to an added competitor. Arithmetic
remained 29/48, fork/join 53/72, and role binding 40/48; cataphoric decisions
improved from 36/48 to 47/48 with type eligibility. This pilot does not establish
runtime accuracy, since it supplies gold reference and definition spans. It
does not support expanding relation training as the next repair by itself.

## Argument Pointer Repair

The operation-only repair left the argument pointer trained under contradictory
labels. The existing proposal refit now accepts an explicit pointer-refit mode.
It rebuilds that pointer from source training, refits dependent proposal heads
against the new proposed spans, and calibrates their scale on source validation.
Operation recognition, register relations, input grounding, and graph rules
remain unchanged. The default proposal-only mode retains its previous behavior.

The CLI still verifies the exact original training and validation source hashes.
The new receipt records the parent, split hashes, corrected label policy, and
dependent-head refit. Test examples do not enter optimization or calibration.
Neither the refit nor its receipt grants serving authority.

Focused checks: 23 passed, including a real small-data fit showing identical
coefficients when an already-clean model is refitted and identical receipts
with test examples omitted. Smoke: 164 passed, one skipped. Compilation, Ruff,
and writing gates passed. The full source refit and evaluation are pending at
this entry; G03 remains open.

## Completed Evaluation

The full refit completed with 728 training and 500 validation examples. Its
dependent proposal fit used 3,296 positive and 52,736 negative rows; the selected
proposal scale remained 0.875. Only argument-pointer and proposal-head
coefficients changed relative to the operation-pointer parent.

The candidate is rejected. It produced 47/48 weave programs and answers, losing
one answer relative to the operation-only candidate. On 500 source-validation
examples it produced 342 correct answers against the frozen parent's 346, with
44 gains and 48 regressions. The operation-only candidate had scored 377.

| Source family | Examples | Frozen parent answers | Argument refit answers |
| --- | ---: | ---: | ---: |
| Arithmetic | 128 | 97 | 82 |
| Cataphoric | 48 | 4 | 15 |
| Fork/join | 192 | 155 | 135 |
| Role binding | 48 | 24 | 37 |
| Reserved alias | 48 | 30 | 37 |
| Natural source | 24 | 24 | 24 |
| Natural alias source | 12 | 12 | 12 |

A 24-example source pilot isolated pointer and proposal coefficients. On its
12 arithmetic examples, the combined refit scored 9 answers, the new pointer
alone 11, and new proposal heads alone 12. Reducing the combined candidate's
pointer score scale to 1e-6 still scored 9. All four variants scored 12/12 on
the pilot's fork/join examples. These selected pilot results cannot establish
global admission or identify one exclusive cause. They show that suppressing
the pointer score alone does not recover the observed arithmetic loss.

The candidate and outcomes are hash-bound by
`argument-pointer-evidence-receipt.json` in the evidence directory above. The
artifact is retained outside the source tree; no serving authority was granted.
The next development comparison uses the stronger operation-only candidate.
The corrected sampler remains the training implementation, but label
consistency alone has not solved source-general argument binding. G03 is open.

## Remaining Definition Localization

The existing definition-relation diagnostic was run on the operation-only
candidate's 123 source-validation failures. It supplies gold operation and
reference spans; these counts are diagnostic links, not runtime answers.

| Family | Reference links | Learned definition top-1 | Gold definition top-1 |
| --- | ---: | ---: | ---: |
| Arithmetic | 124 | 39 | 75 |
| Cataphoric | 120 | 49 | 95 |
| Fork/join | 258 | 83 | 201 |
| Role binding | 32 | 17 | 27 |
| Reserved alias | 44 | 25 | 33 |

The definition pointer chooses a single register span before relation scoring;
its score scale of zero does not disable that selection. These results support
testing definition localization separately from relation-tissue retraining.
Gold-definition errors remain, so localization cannot explain all failures.
The report is `remaining-source-relation-diagnostic.json` in the same artifact
directory, SHA-256
`d286a1240ccc2c8bbdcf9c49f3dbad226d053efcf917d0f717b07f7fd0ea6629`.

## Definition Supervision Coverage

The follow-up coverage diagnostic found every arithmetic, cataphoric,
fork/join, and reserved-alias gold definition among the preselection candidates.
Their targets are anchor-based, and all those examples are excluded by the
symbolic-only definition-training filter. Exact boundary selection was 144/640,
48/240, 0/1,344, and 20/240 respectively. These boundary counts do not imply
equivalent answer failure rates. Role binding had 192/240 gold definitions
present and 27/240 selected. Its examples also fail the all-registers-symbolic
filter, despite containing a mixture of anchor and separate-name targets.

The opt-in definition-pointer refit now uses every source training example,
without retraining the relation coefficients or changing their scale. Its
receipt separates training from validation contract checking; no validation or
test rows enter pointer optimization. Full-data candidate measurement remains
pending. The executable unit checks include mixed anchor/name supervision.

Fresh fitting also exposed a module-extraction regression: the relation-tissue
module lacked its register-definition helper, training constants, and `math`.
These dependencies are restored through the existing deferred-import pattern.
The real small-data fit now passes alongside the new pointer tests: 21 focused
checks passed. Smoke passed 164 with one skip. No G03 closure is claimed.
