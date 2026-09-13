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
