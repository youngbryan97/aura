# G03 source-only proposal refit

Development diagnostic only. G03 remains open; no serving package changed.

Training previously built global argument candidates while runtime merged
global and per-operation clause candidates. The training builder now reuses
the runtime proposal builder. A targeted refit API preserves all other heads
and records its parent, source split hashes, calibration and new coefficients.
New calibration and coefficients are published atomically in the receipt.

The frozen weave parent was refitted using `semantic-natural-alias-source-v1`:
24 training examples and 12 validation examples, with zero test examples used.
The source manifest is
`363dde7ba161f02720078dd6722cacff747bbe0540c93c4da1524b96a5bf613d`.
Refitting took 4.44 seconds. Validation cross-entropy selected proposal scale
1.5, compared with the parent's 0.875. No language model was loaded.

On the separate 12-example source test split, both parent and candidate scored
12/12 exact programs and answers. This establishes only source preservation.

On the already exposed `semantic-natural-weave-replication-v2` validation
split, the candidate accepted 2/24 and scored 1/24 exact programs and answers,
taking 163.10 seconds. The remaining test split was interrupted after this
failure; no complete 48-example result is claimed. The candidate is rejected
for promotion. Source row-level calibration did not predict deeper composition.

Seven focused proposal tests passed, including train/validation separation,
test exclusion, coefficient preservation and changed-scale receipt reload.
The combined proposal, qualification and shadow suites passed 20 tests.
Smoke passed 164 tests with one skip; touched Python files passed Ruff and
`git diff --check`. These checks do not establish general reasoning gain.
