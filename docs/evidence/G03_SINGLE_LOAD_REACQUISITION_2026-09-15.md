# One-load source reacquisition

The counterfactual feature bank and the older seven source banks do not share
a verified worker representation. This change reacquires their exact public
cohorts on one worker instead of granting a compatibility exception.

`tools/materialize_semantic_program_features.py --plan PLAN --model MODEL
--output ROOT` now accepts named source manifests. The existing seeded corpus
builder reconstructs each cohort and verifies the manifest, config, selected
identities and public corpus hash before loading the worker. It does not load
the old hidden-state arrays or substitute held-out examples into training.

The same collector, tokenizer and exclusive worker handle process every job.
Warmup and shutdown each have one owner. A partial cohort stops the batch,
preserves its status, and leaves later cohorts untouched. Failures preserve
the original exception even if cleanup also fails. Output paths cannot escape
the named root or overwrite a source manifest.

The partial-status path also had an actual asynchronous write defect: it sent
an async writer through `to_thread`, obtaining a coroutine without awaiting
it. It now awaits the existing governed writer directly. A busy-worker test
reads the resulting durable status and verifies that it is incomplete.

Focused verification: 38 tests passed across the materialization and new
reacquisition suites. Tests acquire two real synthetic feature bundles through
one instrumented worker lifecycle, independently reload them, and reject
source relabelling, path escapes and duplicate cohort names.

Model acquisition and counterfactual training are not claimed by these CPU
checks. This implementation advances S05; neither S05 nor master G03 is closed.
