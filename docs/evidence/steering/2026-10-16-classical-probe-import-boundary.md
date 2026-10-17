# Classical probe import boundary, 16 October 2026

The existing CPU campaign envelope prohibits model workers, requests CPU
vision and requires a measured software browser driver. Worker admission
alone does not prevent a model backend from loading inside the runner.

`core/runtime/classical_compute_boundary.py` adds an interpreter-scoped
import finder for known model backends. Admission refuses an already-loaded
backend. Ordinary imports are rejected before the backend package executes.
The runner checks that the finder remains installed and that no forbidden
module has appeared. Scoped cleanup removes only its own finder; nested
owners and unrelated import finders remain intact.

The `--cpu-only` campaign installs the boundary before campaign lease and
asset preparation. CPU preparation refuses a missing boundary. Each case
retains the actual boundary report after play, including refused imports
and loaded backend roots. The existing worker, CPU-vision and software
graphics requirements remain in force. The model-endpoint option remains
incompatible with CPU-only work.

This controls ordinary imports in this Python interpreter. It does not
police native code, external services or arbitrary child interpreters.
It changes no other running process and supplies no evidence of gameplay
success. The protected G09 model job was not changed.

Focused checks: 81 passed, two skipped. The skips require separately
requested native CPU OCR and software-browser execution. Tests used fake
backend packages whose first instruction would fail if executed; the
boundary refused them first. Other tests cover preloaded modules, nested
owners, ordinary exceptions, removal, external module insertion and the
actual CPU campaign report. No model backend was loaded by these tests.

The focused supervisor artifact is
`artifacts/flash-campaign/2026-10-16-classical-import-boundary/final-tests-supervisor.json`.
A fresh recorded CPU browser attempt follows this checkpoint. It cannot
qualify an ordinary native-app demo or the 56-game campaign.

Checkpoint gates: smoke 165 passed, one skipped; lint, compile, layering,
generated DEPS consistency and writing passed. Governance still reports
the same 131 existing regressions. Its baseline was not changed.
