# CPU OCR and short play budgets

A software browser driver alone does not constrain native OCR. Both Apple
Vision request paths previously left device choice to Vision. Apple documents
`usesCPUOnly` as an explicit CPU requirement, with a default that permits
GPU acceleration:
[VNRequest documentation](https://developer.apple.com/documentation/vision/vnrequest/usescpuonly).
The API is deprecated, so unsupported requests must not silently ignore it.

`AURA_VISION_CPU_ONLY=1` now sets the requirement and checks the request's
acknowledgement before its handler can execute. Missing setters, getters or
acknowledgement raise before processing. The default remains the platform's
choice. Both pixel OCR and screenshot OCR use the same helper. The archived
runner exposes `--vision-cpu` and retains that policy in its run identity.

The opted-in native test passed on this Mac: the request acknowledged CPU
execution and its actual OCR returned `READY 42` from generated pixels.
The combined policy, campaign-budget and imported corpus-regression set had
54 passes and one skip in 8.38 seconds. The skipped test was the separately
opted-in real browser test, already recorded in the preceding evidence log.
An initial test-collection failure used pytest's reserved parameter name;
that test was corrected before the final run.

The runner had another timing defect. It set the play budget but left the
six-minute minimum share in place; requests whose terms used that branch
could exceed a short test's declared play budget. Each case now caps the
minimum share to that budget and restores both globals after success or
failure. A fake ordinary-play entry checks a 30-second request against the
former 360-second share and verifies restoration on both paths. This bounds
the allocation handed to play; the outer supervisor still owns the process
deadline, including browser startup and cleanup.

Smoke had 165 passes and one skip in 49.38 seconds. Full lint and compilation
passed. Layering, dependency consistency, changed-method size and writing
checks passed. New native OCR receipts are under this worktree's
`artifacts/flash-campaign/2026-10-15-software-renderer/native-vision/`.

No original program was opened for these checks. No language model was
loaded or called, and no G09 control request or runtime restart was sent.
The next original-game probe can request software graphics, CPU OCR and the
classical point tracker explicitly. Such a probe still cannot establish
ordinary native Aura prompt performance by itself.
