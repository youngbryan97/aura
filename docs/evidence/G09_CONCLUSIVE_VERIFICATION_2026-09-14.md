# G09: conclusive verification follows the measured verdict

The shared `VerificationResult` could report `verdict=UNVERIFIABLE` and
`conclusively_ok=True` for the same result. Combining one successful check
with a second engine's infrastructure failure sets `checked=True`, `ok=True`,
and `infrastructure_failed=True`. The old conclusive property ignored the
last field.

`conclusively_ok` now uses the existing verdict. The change preserves `ok`
and soft scores for candidate ranking; it does not discard a candidate merely
because a check was unavailable. It prevents that candidate from becoming
conclusive correctness evidence through this property.

The focused suite checks all four verdicts, combined partial success with an
engine failure, and an unnecessary check beside a successful check. Together
with absent-check, verifier-foundry, and proof-obligation regressions, 105 tests
passed. Lint, compile, governance, and layering passed. Smoke reported 163
passed, one skipped, and the existing resident-manifest activation alarm.

This repairs a shared evidence contract. It does not establish an independent
task assessor for every procedure, close the procedure outcome-learning loop,
or prove broad reasoning gain. G09 remains open.
