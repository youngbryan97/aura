# Structural checks survive procedure feedback

The shared procedure executor can validate structural type declarations before
calling a backend. Its task-feedback wrapper did not accept or forward that
option, so a caller using outcome learning could not request the same check.

`execute_valued_procedure_plan` now forwards `closed_types` to the common
executor. Invalid mode values fail before opening an outcome receipt. An
unsupported declaration fails before backend effects; its attempted execution
remains pending and does not become a measured failure or success by itself.
Existing nominal-type callers retain their default behavior.

Thirty-eight focused tests pass across task feedback, alternative plans and
closed procedure types. A deliberately incorrect but well-typed computation
still executes and remains ungraded. Structural validity is not correctness.
The existing external outcome assessment remains the learning authority.

This closes the wrapper mismatch, not G09. The feedback API still needs a
production owner that supplies task-grounded alternatives and independently
observed outcomes; these component tests do not establish broad reasoning gain.
