# Procedure dataflow in the cognitive event graph

The common executor now records actual state and context reads, declared
writes, backend identity and duration in the existing cognitive event graph.
The registered semantic runtime uses this executor. Its shadow response names
the process-local event ids separately from the deterministic semantic runtime
receipt. No second event store was added.

Each execution owns its writer map. A step that reads a prior result depends
on that result's latest writer; an unrelated earlier step is not automatically
an ancestor. Copying or enumerating the whole input mapping is recorded
conservatively. Graph records retain value digests, not raw inputs or backend
evidence. A present false value differs from an absent binding.

This establishes observed dataflow, not semantic necessity. An event says
`executed`, not `correct`. It does not grade a procedure, train a policy, or
claim that a returned answer means what the user intended.

Telemetry cannot turn successful backend execution into failure. Digest,
event-recording and degradation-reporting failures leave the computed answer
intact and mark trace completeness false. When a replacement writer's event
is missing, later reads do not attach to the older writer. They record the
missing provenance. Backend failures remain backend failures even when trace
recording also fails.

## Verification

The focused pass passed 167 tests in 13.08 seconds. These include real
universal-floor execution through the registered runtime, cross-backend
composition, dependency ancestry, overwritten values, request isolation,
missing bindings, injected telemetry failures and the existing event graph.
The local-value invariant is executable.

Lint, compile, governance, layering and writing passed. Smoke passed 163,
skipped one and failed the current resident-manifest activation alarm. The
failure still names `resident_manifest_drift` with no drifted bound source
files. This change did not re-seal the earlier activation or touch its model.

These are CPU integration results. No live runtime was restarted or model
loaded for this checkpoint. Independent task-outcome learning, broad language
interpretation, broad comparative measurements and current-model serving
qualification remain unproved. G09-G12 remain open.
