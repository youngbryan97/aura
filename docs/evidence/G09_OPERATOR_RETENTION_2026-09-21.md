# Retained operator semantics

The operator search added in the prior checkpoint reused installed floor
terms, but `what_she_gave_meaning` did not save them. The sequence trial also
kept a smaller rollback snapshot than the developmental action wrapper. An
operator could therefore disappear on restart or survive a rejected trial.

The existing language store now retains operator recipes in installation order.
It uses the floor's data reader, never pickle or Python evaluation. Restoring
the recipes reconstructs the callables and rollback lineage. It does not add
successful verdicts or pretend that recall is a new qualification. A batch with
an unreadable term, conflicting identity, or invalid lineage installs nothing.
An opaque Python callable prevents the new save from replacing the old one.

Sequence trials now use `what_she_can_take_back`, as their action wrapper does.
Operator records are immutable. Rolling back an ancestor removes subsequent
rollback snapshots, so a later rollback cannot resurrect its descendants.
Saving an empty language replaces a previous durable snapshot; deleting its
last entry no longer leaves an old copy available at the next boot.

The novelty test had a quantifier error. It rejected a candidate when, for each
probe, some existing operator matched. It now rejects when one existing
operator matches the complete output vector. Constant-zero and constant-one
operators do not refute identity merely because the probes are zero and one.
Admission also refuses an existing name and rechecks the installed operator
set before committing a candidate evaluated outside the lock.

## Checks and limits

The combined regression run passes 155 tests. Smoke passes 164 with one skip;
lint, compile, governance, layering, and writing pass. Combining the suites
exposed leaked proposer state and a removed addressing word. The shared
snapshot now also covers the four mutable search/decision terms, and the
sequence-answer fixture restores the complete developmental state. These
were order-dependent defects, not discarded retries.

`tests/test_operator_retention.py` covers recipe recovery, idempotence,
lineage, atomic rejection, name conflicts, empty-state persistence, complete
rollback, immutable records, the novelty quantifier, and fresh-process recovery.
The existing conductor already recalls the joined knowledge store at boot and
calls its keeper through a worker thread every 300 seconds. This change extends
that store; it adds no scheduler or second persistence system.

This establishes retained component behavior, not a resident-app result or a
broad reasoning gain. No G checkbox is closed. Fresh cross-family acquisition,
selection, cost-matched lesion/rescue, and independent outcome measurements
remain required. The prior structural compression count is not a claim about
Kolmogorov complexity or measured runtime savings.
