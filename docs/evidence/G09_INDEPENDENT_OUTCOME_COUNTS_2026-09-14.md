# G09: count observations, not pending retries

The outcome ledger collapses repeated requests while one observation is
pending. Its statistics then used that request count as the observation's
weight. One resolved success after 101 retries therefore counted as 101
successes, although only one postcondition had been observed. This defeated
the purpose of collapsing repeated requests and distorted both action-value
means and their effective sample sizes.

Each measured, resolved receipt now contributes one sample to the common
action statistics. Request counts remain in the original records; no history
is rewritten. Contextual and marginal estimates use the same rule. Persisted
statistics also enforce the receipt evidence contract: unresolved,
unobserved, nonnumeric, nonfinite, and out-of-range outcomes cannot train the
value model.

Fifty-three focused tests passed. The new tests resolve one success after 101
identical pending opens, then one independent failure. Both before and after
SQLite reload, the result is n=2, mean=0.5, and squared-deviation sum=0.5.
The existing ActionValueModel assigns identical estimates to the equivalent
retried and unretried histories. Tests also exercise corrupted persisted
records through both statistic paths. No learner or ranking framework was
added.

This repairs the evidence consumed by shared learning. Independent task
assessment, durable credit for reusable procedures, and broad held-out gain
are still needed for G09 closure.
