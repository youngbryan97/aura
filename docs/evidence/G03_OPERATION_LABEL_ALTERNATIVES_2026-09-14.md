# G03: retaining operation label alternatives

The operation proposer previously retained only the winning class for each
span. An opt-in, receipt-bound policy can now retain more of the learned
distribution. The classifier's mean distribution and legacy winner are
unchanged. Chart search keeps span-label alternatives distinct and never
selects overlapping interpretations as separate instructions.

Two complete source-development trials retained two labels per span. Both
scored 459/500, with zero gains and five regressions against the 464/500
feasible-operation parent. The ordinary chart policy took 104.01 seconds;
preserving arity states took 265.95 seconds. Neither candidate is promoted.
The default remains one label. More proposal coverage did not improve the
selected programs in these trials.

The 25 focused tests cover classifier distribution equivalence, legacy
selection, distinct alternatives, non-overlap, receipt identity, invalid
policies, and runtime consumption. Lint, compile, governance and layering
passed. Smoke reported 163 passed, one skipped, and the existing
resident-manifest activation alarm.

Full row reports and candidate hashes are retained under
`artifacts/rlc/semantic_operation_labels_dev_20260914/`; the receipt names the
local candidate files. These are exposed development results. Fresh transfer,
public gain, current-model serving and G03 closure are not established.
