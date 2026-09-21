# Reuse operation feasibility states

Profiling a retained three-operation development case found repeated signature
aggregation in the interval-chart dynamic program. Feasibility depends on the
operation sequence's arities, argument-type demand and result-type supply, not
on the source spans or coefficient scores. The search now caches those states
within one call. It retains the original ranked prefix order and every existing
per-state limit. Nothing is cached across models or requests.

Exhaustive feasibility tests and repeated-label ordering tests passed: 71 tests.
The new test requires one signature lookup per inventory node plus one per
operation in each distinct sequence, rather than per retained prefix.

On one profiled source case, chart construction took 2.860 seconds before and
0.925 seconds after. Whole profiled decode took 9.979 and 8.716 seconds. These
single profiled observations include instrumentation overhead and concurrent
host work; they are not a production throughput estimate. Both produced an IR.
The exhaustive tests, rather than this timing sample, check ranking preservation.

This implements the exact-search efficiency portion of Q20/Q41. The full
1,264-case candidate run remains frozen on the preceding implementation, so
this change does not alter its in-flight measurement identity or observations.
No G-ledger scientific obligation is closed by the optimization.
