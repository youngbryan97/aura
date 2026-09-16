# Reusable procedure domain contract

The first full reuse measurement completed all 500 validation interpretations
and 15,860 distinct fresh-value probes in 192.40 seconds. It retained 471
proved-equivalent interpretations and registered 283 unique procedures.
Receipt: `c4d2327391bfd757e29c31776485904f42fd0c01711b08841da7dd3c9f0321f1`.
The immutable result is at
`~/.aura/rlc-evidence/semantic-procedure-reuse-dev-20260915/report.json`.

Its v1 evaluator reported 2,593 lowering failures and 3,183 task failures.
All recorded execution exceptions were zero-division or sequence-domain
rejections. Counting every exception as a lowering failure did not establish
that the floor disagreed with the reference program: the reference also has
partial operations. The original receipt is retained, not overwritten.

The v2 evaluator compares values and domains separately. A `Stuck` result
matches reference undefinedness only when the reference really is undefined.
It never counts as a successful task answer. Runtime failures, invalid
contracts and exhausted computation are not converted into domain agreement.
A rejected prediction still fails a task whose target has a defined answer.
All probes remain in the recorded cohort; defined-task coverage and matching
domain rejections have separate counters.

A complete small boundary grid also exposed three real compiler mismatches:
the reference rejects empty `total`, `sorted_up`, and `unique`, whereas the
floor returned zero or an empty sequence. Their floor lowering now preserves
the existing partial reference domain. No primitive was changed to fit the
test data, and the historical result is not relabelled.

Verification: 80 focused tests passed across reuse, floor execution, common
procedure currency and semantic counterexamples. This includes 102 boundary
cases spanning every shipped semantic primitive, empty sequences, zero and
signed integers. A full v2 measurement is still required. This does not close
S06, G03 or broad transfer.
