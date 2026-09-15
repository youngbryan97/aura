# G03 complete-graph relation learning canary

The existing low-rank definition-relation projections now accept training
from witnessed complete-program errors. The decoder retains its selected
mention, definition and competing-definition evidence. The gradient includes
the moving maximum used by the runtime categorical score. Base weights,
base bias, pointer scale, operation heads and proposal heads remain unchanged.

The refit mines a new set of complete-graph contrasts each round and trains
against all retained pairs. Only training sources enter the objective. The
existing parent-bound source loader and candidate serializer own admission
and export. The CLI objective is `graph_relations`.

## Canary measurement

Four previously identified training sources were inspected. Three supplied
different-output witnesses; the fourth supplied equivalent arithmetic graphs
and no negative within its four-graph allowance. Fifty optimizer steps on the
three witnessed pairs produced these correct-minus-incorrect margins:

| Pair | Before | After |
| --- | ---: | ---: |
| Wrong reversed binding | -2.3021699339 | 5.7132480175 |
| Second witnessed graph | 8.9547154954 | 8.7025394141 |
| Third witnessed graph | 4.8049443136 | 5.4888748648 |

The regularized pair loss fell from `0.8019342629633432` to
`0.02551330343055013` after storing the projections in float32.

The candidate then rebuilt all four argument charts and selected a computation
proved equivalent to each source program: three structural symmetries and one
integer polynomial identity. The comparison used no expected final answer in
the decoder. Operation identities and spans were supplied by training
annotations, so this does not measure operation recognition or free decoding.

[Raw receipt](../../artifacts/rlc/semantic_graph_factor_dev_20260915/relation_graph_replay_canary.json)
contains the execution witnesses, fitted margins and rebuilt search results.
The diagnostic exports no serving candidate and changes no live runtime.

## Checks and limits

Tests cover analytic gradients against central differences, replay of the
runtime score formula, selected latent evidence, storage precision, preserved
base heads, source/test separation and candidate serialization. The focused
123-test set passed; 15 source-isolation and relation tests passed after the
CLI isolation change. Final smoke passed 164 tests with one skip in 61.15
seconds. Lint, compile, governance-lint and layering passed.

After the replay, a whole-class certificate was added for connected integer
addition-only or multiplication-only trees with exact single use of every
input and nonsink intermediate. Associativity and commutativity prove every
feasible graph has the same output, so negative mining skips that class after
target reachability is established. Tests reject the certificate for reused
or omitted inputs and for subtraction. The optimization changes training
search cost, not runtime scores or validation metrics.

This establishes that the old three-scalar limitation can be crossed by
training the existing relation tissue. It does not establish a 500-row gain,
fresh transfer, whole-operation graph learning, runtime qualification, or
frontier performance. S03/S04 and the master G03-G12 obligations remain open.
