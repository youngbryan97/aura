# Construct training margins with the fitter's replay

The joint comparison builder summed each graph's variable scores separately.
A shared large term could erase a smaller term before subtraction. The fitter
then used compensated signed accumulation and replayed a different margin.

A regression test supplies a shared -1e16 score and a smaller relation term.
The requested comparison margin is +0.25. Before the repair, construction
followed by replay produces -0.4431471805599453, reversing its sign.

Both joint and relation-only comparison constructors now calculate their
variable contribution through the existing graph_margin evaluator. The fixed
remainder uses compensated signed accumulation too. Both constructors replay
the requested +0.25 in the regression. Fifty-one tests pass across joint
learning, relation learning and batched graph evaluation.

This repairs a constructed numerical mismatch, not proof of a transfer gain.
It cannot recover precision already lost in upstream runtime graph scores.
The earlier frozen diagnostic at 18fb42274 did not contain this change; its
last progress file reports first-step cut round 23 and no candidate was
published. Its process is no longer present. No completed result is inferred.
