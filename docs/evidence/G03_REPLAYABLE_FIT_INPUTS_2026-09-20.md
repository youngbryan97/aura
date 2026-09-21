# Preserve the numerical problem before the first update

Two full-retention diagnostics ended without an accepted-update checkpoint.
Their progress files identified repeated first-step refinements, but did not
retain the comparison arrays needed to reproduce that numerical problem.
Repeating mining to investigate the same fit was avoidable work.

The existing checkpointed graph fitter now writes a checksummed problem
archive before attempting its first projection. It preserves parameter blocks,
all retained relation/operation/argument evidence, shared references, and fit
options. Loading uses numeric NPZ arrays and an explicit evidence-class list;
it does not load pickle or import class names supplied by the archive.

The archive is diagnostic input, not accepted model state or qualification.
Its identity remains bound to the original fit. A different implementation can
explicitly replay those inputs for an experiment without relabeling the old
measurement. Existing accepted-update checkpoint validation is unchanged.

Thirty-two tests pass across problem replay, checkpoint integrity and
minimum-change fitting. They verify identical values and gradients, replayed
stored margins, shared evidence identity, checksum rejection, preservation of
the old archive after a rejected resume, and survival when the first projection
is interrupted. G03 remains open; this makes its unresolved fit reproducible.

Repository gates also pass: smoke 164 passed and one skipped; lint, compile,
governance, layering, and writing checks passed. Consolidating archive writes
removed one effect-ownership debt site rather than adding a new writer.
