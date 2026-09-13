# G03 full-source proposal repair: null result

The exact-source refit completed in 336.70 seconds using 728 training and
500 validation examples. All seven family manifests and both split hashes
matched the frozen parent. It produced 3,296 positive and 52,736 negative
proposal rows and selected scale 0.875.

Candidate receipt:
`a22cb379b2a683ddb185331c438503f00ab2dbffb510c4922f35e81f3d789a5a`.
Every serialized coefficient group is identical to the parent. Only the
training receipt differs. The clause-candidate training repair therefore
does not alter this model: the retained hard-negative shortlist is unchanged.

The complete exposed weave replay produced 21/48 exact answers and 19/48
exact programs in 248.53 seconds, matching the parent. The four-minute bound
was checked between examples; the last in-flight example completed beyond it.
No task was omitted. This is a development replay, not fresh replication.

A separate gold-operation diagnostic found all 480 expected argument spans
in the runtime proposal banks. All ranked within the first 128 pointer
candidates; 177 ranked within the first 16. Gold operation spans were supplied
only to this diagnostic, so it does not measure autonomous chart selection.

The next scoring issue is concrete: `_select_argument_proposal_scale` optimizes
binary cross-entropy of summed raw role/proposal logits. Runtime instead sums
weighted log-sigmoid role/proposal/pointer scores and conditional relation
evidence, followed by constrained graph selection. These are different
objectives. The null refit does not establish which replacement objective will
generalize; that requires a separately measured candidate.

G03 remains open. No serving activation, fusion, broad reasoning or frontier
claim follows from this result.
